# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import Callable, List

import numpy as np

from ...common.inference_engine.onnxruntime.main import OrtInferSession
from ...common.io import load_json
from .preprocess.utils import flatten2d_int32, flatten3d_int32
from .typings import MOSSNanoConfig, MOSSNanoInput

DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS = 0.40
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS = 0.24


class MOSSNanoModel:
    def __init__(self, config: MOSSNanoConfig) -> None:
        self.config = config
        self.model_root_dir = config.model_root_dir

        self.engine_cfg_defaults = config.engine_cfg_defaults or {}

        self.tts_dir = self.model_root_dir / "MOSS-TTS-Nano-100M-ONNX"
        self.codec_dir = self.model_root_dir / "MOSS-Audio-Tokenizer-Nano-ONNX"

        codec_meta_path = self.codec_dir / "codec_browser_onnx_meta.json"
        self.codec_meta = load_json(codec_meta_path)

        manifest_path = self.tts_dir / "browser_poc_manifest.json"
        self.manifest = load_json(manifest_path)

        tts_meta_path = self.tts_dir / "tts_browser_onnx_meta.json"
        self.tts_meta = load_json(tts_meta_path)

        self.sessions = self.create_sessions()

        self.rng = np.random.default_rng(1234)

    def speak(self, inputs: list[MOSSNanoInput]) -> List[np.ndarray]:
        all_waveforms: list[np.ndarray] = []
        all_generated_frames: list[list[int]] = []

        sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
        channels = int(self.codec_meta["codec_config"]["channels"])

        for chunk_index, input in enumerate(inputs):
            generated_frames = self.generate_audio_frames(input.request_rows)
            waveform = self.decode_full_audio_safe(generated_frames)

            all_waveforms.append(np.asarray(waveform, dtype=np.float32))
            all_generated_frames.extend(generated_frames)
            if chunk_index < len(inputs) - 1:
                pause_seconds = self.estimate_voice_clone_inter_chunk_pause_seconds(
                    input.text
                )
                pause_samples = max(0, int(round(sample_rate * pause_seconds)))
                if pause_samples > 0:
                    all_waveforms.append(
                        np.zeros((pause_samples, channels), dtype=np.float32)
                    )
        return all_waveforms

    def create_sessions(self) -> dict[str, OrtInferSession]:
        models = {
            "prefill": (self.tts_dir, "moss_tts_prefill.onnx"),
            "decode": (self.tts_dir, "moss_tts_decode_step.onnx"),
            "local_decoder": (self.tts_dir, "moss_tts_local_decoder.onnx"),
            "local_fixed_sampled_frame": (
                self.tts_dir,
                "moss_tts_local_fixed_sampled_frame.onnx",
            ),
            "local_cached_step": (self.tts_dir, "moss_tts_local_cached_step.onnx"),
            "codec_decode": (
                self.codec_dir,
                "moss_audio_tokenizer_decode_full.onnx",
            ),
            "codec_decode_step": (
                self.codec_dir,
                "moss_audio_tokenizer_decode_step.onnx",
            ),
        }
        return {
            name: self.init_session(dir / fname)
            for name, (dir, fname) in models.items()
        }

    def init_session(self, path_value: Path) -> OrtInferSession:
        return OrtInferSession(
            model_path=path_value,
            engine_cfg=self.engine_cfg_defaults,
            device=self.config.device,
        )

    def generate_audio_frames(
        self,
        request_rows: dict[str, list[list[int]]],
        on_frame: Callable[[list[list[int]], int, list[int]], None] | None = None,
    ) -> list[list[int]]:
        prefill_ids, prefill_dims = flatten3d_int32([request_rows["inputIds"]])
        prefill_mask, prefill_mask_dims = flatten2d_int32(request_rows["attentionMask"])
        outputs = self.sessions["prefill"](
            {
                "input_ids": prefill_ids.reshape(prefill_dims),
                "attention_mask": prefill_mask.reshape(prefill_mask_dims),
            }
        )

        output_names = self.sessions["prefill"].output_names
        named_outputs = dict(zip(output_names, outputs, strict=True))
        global_hidden = self.extract_last_hidden(named_outputs["global_hidden"])

        past_valid_length = sum(int(item) for item in request_rows["attentionMask"][0])
        past_by_name = {
            output_name.replace("present_", "past_"): named_outputs[output_name]
            for output_name in self.tts_meta["onnx"]["prefill_output_names"][1:]
        }

        generated_frames: list[list[int]] = []
        previous_tokens_by_channel = [
            [] for _ in range(int(self.manifest["tts_config"]["n_vq"]))
        ]
        previous_token_sets_by_channel = [
            set() for _ in range(int(self.manifest["tts_config"]["n_vq"]))
        ]

        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        generation_defaults = self.manifest["generation_defaults"]
        for step_index in range(int(generation_defaults["max_new_frames"])):
            should_continue, frame = self.run_local_fixed_sampled_frame(
                global_hidden,
                previous_token_sets_by_channel=previous_token_sets_by_channel,
            )
            if not should_continue:
                break

            for channel_index, sampled_token in enumerate(frame):
                previous_tokens_by_channel[channel_index].append(sampled_token)
                previous_token_sets_by_channel[channel_index].add(sampled_token)

            generated_frames.append(frame)

            next_row = np.full(
                (1, 1, row_width),
                int(self.manifest["tts_config"]["audio_pad_token_id"]),
                dtype=np.int32,
            )
            next_row[0, 0, 0] = int(
                self.manifest["tts_config"]["audio_assistant_slot_token_id"]
            )

            for index, token in enumerate(frame):
                next_row[0, 0, index + 1] = int(token)

            decode_feeds: dict[str, np.ndarray] = {
                "input_ids": next_row,
                "past_valid_lengths": np.asarray([past_valid_length], dtype=np.int32),
            }

            for input_name in self.tts_meta["onnx"]["decode_input_names"][2:]:
                decode_feeds[input_name] = past_by_name[input_name]

            decode_outputs = self.sessions["decode"](decode_feeds)
            decode_output_names = self.sessions["decode"].output_names
            named_decode_outputs = dict(
                zip(decode_output_names, decode_outputs, strict=True)
            )
            global_hidden = self.extract_last_hidden(
                named_decode_outputs["global_hidden"]
            )
            past_valid_length += 1
            past_by_name = {
                output_name.replace("present_", "past_"): named_decode_outputs[
                    output_name
                ]
                for output_name in self.tts_meta["onnx"]["decode_output_names"][1:]
            }
            if on_frame is not None:
                on_frame(generated_frames, step_index, frame)
        return generated_frames

    def estimate_voice_clone_inter_chunk_pause_seconds(self, text_chunk: str) -> float:
        word_count = len(
            [item for item in str(text_chunk or "").strip().split() if item]
        )
        return (
            DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS
            if word_count <= 4
            else DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS
        )

    def run_local_fixed_sampled_frame(
        self,
        global_hidden: np.ndarray,
        *,
        previous_token_sets_by_channel: list[set[int]],
    ) -> tuple[bool, list[int]]:
        audio_codebook_size = int(
            self.tts_meta["model_config"]["audio_codebook_sizes"][0]
        )
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        repetition_seen_mask = np.zeros((1, n_vq, audio_codebook_size), dtype=np.int32)
        for channel_index, token_ids in enumerate(previous_token_sets_by_channel):
            for token_id in token_ids:
                if 0 <= token_id < audio_codebook_size:
                    repetition_seen_mask[0, channel_index, token_id] = 1
        assistant_random_u = np.asarray(
            [min(0.99999994, max(0.0, float(self.rng.random())))], dtype=np.float32
        )
        audio_random_u = np.asarray(
            [
                [
                    min(0.99999994, max(0.0, float(self.rng.random())))
                    for _ in range(n_vq)
                ]
            ],
            dtype=np.float32,
        )
        outputs = self.sessions["local_fixed_sampled_frame"](
            {
                "global_hidden": global_hidden.astype(np.float32, copy=False),
                "repetition_seen_mask": repetition_seen_mask,
                "assistant_random_u": assistant_random_u,
                "audio_random_u": audio_random_u,
            },
        )
        output_names = self.sessions["local_fixed_sampled_frame"].output_names
        named_outputs = dict(zip(output_names, outputs, strict=True))
        frame_token_ids = (
            np.asarray(named_outputs["frame_token_ids"])
            .reshape(-1)
            .astype(np.int32, copy=False)
            .tolist()
        )
        should_continue = bool(
            int(np.asarray(named_outputs["should_continue"]).reshape(-1)[0])
        )
        return should_continue, [int(item) for item in frame_token_ids]

    def decode_full_audio_safe(self, generated_frames: list[list[int]]) -> np.ndarray:
        try:
            channel_arrays, _audio_length = self.decode_full_audio(generated_frames)
            return self.merge_audio_channels(channel_arrays)
        except Exception as exc:
            raise RuntimeError(
                f"full codec decode failed, falling back to incremental decode: {exc}"
            ) from exc

    def decode_full_audio(
        self, generated_frames: list[list[int]]
    ) -> tuple[list[np.ndarray], int]:
        if not generated_frames:
            return [], 0

        audio_codes, dims = flatten3d_int32([generated_frames])
        outputs = self.sessions["codec_decode"](
            {
                "audio_codes": audio_codes.reshape(dims),
                "audio_code_lengths": np.asarray(
                    [len(generated_frames)], dtype=np.int32
                ),
            }
        )

        output_names = self.sessions["codec_decode"].output_names
        named_outputs = dict(zip(output_names, outputs, strict=True))
        audio_length = int(named_outputs["audio_lengths"].reshape(-1)[0])
        return self.slice_channel_major_audio(
            named_outputs["audio"], 0, audio_length
        ), audio_length

    @staticmethod
    def slice_channel_major_audio(
        audio: np.ndarray, start_sample: int = 0, end_sample: int | None = None
    ) -> list[np.ndarray]:
        if audio.ndim != 3 or audio.shape[0] != 1:
            raise ValueError(f"Unexpected audio tensor shape: {audio.shape}")

        channels = int(audio.shape[1])
        total_samples = int(audio.shape[2])

        start = max(0, int(start_sample))
        end = (
            total_samples
            if end_sample is None
            else max(start, min(int(end_sample), total_samples))
        )
        return [
            audio[0, channel_index, start:end].astype(np.float32, copy=False)
            for channel_index in range(channels)
        ]

    @staticmethod
    def merge_audio_channels(channel_arrays: list[np.ndarray]) -> np.ndarray:
        if not channel_arrays:
            return np.zeros((0, 1), dtype=np.float32)

        if len(channel_arrays) == 1:
            return np.asarray(channel_arrays[0], dtype=np.float32).reshape(-1, 1)

        min_length = min(int(channel.shape[0]) for channel in channel_arrays)
        trimmed = [
            np.asarray(channel[:min_length], dtype=np.float32)
            for channel in channel_arrays
        ]
        return np.stack(trimmed, axis=1)

    @staticmethod
    def extract_last_hidden(hidden_states: np.ndarray) -> np.ndarray:
        if hidden_states.ndim == 2:
            return hidden_states.astype(np.float32, copy=False)

        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError(f"Unexpected global_hidden shape: {hidden_states.shape}")

        return hidden_states[:, -1, :].astype(np.float32, copy=False)
