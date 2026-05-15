# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnxruntime as ort

from ...common.io import load_json
from ...common.logger import logging
from .typings import MOSSNanoConfig, MOSSNanoInput

EXECUTION_PROVIDER_CUDA = "cuda"
SAMPLE_MODE_FIXED = "fixed"
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS = 0.40
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS = 0.24


class MOSSNanoModel:
    def __init__(self, config: MOSSNanoConfig) -> None:
        self.config = config
        self.model_root_dir = config.model_root_dir

        self.thread_count = self.config.thread_count
        self.ort_providers = ["CPUExecutionProvider"]
        self.execution_provider = "cpu"

        codec_meta_path = (
            self.model_root_dir
            / "MOSS-Audio-Tokenizer-Nano-ONNX"
            / "codec_browser_onnx_meta.json"
        )
        self.codec_meta = load_json(codec_meta_path)

        self.manifest_path = (
            self.model_root_dir
            / "MOSS-TTS-Nano-100M-ONNX"
            / "browser_poc_manifest.json"
        )
        self.manifest = load_json(self.manifest_path)

        self.tts_meta_path = Path(
            (
                self.model_root_dir
                / "MOSS-TTS-Nano-100M-ONNX"
                / "tts_browser_onnx_meta.json"
            )
        )
        self.tts_meta = load_json(self.tts_meta_path)

        self.sessions = self._create_sessions()

        self.rng = np.random.default_rng(1234)

    def speak(self, inputs: list[MOSSNanoInput]):
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

    def synthesize_single_chunk(self, request_rows):
        generated_frames = self.generate_audio_frames(request_rows)
        waveform = self.decode_full_audio_safe(generated_frames)
        return generated_frames, waveform

    def _create_sessions(self) -> dict[str, ort.InferenceSession]:
        tts_dir = self.model_root_dir / "MOSS-TTS-Nano-100M-ONNX"
        codec_dir = self.model_root_dir / "MOSS-Audio-Tokenizer-Nano-ONNX"
        return {
            "prefill": self._session(tts_dir / "moss_tts_prefill.onnx"),
            "decode": self._session(tts_dir / "moss_tts_decode_step.onnx"),
            "local_decoder": self._session(tts_dir / "moss_tts_local_decoder.onnx"),
            "local_fixed_sampled_frame": self._session(
                tts_dir / "moss_tts_local_fixed_sampled_frame.onnx"
            ),
            "local_cached_step": self._session(
                tts_dir / "moss_tts_local_cached_step.onnx"
            ),
            "codec_encode": self._session(
                codec_dir / "moss_audio_tokenizer_encode.onnx"
            ),
            "codec_decode": self._session(
                codec_dir / "moss_audio_tokenizer_decode_full.onnx"
            ),
            "codec_decode_step": self._session(
                codec_dir / "moss_audio_tokenizer_decode_step.onnx"
            ),
        }

    def _session(self, path_value: Path) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = self.thread_count
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(path_value), sess_options=options, providers=self.ort_providers
        )
        if (
            self.execution_provider == EXECUTION_PROVIDER_CUDA
            and "CUDAExecutionProvider" not in session.get_providers()
        ):
            raise RuntimeError(
                "CUDAExecutionProvider was requested, but ONNX Runtime created a session without CUDA support "
                f"for {path_value}. Actual providers: {session.get_providers()}"
            )
        return session

    def generate_audio_frames(
        self,
        request_rows: dict[str, list[list[int]]],
        on_frame: Callable[[list[list[int]], int, list[int]], None] | None = None,
    ) -> list[list[int]]:
        generation_defaults = self.manifest["generation_defaults"]
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        prefill_ids, prefill_dims = _flatten3d_int32([request_rows["inputIds"]])
        prefill_mask, prefill_mask_dims = _flatten2d_int32(
            request_rows["attentionMask"]
        )
        outputs = self.sessions["prefill"].run(
            None,
            {
                "input_ids": prefill_ids.reshape(prefill_dims),
                "attention_mask": prefill_mask.reshape(prefill_mask_dims),
            },
        )
        output_names = [
            output.name for output in self.sessions["prefill"].get_outputs()
        ]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        global_hidden = _extract_last_hidden(named_outputs["global_hidden"])
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

        for step_index in range(int(generation_defaults["max_new_frames"])):
            frame: list[int] = []
            if "local_greedy_frame" in self.sessions and not bool(
                generation_defaults["do_sample"]
            ):
                should_continue, frame = self.run_local_greedy_frame(
                    global_hidden,
                    previous_token_sets_by_channel=previous_token_sets_by_channel,
                    repetition_penalty=float(
                        generation_defaults["audio_repetition_penalty"]
                    ),
                )
                if not should_continue:
                    break
                for channel_index, sampled_token in enumerate(frame):
                    previous_tokens_by_channel[channel_index].append(sampled_token)
                    previous_token_sets_by_channel[channel_index].add(sampled_token)
            elif (
                "local_fixed_sampled_frame" in self.sessions
                and generation_defaults["sample_mode"] == SAMPLE_MODE_FIXED
            ):
                should_continue, frame = self.run_local_fixed_sampled_frame(
                    global_hidden,
                    previous_token_sets_by_channel=previous_token_sets_by_channel,
                )
                if not should_continue:
                    break
                for channel_index, sampled_token in enumerate(frame):
                    previous_tokens_by_channel[channel_index].append(sampled_token)
                    previous_token_sets_by_channel[channel_index].add(sampled_token)
            elif "local_cached_step" in self.sessions:
                local_past_by_name = self.create_empty_local_cached_past()
                local_past_valid_length = 0
                local_text_logits, _ignored_audio_logits, local_past_by_name = (
                    self.run_local_cached_step(
                        global_hidden,
                        text_token_id=0,
                        audio_token_id=0,
                        channel_index=0,
                        step_type=0,
                        past_valid_lengths=local_past_valid_length,
                        local_past_by_name=local_past_by_name,
                    )
                )
                local_past_valid_length += 1
                next_text_token = _sample_assistant_text_token(
                    local_text_logits,
                    self.manifest,
                    generation_defaults,
                    self.rng,
                )
                if next_text_token != int(
                    self.manifest["tts_config"]["audio_assistant_slot_token_id"]
                ):
                    break
                _unused_text_logits, audio_logits, local_past_by_name = (
                    self.run_local_cached_step(
                        global_hidden,
                        text_token_id=next_text_token,
                        audio_token_id=0,
                        channel_index=0,
                        step_type=1,
                        past_valid_lengths=local_past_valid_length,
                        local_past_by_name=local_past_by_name,
                    )
                )
                local_past_valid_length += 1
                first_channel_logits = self.slice_audio_channel_logits(
                    audio_logits, 0
                ).astype(np.float32, copy=False)
                sampled_token = _sample_audio_token(
                    first_channel_logits,
                    previous_tokens_by_channel[0],
                    previous_token_sets_by_channel[0],
                    generation_defaults,
                    self.rng,
                )
                frame.append(sampled_token)
                previous_tokens_by_channel[0].append(sampled_token)
                previous_token_sets_by_channel[0].add(sampled_token)

                previous_token = sampled_token
                host_sampled_channel_limit = int(self.manifest["tts_config"]["n_vq"])
                for channel_index in range(1, host_sampled_channel_limit):
                    _unused_text_logits, audio_logits, local_past_by_name = (
                        self.run_local_cached_step(
                            global_hidden,
                            text_token_id=0,
                            audio_token_id=previous_token,
                            channel_index=channel_index - 1,
                            step_type=2,
                            past_valid_lengths=local_past_valid_length,
                            local_past_by_name=local_past_by_name,
                        )
                    )
                    local_past_valid_length += 1
                    channel_logits = self.slice_audio_channel_logits(
                        audio_logits, channel_index
                    ).astype(np.float32, copy=False)
                    sampled_token = _sample_audio_token(
                        channel_logits,
                        previous_tokens_by_channel[channel_index],
                        previous_token_sets_by_channel[channel_index],
                        generation_defaults,
                        self.rng,
                    )
                    frame.append(sampled_token)
                    previous_tokens_by_channel[channel_index].append(sampled_token)
                    previous_token_sets_by_channel[channel_index].add(sampled_token)
                    previous_token = sampled_token
            else:
                local_text_logits, _ = self.run_local_decoder(global_hidden, 0, [])
                next_text_token = _sample_assistant_text_token(
                    local_text_logits,
                    self.manifest,
                    generation_defaults,
                    self.rng,
                )
                if next_text_token != int(
                    self.manifest["tts_config"]["audio_assistant_slot_token_id"]
                ):
                    break
                for channel_index in range(int(self.manifest["tts_config"]["n_vq"])):
                    _, audio_logits = self.run_local_decoder(
                        global_hidden, next_text_token, frame
                    )
                    channel_logits = self.slice_audio_channel_logits(
                        audio_logits, channel_index
                    ).astype(np.float32, copy=False)
                    sampled_token = _sample_audio_token(
                        channel_logits,
                        previous_tokens_by_channel[channel_index],
                        previous_token_sets_by_channel[channel_index],
                        generation_defaults,
                        self.rng,
                    )
                    frame.append(sampled_token)
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
            decode_outputs = self.sessions["decode"].run(None, decode_feeds)
            decode_output_names = [
                output.name for output in self.sessions["decode"].get_outputs()
            ]
            named_decode_outputs = dict(
                zip(decode_output_names, decode_outputs, strict=True)
            )
            global_hidden = _extract_last_hidden(named_decode_outputs["global_hidden"])
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
        outputs = self.sessions["local_fixed_sampled_frame"].run(
            None,
            {
                "global_hidden": global_hidden.astype(np.float32, copy=False),
                "repetition_seen_mask": repetition_seen_mask,
                "assistant_random_u": assistant_random_u,
                "audio_random_u": audio_random_u,
            },
        )
        output_names = [
            output.name
            for output in self.sessions["local_fixed_sampled_frame"].get_outputs()
        ]
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
            return _merge_audio_channels(channel_arrays)
        except Exception as exc:
            logging.warning(
                "full codec decode failed, falling back to incremental decode: %s", exc
            )
            self.codec_streaming_session.reset()
            merged_by_channel: list[list[np.ndarray]] = [
                [] for _ in range(int(self.codec_meta["codec_config"]["channels"]))
            ]
            try:
                for start_index in range(0, len(generated_frames), 8):
                    frame_chunk = generated_frames[start_index : start_index + 8]
                    decoded = self.codec_streaming_session.run_frames(frame_chunk)
                    if decoded is None:
                        continue
                    audio, audio_length = decoded
                    if audio_length <= 0:
                        continue
                    for channel_index, channel in enumerate(audio[0, :, :audio_length]):
                        merged_by_channel[channel_index].append(
                            np.asarray(channel, dtype=np.float32)
                        )
            finally:
                self.codec_streaming_session.reset()
            return _merge_audio_channels(
                [
                    np.concatenate(chunks)
                    if chunks
                    else np.zeros((0,), dtype=np.float32)
                    for chunks in merged_by_channel
                ]
            )

    def decode_full_audio(
        self, generated_frames: list[list[int]]
    ) -> tuple[list[np.ndarray], int]:
        if not generated_frames:
            return [], 0
        audio_codes, dims = _flatten3d_int32([generated_frames])
        outputs = self.sessions["codec_decode"].run(
            None,
            {
                "audio_codes": audio_codes.reshape(dims),
                "audio_code_lengths": np.asarray(
                    [len(generated_frames)], dtype=np.int32
                ),
            },
        )
        output_names = [
            output.name for output in self.sessions["codec_decode"].get_outputs()
        ]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        audio_length = int(named_outputs["audio_lengths"].reshape(-1)[0])
        return _slice_channel_major_audio(
            named_outputs["audio"], 0, audio_length
        ), audio_length


def _slice_channel_major_audio(
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


@dataclass
class CodecStreamingDecodeSession:
    codec_meta: dict[str, Any]
    session: ort.InferenceSession

    def __post_init__(self) -> None:
        self.transformer_specs = list(
            self.codec_meta.get("streaming_decode", {}).get("transformer_offsets", [])
        )
        self.attention_specs = list(
            self.codec_meta.get("streaming_decode", {}).get("attention_caches", [])
        )
        self.state_feeds: dict[str, np.ndarray] = {}
        self.reset()

    def reset(self) -> None:
        self.state_feeds = {}
        for spec in self.transformer_specs:
            self.state_feeds[str(spec["input_name"])] = np.zeros(
                tuple(spec["shape"]), dtype=np.int32
            )
        for spec in self.attention_specs:
            self.state_feeds[str(spec["offset_input_name"])] = np.zeros(
                tuple(spec["offset_shape"]), dtype=np.int32
            )
            self.state_feeds[str(spec["cached_keys_input_name"])] = np.zeros(
                tuple(spec["cache_shape"]), dtype=np.float32
            )
            self.state_feeds[str(spec["cached_values_input_name"])] = np.zeros(
                tuple(spec["cache_shape"]), dtype=np.float32
            )
            positions = np.full(tuple(spec["positions_shape"]), -1, dtype=np.int32)
            self.state_feeds[str(spec["cached_positions_input_name"])] = positions

    def run_frames(self, frame_rows: list[list[int]]) -> tuple[np.ndarray, int] | None:
        if not frame_rows:
            return None
        num_quantizers = int(self.codec_meta["codec_config"]["num_quantizers"])
        frame_count = len(frame_rows)
        audio_codes = np.zeros((1, frame_count, num_quantizers), dtype=np.int32)
        for frame_index, frame_row in enumerate(frame_rows):
            for channel_index in range(num_quantizers):
                audio_codes[0, frame_index, channel_index] = int(
                    frame_row[channel_index] if channel_index < len(frame_row) else 0
                )
        feeds: dict[str, np.ndarray] = {
            "audio_codes": audio_codes,
            "audio_code_lengths": np.asarray([frame_count], dtype=np.int32),
        }
        feeds.update(self.state_feeds)
        outputs = self.session.run(None, feeds)
        output_names = [output.name for output in self.session.get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        for spec in self.transformer_specs:
            self.state_feeds[str(spec["input_name"])] = named_outputs[
                str(spec["output_name"])
            ]
        for spec in self.attention_specs:
            self.state_feeds[str(spec["offset_input_name"])] = named_outputs[
                str(spec["offset_output_name"])
            ]
            self.state_feeds[str(spec["cached_keys_input_name"])] = named_outputs[
                str(spec["cached_keys_output_name"])
            ]
            self.state_feeds[str(spec["cached_values_input_name"])] = named_outputs[
                str(spec["cached_values_output_name"])
            ]
            self.state_feeds[str(spec["cached_positions_input_name"])] = named_outputs[
                str(spec["cached_positions_output_name"])
            ]
        return (
            named_outputs["audio"],
            int(named_outputs["audio_lengths"].reshape(-1)[0]),
        )


def _merge_audio_channels(channel_arrays: list[np.ndarray]) -> np.ndarray:
    if not channel_arrays:
        return np.zeros((0, 1), dtype=np.float32)
    if len(channel_arrays) == 1:
        return np.asarray(channel_arrays[0], dtype=np.float32).reshape(-1, 1)
    min_length = min(int(channel.shape[0]) for channel in channel_arrays)
    trimmed = [
        np.asarray(channel[:min_length], dtype=np.float32) for channel in channel_arrays
    ]
    return np.stack(trimmed, axis=1)


def _flatten3d_int32(nested: list[list[list[int]]]) -> tuple[np.ndarray, list[int]]:
    dim0 = len(nested)
    dim1 = len(nested[0])
    dim2 = len(nested[0][0])
    data = np.zeros((dim0 * dim1 * dim2,), dtype=np.int32)
    offset = 0
    for i in range(dim0):
        for j in range(dim1):
            for k in range(dim2):
                data[offset] = int(nested[i][j][k])
                offset += 1
    return data, [dim0, dim1, dim2]


def _flatten2d_int32(nested: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    dim0 = len(nested)
    dim1 = len(nested[0])
    data = np.zeros((dim0 * dim1,), dtype=np.int32)
    offset = 0
    for i in range(dim0):
        for j in range(dim1):
            data[offset] = int(nested[i][j])
            offset += 1
    return data, [dim0, dim1]


def _extract_last_hidden(hidden_states: np.ndarray) -> np.ndarray:
    if hidden_states.ndim == 2:
        return hidden_states.astype(np.float32, copy=False)
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError(f"Unexpected global_hidden shape: {hidden_states.shape}")
    return hidden_states[:, -1, :].astype(np.float32, copy=False)


def _sample_audio_token(
    audio_logits: np.ndarray,
    previous_token_ids: list[int],
    previous_token_set: set[int],
    generation_defaults: dict[str, Any],
    rng: np.random.Generator,
) -> int:
    repetition_penalty = float(generation_defaults["audio_repetition_penalty"])
    if not bool(generation_defaults["do_sample"]):
        return _argmax_with_repetition_penalty(
            audio_logits, previous_token_set, repetition_penalty
        )
    penalized_scores = _apply_repetition_penalty(
        audio_logits, previous_token_ids, repetition_penalty
    )
    return _sample_from_scores(
        penalized_scores,
        do_sample=True,
        temperature=float(generation_defaults["audio_temperature"]),
        top_k=int(generation_defaults["audio_top_k"]),
        top_p=float(generation_defaults["audio_top_p"]),
        rng=rng,
    )


def _argmax_with_repetition_penalty(
    values: np.ndarray, previous_token_set: set[int], repetition_penalty: float
) -> int:
    best_index = 0
    best_value = float("-inf")
    apply_penalty = bool(previous_token_set) and repetition_penalty != 1.0
    for index, value in enumerate(values):
        score = float(value)
        if apply_penalty and index in previous_token_set:
            score = (
                score * repetition_penalty if score < 0 else score / repetition_penalty
            )
        if score > best_value:
            best_value = score
            best_index = index
    return int(best_index)


def _apply_repetition_penalty(
    values: np.ndarray, previous_token_ids: list[int], repetition_penalty: float
) -> np.ndarray:
    if not previous_token_ids or repetition_penalty == 1.0:
        return values
    result = values.copy()
    for token_id in set(int(item) for item in previous_token_ids):
        if token_id < 0 or token_id >= result.shape[0]:
            continue
        result[token_id] = (
            result[token_id] * repetition_penalty
            if result[token_id] < 0
            else result[token_id] / repetition_penalty
        )
    return result


def _sample_from_scores(
    values: np.ndarray,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    if not do_sample:
        return _argmax(values)
    if not (temperature > 0):
        raise ValueError("temperature must be positive when do_sample=True")
    scores = np.asarray(values, dtype=np.float32).copy() / float(temperature)
    if top_k > 0 and top_k < scores.shape[0]:
        sorted_desc = np.sort(scores)[::-1]
        threshold = float(sorted_desc[top_k - 1])
        scores[scores < threshold] = float("-inf")
    if top_p > 0 and top_p < 1:
        indexed = list(enumerate(scores.tolist()))
        indexed.sort(key=lambda item: item[1], reverse=True)
        sorted_scores = np.asarray([item[1] for item in indexed], dtype=np.float32)
        sorted_probs = _softmax(sorted_scores)
        remove_mask = [False] * len(indexed)
        cumulative = 0.0
        for index, probability in enumerate(sorted_probs):
            cumulative += float(probability)
            if cumulative > float(top_p):
                remove_mask[index] = True
        for index in range(len(remove_mask) - 1, 0, -1):
            remove_mask[index] = remove_mask[index - 1]
        if remove_mask:
            remove_mask[0] = False
        for index, should_remove in enumerate(remove_mask):
            if should_remove:
                scores[indexed[index][0]] = float("-inf")
    probabilities = _softmax(scores)
    random_value = float(rng.random())
    for index, probability in enumerate(probabilities):
        random_value -= float(probability)
        if random_value <= 0:
            return int(index)
    return _argmax(scores)


def _softmax(values: np.ndarray) -> np.ndarray:
    max_value = float(np.max(values))
    shifted = np.asarray(values - max_value, dtype=np.float64)
    exps = np.exp(shifted)
    return exps / np.sum(exps, dtype=np.float64)


def _argmax(values: np.ndarray) -> int:
    return int(np.argmax(values))


def _sample_assistant_text_token(
    text_logits: np.ndarray,
    manifest: dict[str, Any],
    generation_defaults: dict[str, Any],
    rng: np.random.Generator,
) -> int:
    candidate_ids = np.asarray(
        [
            int(manifest["tts_config"]["audio_assistant_slot_token_id"]),
            int(manifest["tts_config"]["audio_end_token_id"]),
        ],
        dtype=np.int32,
    )
    candidate_scores = text_logits[candidate_ids]
    sampled_index = _sample_from_scores(
        candidate_scores,
        do_sample=bool(generation_defaults["do_sample"]),
        temperature=float(generation_defaults["text_temperature"]),
        top_k=min(
            int(generation_defaults["text_top_k"]), int(candidate_scores.shape[0])
        ),
        top_p=float(generation_defaults["text_top_p"]),
        rng=rng,
    )
    return int(candidate_ids[sampled_index])
