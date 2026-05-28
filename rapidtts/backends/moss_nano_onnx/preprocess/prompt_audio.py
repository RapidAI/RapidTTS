# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np
import soundfile as sf

from ....common.inference_engine.onnxruntime.main import OrtInferSession
from ....common.io import load_json
from .utils import convert_audio_channels


class PromptAudioProcessor:
    def __init__(
        self,
        tokenizer_dir: Path,
        engine_cfg_defaults: Dict[str, Any],
        manifest: Dict[str, Any],
        device: str = "cpu",
    ):
        self.manifest = manifest
        codec_meta_path = tokenizer_dir / "codec_browser_onnx_meta.json"
        self.codec_meta = load_json(codec_meta_path)

        self._session = None
        self.tokenizer_dir = tokenizer_dir
        self.engine_cfg_defaults = engine_cfg_defaults
        self.device = device

    @property
    def session(self):
        if self._session is None:
            tokenizer_encode_path = (
                self.tokenizer_dir / "moss_audio_tokenizer_encode.onnx"
            )
            self._session = OrtInferSession(
                tokenizer_encode_path,
                engine_cfg=self.engine_cfg_defaults,
                device=self.device,
            )
        return self._session

    def resolve_prompt_audio_codes(
        self,
        *,
        voice: Optional[str],
        prompt_audio_path: Optional[Union[str, Path]] = None,
    ) -> List[List[int]]:
        if prompt_audio_path:
            return self.encode_reference_audio(prompt_audio_path)

        resolved_voice = str(voice or self.builtin_voices[0]["voice"])
        voice_row = self.require_builtin_voice(resolved_voice)
        return list(voice_row["prompt_audio_codes"])

    @cached_property
    def builtin_voices(self) -> List[Dict[str, Any]]:
        return list(self.manifest["builtin_voices"])

    def require_builtin_voice(self, voice: Optional[str]) -> Dict[str, Any]:
        for voice_row in self.builtin_voices:
            if voice_row["voice"] == voice:
                return voice_row

        raise ValueError(f"Built-in voice not found: {voice}")

    def encode_reference_audio(
        self, reference_audio_path: Union[str, Path]
    ) -> List[List[int]]:
        waveform = self.load_reference_audio(reference_audio_path)
        waveform_length = int(waveform.shape[-1])

        outputs = self.session(
            {
                "waveform": waveform,
                "input_lengths": np.asarray([waveform_length], dtype=np.int32),
            }
        )

        output_names = self.session.output_names
        named_outputs = dict(zip(output_names, outputs, strict=True))
        audio_codes = np.asarray(named_outputs["audio_codes"], dtype=np.int32)
        audio_code_lengths = np.asarray(
            named_outputs["audio_code_lengths"], dtype=np.int32
        )
        code_length = int(audio_code_lengths.reshape(-1)[0])

        num_quantizers = int(self.codec_meta["codec_config"]["num_quantizers"])

        prompt_audio_codes = []
        for frame_index in range(code_length):
            prompt_audio_codes.append(
                [
                    int(audio_codes[0, frame_index, quantizer_index])
                    for quantizer_index in range(num_quantizers)
                ]
            )
        return prompt_audio_codes

    def load_reference_audio(
        self, reference_audio_path: Union[str, Path]
    ) -> np.ndarray:
        audio_path = Path(reference_audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Reference audio file not found: {audio_path}")

        waveform, sample_rate = sf.read(
            str(audio_path), dtype="float32", always_2d=True
        )
        waveform = waveform.T

        target_sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
        if sample_rate != target_sample_rate:
            waveform = librosa.resample(
                waveform.astype(np.float32, copy=False),
                orig_sr=sample_rate,
                target_sr=target_sample_rate,
                axis=-1,
                res_type="soxr_hq",
                fix=True,
                scale=False,
            )

        target_channels = int(self.codec_meta["codec_config"]["channels"])
        waveform = convert_audio_channels(waveform, target_channels)
        return waveform[None, ...].astype(np.float32, copy=False)
