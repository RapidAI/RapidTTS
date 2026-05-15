# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from ...common.errors import BackendNotLoadedError
from ...core.backend import BaseTTSBackend
from ...core.request import SynthesisRequest
from ...core.response import SynthesisResponse
from ...core.typings import ModelCapability, TextNormalizerType, TTSLanguage
from .model import KokoroONNXModel
from .postprocess import KokoroONNXPostprocessor
from .preprocess import KokoroONNXPreprocessor
from .typings import KokoroONNXConfig, KokoroONNXInput


class KokoroONNXBackend(BaseTTSBackend):
    def __init__(
        self,
        model_root_dir: Union[str, Path],
        device: str = "cpu",
        request_defaults: Optional[Dict[str, Any]] = None,
        text_normalizer_type: str = "wetext",
    ) -> None:
        self.request_defaults = request_defaults or {}

        voices_path = Path(model_root_dir) / "voices-v1.1-zh.bin"
        self.voices = np.load(voices_path)

        self.model_root_dir = Path(model_root_dir)
        tts_model_path = self.model_root_dir / "kokoro-v1.1-zh.onnx"
        self.model = KokoroONNXModel(
            KokoroONNXConfig(model_path=str(tts_model_path), device=device)
        )

        max_phoneme_length = self.request_defaults["extra_params"].get(
            "max_phoneme_length", 300
        )
        self.preprocessor = KokoroONNXPreprocessor(
            self.model_root_dir,
            max_phoneme_length=max_phoneme_length,
            voices=self.voices,
            text_normalizer_type=TextNormalizerType(text_normalizer_type),
        )
        self.postprocessor = KokoroONNXPostprocessor()

    def infer(self, model_inputs: list[KokoroONNXInput]) -> list[np.ndarray]:
        return self.model.speak(model_inputs)

    def preprocess(self, request: SynthesisRequest):
        if self.preprocessor is None:
            raise BackendNotLoadedError("KokoroONNXBackend is not loaded")

        if request.language is None:
            request.language = self.language

        return self.preprocessor.run(request)

    def postprocess(self, audio_list, sample_rate, speed) -> SynthesisResponse:
        if self.postprocessor is None:
            raise BackendNotLoadedError("KokoroONNXBackend is not loaded")

        return self.postprocessor.run(audio_list, sample_rate, speed)

    def synthesize(self, request: SynthesisRequest) -> SynthesisResponse:
        request = self.normalize_request(request)
        return super().synthesize(request)

    def get_voices(self) -> list[str]:
        return sorted(self.voices.keys())

    def get_capability(self) -> ModelCapability:
        defaults = self.request_defaults
        return ModelCapability(
            name="kokoro_onnx",
            languages=[
                TTSLanguage.ZH.value,
                TTSLanguage.EN.value,
                TTSLanguage.ZH_MIX_EN.value,
            ],
            default_language=defaults["language"],
            voices=self.get_voices(),
            default_voice=defaults.get("voice"),
            voice_source="voices-v1.1-zh.bin",
        )

    def normalize_request(self, request: SynthesisRequest) -> SynthesisRequest:
        defaults = self.request_defaults
        return SynthesisRequest(
            text=request.text,
            language=request.language or TTSLanguage(defaults["language"]),
            voice=request.voice or defaults.get("voice"),
            speed=request.speed if request.speed is not None else defaults["speed"],
            sample_rate=(
                request.sample_rate
                if request.sample_rate is not None
                else defaults["sample_rate"]
            ),
            audio_format=request.audio_format or defaults["audio_format"],
            extras=request.extras,
        )
