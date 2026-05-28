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
from ...core.typings import ModelCapability, TextNormalizerType, TTSLanguage, TTSModel
from .model import MOSSNanoModel
from .postprocess import MOSSNanoPostprocessor
from .preprocess import MOSSNanoPreprocessor
from .typings import MOSSNanoConfig, MOSSNanoInput


class MOSSNanoBackend(BaseTTSBackend):
    def __init__(
        self,
        model_root_dir: Union[str, Path],
        device: str = "cpu",
        request_defaults: Optional[Dict[str, Any]] = None,
        engine_cfg_defaults: Optional[Dict[str, Any]] = None,
        text_normalizer_type: str = "wetext",
    ):
        self.request_defaults = request_defaults or {}
        self.engine_cfg_defaults = engine_cfg_defaults or {}

        self.model_root_dir = Path(model_root_dir)
        self.model = MOSSNanoModel(
            MOSSNanoConfig(
                model_root_dir=self.model_root_dir,
                device=device,
                engine_cfg_defaults=self.engine_cfg_defaults,
            )
        )

        self.preprocessor = MOSSNanoPreprocessor(
            self.model_root_dir,
            text_normalizer_type=TextNormalizerType(text_normalizer_type),
            engine_cfg_defaults=self.engine_cfg_defaults,
            device=device,
        )
        self.postprocessor = MOSSNanoPostprocessor()

    def infer(self, model_inputs: list[MOSSNanoInput]) -> list[np.ndarray]:
        return self.model.speak(model_inputs)

    def preprocess(self, request: SynthesisRequest):
        if self.preprocessor is None:
            raise BackendNotLoadedError("MOSSNanoBackend is not loaded")

        if request.language is None:
            request.language = self.language

        return self.preprocessor.run(request)

    def postprocess(self, audio_list, sample_rate, speed) -> SynthesisResponse:
        if self.postprocessor is None:
            raise BackendNotLoadedError("MOSSNanoBackend is not loaded")

        return self.postprocessor.run(audio_list, sample_rate)

    def synthesize(self, request: SynthesisRequest) -> SynthesisResponse:
        request = self.normalize_request(request)
        return super().synthesize(request)

    def get_voices(self) -> list[str]:
        if self.preprocessor is None:
            return []

        return self.preprocessor.get_voices()

    def get_capability(self) -> ModelCapability:
        defaults = self.request_defaults
        return ModelCapability(
            name=TTSModel.MOSS_NANO_ONNX.value,
            languages=[TTSLanguage.ZH_MIX_EN.value],
            default_language=defaults["language"],
            voices=self.get_voices(),
            default_voice=defaults.get("voice"),
            voice_source="configuration.json",
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
            extras=request.extras or defaults.get("extras", {}),
        )
