# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import Any, Optional, Union

from ..common.logger import set_logger_enabled
from .config import (
    get_backend_init_defaults,
    get_backend_request_defaults,
    get_default_backend,
    load_config,
)
from .model_assets import ensure_model_assets
from .request import SynthesisRequest
from .response import SynthesisResponse
from .typings import ModelCapability, TTSModel


class RapidTTS:
    def __init__(
        self,
        model: Optional[TTSModel] = None,
        config_path: Union[str, Path, None] = None,
        enable_log: bool = True,
        **kwargs: Any,
    ) -> None:
        set_logger_enabled(enable_log)
        self.cfg = load_config(config_path)

        if model is None:
            model = TTSModel(get_default_backend(self.cfg))
        backend_name = model.value
        init_defaults = get_backend_init_defaults(self.cfg, backend_name)
        request_defaults = get_backend_request_defaults(self.cfg, backend_name)

        init_kwargs = {
            **init_defaults,
            **kwargs,
            "request_defaults": request_defaults,
        }
        if "model_root_dir" not in kwargs:
            init_kwargs["model_root_dir"] = ensure_model_assets(backend_name)

        self.backend = self.create_backend(model, **init_kwargs)

    def create_backend(self, model: TTSModel, **kwargs: Any) -> Any:
        if model == TTSModel.MELO_ONNX:
            from ..backends.melo_onnx.backend import MeloONNXBackend

            return MeloONNXBackend(**kwargs)

        if model == TTSModel.KOKORO_ONNX:
            from ..backends.kokoro_onnx.backend import KokoroONNXBackend

            return KokoroONNXBackend(**kwargs)

        if model == TTSModel.MOSS_NANO_ONNX:
            from ..backends.moss_nano_onnx.backend import MOSSNanoBackend

            return MOSSNanoBackend(**kwargs)

        raise ValueError(f"Unsupported model: {model}")

    def synthesize(self, request: SynthesisRequest) -> SynthesisResponse:
        return self.backend.synthesize(request)

    def get_voices(self) -> list[str]:
        get_backend_voices = getattr(self.backend, "get_voices", None)
        if get_backend_voices is None:
            return []

        return get_backend_voices()

    def get_capability(self) -> ModelCapability:
        get_backend_capability = getattr(self.backend, "get_capability", None)
        if get_backend_capability is None:
            raise NotImplementedError(
                "Current backend does not expose capability metadata"
            )

        return get_backend_capability()
