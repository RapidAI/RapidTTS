# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from onnxruntime import InferenceSession

from .provider_config import get_ep_list
from .session_option import build_session_options


class OrtInferSession:
    def __init__(
        self,
        model_path: Path,
        engine_cfg: Dict[str, Any],
        device: str = "cpu",
    ):
        sess_opts = build_session_options(engine_cfg["onnxruntime"]["session_options"])
        ep_list = get_ep_list(engine_cfg["onnxruntime"]["backends"], device)

        self.session = InferenceSession(
            str(model_path), sess_options=sess_opts, providers=ep_list
        )

        # provider_cfg.verify_providers(self.session.get_providers())

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), [input_content]))
        try:
            return self.session.run(self.get_output_names(), input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    def get_input_names(self) -> List[str]:
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(self) -> List[str]:
        return [v.name for v in self.session.get_outputs()]


class ONNXRuntimeError(Exception):
    pass
