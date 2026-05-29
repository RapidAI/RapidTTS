# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import traceback
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from onnxruntime import InferenceSession

from .provider_config import get_ep_list
from .session_option import build_session_options


class OrtInferSession:
    def __init__(
        self,
        model_path: Union[str, Path],
        engine_cfg: Dict[str, Any],
        device: str = "cpu",
    ):
        sess_opts = build_session_options(engine_cfg["onnxruntime"]["session_options"])
        ep_list = get_ep_list(engine_cfg["onnxruntime"]["backends"], device)
        self.session = InferenceSession(
            str(model_path), sess_options=sess_opts, providers=ep_list
        )

    def __call__(self, input_content: Union[np.ndarray, Dict[str, Any]]) -> Any:
        try:
            if isinstance(input_content, np.ndarray):
                if len(self.input_names) != 1:
                    raise ONNXRuntimeError(
                        f"Model has multiple inputs {self.input_names}, but only one input provided."
                    )

                input_dict = dict(zip(self.input_names, [input_content]))
                return self.session.run(self.output_names, input_dict)

            return self.session.run(None, input_content)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    @cached_property
    def input_names(self) -> List[str]:
        return [v.name for v in self.session.get_inputs()]

    @cached_property
    def output_names(self) -> List[str]:
        return [v.name for v in self.session.get_outputs()]


class ONNXRuntimeError(Exception):
    pass
