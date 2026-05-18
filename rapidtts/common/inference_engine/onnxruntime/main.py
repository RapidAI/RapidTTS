# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from .provider_config import ProviderConfig


class OrtInferSession:
    def __init__(self, cfg: Dict[str, Any]):
        model_root_dir = Path(cfg.get("model_root_dir"))
        if not model_root_dir.exists():
            raise FileNotFoundError(f"model_root_dir {model_root_dir} does not exist")

        sess_opt = self._init_sess_opts(cfg.engine_cfg)
        provider_cfg = ProviderConfig(engine_cfg=cfg.engine_cfg)
        self.session = InferenceSession(
            str(model_path),
            sess_options=sess_opt,
            providers=provider_cfg.get_ep_list(),
        )
        provider_cfg.verify_providers(self.session.get_providers())

    @staticmethod
    def _init_sess_opts(cfg: Dict[str, Any]) -> SessionOptions:
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = cfg.enable_cpu_mem_arena
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cpu_nums = os.cpu_count()
        intra_op_num_threads = cfg.get("intra_op_num_threads", -1)
        if intra_op_num_threads != -1 and 1 <= intra_op_num_threads <= cpu_nums:
            sess_opt.intra_op_num_threads = intra_op_num_threads

        inter_op_num_threads = cfg.get("inter_op_num_threads", -1)
        if inter_op_num_threads != -1 and 1 <= inter_op_num_threads <= cpu_nums:
            sess_opt.inter_op_num_threads = inter_op_num_threads

        return sess_opt

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
