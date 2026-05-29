# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import Any, Dict, Optional

from onnxruntime import GraphOptimizationLevel, SessionOptions

DEFAULT_SESSION_OPTIONS = {
    "log_severity_level": 4,
    "enable_cpu_mem_arena": True,
    "graph_optimization_level": GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def build_session_options(cfg: Optional[Dict[str, Any]] = None) -> SessionOptions:
    options = SessionOptions()

    cfg = cfg or {}
    option_cfg = cfg.get("session_options", cfg) or {}
    option_values = dict(DEFAULT_SESSION_OPTIONS)
    option_values.update(option_cfg)

    for name, value in option_values.items():
        if value is None or not hasattr(options, name):
            continue

        option_attr = getattr(options, name)
        if callable(option_attr):
            continue

        if name in ("intra_op_num_threads", "inter_op_num_threads"):
            value = int(value)
            if value <= 0:
                continue

        if name == "graph_optimization_level" and isinstance(value, str):
            value = getattr(GraphOptimizationLevel, value)

        setattr(options, name, value)

    return options
