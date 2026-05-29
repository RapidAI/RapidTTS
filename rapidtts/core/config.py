# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import Union

from omegaconf import OmegaConf

cur_dir = Path(__file__).resolve().parent
CONFIG_PATH = cur_dir.parent / "configs" / "config.yaml"
ENGINE_PATH = cur_dir.parent / "configs" / "engine_config.yaml"


def load_config(config_path: Union[str, Path, None] = None):
    path = Path(config_path) if config_path else CONFIG_PATH
    return OmegaConf.load(path)


def get_default_backend(cfg) -> str:
    return OmegaConf.to_container(cfg["global"], resolve=True)["backend"]


def get_backend_init_defaults(cfg, backend_name: str) -> dict:
    return OmegaConf.to_container(
        cfg[backend_name].init, resolve=True, throw_on_missing=True
    )


def get_backend_request_defaults(cfg, backend_name: str) -> dict:
    return OmegaConf.to_container(
        cfg[backend_name].request, resolve=True, throw_on_missing=True
    )


def get_engine_config(config_path: Union[str, Path, None] = None):
    path = Path(config_path) if config_path else ENGINE_PATH
    return OmegaConf.load(path)
