# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MOSSNanoInput:
    text: str
    text_token_ids: list[int]
    prompt_text: str
    prompt_audio_codes: list[list[int]]
    request_rows: dict[str, list[list[int]]]


@dataclass
class MOSSNanoConfig:
    model_root_dir: Path
    device: str = "cpu"
    thread_count: int = 4
