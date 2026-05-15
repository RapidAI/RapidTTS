# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass
from enum import Enum


class TTSModel(Enum):
    MELO_ONNX = "melo_onnx"
    KOKORO_ONNX = "kokoro_onnx"
    MOSS_NANO_ONNX = "moss_nano_onnx"


class TTSLanguage(Enum):
    ZH = "ZH"
    EN = "EN"
    ZH_MIX_EN = "ZH_MIX_EN"


class TextNormalizerType(Enum):
    LEGACY = "legacy"
    WETEXT = "wetext"
    NONE = "none"


@dataclass(frozen=True)
class ModelCapability:
    name: str
    languages: list[str]
    default_language: str
    voices: list[str]
    default_voice: str | None = None
    voice_source: str | None = None
