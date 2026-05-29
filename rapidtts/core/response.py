# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..common.audio import AudioSaveBackend, save_audio


@dataclass
class SynthesisResponse:
    audio: np.ndarray
    sample_rate: int
    audio_format: str
    metadata: dict[str, Any] = field(default_factory=dict)
    audio_save_backend: AudioSaveBackend = AudioSaveBackend.SOUNDFILE

    def __post_init__(self):
        self.audio_save_backend = AudioSaveBackend(self.audio_save_backend)

    def save(self, file_path: str):
        return save_audio(
            file_path,
            self.audio,
            self.sample_rate,
            audio_format=self.audio_format,
            backend=self.audio_save_backend,
        )
