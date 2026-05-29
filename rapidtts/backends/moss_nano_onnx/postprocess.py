# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import numpy as np

from ...common.audio import AudioSaveBackend
from ...core.response import SynthesisResponse


class MOSSNanoPostprocessor:
    def __init__(self):
        pass

    def run(self, audio_list, sample_rate) -> SynthesisResponse:
        waveform = self.concat_waveforms(audio_list)
        return SynthesisResponse(
            audio=waveform,
            sample_rate=sample_rate,
            audio_format="wav",
            audio_save_backend=AudioSaveBackend.WAVE,
        )

    @staticmethod
    def concat_waveforms(waveforms: list[np.ndarray]) -> np.ndarray:
        if not waveforms:
            return np.zeros((0, 1), dtype=np.float32)
        non_empty = [waveform for waveform in waveforms if waveform.size > 0]
        if not non_empty:
            channel_count = (
                int(waveforms[0].shape[1])
                if waveforms[0].ndim == 2 and waveforms[0].shape[1] > 0
                else 1
            )
            return np.zeros((0, channel_count), dtype=np.float32)
        return np.concatenate(non_empty, axis=0)
