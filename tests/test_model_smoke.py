# -*- encoding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pytest

from rapidtts import RapidTTS, SynthesisRequest, TTSModel
from rapidtts.core.model_assets import check_model_assets


MODEL_SMOKE_CASES = [
    pytest.param(
        TTSModel.KOKORO_ONNX,
        SynthesisRequest(text="你好，RapidTTS。"),
        id="kokoro_onnx",
    ),
    pytest.param(
        TTSModel.MELO_ONNX,
        SynthesisRequest(text="你好，RapidTTS。"),
        id="melo_onnx",
    ),
    pytest.param(
        TTSModel.MOSS_NANO_ONNX,
        SynthesisRequest(text="你好，RapidTTS。"),
        id="moss_nano_onnx",
    ),
]


def require_default_model_assets(model: TTSModel):
    check_result = check_model_assets(model.value)
    if check_result.ok:
        return check_result.model_dir

    missing_count = len(check_result.missing_files)
    mismatch_count = len(check_result.hash_mismatch_files)
    pytest.skip(
        f"{model.value} default model assets are not ready in "
        f"{check_result.model_dir}: {missing_count} missing, "
        f"{mismatch_count} hash mismatch"
    )


@pytest.mark.parametrize("model, synthesis_request", MODEL_SMOKE_CASES)
def test_default_model_directory_can_synthesize_when_assets_exist(
    model: TTSModel, synthesis_request: SynthesisRequest
) -> None:
    model_root_dir = require_default_model_assets(model)

    tts = RapidTTS(model=model, model_root_dir=model_root_dir, enable_log=False)
    response = tts.synthesize(synthesis_request)

    assert response.sample_rate > 0
    assert response.audio_format == "wav"
    assert isinstance(response.audio, np.ndarray)
    assert response.audio.dtype == np.float32
    assert response.audio.size > 0
