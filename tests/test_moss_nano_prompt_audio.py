# -*- encoding: utf-8 -*-
from pathlib import Path

import numpy as np

from rapidtts.backends.moss_nano_onnx.preprocess import prompt_audio
from rapidtts.backends.moss_nano_onnx.preprocess.prompt_audio import (
    PromptAudioProcessor,
)


def make_manifest():
    return {
        "builtin_voices": [
            {
                "voice": "Junhao",
                "prompt_audio_codes": [[1, 2], [3, 4]],
            },
            {
                "voice": "Alice",
                "prompt_audio_codes": [[5, 6]],
            },
        ]
    }


def make_codec_meta():
    return {
        "codec_config": {
            "channels": 1,
            "num_quantizers": 2,
            "sample_rate": 16000,
        }
    }


def test_builtin_voice_uses_manifest_codes_without_initializing_encoder(monkeypatch):
    created_sessions = []

    def fake_ort_infer_session(*args, **kwargs):
        created_sessions.append((args, kwargs))
        raise AssertionError("encoder session should not be initialized")

    monkeypatch.setattr(prompt_audio, "load_json", lambda path: make_codec_meta())
    monkeypatch.setattr(prompt_audio, "OrtInferSession", fake_ort_infer_session)

    processor = PromptAudioProcessor(
        tokenizer_dir=Path("MOSS-Audio-Tokenizer-Nano-ONNX"),
        engine_cfg_defaults={},
        manifest=make_manifest(),
        device="cpu",
    )

    assert processor.resolve_prompt_audio_codes(voice="Junhao") == [[1, 2], [3, 4]]
    assert processor._session is None
    assert created_sessions == []


def test_prompt_audio_path_initializes_encoder_lazily_and_reuses_it(monkeypatch):
    created_sessions = []
    session_calls = []

    class FakeEncoderSession:
        output_names = ["audio_codes", "audio_code_lengths"]

        def __init__(self, model_path, engine_cfg, device):
            created_sessions.append(
                {
                    "model_path": model_path,
                    "engine_cfg": engine_cfg,
                    "device": device,
                }
            )

        def __call__(self, feeds):
            session_calls.append(feeds)
            return [
                np.array([[[10, 11], [12, 13], [14, 15]]], dtype=np.int32),
                np.array([2], dtype=np.int32),
            ]

    monkeypatch.setattr(prompt_audio, "load_json", lambda path: make_codec_meta())
    monkeypatch.setattr(prompt_audio, "OrtInferSession", FakeEncoderSession)

    processor = PromptAudioProcessor(
        tokenizer_dir=Path("MOSS-Audio-Tokenizer-Nano-ONNX"),
        engine_cfg_defaults={"provider": "test"},
        manifest=make_manifest(),
        device="cpu",
    )
    monkeypatch.setattr(
        processor,
        "load_reference_audio",
        lambda path: np.ones((1, 1, 3), dtype=np.float32),
    )

    first_codes = processor.resolve_prompt_audio_codes(
        voice="Junhao", prompt_audio_path="reference.wav"
    )
    second_codes = processor.resolve_prompt_audio_codes(
        voice="Alice", prompt_audio_path="another-reference.wav"
    )

    assert first_codes == [[10, 11], [12, 13]]
    assert second_codes == [[10, 11], [12, 13]]
    assert len(created_sessions) == 1
    assert created_sessions[0] == {
        "model_path": Path(
            "MOSS-Audio-Tokenizer-Nano-ONNX/moss_audio_tokenizer_encode.onnx"
        ),
        "engine_cfg": {"provider": "test"},
        "device": "cpu",
    }
    assert len(session_calls) == 2
    assert session_calls[0]["input_lengths"].tolist() == [3]
