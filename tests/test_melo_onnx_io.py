# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

cur_dir = Path(__file__).parent.resolve()
sys.path.append(str(cur_dir.parent))

from rapidtts.backends.melo_onnx.backend import MeloONNXBackend
from rapidtts.backends.melo_onnx.kernel.chinese_mix_en_kernel import ChineseMixEnKernel
from rapidtts.backends.melo_onnx.kernel.english_kernel import EnglishKernel
from rapidtts.backends.melo_onnx.model import MeloONNXModel
from rapidtts.backends.melo_onnx.postprocess import MeloONNXPostprocessor
from rapidtts.backends.melo_onnx.preprocess import MeloONNXPreprocessor
from rapidtts.backends.melo_onnx.typings import MeloONNXInput
from rapidtts.common.errors import BackendNotLoadedError
from rapidtts.core.backend import BaseTTSBackend
from rapidtts.core.request import SynthesisRequest
from rapidtts.core.response import SynthesisResponse
from rapidtts.core.typings import TTSLanguage


def make_melo_input(length: int = 4) -> MeloONNXInput:
    return MeloONNXInput(
        x=np.arange(length, dtype=np.int64)[None, :],
        x_lengths=np.array([length], dtype=np.int64),
        sid=np.array([0], dtype=np.int64),
        tone=np.zeros((1, length), dtype=np.int64),
        language=np.zeros((1, length), dtype=np.int64),
        bert=np.zeros((1, 1024, length), dtype=np.float32),
        ja_bert=np.zeros((1, 1024, length), dtype=np.float32),
        noise_scale=np.array([0.6], dtype=np.float32),
        length_scale=np.array([1.0], dtype=np.float32),
        noise_scale_w=np.array([0.8], dtype=np.float32),
        sdp_ratio=np.array([0.2], dtype=np.float32),
    )


def make_melo_onnx_regression_input() -> MeloONNXInput:
    length = 4
    return MeloONNXInput(
        x=np.array([[1, 2, 3, 4]], dtype=np.int64),
        x_lengths=np.array([length], dtype=np.int64),
        sid=np.array([0], dtype=np.int64),
        tone=np.zeros((1, length), dtype=np.int64),
        language=np.zeros((1, length), dtype=np.int64),
        bert=np.zeros((1, 1024, length), dtype=np.float32),
        ja_bert=np.zeros((1, 768, length), dtype=np.float32),
        noise_scale=np.array(0.0, dtype=np.float32),
        length_scale=np.array(1.0, dtype=np.float32),
        noise_scale_w=np.array(0.0, dtype=np.float32),
        sdp_ratio=np.array(0.0, dtype=np.float32),
    )


def make_backend_without_init() -> MeloONNXBackend:
    backend = MeloONNXBackend.__new__(MeloONNXBackend)
    backend.language = "ZH_MIX_EN"
    backend.request_defaults = {
        "language": "ZH_MIX_EN",
        "voice": "zf_001",
        "speed": 1.0,
        "sample_rate": 44100,
        "audio_format": "wav",
        "sdp_ratio": 0.2,
        "noise_scale": 0.6,
        "noise_scale_w": 0.8,
    }
    backend.preprocessor = None
    backend.postprocessor = None
    return backend


class FakeModel:
    def __init__(self) -> None:
        self.seen_inputs = None

    def speak(self, inputs: list[MeloONNXInput]) -> list[np.ndarray]:
        self.seen_inputs = inputs
        return [np.array([0.1, 0.2, 0.3], dtype=np.float32) for _ in inputs]


class FakePreprocessor:
    def __init__(self) -> None:
        self.seen_request = None
        self.output = [make_melo_input()]

    def run(self, request: SynthesisRequest) -> list[MeloONNXInput]:
        self.seen_request = request
        return self.output

    def get_voices(self) -> list[str]:
        return ["zf_001"]


class FakePostprocessor:
    def __init__(self) -> None:
        self.seen_audio_list = None
        self.seen_sample_rate = None
        self.seen_speed = None

    def run(
        self, audio_list: list[np.ndarray], sample_rate: int, speed: float
    ) -> SynthesisResponse:
        self.seen_audio_list = audio_list
        self.seen_sample_rate = sample_rate
        self.seen_speed = speed
        return SynthesisResponse(
            audio=np.concatenate(audio_list).astype(np.float32),
            sample_rate=sample_rate,
            audio_format="wav",
        )


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        class Encoded:
            tokens = text.lower().split()

        return Encoded()


class FakeSynthesisBackend(BaseTTSBackend):
    def __init__(self) -> None:
        self.seen_postprocess_speed = None

    def preprocess(self, request: SynthesisRequest) -> str:
        return request.text

    def infer(self, model_input: str) -> list[np.ndarray]:
        return [np.array([0.1, 0.2], dtype=np.float32)]

    def postprocess(
        self, audio_list: list[np.ndarray], sample_rate: int, speed: float
    ) -> SynthesisResponse:
        self.seen_postprocess_speed = speed
        return SynthesisResponse(
            audio=np.concatenate(audio_list),
            sample_rate=sample_rate,
            audio_format="wav",
        )


def test_melo_input_contains_expected_array_shapes() -> None:
    model_input = make_melo_input(length=5)

    assert model_input.x.shape == (1, 5)
    assert model_input.x_lengths.tolist() == [5]
    assert model_input.bert.shape == (1, 1024, 5)
    assert model_input.ja_bert.dtype == np.float32
    assert model_input.length_scale.tolist() == [1.0]


def test_postprocessor_returns_synthesis_response_with_concat_audio() -> None:
    postprocessor = MeloONNXPostprocessor()
    audio_list = [
        np.array([[0.1, 0.2]], dtype=np.float32),
        np.array([[0.3]], dtype=np.float32),
    ]

    response = postprocessor.run(audio_list, sample_rate=20, speed=1.0)

    assert isinstance(response, SynthesisResponse)
    assert response.sample_rate == 20
    assert response.audio_format == "wav"
    assert response.audio.dtype == np.float32
    assert response.audio.tolist() == pytest.approx([0.1, 0.2, 0.0, 0.3, 0.0])


def test_synthesize_passes_request_speed_to_postprocess() -> None:
    backend = FakeSynthesisBackend()
    request = SynthesisRequest(
        text="hello",
        speed=1.5,
        sample_rate=24000,
        extras={"speed": 0.5},
    )

    response = backend.synthesize(request)

    assert backend.seen_postprocess_speed == 1.5
    assert response.sample_rate == 24000
    assert response.audio.tolist() == pytest.approx([0.1, 0.2])


def test_melo_backend_normalize_request_adds_default_voice() -> None:
    backend = make_backend_without_init()

    request = backend.normalize_request(SynthesisRequest(text="hello"))

    assert request.language == TTSLanguage.ZH_MIX_EN
    assert request.speed == 1.0
    assert request.sample_rate == 44100
    assert request.audio_format == "wav"
    assert request.voice == "zf_001"
    assert request.extras == {}


def test_melo_backend_normalize_request_allows_voice_override() -> None:
    backend = make_backend_without_init()

    request = backend.normalize_request(
        SynthesisRequest(text="hello", voice="custom_voice")
    )

    assert request.voice == "custom_voice"
    assert "voice" not in request.extras


def test_melo_preprocessor_exposes_and_resolves_default_voice_alias() -> None:
    preprocessor = MeloONNXPreprocessor.__new__(MeloONNXPreprocessor)
    preprocessor.params = {"data": {"spk2id": {"ZH_MIX_EN": 1}}}
    request = SynthesisRequest(
        text="hello", language=TTSLanguage.ZH_MIX_EN, voice="zf_001"
    )

    assert preprocessor.get_voices() == ["zf_001"]
    assert preprocessor.resolve_speaker_id(request) == 1


def test_melo_preprocessor_resolves_request_voice_alias() -> None:
    preprocessor = MeloONNXPreprocessor.__new__(MeloONNXPreprocessor)
    preprocessor.params = {"data": {"spk2id": {"ZH_MIX_EN": 1}}}
    request = SynthesisRequest(
        text="hello", language=TTSLanguage.ZH_MIX_EN, voice="zf_001"
    )

    assert preprocessor.resolve_speaker_id(request) == 1


def test_melo_preprocessor_rejects_unknown_voice() -> None:
    preprocessor = MeloONNXPreprocessor.__new__(MeloONNXPreprocessor)
    preprocessor.params = {"data": {"spk2id": {"ZH_MIX_EN": 1}}}
    request = SynthesisRequest(
        text="hello", language=TTSLanguage.ZH_MIX_EN, voice="bad_voice"
    )

    with pytest.raises(ValueError, match="Unsupported Melo voice: bad_voice"):
        preprocessor.resolve_speaker_id(request)


def test_melo_backend_get_voices_delegates_to_preprocessor() -> None:
    backend = make_backend_without_init()
    backend.preprocessor = FakePreprocessor()

    assert backend.get_voices() == ["zf_001"]


def test_melo_backend_capability_describes_language_and_voices() -> None:
    backend = make_backend_without_init()
    backend.preprocessor = FakePreprocessor()

    capability = backend.get_capability()

    assert capability.name == "melo_onnx"
    assert capability.languages == ["ZH_MIX_EN"]
    assert capability.default_language == "ZH_MIX_EN"
    assert capability.voices == ["zf_001"]
    assert capability.default_voice == "zf_001"
    assert capability.voice_source == "configuration.json"


def test_backend_preprocess_sets_default_language_and_returns_inputs() -> None:
    backend = make_backend_without_init()
    preprocessor = FakePreprocessor()
    backend.preprocessor = preprocessor
    request = SynthesisRequest(text="你好")

    result = backend.preprocess(request)

    assert result == preprocessor.output
    assert preprocessor.seen_request is request
    assert request.language == "ZH_MIX_EN"


def test_chinese_segment_processing_removes_spaces_around_english() -> None:
    kernel = ChineseMixEnKernel.__new__(ChineseMixEnKernel)

    assert kernel.process_seg("改到下午 ") == "改到下午"
    assert kernel.process_seg(" API ") == ""


def test_english_kernel_reads_uppercase_single_letters_as_letter_names() -> None:
    kernel = EnglishKernel.__new__(EnglishKernel)
    kernel.eng_dict = {"A": [["AH0"]]}

    phones, tones, word2ph = kernel.g2p_en(
        pad_start_end=False,
        tokenized=["A", "D", "I"],
    )

    assert phones == ["ey", "d", "iy", "ay"]
    assert tones == [2, 0, 2, 2]
    assert word2ph == [1, 2, 1]


def test_chinese_mix_en_tokenizer_preserves_standalone_uppercase_letters() -> None:
    kernel = ChineseMixEnKernel.__new__(ChineseMixEnKernel)
    kernel.tokenizer = FakeTokenizer()

    assert kernel.tokenize_english_segment("A P I ") == ["A", "P", "I"]
    assert kernel.tokenize_english_segment("The C P U usage") == [
        "the",
        "C",
        "P",
        "U",
        "usage",
    ]
    assert kernel.tokenize_english_segment("Apple") == ["apple"]


def test_backend_infer_delegates_inputs_to_model() -> None:
    backend = make_backend_without_init()
    model = FakeModel()
    backend.model = model
    inputs = [make_melo_input(), make_melo_input(length=2)]

    result = backend.infer(inputs)

    assert model.seen_inputs is inputs
    assert len(result) == 2
    assert all(audio.dtype == np.float32 for audio in result)
    assert result[0].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_melo_onnx_model_fixed_input_matches_golden_output() -> None:
    ort = pytest.importorskip("onnxruntime")
    model_path = Path("models/melotts_zh_mix_en_onnx/tts_model.onnx")
    if not model_path.exists():
        pytest.skip(f"Missing Melo ONNX model: {model_path}")

    model = MeloONNXModel.__new__(MeloONNXModel)
    model.tts_infer = ort.InferenceSession(
        str(model_path),
        sess_options=MeloONNXModel.get_def_onnx_session_options(),
        providers=["CPUExecutionProvider"],
    )

    audio = model.speak([make_melo_onnx_regression_input()])[0]
    audio_digest = hashlib.sha256(
        np.ascontiguousarray(audio).view(np.uint8)
    ).hexdigest()

    assert audio.shape == (8192,)
    assert audio.dtype == np.float32
    assert float(np.sum(audio)) == pytest.approx(-0.012945152819156647)
    assert float(np.mean(audio)) == pytest.approx(-1.5802188499947079e-06)
    assert float(np.std(audio)) == pytest.approx(7.558455763501115e-07)
    np.testing.assert_allclose(
        audio[:16],
        np.array(
            [
                -1.6078056432888843e-05,
                1.2582831914187409e-05,
                1.2796539522241801e-05,
                1.4724619177286513e-05,
                1.3613197324957582e-06,
                -4.071927378390683e-06,
                -1.4664054504009982e-07,
                -6.577649855898926e-06,
                -5.299784334056312e-06,
                2.005345550060156e-06,
                -2.1073381049063755e-06,
                2.2342144347931026e-06,
                5.590511591435643e-06,
                -4.070816885359818e-06,
                -3.868695330311311e-06,
                2.868731598937302e-06,
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        audio[-16:],
        np.array(
            [
                -1.6148676422744757e-06,
                -8.104090738925152e-06,
                -8.704676474735606e-06,
                -2.5045826532732463e-06,
                -5.208885795582319e-06,
                -1.0440850928716827e-05,
                -8.463841368211433e-06,
                -9.852518815023359e-06,
                -5.727853931603022e-06,
                3.0005773510310974e-07,
                -4.092706603842089e-06,
                -1.4932963949831901e-06,
                1.2029446224914864e-05,
                7.313555670407368e-06,
                -2.880724139231461e-07,
                -9.353571840620134e-06,
            ],
            dtype=np.float32,
        ),
    )
    assert (
        audio_digest
        == "3c146f0218b528cbe63bbc8c83f85e0a76fd3cc80c17e2c072df718944a41bab"
    )


def test_backend_postprocess_delegates_audio_to_postprocessor() -> None:
    backend = make_backend_without_init()
    postprocessor = FakePostprocessor()
    backend.postprocessor = postprocessor
    audio_list = [np.array([0.1], dtype=np.float32), np.array([0.2], dtype=np.float32)]

    response = backend.postprocess(audio_list, sample_rate=8000, speed=1.2)

    assert postprocessor.seen_audio_list is audio_list
    assert postprocessor.seen_sample_rate == 8000
    assert postprocessor.seen_speed == 1.2
    assert response.audio.tolist() == pytest.approx([0.1, 0.2])


def test_backend_preprocess_requires_loaded_preprocessor() -> None:
    backend = make_backend_without_init()

    with pytest.raises(BackendNotLoadedError):
        backend.preprocess(SynthesisRequest(text="你好"))


def test_backend_postprocess_requires_loaded_postprocessor() -> None:
    backend = make_backend_without_init()

    with pytest.raises(BackendNotLoadedError):
        backend.postprocess(
            [np.array([0.1], dtype=np.float32)], sample_rate=8000, speed=1.0
        )


def test_backend_init_fails_for_missing_model_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MeloONNXBackend(model_root_dir=tmp_path)
