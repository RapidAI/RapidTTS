# -*- encoding: utf-8 -*-
import numpy as np
import pytest

from rapidtts.backends.kokoro_onnx import postprocess as kokoro_postprocess
from rapidtts.backends.kokoro_onnx.backend import KokoroONNXBackend
from rapidtts.backends.kokoro_onnx.model import KokoroONNXModel
from rapidtts.backends.kokoro_onnx.preprocess import KokoroONNXPreprocessor
from rapidtts.backends.kokoro_onnx.tokenizer import MAX_PHONEME_LENGTH, Tokenizer
from rapidtts.backends.kokoro_onnx.typings import KokoroONNXInput
from rapidtts.common.errors import BackendNotLoadedError
from rapidtts.core.request import SynthesisRequest
from rapidtts.core.response import SynthesisResponse
from rapidtts.core.typings import TTSLanguage


class FakeTextNormalizer:
    def normalize(self, text):
        return text


class FakeZHG2P:
    unk = "?"
    en_callable = None

    def __init__(self):
        self.frontend = self._frontend

    def map_punctuation(self, text):
        return text

    def _frontend(self, text):
        return f"ZH({text})", None


class FakeTokenizer:
    def __init__(self, tokens=None):
        self.tokens = tokens or [10, 20]

    def phonemize(self, text):
        return f"EN({text})"

    def tokenize(self, phonemes):
        return self.tokens


class FakePreprocessor:
    def __init__(self):
        self.seen_request = None
        self.output = [make_kokoro_input()]

    def run(self, request):
        self.seen_request = request
        return self.output


class FakePostprocessor:
    def __init__(self):
        self.seen_audio_list = None
        self.seen_sample_rate = None
        self.seen_speed = None

    def run(self, audio_list, sample_rate, speed):
        self.seen_audio_list = audio_list
        self.seen_sample_rate = sample_rate
        self.seen_speed = speed
        return SynthesisResponse(
            audio=np.concatenate(audio_list).astype(np.float32),
            sample_rate=sample_rate,
            audio_format="wav",
        )


class FakeModel:
    def __init__(self):
        self.seen_inputs = None

    def speak(self, inputs):
        self.seen_inputs = inputs
        return [np.array([0.1, 0.2], dtype=np.float32) for _ in inputs]


class FakeONNXSession:
    def __init__(self):
        self.seen_feeds = []

    def run(self, output_names, input_feed):
        self.seen_feeds.append(input_feed)
        return [np.array([[0.1, 0.2, 0.3]], dtype=np.float32)]


def make_kokoro_input():
    return KokoroONNXInput(
        tokens=[[0, 1, 2, 0]],
        style=np.array([0.1, 0.2], dtype=np.float32),
        speed=np.array([1], dtype=np.int32),
    )


def make_backend_without_init():
    backend = KokoroONNXBackend.__new__(KokoroONNXBackend)
    backend.language = TTSLanguage.ZH
    backend.request_defaults = {
        "language": "ZH",
        "speed": 1.0,
        "sample_rate": 24000,
        "audio_format": "wav",
        "voice": "zf_001",
        "extra_params": {"max_phoneme_length": 510},
    }
    backend.preprocessor = None
    backend.postprocessor = None
    return backend


def test_kokoro_g2p_falls_back_to_tokenizer_for_english_segments():
    preprocessor = KokoroONNXPreprocessor.__new__(KokoroONNXPreprocessor)
    preprocessor.text_normalizer = FakeTextNormalizer()
    preprocessor.zhg2p = FakeZHG2P()
    preprocessor.tokenizer = FakeTokenizer()

    phonemes = preprocessor.g2p("你好 hello 123")

    assert phonemes == "ZH(你好) EN(hello) ZH(123)"


def test_kokoro_g2p_uses_misaki_english_callable_when_available():
    preprocessor = KokoroONNXPreprocessor.__new__(KokoroONNXPreprocessor)
    preprocessor.text_normalizer = FakeTextNormalizer()
    preprocessor.zhg2p = FakeZHG2P()
    preprocessor.zhg2p.en_callable = lambda text: f"MISAKI_EN({text})"
    preprocessor.tokenizer = FakeTokenizer()

    phonemes = preprocessor.g2p("你好 hello")

    assert phonemes == "ZH(你好) MISAKI_EN(hello)"


def test_kokoro_create_audio_wraps_tokens_and_selects_style_by_token_count():
    preprocessor = KokoroONNXPreprocessor.__new__(KokoroONNXPreprocessor)
    preprocessor.max_phoneme_length = 10
    preprocessor.tokenizer = FakeTokenizer(tokens=[10, 20, 30])
    voice = np.arange(20, dtype=np.float32).reshape(5, 4)

    result = preprocessor.create_audio("abc", voice=voice, speed=1.5)

    assert result.tokens == [[0, 10, 20, 30, 0]]
    assert result.style.dtype == np.float32
    assert result.style.tolist() == pytest.approx(voice[3].tolist())
    assert result.speed.dtype == np.int32
    assert result.speed.tolist() == [1]


def test_kokoro_run_rejects_speed_outside_supported_range():
    preprocessor = KokoroONNXPreprocessor.__new__(KokoroONNXPreprocessor)

    with pytest.raises(ValueError, match="Speed must be between 0.5 and 2.0"):
        preprocessor.run(SynthesisRequest(text="hello", speed=0.49))

    with pytest.raises(ValueError, match="Speed must be between 0.5 and 2.0"):
        preprocessor.run(SynthesisRequest(text="hello", speed=2.01))


def test_kokoro_split_phonemes_prefers_punctuation_boundaries():
    preprocessor = KokoroONNXPreprocessor.__new__(KokoroONNXPreprocessor)
    preprocessor.max_phoneme_length = 10

    result = preprocessor.split_phonemes("abc def. ghi jkl!")

    assert result == ["abc def.", "ghi jkl!"]


def test_kokoro_tokenizer_filters_unknown_symbols_and_enforces_limit():
    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.vocab = {"a": 1, " ": 2, "b": 3}

    assert tokenizer.tokenize("a? b") == [1, 2, 3]

    with pytest.raises(ValueError, match="must be less than"):
        tokenizer.tokenize("a" * (MAX_PHONEME_LENGTH + 1))


def test_kokoro_postprocessor_trims_and_concatenates_audio(monkeypatch):
    seen_audio = []

    def fake_trim(audio):
        seen_audio.append(audio)
        return audio[1:], np.array([1, len(audio)])

    monkeypatch.setattr(kokoro_postprocess, "trim_audio", fake_trim)
    postprocessor = kokoro_postprocess.KokoroONNXPostprocessor()

    response = postprocessor.run(
        [
            np.array([0.0, 0.1, 0.2], dtype=np.float32),
            np.array([0.0, 0.3], dtype=np.float32),
        ],
        sample_rate=24000,
        speed=1.0,
    )

    assert len(seen_audio) == 2
    assert response.sample_rate == 24000
    assert response.audio_format == "wav"
    assert response.audio.dtype == np.float32
    assert response.audio.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_kokoro_backend_normalize_request_applies_defaults_and_extras():
    backend = make_backend_without_init()

    request = backend.normalize_request(SynthesisRequest(text="hello"))

    assert request.language == TTSLanguage.ZH
    assert request.speed == 1.0
    assert request.sample_rate == 24000
    assert request.audio_format == "wav"
    assert request.voice == "zf_001"
    assert request.speed == 1.0
    assert request.extras == {"max_phoneme_length": 510}


def test_kokoro_backend_normalize_request_allows_request_overrides():
    backend = make_backend_without_init()

    request = backend.normalize_request(
        SynthesisRequest(
            text="hello",
            language=TTSLanguage.EN,
            speed=1.25,
            sample_rate=16000,
            audio_format="pcm",
            voice="zm_009",
        )
    )

    assert request.language == TTSLanguage.EN
    assert request.speed == 1.25
    assert request.sample_rate == 16000
    assert request.audio_format == "pcm"
    assert request.voice == "zm_009"


def test_kokoro_backend_get_voices_returns_sorted_voice_names():
    backend = make_backend_without_init()
    backend.voices = {
        "zm_009": np.zeros((4, 2), dtype=np.float32),
        "zf_001": np.zeros((4, 2), dtype=np.float32),
        "zf_003": np.zeros((4, 2), dtype=np.float32),
    }

    assert backend.get_voices() == ["zf_001", "zf_003", "zm_009"]


def test_kokoro_backend_capability_describes_languages_and_voices():
    backend = make_backend_without_init()
    backend.voices = {
        "zm_009": np.zeros((4, 2), dtype=np.float32),
        "zf_001": np.zeros((4, 2), dtype=np.float32),
    }

    capability = backend.get_capability()

    assert capability.name == "kokoro_onnx"
    assert capability.languages == ["ZH", "EN", "ZH_MIX_EN"]
    assert capability.default_language == "ZH"
    assert capability.voices == ["zf_001", "zm_009"]
    assert capability.default_voice == "zf_001"
    assert capability.voice_source == "voices-v1.1-zh.bin"


def test_kokoro_backend_preprocess_and_infer_delegate_to_components():
    backend = make_backend_without_init()
    preprocessor = FakePreprocessor()
    model = FakeModel()
    backend.preprocessor = preprocessor
    backend.model = model
    request = SynthesisRequest(text="你好")

    model_inputs = backend.preprocess(request)
    audio_list = backend.infer(model_inputs)

    assert request.language == TTSLanguage.ZH
    assert model_inputs is preprocessor.output
    assert preprocessor.seen_request is request
    assert model.seen_inputs is model_inputs
    assert audio_list[0].tolist() == pytest.approx([0.1, 0.2])


def test_kokoro_backend_raises_when_components_are_not_loaded():
    backend = make_backend_without_init()

    with pytest.raises(BackendNotLoadedError):
        backend.preprocess(SynthesisRequest(text="你好"))

    with pytest.raises(BackendNotLoadedError):
        backend.postprocess([np.array([0.1], dtype=np.float32)], 24000, 1.0)


def test_kokoro_model_speak_passes_expected_onnx_inputs():
    model = KokoroONNXModel.__new__(KokoroONNXModel)
    model.tts_infer = FakeONNXSession()
    model_input = make_kokoro_input()

    result = model.speak([model_input])

    feed = model.tts_infer.seen_feeds[0]
    assert feed["input_ids"] == [[0, 1, 2, 0]]
    assert feed["style"].dtype == np.float32
    assert feed["speed"].dtype == np.int32
    assert len(result) == 1
    assert result[0].dtype == np.float32
    np.testing.assert_allclose(result[0], np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
