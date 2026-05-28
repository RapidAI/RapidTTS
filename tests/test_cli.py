# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

cur_dir = Path(__file__).parent.resolve()
sys.path.append(str(cur_dir.parent))

from rapidtts import cli
from rapidtts.core import model_assets
from rapidtts.core.typings import ModelCapability


@dataclass
class FakeSynthesisResponse:
    audio: np.ndarray
    sample_rate: int


def test_download_cli_passes_model_dir_and_progress_options(monkeypatch, tmp_path):
    seen = {}

    def fake_ensure_model_assets(model_name, local_dir=None, show_progress=True):
        seen["model_name"] = model_name
        seen["local_dir"] = local_dir
        seen["show_progress"] = show_progress
        return Path(local_dir)

    monkeypatch.setattr(cli, "ensure_model_assets", fake_ensure_model_assets)

    exit_code = cli.main(
        [
            "download",
            "melo_onnx",
            "--save-dir",
            str(tmp_path),
            "--no-progress",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert seen == {
        "model_name": "melo_onnx",
        "local_dir": str(tmp_path),
        "show_progress": False,
    }


def test_cli_uses_configured_default_model(monkeypatch):
    parser = cli.build_parser()

    download_args = parser.parse_args(["download"])
    check_args = parser.parse_args(["check"])
    info_args = parser.parse_args(["info"])
    voices_args = parser.parse_args(["voices"])
    text_args = parser.parse_args(["text", "hello", "out.wav"])
    voice_args = parser.parse_args(["text", "hello", "out.wav", "--voice", "zm_009"])

    assert download_args.model == "kokoro_onnx"
    assert check_args.model == "kokoro_onnx"
    assert info_args.model == "kokoro_onnx"
    assert voices_args.model == "kokoro_onnx"
    assert text_args.model == "kokoro_onnx"
    assert text_args.voice is None
    assert voice_args.voice == "zm_009"


def test_cli_parser_default_model_is_config_driven(monkeypatch):
    monkeypatch.setattr(cli, "get_default_model_name", lambda: "melo_onnx")

    parser = cli.build_parser()

    assert parser.parse_args(["download"]).model == "melo_onnx"
    assert parser.parse_args(["check"]).model == "melo_onnx"
    assert parser.parse_args(["info"]).model == "melo_onnx"
    assert parser.parse_args(["voices"]).model == "melo_onnx"
    assert parser.parse_args(["text", "hello", "out.wav"]).model == "melo_onnx"


def test_check_cli_success_without_backend_init(monkeypatch, capsys):
    backend_called = False

    monkeypatch.setattr(cli, "_check_package_import", lambda: True)
    monkeypatch.setattr(cli, "_check_dependencies", lambda model_name: True)
    monkeypatch.setattr(cli, "_check_configs", lambda model_name: True)
    monkeypatch.setattr(cli, "_check_model_files", lambda model_name, model_dir: True)

    def fake_check_backend_init(model_name, model_dir):
        nonlocal backend_called
        backend_called = True
        return True

    monkeypatch.setattr(cli, "_check_backend_init", fake_check_backend_init)

    exit_code = cli.main(["check", "melo_onnx", "--quiet"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "RapidTTS installation looks ready." in captured.out
    assert backend_called is False


def test_check_cli_runs_backend_init_when_requested(monkeypatch):
    seen = {}

    monkeypatch.setattr(cli, "_check_package_import", lambda: True)
    monkeypatch.setattr(cli, "_check_dependencies", lambda model_name: True)
    monkeypatch.setattr(cli, "_check_configs", lambda model_name: True)
    monkeypatch.setattr(cli, "_check_model_files", lambda model_name, model_dir: True)

    def fake_check_backend_init(model_name, model_dir):
        seen["model_name"] = model_name
        seen["model_dir"] = model_dir
        return True

    monkeypatch.setattr(cli, "_check_backend_init", fake_check_backend_init)

    exit_code = cli.main(
        [
            "check",
            "melo_onnx",
            "--model-dir",
            "custom_model_dir",
            "--init-backend",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert seen == {"model_name": "melo_onnx", "model_dir": "custom_model_dir"}


def test_check_cli_returns_failure_when_model_assets_fail(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_check_package_import", lambda: True)
    monkeypatch.setattr(cli, "_check_dependencies", lambda model_name: True)
    monkeypatch.setattr(cli, "_check_configs", lambda model_name: True)
    monkeypatch.setattr(cli, "_check_model_files", lambda model_name, model_dir: False)

    exit_code = cli.main(["check", "melo_onnx", "--quiet"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "RapidTTS installation check failed." in captured.out


def test_check_dependencies_uses_kokoro_dependency_set(monkeypatch, capsys):
    seen = []

    def fake_find_spec(name):
        seen.append(name)
        return object()

    monkeypatch.setattr(cli.importlib.util, "find_spec", fake_find_spec)

    assert cli._check_dependencies("kokoro_onnx") is True

    captured = capsys.readouterr()
    assert "[OK] required dependencies for kokoro_onnx" in captured.out
    assert "misaki" in seen
    assert "phonemizer" in seen
    assert "espeakng_loader" in seen
    assert "librosa" not in seen
    assert "tokenizers" not in seen


def test_check_dependencies_reports_kokoro_extra_install_hint(monkeypatch, capsys):
    def fake_find_spec(name):
        if name == "phonemizer":
            return None
        return object()

    monkeypatch.setattr(cli.importlib.util, "find_spec", fake_find_spec)

    assert cli._check_dependencies("kokoro_onnx") is False

    captured = capsys.readouterr()
    assert "[FAIL] required dependencies for kokoro_onnx" in captured.out
    assert "  - phonemizer" in captured.out
    assert 'pip install "rapidtts[kokoro]"' in captured.out


def test_check_dependencies_reports_melo_extra_install_hint(monkeypatch, capsys):
    def fake_find_spec(name):
        if name == "g2p_en":
            return None
        return object()

    monkeypatch.setattr(cli.importlib.util, "find_spec", fake_find_spec)

    assert cli._check_dependencies("melo_onnx") is False

    captured = capsys.readouterr()
    assert "[FAIL] required dependencies for melo_onnx" in captured.out
    assert "  - g2p_en" in captured.out
    assert 'pip install "rapidtts[melo]"' in captured.out


def test_check_dependencies_reports_moss_nano_extra_install_hint(monkeypatch, capsys):
    def fake_find_spec(name):
        if name == "sentencepiece":
            return None
        return object()

    monkeypatch.setattr(cli.importlib.util, "find_spec", fake_find_spec)

    assert cli._check_dependencies("moss_nano_onnx") is False

    captured = capsys.readouterr()
    assert "[FAIL] required dependencies for moss_nano_onnx" in captured.out
    assert "  - sentencepiece" in captured.out
    assert 'pip install "rapidtts[moss_nano]"' in captured.out


def test_text_cli_synthesizes_and_saves_audio(monkeypatch, tmp_path):
    seen = {}
    response = FakeSynthesisResponse(
        audio=np.array([0.1, 0.2], dtype=np.float32),
        sample_rate=16000,
    )

    def fake_run_tts(
        text,
        model_name,
        model_dir=None,
        language=None,
        voice=None,
        speed=None,
        sample_rate=None,
        enable_log=True,
    ):
        seen["run_tts"] = {
            "text": text,
            "model_name": model_name,
            "model_dir": model_dir,
            "language": language,
            "voice": voice,
            "speed": speed,
            "sample_rate": sample_rate,
            "enable_log": enable_log,
        }
        return response

    def fake_save_audio(output_path, audio, sample_rate):
        seen["save_audio"] = {
            "output_path": output_path,
            "audio": audio,
            "sample_rate": sample_rate,
        }
        return Path(output_path)

    monkeypatch.setattr(cli, "_run_tts", fake_run_tts)
    monkeypatch.setattr(cli, "_save_audio", fake_save_audio)

    output_path = tmp_path / "1.wav"
    exit_code = cli.main(
        [
            "text",
            "hello",
            str(output_path),
            "--model",
            "melo_onnx",
            "--model-dir",
            "custom_model_dir",
            "--language",
            "EN",
            "--voice",
            "zf_001",
            "--speed",
            "1.2",
            "--sample-rate",
            "16000",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert seen["run_tts"] == {
        "text": "hello",
        "model_name": "melo_onnx",
        "model_dir": "custom_model_dir",
        "language": "EN",
        "voice": "zf_001",
        "speed": 1.2,
        "sample_rate": 16000,
        "enable_log": False,
    }
    assert seen["save_audio"]["output_path"] == str(output_path)
    assert seen["save_audio"]["audio"] is response.audio
    assert seen["save_audio"]["sample_rate"] == 16000


def test_run_tts_passes_voice_to_synthesis_request(monkeypatch):
    import rapidtts

    seen = {}
    response = FakeSynthesisResponse(
        audio=np.array([0.1, 0.2], dtype=np.float32),
        sample_rate=24000,
    )

    class FakeRapidTTS:
        def __init__(self, model, enable_log=True, **kwargs):
            seen["model"] = model.value
            seen["enable_log"] = enable_log
            seen["kwargs"] = kwargs

        def synthesize(self, request):
            seen["request"] = request
            return response

    monkeypatch.setattr(rapidtts, "RapidTTS", FakeRapidTTS)

    result = cli._run_tts(
        text="hello",
        model_name="kokoro_onnx",
        model_dir="custom_model_dir",
        language="EN",
        voice="zm_009",
        speed=1.2,
        sample_rate=16000,
        enable_log=False,
    )

    assert result is response
    assert seen["model"] == "kokoro_onnx"
    assert seen["enable_log"] is False
    assert seen["kwargs"] == {"model_root_dir": "custom_model_dir"}
    assert seen["request"].text == "hello"
    assert seen["request"].language.value == "EN"
    assert seen["request"].voice == "zm_009"
    assert seen["request"].speed == 1.2
    assert seen["request"].sample_rate == 16000


def test_info_cli_prints_model_capability(monkeypatch, capsys):
    capability = ModelCapability(
        name="kokoro_onnx",
        languages=["ZH", "EN", "ZH_MIX_EN"],
        default_language="ZH_MIX_EN",
        voices=["zf_001", "zm_009"],
        default_voice="zf_001",
        voice_source="voices-v1.1-zh.bin",
    )

    monkeypatch.setattr(
        cli,
        "_get_model_capability",
        lambda model_name, model_dir=None, enable_log=True: capability,
    )

    exit_code = cli.main(["info", "kokoro_onnx", "--quiet"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Model: kokoro_onnx" in captured.out
    assert "Languages: ZH, EN, ZH_MIX_EN" in captured.out
    assert "Voices: 2 available" in captured.out
    assert "Default voice: zf_001" in captured.out
    assert "rapidtts voices kokoro_onnx" in captured.out


def test_voices_cli_prints_voice_list(monkeypatch, capsys):
    capability = ModelCapability(
        name="melo_onnx",
        languages=["ZH_MIX_EN"],
        default_language="ZH_MIX_EN",
        voices=["zf_001"],
        default_voice="zf_001",
        voice_source="configuration.json",
    )

    monkeypatch.setattr(
        cli,
        "_get_model_capability",
        lambda model_name, model_dir=None, enable_log=True: capability,
    )

    exit_code = cli.main(["voices", "melo_onnx", "--quiet"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "Available voices for melo_onnx:\n  zf_001\n"


def test_check_model_assets_reports_missing_and_hash_mismatch(monkeypatch, tmp_path):
    good_file = tmp_path / "good.txt"
    bad_file = tmp_path / "bad.txt"
    good_file.write_text("ok", encoding="utf-8")
    bad_file.write_text("bad", encoding="utf-8")
    good_sha256 = hashlib.sha256(b"ok").hexdigest()

    cfg = OmegaConf.create(
        {
            "melo_onnx": {
                "base_url": "https://example.com/models",
                "local_dir": "unused",
                "files": [
                    {
                        "name": "good.txt",
                        "sha256": good_sha256,
                    },
                    {
                        "name": "bad.txt",
                        "sha256": (
                            "00000000000000000000000000000000"
                            "00000000000000000000000000000000"
                        ),
                    },
                    {
                        "name": "missing.txt",
                        "sha256": (
                            "11111111111111111111111111111111"
                            "11111111111111111111111111111111"
                        ),
                    },
                ],
            }
        }
    )
    monkeypatch.setattr(model_assets, "load_models_config", lambda: cfg)

    result = model_assets.check_model_assets("melo_onnx", local_dir=tmp_path)

    assert result.ok is False
    assert [item.name for item in result.missing_files] == ["missing.txt"]
    assert [item.name for item in result.hash_mismatch_files] == ["bad.txt"]


def test_check_model_assets_rejects_unknown_model(monkeypatch):
    monkeypatch.setattr(
        model_assets, "load_models_config", lambda: OmegaConf.create({})
    )

    with pytest.raises(ValueError, match="Unsupported model"):
        model_assets.check_model_assets("unknown")
