# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import contextlib
import io
import importlib.util
from pathlib import Path
from typing import Optional

from .common.logger import logger, set_logger_enabled
from .core.config import CONFIG_PATH, get_default_backend, load_config
from .core.model_assets import (
    MODELS_CONFIG_PATH,
    available_model_names,
    check_model_assets,
    ensure_model_assets,
    load_models_config,
)

COMMON_DEPENDENCIES = [
    "numpy",
    "onnxruntime",
    "omegaconf",
    "soundfile",
    "colorlog",
    "wetext",
]

MODEL_DEPENDENCIES = {
    "kokoro_onnx": [
        "misaki",
        "phonemizer",
        "espeakng_loader",
    ],
    "melo_onnx": [
        "cn2an",
        "g2p_en",
        "jieba",
        "librosa",
        "pypinyin",
        "tokenizers",
    ],
    "moss_nano_onnx": [
        "librosa",
        "sentencepiece",
    ],
}

MODEL_INSTALL_HINTS = {
    "kokoro_onnx": 'pip install "rapidtts[kokoro]"',
    "melo_onnx": 'pip install "rapidtts[melo]"',
    "moss_nano_onnx": 'pip install "rapidtts[moss_nano]"',
}

def get_default_model_name() -> str:
    return get_default_backend(load_config())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rapidtts")
    subparsers = parser.add_subparsers(dest="command")
    default_model = get_default_model_name()

    download_parser = subparsers.add_parser(
        "download",
        help="Download model assets declared in rapidtts/configs/models.yaml.",
    )
    download_parser.add_argument(
        "model",
        nargs="?",
        default=default_model,
        choices=available_model_names(),
        help="Model name to download.",
    )
    download_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable download progress display.",
    )
    download_parser.add_argument(
        "--save-dir",
        help="Directory to save model assets. Defaults to the RapidTTS package directory.",
    )
    download_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable RapidTTS logs.",
    )
    download_parser.set_defaults(func=download_model)

    check_parser = subparsers.add_parser(
        "check",
        help="Check whether RapidTTS is installed and ready to use.",
    )
    check_parser.add_argument(
        "model",
        nargs="?",
        default=default_model,
        choices=available_model_names(),
        help="Model name to check.",
    )
    check_parser.add_argument(
        "--model-dir",
        help="Directory containing model assets. Defaults to the RapidTTS package directory.",
    )
    check_parser.add_argument(
        "--init-backend",
        action="store_true",
        help="Also initialize RapidTTS backend. This may be slower.",
    )
    check_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable RapidTTS logs.",
    )
    check_parser.set_defaults(func=check_installation)

    info_parser = subparsers.add_parser(
        "info",
        help="Show languages and voices supported by a model.",
    )
    info_parser.add_argument(
        "model",
        nargs="?",
        default=default_model,
        choices=available_model_names(),
        help="Model name to inspect.",
    )
    info_parser.add_argument(
        "--model-dir",
        help="Directory containing model assets. Defaults to the RapidTTS package directory.",
    )
    info_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable RapidTTS logs.",
    )
    info_parser.set_defaults(func=show_model_info)

    voices_parser = subparsers.add_parser(
        "voices",
        help="List voices supported by a model.",
    )
    voices_parser.add_argument(
        "model",
        nargs="?",
        default=default_model,
        choices=available_model_names(),
        help="Model name to inspect.",
    )
    voices_parser.add_argument(
        "--model-dir",
        help="Directory containing model assets. Defaults to the RapidTTS package directory.",
    )
    voices_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable RapidTTS logs.",
    )
    voices_parser.set_defaults(func=show_model_voices)

    text_parser = subparsers.add_parser(
        "text",
        help="Synthesize text and save it to an audio file.",
    )
    text_parser.add_argument("text", help="Text to synthesize.")
    text_parser.add_argument("output", help="Output audio file path.")
    text_parser.add_argument(
        "--model",
        default=default_model,
        choices=available_model_names(),
        help="Model name to use.",
    )
    text_parser.add_argument(
        "--model-dir",
        help="Directory containing model assets. Defaults to the RapidTTS package directory.",
    )
    text_parser.add_argument(
        "--language",
        choices=["ZH", "EN", "ZH_MIX_EN"],
        help="Text language.",
    )
    text_parser.add_argument("--voice", help="Voice name.")
    text_parser.add_argument("--speed", type=float, help="Speech speed.")
    text_parser.add_argument("--sample-rate", type=int, help="Output sample rate.")
    text_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable RapidTTS logs.",
    )
    text_parser.set_defaults(func=synthesize_text)

    return parser


def download_model(args: argparse.Namespace) -> int:
    set_logger_enabled(not args.quiet)

    model_dir = ensure_model_assets(
        args.model,
        local_dir=args.save_dir,
        show_progress=not args.no_progress,
    )
    logger.info('Model "%s" is ready at "%s"', args.model, model_dir)
    return 0


def check_installation(args: argparse.Namespace) -> int:
    set_logger_enabled(not args.quiet)

    ok = True
    ok &= _check_package_import()
    ok &= _check_dependencies(args.model)
    ok &= _check_configs(args.model)
    ok &= _check_model_files(args.model, args.model_dir)

    if args.init_backend:
        ok &= _check_backend_init(args.model, args.model_dir)

    if ok:
        print("RapidTTS installation looks ready.")
        return 0

    print("RapidTTS installation check failed.")
    return 1


def show_model_info(args: argparse.Namespace) -> int:
    set_logger_enabled(not args.quiet)
    capability = _get_model_capability(
        model_name=args.model,
        model_dir=args.model_dir,
        enable_log=not args.quiet,
    )
    print(_format_model_info(capability))
    return 0


def show_model_voices(args: argparse.Namespace) -> int:
    set_logger_enabled(not args.quiet)
    capability = _get_model_capability(
        model_name=args.model,
        model_dir=args.model_dir,
        enable_log=not args.quiet,
    )
    print(_format_model_voices(capability))
    return 0


def synthesize_text(args: argparse.Namespace) -> int:
    set_logger_enabled(not args.quiet)

    response = _run_tts(
        text=args.text,
        model_name=args.model,
        model_dir=args.model_dir,
        language=args.language,
        voice=args.voice,
        speed=args.speed,
        sample_rate=args.sample_rate,
        enable_log=not args.quiet,
    )
    output_path = _save_audio(args.output, response.audio, response.sample_rate)
    logger.info('Audio saved to "%s"', output_path)
    return 0


def _run_tts(
    text: str,
    model_name: str,
    model_dir: Optional[str] = None,
    language: Optional[str] = None,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
    sample_rate: Optional[int] = None,
    enable_log: bool = True,
):
    from rapidtts import RapidTTS, SynthesisRequest, TTSLanguage, TTSModel

    model = TTSModel(model_name)
    kwargs = {"model_root_dir": model_dir} if model_dir else {}
    tts = RapidTTS(model=model, enable_log=enable_log, **kwargs)

    request = SynthesisRequest(
        text=text,
        language=TTSLanguage(language) if language else None,
        voice=voice,
        speed=speed,
        sample_rate=sample_rate,
    )
    return tts.synthesize(request)


def _get_model_capability(
    model_name: str,
    model_dir: Optional[str] = None,
    enable_log: bool = True,
):
    from rapidtts import RapidTTS, TTSModel

    model = TTSModel(model_name)
    kwargs = {"model_root_dir": model_dir} if model_dir else {}
    if enable_log:
        return RapidTTS(model=model, enable_log=enable_log, **kwargs).get_capability()

    with contextlib.redirect_stdout(io.StringIO()):
        tts = RapidTTS(model=model, enable_log=enable_log, **kwargs)
    return tts.get_capability()


def _format_model_info(capability) -> str:
    voice_count = len(capability.voices)
    lines = [
        f"Model: {capability.name}",
        f"Languages: {', '.join(capability.languages)}",
        f"Default language: {capability.default_language}",
        f"Voices: {voice_count} available",
    ]

    if capability.default_voice:
        lines.append(f"Default voice: {capability.default_voice}")
    if capability.voice_source:
        lines.append(f"Voice source: {capability.voice_source}")
    if voice_count:
        lines.extend(
            [
                "Run:",
                f"  rapidtts voices {capability.name}",
            ]
        )
    return "\n".join(lines)


def _format_model_voices(capability) -> str:
    if not capability.voices:
        return f"No voices are declared for {capability.name}."

    lines = [f"Available voices for {capability.name}:"]
    lines.extend(f"  {voice}" for voice in capability.voices)
    return "\n".join(lines)


def _save_audio(output_path: str, audio, sample_rate: int) -> Path:
    import soundfile

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, audio, samplerate=sample_rate)
    return path


def _check_package_import() -> bool:
    try:
        from rapidtts import RapidTTS, SynthesisRequest, TTSModel  # noqa: F401
    except Exception as e:
        print(f"[FAIL] rapidtts package import: {e}")
        return False

    print("[OK] rapidtts package import")
    return True


def _check_dependencies(model_name: str) -> bool:
    dependencies = COMMON_DEPENDENCIES + MODEL_DEPENDENCIES.get(model_name, [])
    missing = [
        dependency
        for dependency in dependencies
        if importlib.util.find_spec(dependency) is None
    ]
    if missing:
        print(f"[FAIL] required dependencies for {model_name}")
        for dependency in missing:
            print(f"  - {dependency}")
        install_hint = MODEL_INSTALL_HINTS.get(model_name)
        if install_hint:
            print("Run:")
            print(f"  {install_hint}")
        return False

    print(f"[OK] required dependencies for {model_name}")
    return True


def _check_configs(model_name: str) -> bool:
    try:
        cfg = load_config()
        models_cfg = load_models_config()

        if model_name not in cfg:
            raise ValueError(f'"{model_name}" not found in {CONFIG_PATH}')
        if model_name not in models_cfg:
            raise ValueError(f'"{model_name}" not found in {MODELS_CONFIG_PATH}')

        model_cfg = models_cfg[model_name]
        for key in ("base_url", "local_dir", "files"):
            if key not in model_cfg:
                raise ValueError(f'"{key}" missing in {MODELS_CONFIG_PATH}')

        for file_cfg in model_cfg.files:
            if "name" not in file_cfg or "sha256" not in file_cfg:
                raise ValueError(f'Invalid file entry in "{MODELS_CONFIG_PATH}"')
    except Exception as e:
        print(f"[FAIL] config files: {e}")
        return False

    print("[OK] config files")
    return True


def _check_model_files(model_name: str, model_dir: Optional[str]) -> bool:
    try:
        result = check_model_assets(model_name, local_dir=model_dir)
    except Exception as e:
        print(f"[FAIL] model assets: {model_name}: {e}")
        return False

    if result.ok:
        print(f'[OK] model assets: {model_name} at "{result.model_dir}"')
        return True

    print(f'[FAIL] model assets: {model_name} at "{result.model_dir}"')
    if result.missing_files:
        print("Missing files:")
        for item in result.missing_files:
            print(f"  - {item.name}")

    if result.hash_mismatch_files:
        print("SHA256 mismatch files:")
        for item in result.hash_mismatch_files:
            print(f"  - {item.name}")

    print("Run:")
    command = f"  rapidtts download {model_name}"
    if model_dir:
        command += f" --save-dir {model_dir}"
    print(command)
    return False


def _check_backend_init(model_name: str, model_dir: Optional[str]) -> bool:
    try:
        from rapidtts import RapidTTS, TTSModel

        model = TTSModel(model_name)
        kwargs = {"model_root_dir": model_dir} if model_dir else {}
        RapidTTS(model=model, enable_log=False, **kwargs)
    except Exception as e:
        print(f"[FAIL] backend init: {e}")
        return False

    print("[OK] backend init")
    return True


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
