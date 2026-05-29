# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

from omegaconf import OmegaConf

from ..common.download import ensure_file, sha256_file

cur_dir = Path(__file__).resolve().parent
PACKAGE_ROOT = cur_dir.parent
MODELS_CONFIG_PATH = cur_dir.parent / "configs" / "models.yaml"


@dataclass
class ModelAssetIssue:
    name: str
    path: Path
    expected_sha256: str
    actual_sha256: Optional[str] = None


@dataclass
class ModelAssetCheckResult:
    model_name: str
    model_dir: Path
    missing_files: list[ModelAssetIssue]
    hash_mismatch_files: list[ModelAssetIssue]

    @property
    def ok(self) -> bool:
        return not self.missing_files and not self.hash_mismatch_files


@lru_cache(maxsize=1)
def load_models_config():
    cfg = OmegaConf.load(MODELS_CONFIG_PATH)
    OmegaConf.set_readonly(cfg, True)
    return cfg


def available_model_names() -> list[Any]:
    cfg = load_models_config()
    return list(cfg.keys())


def ensure_model_assets(
    model_name: str,
    local_dir: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
    groups: Optional[list[str]] = None,
    include_base_files: bool = True,
) -> Path:
    cfg = load_models_config()
    if model_name not in cfg:
        names = ", ".join(available_model_names())
        raise ValueError(f'Unsupported model "{model_name}". Available models: {names}')

    model_cfg = cfg[model_name]
    base_url = str(model_cfg.base_url).rstrip("/")
    if local_dir is None:
        model_dir = _resolve_local_dir(model_cfg.local_dir)
    else:
        model_dir = Path(local_dir)

    for file_cfg in _get_model_files(
        model_cfg, groups=groups, include_base_files=include_base_files
    ):
        ensure_file(
            url=f"{base_url}/{file_cfg.name}",
            save_path=model_dir / file_cfg.name,
            sha256=file_cfg.sha256,
            show_progress=show_progress,
        )

    return model_dir


def check_model_assets(
    model_name: str,
    local_dir: Optional[Union[str, Path]] = None,
    groups: Optional[list[str]] = None,
    include_base_files: bool = True,
) -> ModelAssetCheckResult:
    cfg = load_models_config()
    if model_name not in cfg:
        names = ", ".join(available_model_names())
        raise ValueError(f'Unsupported model "{model_name}". Available models: {names}')

    model_cfg = cfg[model_name]
    if local_dir is None:
        model_dir = _resolve_local_dir(model_cfg.local_dir)
    else:
        model_dir = Path(local_dir)

    missing_files = []
    hash_mismatch_files = []

    for file_cfg in _get_model_files(
        model_cfg, groups=groups, include_base_files=include_base_files
    ):
        path = model_dir / file_cfg.name
        expected_sha256 = str(file_cfg.sha256).lower()

        if not path.exists():
            missing_files.append(
                ModelAssetIssue(
                    name=file_cfg.name,
                    path=path,
                    expected_sha256=expected_sha256,
                )
            )
            continue

        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            hash_mismatch_files.append(
                ModelAssetIssue(
                    name=file_cfg.name,
                    path=path,
                    expected_sha256=expected_sha256,
                    actual_sha256=actual_sha256,
                )
            )

    return ModelAssetCheckResult(
        model_name=model_name,
        model_dir=model_dir,
        missing_files=missing_files,
        hash_mismatch_files=hash_mismatch_files,
    )


def _resolve_local_dir(local_dir: str) -> Path:
    path = Path(local_dir)
    if path.is_absolute():
        return path

    return PACKAGE_ROOT / path


def _get_model_files(
    model_cfg, groups: Optional[list[str]] = None, include_base_files: bool = True
) -> list[dict]:
    files = list(model_cfg.files) if include_base_files else []
    optional_files = model_cfg.get("optional_files", {})

    for group in groups or []:
        if group not in optional_files:
            raise ValueError(
                f'Optional file group "{group}" not found for model. '
                f"Available groups: {', '.join(optional_files.keys())}"
            )
        files.extend(optional_files[group])
    return files
