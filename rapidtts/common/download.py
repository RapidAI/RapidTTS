# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import hashlib
import sys
from pathlib import Path
from typing import Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .logger import logger

CHUNK_SIZE = 1024 * 1024
USER_AGENT = "RapidTTS"


def ensure_file(
    url: str,
    save_path: Union[str, Path],
    sha256: str,
    timeout: float = 30.0,
    show_progress: bool = True,
) -> Path:
    path = Path(save_path)
    expected_sha256 = sha256.lower()

    if path.exists() and sha256_file(path) == expected_sha256:
        logger.info(f'✅ File "{path}" already exists and matches the expected SHA256.')
        return path

    download_file(url, path, timeout=timeout, show_progress=show_progress)

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        path.unlink(missing_ok=True)
        logger.error(
            f'SHA256 mismatch for "{path}": expected {expected_sha256}, got {actual_sha256}'
        )
        raise FileHashMismatchError(
            f'SHA256 mismatch for "{path}": '
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    return path


def sha256_file(file_path: Union[str, Path]) -> str:
    path = Path(file_path)
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_file(
    url: str,
    save_path: Union[str, Path],
    timeout: float = 30.0,
    show_progress: bool = True,
) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_name(f"{path.name}.tmp")
    request = Request(url, headers={"User-Agent": USER_AGENT})

    logger.info(f'Downloading "{url}" to "{path}"...')

    try:
        with urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as f:
            total_size = _get_content_length(response)
            downloaded_size = 0

            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break

                f.write(chunk)
                downloaded_size += len(chunk)

                if show_progress:
                    _render_progress(path.name, downloaded_size, total_size)
    except (HTTPError, URLError, OSError) as e:
        tmp_path.unlink(missing_ok=True)
        logger.error(f'Failed to download "{url}" to "{path}": {e}')
        raise DownloadError(f'Failed to download "{url}" to "{path}": {e}') from e

    if show_progress:
        _finish_progress()

    tmp_path.replace(path)
    return path


def _get_content_length(response) -> Optional[int]:
    content_length = response.headers.get("Content-Length")
    if content_length is None:
        return None

    try:
        return int(content_length)
    except ValueError:
        return None


def _render_progress(
    filename: str, downloaded_size: int, total_size: Optional[int]
) -> None:
    if total_size:
        percent = min(downloaded_size / total_size, 1.0)
        bar_width = 30
        filled_width = int(bar_width * percent)
        bar = "#" * filled_width + "-" * (bar_width - filled_width)
        message = (
            f"\rDownloading {filename} [{bar}] "
            f"{percent * 100:6.2f}% "
            f"{_format_size(downloaded_size)}/{_format_size(total_size)}"
        )
    else:
        message = f"\rDownloading {filename} {_format_size(downloaded_size)}"

    sys.stderr.write(message)
    sys.stderr.flush()


def _finish_progress() -> None:
    sys.stderr.write("\n")
    sys.stderr.flush()


def _format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.2f}{unit}"
        value /= 1024

    return f"{value:.2f}TB"


class DownloadError(RuntimeError):
    """Raised when a file cannot be downloaded."""


class FileHashMismatchError(RuntimeError):
    """Raised when a downloaded file does not match the expected hash."""
