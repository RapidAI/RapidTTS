# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import io
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


class AudioSaveBackend(Enum):
    SOUNDFILE = "soundfile"
    WAVE = "wave"


def save_audio(
    save_path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    audio_format: str = "wav",
    backend: AudioSaveBackend = AudioSaveBackend.SOUNDFILE,
) -> Path:
    backend = AudioSaveBackend(backend)
    output_path = Path(save_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if backend == AudioSaveBackend.WAVE:
        save_wav_with_wave(output_path, audio, sample_rate, audio_format)
    elif backend == AudioSaveBackend.SOUNDFILE:
        import soundfile

        soundfile.write(output_path, audio, samplerate=sample_rate)
    else:
        raise ValueError(f"Unsupported audio save backend: {backend}")

    return output_path


def save_wav_with_wave(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
    audio_format: str,
) -> None:
    import wave

    if audio_format.lower() != "wav":
        raise ValueError("wave backend only supports wav output")

    audio = np.asarray(audio)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)

    if audio.dtype == np.int16:
        pcm16 = audio
    else:
        pcm16 = np.round(np.clip(audio.astype(np.float32), -1.0, 1.0) * 32767.0)
        pcm16 = pcm16.astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(int(pcm16.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def maybe_resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: Optional[int],
) -> tuple[np.ndarray, int]:
    if target_sr is None or target_sr == orig_sr:
        return audio, orig_sr

    import librosa

    out = librosa.resample(
        audio.astype("float32"),
        orig_sr=orig_sr,
        target_sr=target_sr,
    )
    return out.astype("float32"), target_sr
