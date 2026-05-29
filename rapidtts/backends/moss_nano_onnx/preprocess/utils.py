# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import List

import numpy as np


def flatten3d_int32(nested: list[list[list[int]]]) -> tuple[np.ndarray, list[int]]:
    array = np.asarray(nested, dtype=np.int32)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D nested list, got shape {array.shape}")

    return array.reshape(-1), list(array.shape)


def flatten2d_int32(nested: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    array = np.asarray(nested, dtype=np.int32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D nested list, got shape {array.shape}")

    return array.reshape(-1), list(array.shape)


def contains_cjk(text: str) -> bool:
    for character in str(text or ""):
        if (
            "\u4e00" <= character <= "\u9fff"
            or "\u3400" <= character <= "\u4dbf"
            or "\u3040" <= character <= "\u30ff"
            or "\uac00" <= character <= "\ud7af"
        ):
            return True
    return False


def find_preferred_cut_index(
    prefix: str, boundary_chars: set[str], search_window: int = 25
) -> int:
    start = max(0, len(prefix) - search_window)

    for index in range(len(prefix) - 1, start - 1, -1):
        if prefix[index] in boundary_chars:
            return index + 1

    return len(prefix)


def prepare_text_for_sentence_chunking(
    normalized_text: str, sentence_end_punctuation: set[str]
) -> str:
    if not normalized_text:
        raise ValueError("Text prompt cannot be empty.")

    normalized_text = normalized_text.replace("\r", " ").replace("\n", " ")

    while "  " in normalized_text:
        normalized_text = normalized_text.replace("  ", " ")

    if contains_cjk(normalized_text):
        if normalized_text[-1] not in sentence_end_punctuation:
            normalized_text += "。"
        return normalized_text

    if normalized_text[:1].islower():
        normalized_text = normalized_text[:1].upper() + normalized_text[1:]

    if normalized_text[-1].isalnum():
        normalized_text += "."

    if len([item for item in normalized_text.split() if item]) < 5:
        normalized_text = f"        {normalized_text}"

    return normalized_text


def split_text_by_punctuation(
    text: str, punctuation: set[str], closing_punctuation: set[str]
) -> List[str]:
    sentences: List[str] = []
    current_chars: List[str] = []

    index = 0
    normalized_text = str(text or "")
    while index < len(normalized_text):
        character = normalized_text[index]
        current_chars.append(character)

        if character not in punctuation:
            index += 1
            continue

        lookahead = index + 1
        while (
            lookahead < len(normalized_text)
            and normalized_text[lookahead] in closing_punctuation
        ):
            current_chars.append(normalized_text[lookahead])
            lookahead += 1

        sentence = "".join(current_chars).strip()

        if sentence:
            sentences.append(sentence)

        current_chars.clear()
        while lookahead < len(normalized_text) and normalized_text[lookahead].isspace():
            lookahead += 1

        index = lookahead

    tail = "".join(current_chars).strip()
    if tail:
        sentences.append(tail)
    return sentences


def join_sentence_parts(left: str, right: str) -> str:
    if not left:
        return right

    if not right:
        return left

    if contains_cjk(left) or contains_cjk(right):
        return left + right

    return f"{left} {right}"


def convert_audio_channels(waveform: np.ndarray, target_channels: int) -> np.ndarray:
    current_channels = int(waveform.shape[0])

    if current_channels == target_channels:
        return waveform

    if current_channels == 1 and target_channels > 1:
        return np.repeat(waveform, target_channels, axis=0)

    if current_channels > 1 and target_channels == 1:
        return waveform.mean(axis=0, keepdims=True)

    raise ValueError(
        f"Unsupported reference audio channel conversion: "
        f"{current_channels} -> {target_channels}"
    )
