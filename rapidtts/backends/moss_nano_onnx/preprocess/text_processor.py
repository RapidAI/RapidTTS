# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from pathlib import Path
from typing import List

import sentencepiece as spm

from ....common.text.normalization import create_text_normalizer
from ....common.text.post_tn_sanitizer import sanitize_post_tn_text
from ....core.typings import TextNormalizerType
from .utils import (
    find_preferred_cut_index,
    join_sentence_parts,
    prepare_text_for_sentence_chunking,
    split_text_by_punctuation,
)

SENTENCE_END_PUNCTUATION = set(".!?。！？；;")
SEGMENT_SPLIT_PUNCTUATION = set(",，、；;：:")
CLOSING_PUNCTUATION = set("\"'”’)]}）】》」』")
VOICE_CLONE_MAX_TEXT_TOKENS = 75


class TextProcessor:
    def __init__(
        self,
        tts_model_dir: Path,
        text_normalizer_type: TextNormalizerType = TextNormalizerType.WETEXT,
    ):
        self.text_normalizer = create_text_normalizer(text_normalizer_type)

        tokenizer_path = tts_model_dir / "tokenizer.model"
        self.sp_model = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    def normalize(self, text: str) -> str:
        normalized_text = self.text_normalizer.normalize(text)
        sanitized_text = sanitize_post_tn_text(normalized_text)
        return sanitized_text

    def encode_text(self, text: str) -> List[int]:
        return self.sp_model.encode(text or "", out_type=int)

    def split_voice_clone_text(self, text: str, max_tokens: int = 75) -> List[str]:
        normalized_text = text.strip()
        if not normalized_text:
            return []

        token_limit = max(1, int(max_tokens))
        prepared_text = prepare_text_for_sentence_chunking(
            normalized_text, SENTENCE_END_PUNCTUATION
        )

        parts = self.split_text_to_token_budget_parts(prepared_text, token_limit)
        chunks = self.merge_text_parts_by_token_budget(parts, token_limit)
        return chunks if len(chunks) > 1 else [normalized_text]

    def split_text_to_token_budget_parts(
        self, text: str, token_limit: int
    ) -> list[tuple[int, str]]:
        sentences = split_text_by_punctuation(
            text, SENTENCE_END_PUNCTUATION, CLOSING_PUNCTUATION
        ) or [text.strip()]

        parts: list[tuple[int, str]] = []
        for sentence in sentences:
            if not sentence:
                continue

            token_count = self.count_text_tokens(sentence)
            if token_count <= token_limit:
                parts.append((token_count, sentence))
                continue

            segments = split_text_by_punctuation(
                sentence, SEGMENT_SPLIT_PUNCTUATION, CLOSING_PUNCTUATION
            )
            if len(segments) <= 1:
                segments = [sentence]

            for segment in segments:
                if not segment:
                    continue

                token_count = self.count_text_tokens(segment)
                if token_count <= token_limit:
                    parts.append((token_count, segment))
                    continue

                for piece in self.split_text_by_token_budget(segment, token_limit):
                    if piece:
                        parts.append((self.count_text_tokens(piece), piece))

        return parts

    def split_text_by_token_budget(self, text: str, max_tokens: int) -> List[str]:
        if not text:
            return []

        pieces: List[str] = []
        boundary_chars = (
            set(SEGMENT_SPLIT_PUNCTUATION) | set(SENTENCE_END_PUNCTUATION) | {" "}
        )

        while text:
            if self.count_text_tokens(text) <= max_tokens:
                pieces.append(text)
                break

            prefix_length = self.find_max_prefix_length(text, max_tokens)
            cut_index = find_preferred_cut_index(
                text[:prefix_length],
                boundary_chars,
            )

            piece = text[:cut_index].strip()
            pieces.append(piece)
            text = text[cut_index:].strip()
        return pieces

    def find_max_prefix_length(self, text: str, max_tokens: int) -> int:
        low = 1
        high = len(text)
        best = 1

        while low <= high:
            mid = (low + high) // 2
            candidate = text[:mid].strip()

            if not candidate:
                low = mid + 1
                continue

            if self.count_text_tokens(candidate) <= max_tokens:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        return best

    def count_text_tokens(self, text: str) -> int:
        return len(self.encode_text(text))

    def merge_text_parts_by_token_budget(
        self, parts: list[tuple[int, str]], token_limit: int
    ) -> List[str]:
        chunks: List[str] = []

        current_chunk = ""
        current_token_count = 0
        for token_count, text in parts:
            if not current_chunk:
                current_chunk = text
                current_token_count = token_count
                continue

            if current_token_count + token_count > token_limit:
                chunks.append(current_chunk.strip())
                current_chunk = text
                current_token_count = token_count
                continue

            current_chunk = join_sentence_parts(current_chunk, text)
            current_token_count = self.count_text_tokens(current_chunk)

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
