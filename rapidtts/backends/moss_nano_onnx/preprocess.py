# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import librosa
import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import soundfile as sf

from ...common.io import load_json
from ...common.text.normalization import create_text_normalizer
from ...core.request import SynthesisRequest
from ...core.typings import TextNormalizerType
from .tts_robust_normalizer_single_script import normalize_tts_text
from .typings import MOSSNanoInput

SENTENCE_END_PUNCTUATION = set(".!?。！？；;")
CLAUSE_SPLIT_PUNCTUATION = set(",，、；;：:")
CLOSING_PUNCTUATION = set("\"'”’)]}）】》」』")


class MOSSNanoPreprocessor:
    def __init__(
        self,
        model_root_dir: Path,
        text_normalizer_type: TextNormalizerType = TextNormalizerType.WETEXT,
    ) -> None:
        manifest_dir = model_root_dir / "MOSS-TTS-Nano-100M-ONNX"
        self.manifest_path = manifest_dir / "browser_poc_manifest.json"
        self.manifest = load_json(self.manifest_path)

        codec_meta_path = (
            model_root_dir
            / "MOSS-Audio-Tokenizer-Nano-ONNX"
            / "codec_browser_onnx_meta.json"
        )
        self.codec_meta = load_json(codec_meta_path)

        self.text_normalizer = create_text_normalizer(text_normalizer_type)

        tokenizer_path = model_root_dir / "MOSS-TTS-Nano-100M-ONNX" / "tokenizer.model"
        self.sp_model = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

        path = (
            model_root_dir
            / "MOSS-Audio-Tokenizer-Nano-ONNX"
            / "moss_audio_tokenizer_encode.onnx"
        )
        self.sessions = ort.InferenceSession(
            str(path), sess_options=None, providers=["CPUExecutionProvider"]
        )

    def run(self, request: SynthesisRequest) -> List[MOSSNanoInput]:
        text = self.text_normalizer.normalize(request.text)
        prompt_text = self.text_normalizer.normalize(
            request.extras.get("prompt_text", "")
        )
        normalized_text = normalize_tts_text(text)
        normalized_prompt_text = normalize_tts_text(prompt_text)

        prompt_audio_codes = self.resolve_prompt_audio_codes(
            voice=request.voice,
            prompt_audio_path=request.extras.get("prompt_audio_path"),
        )

        voice_clone_max_text_tokens = 75
        text_chunks = self.split_voice_clone_text(
            normalized_text, max_tokens=int(voice_clone_max_text_tokens)
        )

        results = []
        for text_chunk in text_chunks:
            text_token_ids = self.encode_text(text_chunk)
            request_rows = self.build_voice_clone_request_rows(
                prompt_audio_codes=prompt_audio_codes, text_token_ids=text_token_ids
            )
            input = MOSSNanoInput(
                text=text_chunk,
                text_token_ids=text_token_ids,
                prompt_text=normalized_prompt_text,
                prompt_audio_codes=prompt_audio_codes,
                request_rows=request_rows,
            )
            results.append(input)
        return results

    def encode_text(self, text: str) -> list[int]:
        return [
            int(token_id)
            for token_id in self.sp_model.encode(str(text or ""), out_type=int)
        ]

    def count_text_tokens(self, text: str) -> int:
        return len(self.encode_text(text))

    def resolve_prompt_audio_codes(
        self,
        *,
        voice: str | None,
        prompt_audio_path: str | Path | None,
    ) -> list[list[int]]:
        if prompt_audio_path:
            return self.encode_reference_audio(prompt_audio_path)

        resolved_voice = str(voice or self.list_builtin_voices()[0]["voice"])
        voice_row = next(
            (
                item
                for item in self.list_builtin_voices()
                if item["voice"] == resolved_voice
            ),
            None,
        )
        if voice_row is None:
            raise ValueError(f"Built-in voice not found: {resolved_voice}")
        return list(voice_row["prompt_audio_codes"])

    def encode_reference_audio(
        self, reference_audio_path: str | Path
    ) -> list[list[int]]:
        waveform = self._load_reference_audio(reference_audio_path)
        waveform_length = int(waveform.shape[-1])
        outputs = self.sessions.run(
            None,
            {
                "waveform": waveform,
                "input_lengths": np.asarray([waveform_length], dtype=np.int32),
            },
        )
        output_names = [output.name for output in self.sessions.get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        audio_codes = np.asarray(named_outputs["audio_codes"], dtype=np.int32)
        audio_code_lengths = np.asarray(
            named_outputs["audio_code_lengths"], dtype=np.int32
        )
        code_length = int(audio_code_lengths.reshape(-1)[0])
        num_quantizers = int(self.codec_meta["codec_config"]["num_quantizers"])
        prompt_audio_codes: list[list[int]] = []
        for frame_index in range(code_length):
            prompt_audio_codes.append(
                [
                    int(audio_codes[0, frame_index, quantizer_index])
                    for quantizer_index in range(num_quantizers)
                ]
            )
        return prompt_audio_codes

    def _load_reference_audio(self, reference_audio_path: str | Path) -> np.ndarray:
        audio_path = str(Path(reference_audio_path).expanduser().resolve())
        waveform, sample_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True,
        )
        waveform = waveform.T

        target_sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
        target_channels = int(self.codec_meta["codec_config"]["channels"])
        if sample_rate != target_sample_rate:
            waveform = librosa.resample(
                waveform.astype(np.float32, copy=False),
                orig_sr=sample_rate,
                target_sr=target_sample_rate,
                axis=-1,
                res_type="soxr_hq",
                fix=True,
                scale=False,
            )

        current_channels = int(waveform.shape[0])
        if current_channels == target_channels:
            pass
        elif current_channels == 1 and target_channels > 1:
            waveform = np.repeat(waveform, target_channels, axis=0)
        elif current_channels > 1 and target_channels == 1:
            waveform = waveform.mean(axis=0, keepdims=True)
        else:
            raise ValueError(
                f"Unsupported reference audio channel conversion: {current_channels} -> {target_channels}"
            )

        return waveform[None, ...].astype(np.float32, copy=False)

    def list_builtin_voices(self) -> list[dict[str, Any]]:
        return list(self.manifest["builtin_voices"])

    def split_voice_clone_text(self, text: str, max_tokens: int = 75) -> list[str]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return []
        safe_max_tokens = max(1, int(max_tokens))
        prepared_text = _prepare_text_for_sentence_chunking(normalized_text)
        sentence_candidates = _split_text_by_punctuation(
            prepared_text, SENTENCE_END_PUNCTUATION
        ) or [prepared_text.strip()]
        sentence_slices: list[tuple[int, str]] = []
        for sentence_text in sentence_candidates:
            normalized_sentence = sentence_text.strip()
            if not normalized_sentence:
                continue
            sentence_token_count = self.count_text_tokens(normalized_sentence)
            if sentence_token_count <= safe_max_tokens:
                sentence_slices.append((sentence_token_count, normalized_sentence))
                continue
            clause_candidates = _split_text_by_punctuation(
                normalized_sentence, CLAUSE_SPLIT_PUNCTUATION
            )
            if len(clause_candidates) <= 1:
                clause_candidates = [normalized_sentence]
            for clause_text in clause_candidates:
                normalized_clause = clause_text.strip()
                if not normalized_clause:
                    continue
                clause_token_count = self.count_text_tokens(normalized_clause)
                if clause_token_count <= safe_max_tokens:
                    sentence_slices.append((clause_token_count, normalized_clause))
                    continue
                for piece in self.split_text_by_token_budget(
                    normalized_clause, safe_max_tokens
                ):
                    normalized_piece = piece.strip()
                    if normalized_piece:
                        sentence_slices.append(
                            (self.count_text_tokens(normalized_piece), normalized_piece)
                        )
        chunks: list[str] = []
        current_chunk = ""
        current_chunk_token_count = 0
        for sentence_token_count, sentence_text in sentence_slices:
            if not current_chunk:
                current_chunk = sentence_text
                current_chunk_token_count = sentence_token_count
                continue
            if current_chunk_token_count + sentence_token_count > safe_max_tokens:
                chunks.append(current_chunk.strip())
                current_chunk = sentence_text
                current_chunk_token_count = sentence_token_count
            else:
                current_chunk = _join_sentence_parts(current_chunk, sentence_text)
                current_chunk_token_count = self.count_text_tokens(current_chunk)
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks if len(chunks) > 1 else [normalized_text]

    def build_text_rows(self, token_ids: list[int]) -> list[list[int]]:
        rows: list[list[int]] = []
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        for token_id in token_ids:
            row = [int(self.manifest["tts_config"]["audio_pad_token_id"])] * row_width
            row[0] = int(token_id)
            rows.append(row)
        return rows

    def build_audio_prefix_rows(
        self, prompt_audio_codes: list[list[int]], slot_token_id: int | None = None
    ) -> list[list[int]]:
        rows: list[list[int]] = []
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        resolved_slot_token_id = int(
            self.manifest["tts_config"]["audio_user_slot_token_id"]
            if slot_token_id is None
            else slot_token_id
        )
        for code_row in prompt_audio_codes:
            row = [int(self.manifest["tts_config"]["audio_pad_token_id"])] * row_width
            row[0] = resolved_slot_token_id
            for index in range(
                min(len(code_row), int(self.manifest["tts_config"]["n_vq"]))
            ):
                row[index + 1] = int(code_row[index])
            rows.append(row)
        return rows

    def build_voice_clone_request_rows(
        self, prompt_audio_codes: list[list[int]], text_token_ids: list[int]
    ) -> dict[str, list[list[int]]]:
        prefix_text_token_ids = [
            *self.manifest["prompt_templates"]["user_prompt_prefix_token_ids"],
            int(self.manifest["tts_config"]["audio_start_token_id"]),
        ]
        suffix_text_token_ids = [
            int(self.manifest["tts_config"]["audio_end_token_id"]),
            *self.manifest["prompt_templates"]["user_prompt_after_reference_token_ids"],
            *text_token_ids,
            *self.manifest["prompt_templates"]["assistant_prompt_prefix_token_ids"],
            int(self.manifest["tts_config"]["audio_start_token_id"]),
        ]
        rows = [
            *self.build_text_rows(prefix_text_token_ids),
            *self.build_audio_prefix_rows(prompt_audio_codes),
            *self.build_text_rows(suffix_text_token_ids),
        ]
        return {
            "inputIds": rows,
            "attentionMask": [[1 for _ in rows]],
        }


def _prepare_text_for_sentence_chunking(text: str) -> str:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        raise ValueError("Text prompt cannot be empty.")
    normalized_text = normalized_text.replace("\r", " ").replace("\n", " ")
    while "  " in normalized_text:
        normalized_text = normalized_text.replace("  ", " ")
    if _contains_cjk(normalized_text):
        if normalized_text[-1] not in SENTENCE_END_PUNCTUATION:
            normalized_text += "。"
        return normalized_text
    if normalized_text[:1].islower():
        normalized_text = normalized_text[:1].upper() + normalized_text[1:]
    if normalized_text[-1].isalnum():
        normalized_text += "."
    if len([item for item in normalized_text.split() if item]) < 5:
        normalized_text = f"        {normalized_text}"
    return normalized_text


def _contains_cjk(text: str) -> bool:
    for character in str(text or ""):
        if (
            "\u4e00" <= character <= "\u9fff"
            or "\u3400" <= character <= "\u4dbf"
            or "\u3040" <= character <= "\u30ff"
            or "\uac00" <= character <= "\ud7af"
        ):
            return True
    return False


def _split_text_by_punctuation(text: str, punctuation: set[str]) -> list[str]:
    sentences: list[str] = []
    current_chars: list[str] = []
    index = 0
    normalized_text = str(text or "")
    while index < len(normalized_text):
        character = normalized_text[index]
        current_chars.append(character)
        if character in punctuation:
            lookahead = index + 1
            while (
                lookahead < len(normalized_text)
                and normalized_text[lookahead] in CLOSING_PUNCTUATION
            ):
                current_chars.append(normalized_text[lookahead])
                lookahead += 1
            sentence = "".join(current_chars).strip()
            if sentence:
                sentences.append(sentence)
            current_chars.clear()
            while (
                lookahead < len(normalized_text)
                and normalized_text[lookahead].isspace()
            ):
                lookahead += 1
            index = lookahead
            continue
        index += 1
    tail = "".join(current_chars).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _join_sentence_parts(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if _contains_cjk(left) or _contains_cjk(right):
        return left + right
    return f"{left} {right}"
