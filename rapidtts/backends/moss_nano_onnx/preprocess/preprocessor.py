# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ....common.io import load_json
from ....core.request import SynthesisRequest
from ....core.typings import TextNormalizerType
from ..typings import MOSSNanoInput
from .prompt_audio import PromptAudioProcessor
from .request_builder import RequestBuilder
from .text_processor import TextProcessor

VOICE_CLONE_MAX_TEXT_TOKENS = 75


class MOSSNanoPreprocessor:
    def __init__(
        self,
        model_root_dir: Path,
        engine_cfg_defaults: Dict[str, Any],
        device: str = "cpu",
        text_normalizer_type: TextNormalizerType = TextNormalizerType.WETEXT,
    ):
        tts_model_dir = model_root_dir / "MOSS-TTS-Nano-100M-ONNX"
        tokenizer_dir = model_root_dir / "MOSS-Audio-Tokenizer-Nano-ONNX"

        manifest_path = tts_model_dir / "browser_poc_manifest.json"
        self.manifest = load_json(manifest_path)

        self.text_processor = TextProcessor(tts_model_dir, text_normalizer_type)
        self.prompt_audio_processor = PromptAudioProcessor(
            tokenizer_dir, engine_cfg_defaults, self.manifest, device
        )
        self.request_builder = RequestBuilder(self.manifest)

    def run(self, request: SynthesisRequest) -> List[MOSSNanoInput]:
        normalized_text = self.text_processor.normalize(request.text)
        normalized_prompt_text = self.text_processor.normalize(
            request.extras.get("prompt_text", "")
        )

        prompt_audio_codes = self.prompt_audio_processor.resolve_prompt_audio_codes(
            voice=request.voice,
            prompt_audio_path=request.extras.get("prompt_audio_path"),
        )

        text_chunks = self.text_processor.split_voice_clone_text(
            normalized_text, max_tokens=VOICE_CLONE_MAX_TEXT_TOKENS
        )

        results = []
        for text_chunk in text_chunks:
            text_token_ids = self.text_processor.encode_text(text_chunk)
            request_rows = self.request_builder.build_request_rows(
                prompt_audio_codes, text_token_ids
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

    def get_voices(self) -> list[str]:
        return [
            str(voice_row["voice"])
            for voice_row in self.prompt_audio_processor.builtin_voices
            if "voice" in voice_row
        ]
