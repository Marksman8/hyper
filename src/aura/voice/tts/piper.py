"""Piper text-to-speech.

Chosen for the shape of the problem rather than for top-line quality: Piper is
ONNX (no torch), runs on CPU at roughly 0.1x real time, loads in well under a
second, and costs ~65 MB per voice. On a machine with 0.8 GB of free RAM, a
heavier neural TTS would have to evict Whisper to speak.

This module and its siblings are the only places allowed to import a TTS
library (see tests/unit/test_voice_seam.py).
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from aura.logging import get_logger
from aura.voice.tts.base import TTSProvider
from aura.voice.types import SpeechChunk, VoiceInfo

log = get_logger(__name__)

_INT16_MAX = 32768.0

# Split on sentence enders, keeping the punctuation. Deliberately simple: this
# runs on a partial LLM stream, so it must not need a parser, a model, or the
# rest of the sentence. Abbreviations occasionally split early, which costs a
# slightly odd pause and nothing else.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$|(?<=[;:])\s+")

# Trailing forms that end in a period without ending a sentence. Checked
# instead of a minimum length, because a length rule swallows the short
# answers ("Yes." "Done.") that a voice assistant gives most often.
_ABBREVIATION = re.compile(
    r"(?:\b[A-Za-z]|\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|etc|approx|Fig|Inc|Ltd|e\.g|i\.e)|\b\d+)\.\s*$",
    re.IGNORECASE,
)


def _is_abbreviation(head: str) -> bool:
    """True if this apparent sentence end is really 'Dr.', 'e.g.' or '1.'."""
    return bool(_ABBREVIATION.search(head))


class PiperTTS(TTSProvider):
    """Synthesises speech with a Piper ONNX voice."""

    def __init__(
        self,
        model_path: Path,
        config_path: Path | None = None,
        length_scale: float | None = None,
        volume: float = 1.0,
    ) -> None:
        self._model_path = model_path
        self._config_path = config_path or Path(str(model_path) + ".json")
        self._length_scale = length_scale
        self._volume = volume
        self._voice: Any | None = None
        self._sample_rate = 22_050

    def info(self) -> VoiceInfo:
        return VoiceInfo(
            name="piper",
            kind="onnx_cpu",
            model=self._model_path.stem,
            detail=f"{self._sample_rate} Hz",
        )

    @property
    def is_loaded(self) -> bool:
        return self._voice is not None

    def load(self) -> None:
        if self._voice is not None:
            return
        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"Piper voice not found at {self._model_path}. "
                "Run: uv run python scripts/download_voice.py"
            )
        from piper import PiperVoice

        started = time.perf_counter()
        voice = PiperVoice.load(self._model_path, config_path=self._config_path)
        self._voice = voice
        self._sample_rate = int(voice.config.sample_rate)
        log.info(
            "tts.loaded",
            model=self._model_path.stem,
            sample_rate=self._sample_rate,
            load_ms=int((time.perf_counter() - started) * 1000),
        )

    def unload(self) -> None:
        if self._voice is not None:
            self._voice = None
            log.debug("tts.unloaded")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str) -> Iterator[SpeechChunk]:
        text = text.strip()
        if not text:
            return
        self.load()

        from piper import SynthesisConfig

        syn = SynthesisConfig(
            length_scale=self._length_scale,
            volume=self._volume,
            normalize_audio=True,
        )
        started = time.perf_counter()
        first = True
        assert self._voice is not None
        for chunk in self._voice.synthesize(text, syn_config=syn):
            audio = (np.asarray(chunk.audio_int16_array, dtype=np.float32) / _INT16_MAX).astype(
                np.float32
            )
            if first:
                log.debug(
                    "tts.first_chunk",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    chars=len(text),
                )
                first = False
            yield SpeechChunk(audio=audio, sample_rate=self._sample_rate, text=text)


def split_sentences(text: str) -> list[str]:
    """Break text into speakable units.

    Used to start speaking before the model has finished writing. Anything with
    no letters or digits is dropped: a fragment of pure punctuation would make
    Piper emit a click.
    """
    parts = [p.strip() for p in _SENTENCE_END.split(text) if p and p.strip()]
    return [p for p in parts if re.search(r"[A-Za-z0-9]", p)]


class SentenceStreamer:
    """Turns a token stream into complete sentences, as early as possible.

    The LLM emits fragments ("The ", "capital", " of Fr"); Piper needs whole
    clauses to get prosody right. This buffers until a sentence boundary
    appears, then releases -- so synthesis of sentence one overlaps generation
    of sentence two, which is what keeps time-to-first-audio near 1 s instead
    of tracking the full length of the answer.
    """

    def __init__(self, max_chars: int = 240) -> None:
        self._buf = ""
        self._max = max_chars

    def push(self, delta: str) -> list[str]:
        """Add a token; return any sentences that are now complete."""
        self._buf += delta
        out: list[str] = []

        # `search_from` rather than restarting at 0: when a boundary turns out
        # to be an abbreviation we must keep looking *further along* the
        # buffer. An earlier version stopped instead, which silently made short
        # replies ("Yes." "Done.") never get spoken until the stream ended.
        search_from = 0
        while True:
            match = _SENTENCE_END.search(self._buf, search_from)
            if match is None:
                break
            head = self._buf[: match.end()]
            if _is_abbreviation(head):
                search_from = match.end()
                continue
            out.append(head.strip())
            self._buf = self._buf[match.end() :]
            search_from = 0

        # A very long clause with no punctuation would otherwise never be
        # spoken until the stream ended. Release it at a word boundary.
        while len(self._buf) > self._max:
            cut = self._buf.rfind(" ", 0, self._max)
            if cut <= 0:
                break
            out.append(self._buf[:cut].strip())
            self._buf = self._buf[cut:].lstrip()

        return [s for s in out if re.search(r"[A-Za-z0-9]", s)]

    def flush(self) -> list[str]:
        """Release whatever is left at end of stream."""
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest and re.search(r"[A-Za-z0-9]", rest) else []
