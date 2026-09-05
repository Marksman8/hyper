"""A TTS backend that produces silence.

The voice counterpart of `EchoProvider`: it lets the session loop, the sentence
chunker, playback queueing and barge-in all be tested with no models, no audio
hardware and no GPU. If a test needs real speech to pass, something above the
provider seam has reached below it.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from aura.voice.tts.base import TTSProvider
from aura.voice.types import SpeechChunk, VoiceInfo

_SAMPLE_RATE = 22_050
# Roughly conversational pace, so timing-sensitive tests behave like the real
# thing rather than completing instantly.
_SECONDS_PER_CHAR = 0.06


class NullTTS(TTSProvider):
    """Yields correctly-shaped silence, one chunk per call."""

    def __init__(self, sample_rate: int = _SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self.spoken: list[str] = []

    def info(self) -> VoiceInfo:
        return VoiceInfo(name="null", kind="stub", model="silence", detail="test double")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str) -> Iterator[SpeechChunk]:
        text = text.strip()
        if not text:
            return
        self.spoken.append(text)
        samples = max(1, int(len(text) * _SECONDS_PER_CHAR * self._sample_rate))
        yield SpeechChunk(
            audio=np.zeros(samples, dtype=np.float32),
            sample_rate=self._sample_rate,
            text=text,
        )
