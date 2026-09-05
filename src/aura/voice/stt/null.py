"""An STT backend that returns a scripted transcript.

Lets the session loop be driven deterministically in tests: push the phrases
you want "heard", assert on what AURA does with them.
"""

from __future__ import annotations

from collections import deque

from aura.voice.stt.base import STTProvider
from aura.voice.types import AudioArray, Transcript, VoiceInfo


class NullSTT(STTProvider):
    """Returns queued transcripts, then empty ones."""

    def __init__(self, transcripts: list[str] | None = None) -> None:
        self._queue: deque[str] = deque(transcripts or [])
        self.calls = 0

    def info(self) -> VoiceInfo:
        return VoiceInfo(name="null", kind="stub", model="scripted", detail="test double")

    def push(self, text: str) -> None:
        """Queue what the next transcribe() call should return."""
        self._queue.append(text)

    def transcribe(self, audio: AudioArray) -> Transcript:
        self.calls += 1
        text = self._queue.popleft() if self._queue else ""
        return Transcript(
            text=text,
            duration_s=len(audio) / 16_000,
            latency_ms=0,
            confidence=0.0 if text else -5.0,
        )
