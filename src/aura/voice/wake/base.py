"""Wake word interface.

The detector is the only component that runs continuously on live microphone
audio, so it is the one place where cost and privacy are both non-negotiable:
it must stay under ~1% of one core, and nothing it hears may leave the process
or reach a model until it fires.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aura.voice.types import AudioArray, VoiceInfo, WakeEvent


class WakeWordDetector(ABC):
    """Watches a live audio stream for a trigger phrase."""

    @abstractmethod
    def info(self) -> VoiceInfo:
        """Static description. Must not load the model or perform I/O."""

    @abstractmethod
    def process(self, frame: AudioArray) -> WakeEvent | None:
        """Feed one frame; return an event iff the wake word just fired.

        `frame` is mono float32 at `SAMPLE_RATE`, normally `FRAME_SAMPLES`
        long. Called on every frame in real time, so it must be cheap and must
        never block.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state after a detection.

        Without this, the scores that triggered a detection stay in the
        model's buffer and immediately re-trigger on the next frame.
        """

    def load(self) -> None:  # noqa: B027 - optional hook; a stateless backend needs no-op defaults
        """Bring the model into memory. Idempotent."""

    @property
    def is_loaded(self) -> bool:
        return True
