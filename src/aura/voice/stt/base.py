"""Speech-to-text interface.

The §32 rule applied to speech: the session loop depends on this ABC and
nothing else, so `base.en` on CPU, `large-v3` on GPU, a whisper.cpp server on
another machine, or a future fine-tuned model are all config changes.

Implementations must be safe to call from a worker thread and must tolerate
being asked to load lazily -- with 0.8 GB of free RAM, a model that loads at
import time is a model that makes AURA unusable while it is not being used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aura.voice.types import AudioArray, Transcript, VoiceInfo


class STTProvider(ABC):
    """Transcribes an utterance."""

    @abstractmethod
    def info(self) -> VoiceInfo:
        """Static description. Must not load the model or perform I/O."""

    @abstractmethod
    def transcribe(self, audio: AudioArray) -> Transcript:
        """Transcribe mono float32 audio at `SAMPLE_RATE`.

        Blocking and CPU-bound: callers run it off the event loop. Must return
        an empty Transcript rather than raise when the audio contains no
        speech, because silence is a normal outcome of a false wake.
        """

    def load(self) -> None:  # noqa: B027 - optional hook; a stateless backend needs no-op defaults
        """Bring the model into memory. Idempotent.

        Separate from __init__ so the session loop can decide when to pay the
        cost, and can pay it during the wake word rather than after it.
        """

    def unload(self) -> None:  # noqa: B027 - optional hook; a stateless backend needs no-op defaults
        """Release the model. Idempotent.

        Called after an idle period. On a 16 GB machine this is not an
        optimisation, it is what makes the assistant coexist with a browser.
        """

    @property
    def is_loaded(self) -> bool:
        return True
