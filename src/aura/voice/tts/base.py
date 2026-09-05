"""Text-to-speech interface.

`synthesize` returns an iterator rather than one buffer on purpose. The 1.5 s
latency budget is only met by speaking the first sentence while the model is
still generating the second, so the whole pipeline -- LLM tokens, sentence
chunking, synthesis, playback -- has to stream end to end. A backend that can
only produce a whole utterance at once still satisfies this interface by
yielding once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from aura.voice.types import SpeechChunk, VoiceInfo


class TTSProvider(ABC):
    """Turns text into playable audio."""

    @abstractmethod
    def info(self) -> VoiceInfo:
        """Static description. Must not load the model or perform I/O."""

    @abstractmethod
    def synthesize(self, text: str) -> Iterator[SpeechChunk]:
        """Yield audio for `text`, earliest fragment first.

        Blocking and CPU-bound. Yields nothing for text with no speakable
        content, rather than raising -- the sentence chunker upstream can
        legitimately hand over a fragment that is only punctuation.
        """

    def load(self) -> None:  # noqa: B027 - optional hook; a stateless backend needs no-op defaults
        """Bring the voice into memory. Idempotent."""

    def unload(self) -> None:  # noqa: B027 - optional hook; a stateless backend needs no-op defaults
        """Release the voice. Idempotent."""

    @property
    def is_loaded(self) -> bool:
        return True
