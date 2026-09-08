"""The OCR interface.

Same seam discipline as LLMProvider and STTProvider: everything above depends
on this ABC, so RapidOCR today and a vision-language model later are the same
change to one config line.

Only implementations in this package may import an OCR library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TextLine:
    """One recognised line, with how sure the recogniser was."""

    text: str
    confidence: float
    # Bounding box as (left, top, right, bottom) in pixels, when reported.
    box: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class ScreenText:
    """Everything read from one image."""

    lines: list[TextLine] = field(default_factory=list)
    latency_ms: int = 0
    # Lines below the confidence floor, kept separately rather than dropped:
    # the caller can decide, and silently discarding text is its own failure.
    uncertain: list[TextLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def is_empty(self) -> bool:
        return not self.lines and not self.uncertain


class OCRProvider(ABC):
    """Reads text out of an image."""

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """Static description. Must not load the model."""

    @abstractmethod
    def read(self, image: Any) -> ScreenText:
        """Extract text from an RGB image array.

        Blocking and CPU-bound. Returns an empty ScreenText rather than raising
        when there is no text: a blank screen is a normal outcome.
        """

    def load(self) -> None:  # noqa: B027 - optional hook
        """Bring the model into memory. Idempotent."""

    def unload(self) -> None:  # noqa: B027 - optional hook
        """Release it. Idempotent."""
