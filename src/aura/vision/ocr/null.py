"""An OCR backend that returns scripted text.

Lets the capture path, the privacy gate and the tools be tested without a model
or a screen. If a test needs real OCR to pass, something above the seam has
reached below it.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from aura.vision.ocr.base import OCRProvider, ScreenText, TextLine


class NullOCR(OCRProvider):
    """Returns queued text, then nothing."""

    def __init__(self, texts: list[str] | None = None) -> None:
        self._queue: deque[str] = deque(texts or [])
        self.calls = 0

    def info(self) -> dict[str, Any]:
        return {"name": "null", "kind": "stub"}

    def push(self, text: str) -> None:
        self._queue.append(text)

    def read(self, image: Any) -> ScreenText:
        self.calls += 1
        if not self._queue:
            return ScreenText()
        body = self._queue.popleft()
        return ScreenText(
            lines=[TextLine(text=line, confidence=0.99) for line in body.splitlines() if line],
            latency_ms=0,
        )
