"""RapidOCR: text recognition on onnxruntime.

Chosen over a vision-language model for the reasons in
docs/architecture/04-phase5-vision.md §4 — it is ~15 MB of CPU inference that
answers "what does this error say?" without evicting the LLM from VRAM.

This is one of the few modules permitted to import an inference library. Nothing
above it may; see tests/unit/test_seam.py.
"""

from __future__ import annotations

import time
from typing import Any

from aura.logging import get_logger
from aura.vision.ocr.base import OCRProvider, ScreenText, TextLine

log = get_logger(__name__)

# Lines below this are reported as uncertain rather than as fact. OCR's failure
# mode is confident nonsense, and presenting a bad read as a good one is worse
# than saying "I could not read that clearly".
DEFAULT_MIN_CONFIDENCE = 0.5

# Detection resolution cap. RapidOCR's shipped defaults are tuned for phone
# photographs of documents, not for screenshots, and out of the box a
# 1920x1080 capture took **42 seconds** on this machine - past the tool's own
# timeout, so the model got nothing at all.
#
# The default config says `limit_type: min, limit_side_len: 736`, which scales
# the *shorter* side UP to at least 736. A 1080-tall screenshot is already
# above that, so nothing is downscaled and text detection runs at full
# resolution. Capping the *longer* side instead is what makes this usable.
#
# 1600 is a deliberate compromise: 1920 -> 1600 is a 0.83x scale, which leaves
# 12px UI text around 10px - still comfortably readable - while cutting the
# detector's work by roughly a third.
DEFAULT_DET_SIDE = 1600


class RapidOCR(OCRProvider):
    """OCR via rapidocr-onnxruntime."""

    def __init__(
        self,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        max_lines: int = 400,
        det_side: int = DEFAULT_DET_SIDE,
        threads: int = 8,
    ) -> None:
        self._min_confidence = min_confidence
        self._max_lines = max_lines
        self._det_side = det_side
        self._threads = threads
        self._engine: Any = None

    def info(self) -> dict[str, Any]:
        return {
            "name": "rapidocr",
            "kind": "onnxruntime",
            "min_confidence": self._min_confidence,
            "det_side": self._det_side,
            "threads": self._threads,
            "loaded": self._engine is not None,
        }

    def load(self) -> None:
        """Build the engine. Slow the first time; idempotent after."""
        if self._engine is not None:
            return
        from rapidocr_onnxruntime import RapidOCR as _Engine

        started = time.perf_counter()
        self._engine = _Engine(
            # Cap the long side rather than raise the short side. See above.
            det_limit_type="max",
            det_limit_side_len=self._det_side,
            # Angle classification decides whether a line is upside down. That
            # is a real problem for photographed paper and a non-problem for a
            # screenshot, where text is never rotated 180 degrees. It costs a
            # model pass per detected line, so switching it off is free speed.
            use_cls=False,
            # Bounded, not -1. Unbounded is not just impolite to the resident
            # LLM and the voice loop - it is measurably SLOWER here:
            # detection took 6.0s at the -1 default and 1.1s at 4 threads,
            # because 24 logical cores oversubscribe. 8 measured best
            # end-to-end; 16 was worse again.
            intra_op_num_threads=self._threads,
            inter_op_num_threads=self._threads,
        )
        log.info(
            "vision.ocr_loaded",
            ms=int((time.perf_counter() - started) * 1000),
            det_side=self._det_side,
            threads=self._threads,
        )

    def unload(self) -> None:
        self._engine = None

    def read(self, image: Any) -> ScreenText:
        self.load()
        started = time.perf_counter()
        try:
            raw, _elapsed = self._engine(image)
        except Exception as exc:
            # A recogniser that fails should degrade to "I read nothing", not
            # take down the tool call that asked it.
            log.warning("vision.ocr_failed", error=str(exc))
            return ScreenText(latency_ms=int((time.perf_counter() - started) * 1000))

        lines: list[TextLine] = []
        uncertain: list[TextLine] = []
        for entry in raw or []:
            line = _to_line(entry)
            if line is None or not line.text.strip():
                continue
            (lines if line.confidence >= self._min_confidence else uncertain).append(line)
            if len(lines) >= self._max_lines:
                break

        latency = int((time.perf_counter() - started) * 1000)
        log.info(
            "vision.ocr_done", lines=len(lines), uncertain=len(uncertain), ms=latency
        )
        return ScreenText(lines=lines, uncertain=uncertain, latency_ms=latency)


def _to_line(entry: Any) -> TextLine | None:
    """RapidOCR yields [box, text, score]; be defensive about the shape."""
    try:
        box, text, score = entry[0], entry[1], entry[2]
    except (TypeError, IndexError, KeyError):
        return None
    return TextLine(text=str(text), confidence=float(score), box=_to_box(box))


def _to_box(box: Any) -> tuple[int, int, int, int] | None:
    """Four corner points to (left, top, right, bottom)."""
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
    except (TypeError, IndexError, ValueError):
        return None
    if not xs or not ys:
        return None
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
