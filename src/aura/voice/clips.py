"""Loading the pre-rendered micro-expression clips (10B.6).

The clips are WAVs committed to `assets/micro/`, rendered once by
`scripts/render_micro_expressions.py` from language-specific Piper voices that
are then deleted. AURA's own voice cannot pronounce 了解, and loading a second
TTS model at runtime is RAM this machine does not have.

**A missing clip makes its expression unavailable, silently and by design.**
The catalogue is a list of things AURA *could* say; this is what she actually
has audio for. Without that check, adding a phrase to the catalogue and
forgetting to render it would produce a silent pause in the middle of a reply --
a bug that is invisible in a log and obvious in a room.

Everything is loaded once and held. Eight clips is about 300 KB; re-reading a
file to say "yes" would be silly.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from aura.config import PROJECT_ROOT
from aura.logging import get_logger
from aura.voice.personality import MicroExpression
from aura.voice.types import AudioArray

log = get_logger(__name__)

CLIPS_DIR = PROJECT_ROOT / "assets" / "micro"


def load_wav(path: Path) -> tuple[AudioArray, int]:
    """One clip as float32 in [-1, 1], with its sample rate."""
    with wave.open(str(path), "rb") as fh:
        rate = fh.getframerate()
        frames = fh.readframes(fh.getnframes())
        width = fh.getsampwidth()
    if width != 2:
        raise ValueError(f"{path.name} is {width * 8}-bit; the renderer writes 16-bit")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return audio, rate


class ClipLibrary:
    """The clips that exist, keyed by expression id."""

    def __init__(self, directory: Path = CLIPS_DIR) -> None:
        self._dir = directory
        self._audio: dict[str, tuple[AudioArray, int]] = {}
        self._missing: list[str] = []

    def load(self, catalogue: tuple[MicroExpression, ...]) -> None:
        for expression in catalogue:
            path = self._dir / expression.clip
            if not path.is_file():
                self._missing.append(expression.id)
                continue
            try:
                self._audio[expression.id] = load_wav(path)
            except Exception as exc:
                # A corrupt clip is not worth taking the voice process down
                # for. It makes one phrase unavailable, which is the same
                # outcome as never having rendered it.
                self._missing.append(expression.id)
                log.warning("clips.unreadable", clip=expression.clip, error=str(exc))

        log.info(
            "clips.loaded",
            available=sorted(self._audio),
            missing=sorted(self._missing),
            directory=str(self._dir),
        )

    @property
    def available(self) -> frozenset[str]:
        return frozenset(self._audio)

    @property
    def missing(self) -> list[str]:
        """Catalogue entries with no audio. Rendering them is a build step."""
        return sorted(self._missing)

    def get(self, expression_id: str) -> tuple[AudioArray, int] | None:
        return self._audio.get(expression_id)

    def __len__(self) -> int:
        return len(self._audio)
