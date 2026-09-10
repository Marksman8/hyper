"""Vision-language models, behind one seam.

A VLM answers questions about pixels. What it returns is an OBSERVATION -- data
from a model that was looking at whatever happened to be on the screen -- and
never an instruction. See `base.py`.
"""

from __future__ import annotations

from aura.vision.vlm.base import (
    NullVLM,
    ObservationSource,
    VLMHealth,
    VLMObservation,
    VLMProvider,
)

__all__ = [
    "NullVLM",
    "ObservationSource",
    "VLMHealth",
    "VLMObservation",
    "VLMProvider",
]
