"""A wake-word detector that fires only when told to.

Removes timing and acoustics from tests of the session loop: call `trigger()`
and the next processed frame reports a detection.
"""

from __future__ import annotations

import time

from aura.voice.types import AudioArray, VoiceInfo, WakeEvent
from aura.voice.wake.base import WakeWordDetector


class NullWakeWord(WakeWordDetector):
    """Fires on demand."""

    def __init__(self) -> None:
        self._armed = False
        self.frames_seen = 0

    def info(self) -> VoiceInfo:
        return VoiceInfo(name="null", kind="stub", model="manual", detail="test double")

    def trigger(self) -> None:
        """Make the next process() call report a detection."""
        self._armed = True

    def process(self, frame: AudioArray) -> WakeEvent | None:
        self.frames_seen += 1
        if not self._armed:
            return None
        self._armed = False
        return WakeEvent(model="null", score=1.0, at=time.time())

    def reset(self) -> None:
        self._armed = False
