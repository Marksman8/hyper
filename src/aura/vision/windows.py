"""Which window is in front, and what program owns it.

The privacy gate needs this: "don't look at Chrome" can only be enforced if
AURA knows Chrome is what you are looking at. So this runs *before* every
capture, and a failure to identify the window is treated as a reason to refuse
rather than a reason to proceed.

Windows only. Elsewhere it reports nothing, and the gate treats an unknown
window conservatively.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from aura.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ForegroundWindow:
    """The window with focus."""

    title: str = ""
    process_name: str = ""
    pid: int = 0
    # False when the window could not be identified at all, which the gate
    # treats as "do not capture" rather than "capture anyway".
    known: bool = False

    def describe(self) -> str:
        if not self.known:
            return "unknown window"
        if self.process_name and self.title:
            return f"{self.title} ({self.process_name})"
        return self.title or self.process_name or "unknown window"


def foreground_window() -> ForegroundWindow:
    """Identify the focused window. Never raises."""
    if sys.platform != "win32":
        return ForegroundWindow()

    try:
        import win32gui
        import win32process
    except ImportError:
        log.debug("vision.pywin32_missing")
        return ForegroundWindow()

    try:
        handle = win32gui.GetForegroundWindow()
        if not handle:
            return ForegroundWindow()
        title = win32gui.GetWindowText(handle) or ""
        _thread, pid = win32process.GetWindowThreadProcessId(handle)
    except Exception as exc:  # identification must not crash a capture
        log.debug("vision.foreground_failed", error=str(exc))
        return ForegroundWindow()

    return ForegroundWindow(
        title=title, process_name=_process_name(pid), pid=int(pid or 0), known=True
    )


def _process_name(pid: int) -> str:
    if not pid:
        return ""
    try:
        import psutil

        return str(psutil.Process(pid).name())
    except Exception:  # a name is a nicety, not a requirement
        return ""
