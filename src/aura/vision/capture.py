"""Screen capture.

**The gate is checked here, before any pixels are read.** That placement is the
whole privacy design: a refusal means no screenshot was ever taken, so there is
nothing to leak, cache, or forget to filter. Callers cannot bypass it, because
there is no capture function that does not consult it.

The foreground window is re-identified immediately before each grab rather than
trusted from a cache. Windows change focus in the milliseconds between "may I?"
and "click" — a stale answer there is exactly the case where a password manager
gets captured because Chrome was in front a moment ago.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aura.logging import get_logger
from aura.vision.privacy import Decision, PrivacyPolicy
from aura.vision.windows import ForegroundWindow, foreground_window

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    Image = NDArray[np.uint8]
else:
    Image = Any

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Monitor:
    """One display."""

    index: int
    width: int
    height: int
    left: int
    top: int
    is_primary: bool = False

    def describe(self) -> str:
        tag = " (primary)" if self.is_primary else ""
        return f"Monitor {self.index}: {self.width}x{self.height} at {self.left},{self.top}{tag}"


@dataclass(slots=True)
class Screenshot:
    """A captured frame and what was in front when it was taken."""

    image: Image
    monitor: int
    width: int
    height: int
    window: ForegroundWindow = field(default_factory=ForegroundWindow)
    captured_at: float = field(default_factory=time.time)


class CaptureRefusedError(Exception):
    """The privacy gate said no. Carries the user-facing reason."""


class ScreenCapture:
    """Takes screenshots, subject to the policy."""

    def __init__(self, policy: PrivacyPolicy) -> None:
        self._policy = policy
        self._sct: Any = None

    @property
    def policy(self) -> PrivacyPolicy:
        return self._policy

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def _backend(self) -> Any:
        """The mss instance, created lazily.

        Lazy because constructing it opens a display connection, and a machine
        in PRIVATE mode should not be holding one.
        """
        if self._sct is None:
            import mss

            self._sct = mss.mss()
        return self._sct

    # -- enumeration ---------------------------------------------------------

    def monitors(self) -> list[Monitor]:
        """Displays available. Reading the layout is not reading the content.

        Deliberately not gated: knowing a second monitor exists reveals nothing
        about what is on it, and the agent needs the list to ask sensibly.
        """
        try:
            raw = self._backend().monitors
        except Exception as exc:  # report rather than crash a tool
            log.warning("vision.monitor_enumeration_failed", error=str(exc))
            return []

        # mss index 0 is the union of all displays; the real ones start at 1.
        out: list[Monitor] = []
        for index, mon in enumerate(raw[1:], start=1):
            out.append(
                Monitor(
                    index=index,
                    width=int(mon["width"]),
                    height=int(mon["height"]),
                    left=int(mon["left"]),
                    top=int(mon["top"]),
                    is_primary=index == 1,
                )
            )
        return out

    # -- capture -------------------------------------------------------------

    def may_capture(self, monitor: int = 1, requested_by_user: bool = True) -> Decision:
        """Ask the gate without capturing. Used by status displays."""
        window = foreground_window()
        return self._policy.may_capture(
            monitor=monitor,
            window_title=window.title,
            process_name=window.process_name,
            requested_by_user=requested_by_user,
        )

    def grab(self, monitor: int = 1, requested_by_user: bool = True) -> Screenshot:
        """Capture one monitor, if the policy permits.

        Raises CaptureRefused when it does not — and in that case nothing was
        read from the screen at all.
        """
        # Re-identified here, immediately before the grab. A cached answer is
        # a window-switch race, and the case it loses is the one that matters.
        window = foreground_window()
        decision = self._policy.may_capture(
            monitor=monitor,
            window_title=window.title,
            process_name=window.process_name,
            requested_by_user=requested_by_user,
        )
        if not decision:
            log.info("vision.capture_refused", reason=decision.reason, monitor=monitor)
            raise CaptureRefusedError(decision.reason)

        import numpy as np

        backend = self._backend()
        available = backend.monitors
        if monitor < 1 or monitor >= len(available):
            raise CaptureRefusedError(
                f"No monitor {monitor}. Available: 1 to {len(available) - 1}."
            )

        started = time.perf_counter()
        raw = backend.grab(available[monitor])
        # BGRA from mss; drop alpha and reorder to RGB, which is what every
        # consumer downstream expects.
        image = np.asarray(raw, dtype=np.uint8)[:, :, :3][:, :, ::-1]

        log.info(
            "vision.captured",
            monitor=monitor,
            size=f"{raw.width}x{raw.height}",
            window=window.describe(),
            ms=int((time.perf_counter() - started) * 1000),
        )
        return Screenshot(
            image=image,
            monitor=monitor,
            width=raw.width,
            height=raw.height,
            window=window,
        )
