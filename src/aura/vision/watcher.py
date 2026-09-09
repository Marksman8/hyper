"""Foreground window watcher — the `APP_CHANGED` producer.

Publishes which application has focus, so the presence overlay can decide where
to sit. That is the only consumer today, and the only one this is designed for.

**Two restraints matter more than the polling.**

*Process name only, never the window title.* A title quotes document names,
message subjects, and search queries — "Re: redundancy consultation — Outlook"
is not something to put on a bus. The catalog already records this
(`APP_CHANGED.privacy_note`); this is where it is enforced.

*Silent in PRIVATE and SLEEP.* Reading the foreground process is not screen
capture, so the capture gate does not cover it — but a user who said "don't
look at my screen" has not agreed to have their application use tracked
instead. The watcher stops polling entirely, rather than polling and
withholding, so there is nothing collected to leak.

Lives in `aura.vision` rather than `aura.core` because it needs `win32gui`, and
the seam test keeps Windows APIs inside this package.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from aura.logging import get_logger
from aura.vision.privacy import VisionMode
from aura.vision.windows import foreground_window

log = get_logger(__name__)

# How often focus is sampled. A person switches applications a few times a
# minute at most; the presence overlay rate-limits movement to one move every
# three seconds anyway, so polling faster buys nothing and costs a wakeup.
POLL_S = 1.0


class WindowWatcher:
    """Polls the foreground window and announces changes.

    Passive about privacy: it asks the policy on every tick rather than
    caching, so "don't look" takes effect within one second of being said
    rather than at the next restart.
    """

    def __init__(
        self,
        emitter: Any,
        policy: Any,
        *,
        poll_s: float = POLL_S,
    ) -> None:
        self._emitter = emitter
        self._policy = policy
        self._poll_s = poll_s
        self._current = ""
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def current(self) -> str:
        return self._current

    def _may_watch(self) -> bool:
        """Privacy check, on every tick.

        PRIVATE and SLEEP stop the watcher entirely. Not a filter on the
        output -- a filter would still be collecting.
        """
        mode = getattr(self._policy, "mode", None)
        return mode not in (VisionMode.PRIVATE, VisionMode.SLEEP)

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="window-watcher")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception as exc:
                # Watching is a convenience. It must never take down the
                # process that is actually doing the work.
                log.debug("watcher.poll_failed", error=str(exc))
            await asyncio.sleep(self._poll_s)

    def poll(self) -> str | None:
        """One sample. Returns the new application name if it changed.

        Separated from the loop so it can be tested without a clock.
        """
        if not self._may_watch():
            # Forget what was last seen, so leaving private mode announces the
            # current application rather than assuming it never changed.
            self._current = ""
            return None

        window = foreground_window()
        # Process name only. Never `window.title` -- see the module docstring.
        name = window.process_name
        if not name or name == self._current:
            return None

        self._current = name
        self._emitter.app_changed(name)
        log.debug("watcher.app_changed", application=name)
        return name
