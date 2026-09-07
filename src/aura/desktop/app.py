"""The desktop shell.

**The threading rule, written down because getting it wrong fails weirdly.**

tkinter is not thread-safe. Every widget call must happen on the thread running
`mainloop()`, and a violation does not raise there — it corrupts Tcl's state and
crashes somewhere unrelated, later, in a way that looks like a random bug.

So:

    main thread          tkinter mainloop, every widget call
    worker thread        HTTP to AURA Core (status polls, agent tasks)
    queue.Queue          the only channel between them
    root.after(...)      drains the queue on the main thread

Nothing in a worker touches a widget. Workers put a callable on the queue; the
tick drains it. That is the whole discipline, and it is why the shell can do a
three-minute agent task without freezing the window.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from typing import Any

from aura.desktop.approval import ApprovalDialog
from aura.desktop.client import CoreClient, TaskResult
from aura.desktop.overlay import Overlay
from aura.logging import get_logger

log = get_logger(__name__)

# How often the queue is drained and the pulse advanced. 120 ms is fast enough
# to feel live and slow enough to be invisible in a CPU graph.
TICK_MS = 120
STATUS_EVERY_TICKS = 25  # ~3 seconds


class DesktopApp:
    """Tray presence, status overlay, and the approval prompt."""

    def __init__(self) -> None:
        self._client = CoreClient()
        self._queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._session_id: str | None = None
        self._busy = False
        self._ticks = 0
        self._estop = False
        self._tray: Any = None

        self._root = tk.Tk()
        self._root.withdraw()  # placed and shown by the overlay
        self._overlay = Overlay(
            self._root,
            on_submit=self._submit_task,
            on_estop=self._toggle_estop,
            on_hide=self.hide,
        )
        self._root.deiconify()
        self._root.protocol("WM_DELETE_WINDOW", self.hide)

    # -- worker plumbing -----------------------------------------------------

    def _in_background(self, work: Callable[[], Callable[[], None]]) -> None:
        """Run `work` off the main thread; it returns a UI update to queue.

        The two-stage shape is deliberate: the worker computes, and the thing
        it hands back is the only code allowed to touch a widget.
        """

        def runner() -> None:
            update: Callable[[], None]
            try:
                update = work()
            except Exception as exc:
                log.error("desktop.worker_failed", error=str(exc))
                update = self._error_update(str(exc))
            self._queue.put(update)

        threading.Thread(target=runner, daemon=True).start()

    def _error_update(self, message: str) -> Callable[[], None]:
        """A UI update reporting a worker failure.

        Built on the worker, run on the main thread — the same rule every other
        update follows.
        """

        def update() -> None:
            self._overlay.set_state("error", message[:110])

        return update

    def _tick(self) -> None:
        """Main-thread heartbeat: drain the queue, advance the animation."""
        while True:
            try:
                self._queue.get_nowait()()
            except queue.Empty:
                break
            except Exception as exc:
                log.error("desktop.update_failed", error=str(exc))

        self._overlay.pulse()
        # Layout has settled by now, so the overflow test is meaningful.
        self._overlay.sync_scrollbar()

        self._ticks += 1
        if self._ticks % STATUS_EVERY_TICKS == 0 and not self._busy:
            self._refresh_status()

        self._root.after(TICK_MS, self._tick)

    # -- status --------------------------------------------------------------

    def _refresh_status(self) -> None:
        def work() -> Callable[[], None]:
            status = self._client.status()

            def update() -> None:
                self._estop = status.emergency_stop
                if not status.online:
                    self._overlay.set_state("offline", "AURA Core is not running")
                    self._overlay.set_resources("")
                    return
                if status.emergency_stop:
                    self._overlay.set_state("stopped", "emergency stop engaged")
                elif not status.model_available:
                    self._overlay.set_state("error", status.detail[:110] or "model unavailable")
                else:
                    self._overlay.set_state("idle", status.model)

                bits = [f"RAM {status.ram_percent:.0f}%"]
                if status.vram_total_gb:
                    bits.append(f"VRAM {status.vram_used_gb:.1f}/{status.vram_total_gb:.0f} GB")
                self._overlay.set_resources("   ".join(bits))

            return update

        self._in_background(work)

    # -- tasks ---------------------------------------------------------------

    def _submit_task(self, text: str, approve: list[str] | None = None) -> None:
        if self._busy:
            return
        self._busy = True
        self._overlay.clear_input()
        self._overlay.set_state("thinking", text[:90])
        self._overlay.set_answer("")

        def work() -> Callable[[], None]:
            result = self._client.run_task(text, approve=approve, session_id=self._session_id)

            def update() -> None:
                self._busy = False
                self._finish_task(text, result, already_approved=approve or [])

            return update

        self._in_background(work)

    def _finish_task(self, task: str, result: TaskResult, already_approved: list[str]) -> None:
        if not result.ok:
            self._overlay.set_state("error", result.error[:110])
            self._overlay.set_answer(result.error)
            return

        self._session_id = result.session_id or self._session_id
        self._overlay.set_answer(result.content)

        pending = [t for t in result.pending_confirmations if t not in already_approved]
        if pending:
            # The dialog is modal and runs on the main thread, which is exactly
            # where this callback already is.
            self._overlay.set_state("idle", "waiting for your approval")
            approved = ApprovalDialog(
                self._root, tools=pending, task=task, explanation=result.content
            ).ask()
            if approved:
                log.info("desktop.approval_granted", tools=pending)
                self._submit_task(task, approve=already_approved + pending)
                return
            log.info("desktop.approval_denied", tools=pending)
            self._overlay.set_state("idle", "not approved")
            return

        summary = f"{len(result.tool_calls)} tool calls  {result.latency_ms} ms"
        if result.stopped_early:
            summary += "  (stopped early)"
        self._overlay.set_state("idle", summary)

    # -- controls ------------------------------------------------------------

    def _toggle_estop(self) -> None:
        target = not self._estop

        def work() -> Callable[[], None]:
            engaged = self._client.set_emergency_stop(target)

            def update() -> None:
                self._estop = engaged
                self._overlay.set_state(
                    "stopped" if engaged else "idle",
                    "emergency stop engaged" if engaged else "released",
                )

            return update

        self._in_background(work)

    def show(self) -> None:
        self._overlay.focus_input()

    def hide(self) -> None:
        self._root.withdraw()

    def quit(self) -> None:
        if self._tray is not None:
            self._tray.stop()
        self._client.close()
        self._root.quit()
        self._root.destroy()

    # -- tray ----------------------------------------------------------------

    def _start_tray(self) -> None:
        """System tray icon, if pystray is installed.

        Optional by design: the shell is fully usable without it, so a missing
        extra degrades the presence rather than preventing startup.
        """
        try:
            import pystray
            from PIL import Image, ImageDraw
        except ImportError:
            log.info("desktop.tray_unavailable", hint="uv sync --extra desktop")
            return

        image = Image.new("RGB", (64, 64), "#111318")
        draw = ImageDraw.Draw(image)
        draw.ellipse((14, 14, 50, 50), fill="#4ade80")

        menu = pystray.Menu(
            pystray.MenuItem("Show AURA", lambda: self._queue.put(self.show), default=True),
            pystray.MenuItem("Hide", lambda: self._queue.put(self.hide)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Emergency stop", lambda: self._queue.put(self._toggle_estop)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda: self._queue.put(self.quit)),
        )
        self._tray = pystray.Icon("aura", image, "AURA", menu)
        # pystray wants its own thread; its callbacks only enqueue, so nothing
        # it does touches a widget directly.
        threading.Thread(target=self._tray.run, daemon=True).start()

    # -- run -----------------------------------------------------------------

    def run(self) -> None:
        self._start_tray()
        self._refresh_status()
        self._root.after(TICK_MS, self._tick)
        self._root.mainloop()
