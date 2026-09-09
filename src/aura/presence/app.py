"""The presence process.

    /v1/events  →  CoreEventClient  →  PresenceController  →  Overlay

**No reasoning lives here.** The controller translates events into calls on the
overlay and the placement policy, and nothing else. Every decision about *what
state AURA is in* was already made by `core/presence.py` and arrived as a
`PRESENCE_STATE` event; this process only draws the answer.

That is why killing this process costs AURA a face and nothing else — and why
the whole of it can be replaced by a different toolkit without touching Core.
"""

from __future__ import annotations

import sys
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import QApplication

from aura.config import get_settings
from aura.core.presence import PresenceState
from aura.logging import configure_logging, get_logger
from aura.presence.avatar.silhouette import SilhouetteAvatar
from aura.presence.client import CoreEventClient
from aura.presence.overlay import PresenceOverlay
from aura.presence.placement import Anchor, MoveReason, PlacementPolicy, Screen

log = get_logger(__name__)

# How often the cursor is sampled for gaze. Only runs when gaze is enabled, and
# it moves the eyes -- never the window.
GAZE_MS = 90

# States in which AURA is not visible at all.
HIDDEN = frozenset({PresenceState.PRIVATE, PresenceState.SLEEP})


class PresenceController(QObject):
    """Turns events into presentation. Holds no state of its own beyond what
    is needed to draw."""

    def __init__(
        self,
        overlay: PresenceOverlay,
        policy: PlacementPolicy,
        *,
        cursor_awareness: bool = False,
        focus_dim: float = 0.45,
    ) -> None:
        super().__init__()
        self._overlay = overlay
        self._policy = policy
        self._focus_dim = focus_dim
        self._state = PresenceState.IDLE
        self._focus_mode = False
        self._connected = False

        overlay.dragged.connect(self._on_dragged)

        # Gaze is OFF by default, and even when on it moves only the eyes.
        # Continuous body-following is the fastest way to make a desktop
        # companion intolerable.
        self._gaze_timer: QTimer | None = None
        if cursor_awareness:
            self._gaze_timer = QTimer(self)
            self._gaze_timer.timeout.connect(self._sample_cursor)
            self._gaze_timer.start(GAZE_MS)

    # -- events --------------------------------------------------------------

    def on_event(self, payload: dict[str, Any]) -> None:
        """One event off the wire.

        Untrusted payloads are rendered, never obeyed. Nothing in this method
        can start a task, grant a permission, or call a tool -- there is
        nothing here that could.
        """
        kind = payload.get("type")
        body = payload.get("payload") or {}

        if kind == "snapshot":
            # Requirement 10: recover from what is TRUE NOW.
            self._apply_state(str(body.get("presence_state", "idle")), immediate=True)
            if body.get("focus_mode"):
                self._set_focus_mode(True)
            return

        if kind == "presence_state":
            self._apply_state(str(body.get("state", "idle")))
        elif kind == "focus_mode":
            self._set_focus_mode(bool(body.get("enabled")))
        elif kind == "app_changed":
            # A permitted movement trigger. The payload is an OBSERVATION, and
            # it is used only to decide *where to sit* -- never what to do.
            self._consider_move(MoveReason.APP_CHANGED)

    def _apply_state(self, name: str, *, immediate: bool = False) -> None:
        try:
            state = PresenceState(name)
        except ValueError:
            log.debug("presence.unknown_state", name=name)
            return
        if state is self._state and not immediate:
            return
        self._state = state

        # PRIVATE and SLEEP hide, without a fade. A graceful transition means
        # AURA is still on screen after being told to stop, which is the one
        # place where "graceful" is the wrong answer.
        if state in HIDDEN:
            self._overlay.hide()
            log.info("presence.hidden", state=str(state))
            return

        self._overlay.set_state(state, immediate=immediate)
        if not self._overlay.isVisible():
            self._overlay.show()
        self._consider_move(MoveReason.NEEDS_ATTENTION)

    def _set_focus_mode(self, enabled: bool) -> None:
        """Minimised and quiet, but still capable. Focus reduces intrusion, it
        does not turn presence off."""
        if enabled == self._focus_mode:
            return
        self._focus_mode = enabled
        self._overlay.setWindowOpacity(self._focus_dim if enabled else 1.0)
        log.info("presence.focus_mode", enabled=enabled)

    def on_connection(self, connected: bool) -> None:
        self._connected = connected
        if not connected:
            # Core is gone. Showing a confident IDLE avatar would be claiming
            # to know something we do not.
            self._overlay.setWindowOpacity(0.35)
        else:
            self._overlay.setWindowOpacity(self._focus_dim if self._focus_mode else 1.0)

    # -- placement -----------------------------------------------------------

    def _screen(self) -> Screen:
        geometry = QApplication.primaryScreen().availableGeometry()
        return Screen(
            x=geometry.x(), y=geometry.y(), width=geometry.width(), height=geometry.height()
        )

    def _consider_move(self, reason: MoveReason) -> None:
        import time

        size = (self._overlay.width(), self._overlay.height())
        placement = self._policy.consider(
            reason, self._screen(), size, time.monotonic(), state=self._state
        )
        if placement is not None:
            self._overlay.move(placement.x, placement.y)

    def place_initial(self) -> None:
        x, y = self._policy.resolve(
            self._screen(), (self._overlay.width(), self._overlay.height())
        )
        self._overlay.move(x, y)

    def check_anchor(self) -> None:
        """A display was unplugged or the resolution changed."""
        size = (self._overlay.width(), self._overlay.height())
        at = (self._overlay.x(), self._overlay.y())
        if not self._policy.anchor_is_valid(self._screen(), size, at):
            self._consider_move(MoveReason.ANCHOR_INVALID)

    def _on_dragged(self, x: int, y: int) -> None:
        import time

        self._policy.pin(x, y, time.monotonic())

    # -- gaze ----------------------------------------------------------------

    def _sample_cursor(self) -> None:
        """Eyes only. The body never follows."""
        if not self._overlay.isVisible():
            return
        from PySide6.QtGui import QCursor

        cursor = QCursor.pos()
        centre = self._overlay.geometry().center()
        span = max(400.0, float(self._overlay.width()) * 4.0)
        self._overlay.look_at(
            (cursor.x() - centre.x()) / span, (cursor.y() - centre.y()) / span
        )


def run(cursor_awareness: bool | None = None, anchor: str = "bottom_right") -> int:
    """Start the presence process. Blocks until the overlay is closed."""
    cfg = get_settings()
    configure_logging(
        level=cfg.logging.level,
        fmt=cfg.logging.format,
        log_file=cfg.logging.resolved_file(),
        redact_keys=tuple(cfg.logging.redact_keys),
    )

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    # A hidden overlay (PRIVATE) must not end the process.
    app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, True)

    overlay = PresenceOverlay(SilhouetteAvatar())
    policy = PlacementPolicy(Anchor(anchor))
    controller = PresenceController(
        overlay,
        policy,
        cursor_awareness=bool(cursor_awareness),
    )
    controller.place_initial()
    overlay.show()

    client = CoreEventClient()
    client.received.connect(controller.on_event)
    client.connected.connect(controller.on_connection)
    client.start()

    # A display change can invalidate the anchor; that is the fourth permitted
    # movement trigger.
    screen = app.primaryScreen()
    screen.geometryChanged.connect(lambda _: controller.check_anchor())
    app.screenAdded.connect(lambda _: controller.check_anchor())
    app.screenRemoved.connect(lambda _: controller.check_anchor())

    log.info("presence.started", anchor=anchor, cursor_awareness=bool(cursor_awareness))
    try:
        return int(app.exec())
    finally:
        client.stop()
        client.wait(2000)
        log.info("presence.stopped")
