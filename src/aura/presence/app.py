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
from aura.core.envelope import FRAME_MS as ENVELOPE_FRAME_MS
from aura.core.envelope import openness
from aura.core.presence import PresenceState
from aura.logging import configure_logging, get_logger
from aura.presence.avatar.silhouette import SilhouetteAvatar
from aura.presence.client import CoreEventClient
from aura.presence.overlay import PresenceOverlay
from aura.presence.placement import Anchor, MoveReason, PlacementPolicy, Screen
from aura.presence.tray import PresenceTray

log = get_logger(__name__)

# How often the cursor is sampled for gaze. Only runs when gaze is enabled, and
# it moves the eyes -- never the window.
GAZE_MS = 90

# States in which AURA is not visible at all.
HIDDEN = frozenset({PresenceState.PRIVATE, PresenceState.SLEEP})

# How long a finished line stays up, per character.
#
# ~22 characters a second, which is brisk but not a race. The subtitle is a
# fallback for muted audio rather than the primary channel, so it does not need
# to sit there indefinitely -- but clearing on the audio's last sample would
# cut off a reader mid-sentence, which is the one thing it exists to prevent.
READ_MS_PER_CHAR = 45
MIN_HOLD_MS = 1_200
MAX_HOLD_MS = 5_000

# An interrupted reply clears fast. Its last sentence was cut off, and leaving
# a fragment on screen presents it as though it were the final word.
INTERRUPTED_HOLD_MS = 400


def _read_ms(text: str) -> int:
    """How long to leave `text` up after the audio stops."""
    return max(MIN_HOLD_MS, min(MAX_HOLD_MS, len(text) * READ_MS_PER_CHAR))


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
        # A LOCAL "get out of my way". Deliberately separate from PRIVATE:
        # conflating them would let a cosmetic tray click silently change a
        # privacy setting.
        self._user_hidden = False
        self.tray: Any = None
        # Clears the last spoken line once it has had time to be read.
        self._subtitle_timer: QTimer | None = None
        # Drives the mouth through one sentence's amplitude envelope.
        self._mouth_timer: QTimer | None = None
        self._envelope = ""
        self._frame = 0

        overlay.dragged.connect(self._on_dragged)
        overlay.resized.connect(self._reanchor)

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
            # Both directions: a reconnect must be able to turn focus OFF too,
            # or the overlay stays dimmed after Core says it is finished.
            self._set_focus_mode(bool(body.get("focus_mode")))
            # SPEECH_* are occurrences and are deliberately not in the snapshot,
            # so after a reconnect we do not know whether she is still talking.
            # A line left over from before the drop would be the overlay
            # asserting something it cannot know.
            self._cancel_subtitle_clear()
            self._overlay.set_subtitle("")
            self._stop_mouth()
            return

        if kind == "presence_state":
            self._apply_state(str(body.get("state", "idle")))
        elif kind == "focus_mode":
            self._set_focus_mode(bool(body.get("enabled")))
        elif kind == "app_changed":
            # A permitted movement trigger. The payload is an OBSERVATION, and
            # it is used only to decide *where to sit* -- never what to do.
            self._consider_move(MoveReason.APP_CHANGED)
        elif kind == "speech_started":
            self._speech_started()
        elif kind == "speech_chunk":
            self._speech_chunk(str(body.get("text", "")), str(body.get("envelope", "")))
        elif kind == "speech_ended":
            self._speech_ended(completed=bool(body.get("completed", True)))

    # -- speech (10B.2) --------------------------------------------------------
    #
    # The requirement this serves, in the user's words: *"sometimes I may not
    # be able to hear the audio."* So the subtitle is not decoration -- with
    # sound muted it is the only copy of what AURA said, and every decision
    # below follows from treating it that way.

    def _speech_started(self) -> None:
        """Clear the decks. The previous reply's last line must not linger into
        the next one, which would attribute old words to a new answer."""
        self._cancel_subtitle_clear()
        self._overlay.set_subtitle("")

    def _speech_chunk(self, text: str, envelope: str = "") -> None:
        """One sentence, replacing the last.

        Deliberately not accumulated. A subtitle holding a whole reply grows
        past the screen and stops being readable at exactly the length where
        reading it matters most; the sentence being spoken *now* is the one
        worth showing.
        """
        if not text or self._state in HIDDEN:
            # Nothing is drawn while private, and nothing is retained either --
            # a stored line would reappear the moment privacy ended.
            return
        self._cancel_subtitle_clear()
        self._overlay.set_subtitle(text)
        self._start_mouth(envelope)

    # -- the mouth (10B.3) -----------------------------------------------------

    def _start_mouth(self, envelope: str) -> None:
        """Replay a sentence's amplitude against a local clock.

        The envelope arrives with the sentence, at the moment that sentence's
        audio begins, so playing it from now is as close to in-sync as two
        processes get without a shared clock. Approximate is the right target:
        a mouth that moves while she talks and stops when she stops reads as
        correct, and nobody is lip-reading a silhouette.
        """
        self._stop_mouth()
        if not envelope:
            # No envelope is not an error -- a sentence spoken outside the
            # normal turn flow has no PCM to measure. A still mouth is a better
            # answer than a fabricated one.
            return

        self._envelope = envelope
        self._frame = 0
        timer = QTimer(self)
        timer.timeout.connect(self._advance_mouth)
        timer.start(ENVELOPE_FRAME_MS)
        self._mouth_timer = timer

    def _advance_mouth(self) -> None:
        if self._frame >= len(self._envelope):
            self._stop_mouth()
            return
        self._overlay.set_mouth(openness(self._envelope, self._frame))
        self._frame += 1

    def _stop_mouth(self) -> None:
        """Close the mouth and stop the clock.

        Called on a new sentence, on speech ending, on privacy, and on a
        reconnect. An open mouth left behind by any of those reads as a frozen
        avatar rather than a finished one.
        """
        if self._mouth_timer is not None:
            self._mouth_timer.stop()
            self._mouth_timer = None
        self._envelope = ""
        self._frame = 0
        self._overlay.set_mouth(0.0)

    def _speech_ended(self, *, completed: bool) -> None:
        """Hold the last line long enough to read, then clear.

        The audio finishing is not the same as the reader finishing: the last
        sentence arrives and the room goes quiet, and someone reading rather
        than listening is still mid-line. So the hold scales with length.

        An interrupted reply gets a much shorter hold. Its last sentence was
        cut off, and leaving it on screen presents a fragment as if it were the
        final word.
        """
        self._stop_mouth()
        if not completed:
            self._schedule_subtitle_clear(INTERRUPTED_HOLD_MS)
            return
        self._schedule_subtitle_clear(_read_ms(self._overlay.subtitle))

    def _schedule_subtitle_clear(self, delay_ms: int) -> None:
        self._cancel_subtitle_clear()
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self._overlay.set_subtitle(""))
        timer.start(delay_ms)
        self._subtitle_timer = timer

    def _cancel_subtitle_clear(self) -> None:
        if self._subtitle_timer is not None:
            self._subtitle_timer.stop()
            self._subtitle_timer = None

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
            # Dropped, not merely hidden. A retained line would reappear the
            # moment privacy ended -- and what it would reappear with is the
            # last thing AURA said before being told to stop.
            self._cancel_subtitle_clear()
            self._overlay.set_subtitle("")
            self._stop_mouth()
            log.info("presence.hidden", state=str(state))
            self._sync_tray()
            return

        self._overlay.set_state(state, immediate=immediate)
        if not self._overlay.isVisible() and not self._user_hidden:
            self._overlay.show()
        self._consider_move(MoveReason.NEEDS_ATTENTION)
        self._sync_tray()

    def _set_focus_mode(self, enabled: bool) -> None:
        """Minimised and quiet, but still capable. Focus reduces intrusion, it
        does not turn presence off."""
        if enabled == self._focus_mode:
            return
        self._focus_mode = enabled
        self._overlay.setWindowOpacity(self._focus_dim if enabled else 1.0)
        if self.tray is not None:
            self.tray.set_focus_checked(enabled)
        log.info("presence.focus_mode", enabled=enabled)

    def toggle_visible(self) -> None:
        """Tray click. Local only -- it does not touch privacy mode.

        While PRIVATE or SLEEP, showing is refused: the user asked not to be
        observed, and a tray click is not a withdrawal of that.
        """
        if self._overlay.isVisible():
            self._user_hidden = True
            self._overlay.hide()
        elif self._state in HIDDEN:
            log.info("presence.show_refused", state=str(self._state))
        else:
            self._user_hidden = False
            self._overlay.show()
        self._sync_tray()

    def _sync_tray(self) -> None:
        if self.tray is None:
            return
        from aura.presence.avatar.silhouette import PALETTE

        self.tray.set_visible_label(overlay_visible=self._overlay.isVisible())
        self.tray.set_state_colour(PALETTE.get(self._state, (110, 125, 150)))

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

    def _reanchor(self) -> None:
        """Re-apply the current anchor after a size change.

        **This is deliberately not a move**, and so deliberately not routed
        through `consider()`. The four-trigger rule and the three-second rate
        limit exist to stop AURA *relocating*; holding the same anchor while
        her panel grows a subtitle is the opposite -- it is what stops her
        appearing to slide across the screen every time she says something.
        Rate-limiting it would leave the panel clipped for three seconds.
        """
        self.place_initial()

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

    tray = PresenceTray()
    controller.tray = tray
    tray.toggle_visible.connect(controller.toggle_visible)
    tray.quit_requested.connect(app.quit)
    tray.show()

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
        tray.hide()
        log.info("presence.stopped")
