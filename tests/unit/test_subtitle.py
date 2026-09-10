"""The live subtitle (10B.2).

10B.1 put speech on the bus; nothing consumed it. This is the consumer, and it
is the milestone that delivers the requirement stated in the brief as *"sometimes
I may not be able to hear the audio."*

That framing decides every test here. With sound muted the subtitle is not a
nicety layered over the audio -- it is the only copy of what AURA said. So the
properties worth asserting are about **what survives and what must not**: a line
must stay long enough to read, must not linger into the next reply, must not
outlive privacy, and must not reappear after a reconnect claiming to describe a
turn nobody can vouch for.

`PresenceController` is tested against a fake overlay rather than a real widget.
The routing is the logic; the painting is Qt's problem.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

# Offscreen, so the suite needs no display and opens no window. Set before
# PySide6 is imported below -- Qt reads it when the platform plugin loads.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from aura.core.presence import PresenceState
from aura.presence.app import (
    INTERRUPTED_HOLD_MS,
    MAX_HOLD_MS,
    MIN_HOLD_MS,
    PresenceController,
    _read_ms,
)
from aura.presence.placement import Anchor, PlacementPolicy, Screen

SCREEN = Screen(x=0, y=0, width=1920, height=1040)
SIZE = (132, 176)


class FakeOverlay:
    """Records what it was told to draw. No Qt, no display."""

    def __init__(self) -> None:
        self.subtitle = ""
        self.state: PresenceState | None = None
        self.visible = True
        self.opacity = 1.0
        self.dragged = _NullSignal()
        self.resized = _NullSignal()
        self.moves: list[tuple[int, int]] = []
        self._h = SIZE[1]

    def set_subtitle(self, text: str) -> None:
        self.subtitle = text
        # A real overlay grows to fit the line. Modelled here because the
        # first version of this fake had a fixed size, which is precisely why
        # the clipping bug below reached a live screen.
        self._h = SIZE[1] + (34 if text else 0)

    def set_state(self, state: PresenceState, *, immediate: bool = False) -> None:
        self.state = state

    def show(self) -> None:
        self.visible = True

    def hide(self) -> None:
        self.visible = False

    def isVisible(self) -> bool:  # noqa: N802 - Qt's spelling
        return self.visible

    def setWindowOpacity(self, value: float) -> None:  # noqa: N802 - Qt's spelling
        self.opacity = value

    def move(self, x: int, y: int) -> None:
        self.moves.append((x, y))

    def size(self) -> Any:
        return _Size(*SIZE)

    def width(self) -> int:
        return SIZE[0]

    def height(self) -> int:
        return self._h

    def x(self) -> int:
        return self.moves[-1][0] if self.moves else 0

    def y(self) -> int:
        return self.moves[-1][1] if self.moves else 0

    def screen(self) -> Any:
        return _FakeScreen()


class _FakeScreen:
    def availableGeometry(self) -> Any:  # noqa: N802 - Qt's spelling
        return _Rect(SCREEN.x, SCREEN.y, SCREEN.width, SCREEN.height)


class _Rect:
    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self._x, self._y, self._w, self._h = x, y, w, h

    def x(self) -> int:
        return self._x

    def y(self) -> int:
        return self._y

    def width(self) -> int:
        return self._w

    def height(self) -> int:
        return self._h


class _NullSignal:
    def connect(self, _slot: Any) -> None:
        return None


class _Size:
    def __init__(self, w: int, h: int) -> None:
        self._w, self._h = w, h

    def width(self) -> int:
        return self._w

    def height(self) -> int:
        return self._h


@pytest.fixture(scope="session")
def qapp() -> Any:
    """One QApplication for the module. QTimer needs an application object to
    attach to; nothing here runs an event loop or shows a window."""
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(qapp: Any) -> tuple[PresenceController, FakeOverlay]:
    """A controller with a fake overlay. `qapp` because QTimer needs an app."""
    overlay = FakeOverlay()
    policy = PlacementPolicy(Anchor.BOTTOM_RIGHT)
    return PresenceController(overlay, policy), overlay  # type: ignore[arg-type]


def _speak(controller: PresenceController, *sentences: str) -> None:
    controller.on_event({"type": "speech_started", "payload": {"length": 20}})
    for text in sentences:
        controller.on_event({"type": "speech_chunk", "payload": {"text": text}})


# ===========================================================================
# What gets shown
# ===========================================================================


def test_a_spoken_sentence_appears(controller: Any) -> None:
    ctl, overlay = controller
    _speak(ctl, "The build finished.")

    assert overlay.subtitle == "The build finished."


def test_each_sentence_replaces_the_last(controller: Any) -> None:
    """Not accumulated. A subtitle holding a whole reply grows past the screen
    and stops being readable at exactly the length where reading it matters."""
    ctl, overlay = controller
    _speak(ctl, "First sentence.", "Second sentence.", "Third sentence.")

    assert overlay.subtitle == "Third sentence."


def test_a_new_reply_does_not_inherit_the_previous_line(controller: Any) -> None:
    """Otherwise old words are attributed to a new answer."""
    ctl, overlay = controller
    _speak(ctl, "Answer to the first question.")
    ctl.on_event({"type": "speech_ended", "payload": {"completed": True}})

    ctl.on_event({"type": "speech_started", "payload": {"length": 10}})

    assert overlay.subtitle == ""


def test_an_empty_chunk_does_not_blank_the_line(controller: Any) -> None:
    """The sentence chunker can hand over punctuation alone; that must not
    wipe the line currently being read."""
    ctl, overlay = controller
    _speak(ctl, "Something worth reading.", "")

    assert overlay.subtitle == "Something worth reading."


# ===========================================================================
# Reading time -- the audio ending is not the reader finishing
# ===========================================================================


def test_a_finished_line_is_held_not_cleared(controller: Any) -> None:
    """Clearing on the audio's last sample cuts off a reader mid-sentence,
    which is the one thing the subtitle exists to prevent."""
    ctl, overlay = controller
    _speak(ctl, "A sentence somebody is still reading.")
    ctl.on_event({"type": "speech_ended", "payload": {"completed": True}})

    assert overlay.subtitle == "A sentence somebody is still reading."


def test_the_hold_scales_with_length() -> None:
    short = _read_ms("Done.")
    long = _read_ms("A considerably longer sentence that takes real time to read.")

    assert short < long
    assert short >= MIN_HOLD_MS


def test_the_hold_is_bounded_at_both_ends() -> None:
    """A one-word answer should not vanish instantly; a very long sentence
    should not pin the subtitle up indefinitely."""
    assert _read_ms("") == MIN_HOLD_MS
    assert _read_ms("x" * 10_000) == MAX_HOLD_MS


def test_an_interrupted_reply_clears_quickly(controller: Any) -> None:
    """Its last sentence was cut off. Leaving a fragment on screen presents it
    as though it were the final word."""
    ctl, _ = controller
    _speak(ctl, "I was part way through saying")
    ctl.on_event({"type": "speech_ended", "payload": {"completed": False}})

    assert ctl._subtitle_timer is not None
    assert ctl._subtitle_timer.interval() == INTERRUPTED_HOLD_MS


def test_a_completed_reply_is_held_longer_than_an_interrupted_one(
    controller: Any,
) -> None:
    ctl, _ = controller
    line = "The deployment finished successfully."

    _speak(ctl, line)
    ctl.on_event({"type": "speech_ended", "payload": {"completed": True}})

    assert ctl._subtitle_timer.interval() > INTERRUPTED_HOLD_MS


def test_a_new_sentence_cancels_a_pending_clear(controller: Any) -> None:
    """Otherwise a clear scheduled for the previous reply fires part way
    through the next one."""
    ctl, overlay = controller
    _speak(ctl, "First.")
    ctl.on_event({"type": "speech_ended", "payload": {"completed": True}})
    assert ctl._subtitle_timer is not None

    _speak(ctl, "A new reply entirely.")

    assert ctl._subtitle_timer is None
    assert overlay.subtitle == "A new reply entirely."


# ===========================================================================
# Privacy -- dropped, not merely hidden
# ===========================================================================


def test_private_drops_the_line(controller: Any) -> None:
    ctl, overlay = controller
    _speak(ctl, "Something said before privacy was asked for.")

    ctl.on_event({"type": "presence_state", "payload": {"state": "private"}})

    assert overlay.visible is False
    assert overlay.subtitle == "", "the line was hidden rather than dropped"


def test_leaving_private_does_not_restore_the_line(controller: Any) -> None:
    """The retained-line bug in full: what would reappear is the last thing
    AURA said before being told to stop."""
    ctl, overlay = controller
    _speak(ctl, "Confidential sentence.")
    ctl.on_event({"type": "presence_state", "payload": {"state": "private"}})

    ctl.on_event({"type": "presence_state", "payload": {"state": "idle"}})

    assert overlay.visible is True
    assert overlay.subtitle == ""


def test_speech_while_private_is_not_retained(controller: Any) -> None:
    """Nothing is drawn while private, and nothing is stored either."""
    ctl, overlay = controller
    ctl.on_event({"type": "presence_state", "payload": {"state": "private"}})

    _speak(ctl, "Spoken while the overlay was hidden.")
    ctl.on_event({"type": "presence_state", "payload": {"state": "idle"}})

    assert overlay.subtitle == ""


@pytest.mark.parametrize("hidden", ["private", "sleep"])
def test_both_hidden_states_drop_the_line(hidden: str, controller: Any) -> None:
    ctl, overlay = controller
    _speak(ctl, "A line.")
    ctl.on_event({"type": "presence_state", "payload": {"state": hidden}})

    assert overlay.subtitle == ""


# ===========================================================================
# Reconnect
# ===========================================================================


def test_a_snapshot_clears_the_line(controller: Any) -> None:
    """SPEECH_* are occurrences and are deliberately absent from the snapshot,
    so after a reconnect the overlay does not know whether she is still
    talking. Keeping the old line would assert something it cannot know."""
    ctl, overlay = controller
    _speak(ctl, "Said before the connection dropped.")

    ctl.on_event({"type": "snapshot", "payload": {"presence_state": "idle"}})

    assert overlay.subtitle == ""


def test_a_snapshot_cancels_a_pending_clear(controller: Any) -> None:
    ctl, _ = controller
    _speak(ctl, "Something.")
    ctl.on_event({"type": "speech_ended", "payload": {"completed": True}})

    ctl.on_event({"type": "snapshot", "payload": {"presence_state": "idle"}})

    assert ctl._subtitle_timer is None


# ===========================================================================
# Staying on the screen
# ===========================================================================


def test_growing_a_subtitle_re_anchors_the_panel(controller: Any) -> None:
    """Found on a live screen, not in a test.

    `resize()` moves the bottom and right edges and leaves the top-left alone,
    so a bottom-right-anchored overlay that grows a subtitle grows straight off
    the screen. The first live run had both the avatar and its line clipped by
    the screen edge.
    """
    ctl, overlay = controller
    ctl.place_initial()
    before_bottom = overlay.y() + overlay.height()

    _speak(ctl, "A line long enough to change the panel's height.")
    ctl._reanchor()

    after_bottom = overlay.y() + overlay.height()
    assert after_bottom <= SCREEN.y + SCREEN.height, "the panel grew off-screen"
    assert abs(after_bottom - before_bottom) < 40, (
        "the anchored edge moved; the panel should grow upward, not downward"
    )


def test_re_anchoring_is_not_rate_limited(controller: Any) -> None:
    """The three-second limit governs AURA *relocating*. Holding the same
    anchor while the panel resizes is the opposite of a move, and rate-limiting
    it would leave the panel clipped for three seconds."""
    ctl, overlay = controller
    ctl.place_initial()

    for i in range(5):
        _speak(ctl, f"Sentence number {i} of a reply.")
        ctl._reanchor()
        assert overlay.y() + overlay.height() <= SCREEN.y + SCREEN.height


def test_re_anchoring_does_not_count_as_a_move(controller: Any) -> None:
    """Otherwise a talkative reply would exhaust the movement budget and a
    genuine trigger -- an app change, an error -- would be suppressed."""
    ctl, _ = controller
    ctl.place_initial()
    before = ctl._policy.move_count

    for i in range(5):
        _speak(ctl, f"Line {i}.")
        ctl._reanchor()

    assert ctl._policy.move_count == before


# ===========================================================================
# The overlay draws, it does not interpret
# ===========================================================================


def test_the_subtitle_is_drawn_as_plain_text() -> None:
    """A panel that interpreted markup would be a rendering engine pointed at
    a message stream, which is a wider surface than a subtitle needs."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "src" / "aura" / "presence" / "overlay.py"
    ).read_text(encoding="utf-8")

    for rich in ("setHtml", "QTextDocument", "setTextFormat", "RichText"):
        assert rich not in source, f"overlay.py uses {rich}"
    assert "drawText" in source


def test_speech_cannot_change_presence_state(controller: Any) -> None:
    """Speaking is not a state she is in INSTEAD of working. There is no
    SPEAKING state, and a speech event must not invent one."""
    ctl, overlay = controller
    ctl.on_event({"type": "presence_state", "payload": {"state": "working"}})

    _speak(ctl, "Still working on it.")

    assert overlay.state is PresenceState.WORKING
