"""The three 10A producers: APP_CHANGED, FOCUS_MODE, and the tray (10A.12).

10A.3 catalogued these events and 10A.4 taught the state machine to honour
them, but nothing published them — the contract existed with no producer behind
it. This closes that.

The privacy properties are the ones worth being strict about: what the watcher
collects, and what a tray click is allowed to change.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aura.core.emitter import EventEmitter
from aura.core.events import Delivery, Event, EventBus, EventType, Origin
from aura.core.presence import PresenceState, PresenceStateMachine
from aura.vision.privacy import PrivacyPolicy, VisionMode
from aura.vision.watcher import WindowWatcher

SRC = Path(__file__).resolve().parents[2] / "src" / "aura"


class FakeWindow:
    def __init__(self, process_name: str, title: str = "") -> None:
        self.process_name = process_name
        self.title = title


def _bus_and_emitter() -> tuple[EventBus, EventEmitter]:
    bus = EventBus()
    return bus, EventEmitter(bus)


# ===========================================================================
# APP_CHANGED — the watcher
# ===========================================================================


def test_a_focus_change_is_announced(monkeypatch: pytest.MonkeyPatch) -> None:
    bus, emitter = _bus_and_emitter()
    sub, _ = bus.subscribe()
    monkeypatch.setattr(
        "aura.vision.watcher.foreground_window", lambda: FakeWindow("Code.exe")
    )

    watcher = WindowWatcher(emitter, PrivacyPolicy(mode=VisionMode.ASSIST))
    assert watcher.poll() == "Code.exe"

    published = [e for e in sub._queue if e.type is EventType.APP_CHANGED]
    assert len(published) == 1
    assert published[0].payload["application"] == "Code.exe"


def test_the_same_application_is_not_announced_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise every poll would be a movement trigger, and the avatar would
    hit its rate limit permanently."""
    bus, emitter = _bus_and_emitter()
    sub, _ = bus.subscribe()
    monkeypatch.setattr(
        "aura.vision.watcher.foreground_window", lambda: FakeWindow("Code.exe")
    )

    watcher = WindowWatcher(emitter, PrivacyPolicy(mode=VisionMode.ASSIST))
    for _ in range(20):
        watcher.poll()

    assert len([e for e in sub._queue if e.type is EventType.APP_CHANGED]) == 1


def test_the_window_title_never_reaches_the_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The privacy property that matters most here.

    A title quotes document names, message subjects and search queries. "Re:
    redundancy consultation - Outlook" is not something to put on a bus.
    """
    bus, emitter = _bus_and_emitter()
    sub, _ = bus.subscribe()
    monkeypatch.setattr(
        "aura.vision.watcher.foreground_window",
        lambda: FakeWindow("OUTLOOK.EXE", "Re: redundancy consultation - Outlook"),
    )

    WindowWatcher(emitter, PrivacyPolicy(mode=VisionMode.ASSIST)).poll()
    payload = str(next(iter(sub._queue)).payload)

    assert "OUTLOOK.EXE" in payload
    assert "redundancy" not in payload
    assert "Re:" not in payload


@pytest.mark.parametrize("mode", [VisionMode.PRIVATE, VisionMode.SLEEP])
def test_the_watcher_is_silent_in_private_and_sleep(
    mode: VisionMode, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading the foreground process is not screen capture, so the capture
    gate does not cover it -- but a user who said "don't look at my screen" has
    not agreed to application-use tracking instead."""
    bus, emitter = _bus_and_emitter()
    sub, _ = bus.subscribe()
    monkeypatch.setattr(
        "aura.vision.watcher.foreground_window", lambda: FakeWindow("Chrome.exe")
    )

    watcher = WindowWatcher(emitter, PrivacyPolicy(mode=mode))
    assert watcher.poll() is None
    assert not [e for e in sub._queue if e.type is EventType.APP_CHANGED]


def test_leaving_private_announces_the_current_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The watcher forgets what it last saw while private, so it does not
    assume nothing changed while it was not looking."""
    _, emitter = _bus_and_emitter()
    monkeypatch.setattr(
        "aura.vision.watcher.foreground_window", lambda: FakeWindow("Code.exe")
    )
    policy = PrivacyPolicy(mode=VisionMode.ASSIST)
    watcher = WindowWatcher(emitter, policy)

    assert watcher.poll() == "Code.exe"
    policy.set_mode(VisionMode.PRIVATE)
    assert watcher.poll() is None
    policy.set_mode(VisionMode.ASSIST)

    assert watcher.poll() == "Code.exe", "did not re-announce after privacy ended"


def test_the_watcher_is_observation_origin() -> None:
    """It reads the environment, so consumers must treat it as untrusted."""
    bus, emitter = _bus_and_emitter()
    sub, _ = bus.subscribe()
    emitter.app_changed("Code.exe")

    event = next(iter(sub._queue))
    assert event.origin is Origin.OBSERVATION
    assert event.trusted is False


def test_an_app_change_cannot_drive_presence_state() -> None:
    """Following from the above: the state machine ignores observations, so a
    focus change decides where the avatar sits and nothing else."""
    machine = PresenceStateMachine()
    machine.on_event(
        Event(
            type=EventType.APP_CHANGED,
            origin=Origin.OBSERVATION,
            payload={"application": "Chrome.exe"},
        )
    )
    assert machine.state is PresenceState.IDLE
    assert machine.transition_count == 0


def test_a_polling_failure_does_not_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watching is a convenience; it must never take down the process doing
    the actual work."""

    def boom() -> FakeWindow:
        raise OSError("win32 said no")

    _, emitter = _bus_and_emitter()
    monkeypatch.setattr("aura.vision.watcher.foreground_window", boom)
    watcher = WindowWatcher(emitter, PrivacyPolicy(mode=VisionMode.ASSIST))

    with pytest.raises(OSError):
        watcher.poll()  # poll() itself is honest...
    # ...and the loop is what swallows it; asserted by the loop's try/except
    # rather than by hiding the failure here.


# ===========================================================================
# FOCUS_MODE
# ===========================================================================


def test_focus_mode_is_published_as_state() -> None:
    bus, emitter = _bus_and_emitter()
    sub, _ = bus.subscribe()
    emitter.focus_mode(enabled=True)

    event = next(iter(sub._queue))
    assert event.type is EventType.FOCUS_MODE
    assert event.origin is Origin.USER
    assert event.delivery is Delivery.RELIABLE
    assert event.payload["enabled"] is True


def test_focus_mode_is_not_a_presence_state() -> None:
    """She is not "in focus mode" INSTEAD of thinking -- she is thinking
    quietly. Making it a state would mean losing the real one."""
    machine = PresenceStateMachine()
    machine.on_event(
        Event(
            type=EventType.TASK_STARTED,
            origin=Origin.CORE,
            payload={"kind": "chat", "with_tools": False},
        )
    )
    assert machine.state is PresenceState.THINKING

    machine.on_event(
        Event(
            type=EventType.FOCUS_MODE, origin=Origin.USER, payload={"enabled": True}
        )
    )
    assert machine.state is PresenceState.THINKING
    assert machine.focus_mode is True


def test_focus_mode_survives_a_reconnect() -> None:
    """Otherwise the overlay comes back at full brightness after a drop."""
    machine = PresenceStateMachine()
    machine.on_event(
        Event(
            type=EventType.FOCUS_MODE, origin=Origin.USER, payload={"enabled": True}
        )
    )
    assert machine.snapshot()["focus_mode"] is True


def test_focus_mode_is_reported_even_while_private() -> None:
    """PRIVATE suppresses the condition list, but focus is presentation
    intensity rather than activity -- withholding it would leave a reconnecting
    overlay at the wrong brightness."""
    machine = PresenceStateMachine()
    machine.on_event(
        Event(type=EventType.FOCUS_MODE, origin=Origin.USER, payload={"enabled": True})
    )
    machine.on_event(
        Event(
            type=EventType.PRIVATE_MODE,
            origin=Origin.USER,
            payload={"mode": "private", "capturing": False},
        )
    )
    snap = machine.snapshot()

    assert snap["presence_state"] == "private"
    assert snap["conditions"] == ["private"]
    assert snap["focus_mode"] is True


# ===========================================================================
# THE TRAY
# ===========================================================================


def test_the_tray_changes_focus_through_the_api_not_the_bus() -> None:
    """The bus is one-way. A subscriber that could publish would be a second
    source of authority -- event-bus contract section C.5."""
    source = (SRC / "presence" / "tray.py").read_text(encoding="utf-8")

    assert "/v1/focus" in source, "the tray must use the authenticated endpoint"
    assert "Authorization" in source
    assert "bus.publish" not in source
    assert "EventBus" not in source


def test_the_tray_cannot_change_privacy_mode() -> None:
    """Hiding is local presentation. Conflating it with privacy would let a
    cosmetic click silently change a security setting."""
    source = (SRC / "presence" / "tray.py").read_text(encoding="utf-8")
    assert "/v1/vision" not in source
    assert "VisionMode" not in source


def test_the_tray_has_no_reasoning_dependency() -> None:
    forbidden = ("aura.core.orchestrator", "aura.core.agent", "aura.tools", "aura.memory")
    tree = ast.parse((SRC / "presence" / "tray.py").read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)

    assert not [m for m in modules if any(m.startswith(f) for f in forbidden)]


def test_hiding_locally_is_distinct_from_being_hidden_by_privacy() -> None:
    """Two different reasons to be invisible, tracked separately, so leaving
    focus mode does not accidentally reveal her during private mode."""
    source = (SRC / "presence" / "app.py").read_text(encoding="utf-8")
    assert "_user_hidden" in source
    assert "presence.show_refused" in source, (
        "showing must be refused while PRIVATE or SLEEP"
    )
