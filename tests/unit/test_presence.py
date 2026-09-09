"""The presence state machine (10A.4).

A state-resolution system, so the tests are written against *resolution* rather
than against a sequence of assignments: given these conditions, what does AURA
look like she is doing, and why.

Time is injected everywhere. A test that slept would be slow and flaky, and
dwell and expiry are exactly the properties worth testing precisely.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aura.core.events import Delivery, Event, EventBus, EventType, Origin
from aura.core.presence import (
    ACTIVITY,
    DURATION,
    MIN_DWELL_S,
    PRIORITY,
    PresenceDriver,
    PresenceState,
    PresenceStateMachine,
)

SRC = Path(__file__).resolve().parents[2] / "src" / "aura"


class Clock:
    """A hand-cranked monotonic clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def machine(clock: Clock) -> PresenceStateMachine:
    return PresenceStateMachine(clock)


def ev(event_type: EventType, origin: Origin = Origin.CORE, **payload: object) -> Event:
    """`event_type`, not `kind` -- several payloads carry a `kind` key."""
    return Event(
        type=event_type, origin=origin, delivery=Delivery.RELIABLE, payload=payload
    )


def spoke() -> Event:
    return ev(EventType.USER_SPOKE, Origin.USER, length=10, spoken=True)


def started(with_tools: bool = False) -> Event:
    return ev(EventType.TASK_STARTED, kind="chat", with_tools=with_tools)


def completed(tool_calls: int = 0, stop_reason: str = "") -> Event:
    return ev(
        EventType.TASK_COMPLETED,
        kind="chat",
        latency_ms=100,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
    )


# ===========================================================================
# THE NAMED SEQUENCES
# ===========================================================================


def test_thinking_to_success_to_idle(machine: PresenceStateMachine, clock: Clock) -> None:
    """THINKING -> TASK_COMPLETED -> SUCCESS -> IDLE."""
    machine.on_event(started())
    assert machine.state is PresenceState.THINKING

    clock.advance(MIN_DWELL_S + 0.1)
    machine.on_event(completed())
    assert machine.state is PresenceState.SUCCESS

    # SUCCESS is transient: it expires rather than persisting.
    clock.advance(DURATION[PresenceState.SUCCESS] + 0.1)
    machine.tick()
    assert machine.state is PresenceState.IDLE


def test_interruption_ends_the_work(machine: PresenceStateMachine, clock: Clock) -> None:
    """THINKING -> USER_INTERRUPTED -> LISTENING.

    LISTENING is *lower* priority than THINKING, so this only works because
    interruption CLEARS the activity rather than outranking it.
    """
    machine.on_event(started())
    assert machine.state is PresenceState.THINKING
    assert PRIORITY[PresenceState.LISTENING] < PRIORITY[PresenceState.THINKING]

    clock.advance(MIN_DWELL_S + 0.1)
    machine.on_event(
        ev(EventType.USER_INTERRUPTED, Origin.USER, method="barge_in")
    )
    assert machine.state is PresenceState.LISTENING


def test_permission_request_moves_working_to_waiting(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """WORKING -> PERMISSION_REQUIRED -> WAITING."""
    machine.on_event(started(with_tools=True))
    assert machine.state is PresenceState.WORKING

    clock.advance(MIN_DWELL_S + 0.1)
    machine.on_event(
        ev(EventType.PERMISSION_REQUIRED, tool="write_file", level=2, summary="s")
    )
    assert machine.state is PresenceState.WAITING


def test_waiting_returns_to_working_when_the_task_resumes(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """WAITING -> the tool actually runs -> WORKING."""
    machine.on_event(started(with_tools=True))
    machine.on_event(
        ev(EventType.PERMISSION_REQUIRED, tool="write_file", level=2, summary="s")
    )
    assert machine.state is PresenceState.WAITING

    clock.advance(MIN_DWELL_S + 0.1)
    # The approval was granted and the tool is now running.
    machine.on_event(ev(EventType.AGENT_WAITING, reason="tool", tool="write_file"))
    machine._clear(PresenceState.WAITING)
    machine.tick()
    assert machine.state is PresenceState.WORKING


def test_completion_clears_a_pending_permission(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """A task that ends while WAITING must not leave AURA stuck waiting."""
    machine.on_event(started(with_tools=True))
    machine.on_event(
        ev(EventType.PERMISSION_REQUIRED, tool="write_file", level=2, summary="s")
    )
    assert machine.state is PresenceState.WAITING

    clock.advance(MIN_DWELL_S + 0.1)
    machine.on_event(completed())
    assert machine.state is PresenceState.SUCCESS
    assert PresenceState.WAITING not in {c.state for c in machine.active_conditions()}


# ===========================================================================
# PRECEDENCE
# ===========================================================================


def test_error_overrides_ordinary_activity(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    machine.on_event(started(with_tools=True))
    assert machine.state is PresenceState.WORKING

    machine.on_event(ev(EventType.ERROR, code="model_unavailable"))
    assert machine.state is PresenceState.ERROR


def test_error_is_not_masked_by_a_success_animation(
    machine: PresenceStateMachine,
) -> None:
    """Dwell must be preemptable, or a 2.5s SUCCESS could hide a failure."""
    machine.on_event(completed())
    assert machine.state is PresenceState.SUCCESS

    # No time has passed: we are inside the dwell window.
    machine.on_event(ev(EventType.ERROR, code="disk_full"))
    assert machine.state is PresenceState.ERROR


def test_private_overrides_everything_including_error(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """The deliberate choice recorded in Priority's docstring.

    Privacy is an instruction; an error is information. An overlay that
    reappears to report a problem after the user asked for privacy has broken
    the promise -- and the error is still logged, audited and on the bus.
    """
    machine.on_event(ev(EventType.ERROR, code="something_bad"))
    assert machine.state is PresenceState.ERROR

    clock.advance(MIN_DWELL_S + 0.1)
    machine.on_event(ev(EventType.PRIVATE_MODE, Origin.USER, mode="private", capturing=False))
    assert machine.state is PresenceState.PRIVATE


def test_private_suppresses_activity_that_would_imply_observation(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """Requirement 8. Nothing that looks like watching or listening may show
    while the user has asked for privacy."""
    machine.on_event(ev(EventType.PRIVATE_MODE, Origin.USER, mode="private", capturing=False))

    for event in (spoke(), started(with_tools=True), ev(EventType.AGENT_THINKING, step=1)):
        clock.advance(1.0)
        machine.on_event(event)
        assert machine.state is PresenceState.PRIVATE, (
            f"{event.type} became visible during private mode"
        )


def test_leaving_private_returns_to_what_is_actually_true(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    machine.on_event(started(with_tools=True))
    machine.on_event(ev(EventType.PRIVATE_MODE, Origin.USER, mode="private", capturing=False))
    assert machine.state is PresenceState.PRIVATE

    clock.advance(MIN_DWELL_S + 0.1)
    machine.on_event(ev(EventType.PRIVATE_MODE, Origin.USER, mode="assist", capturing=True))
    # The work was still going on underneath.
    assert machine.state is PresenceState.WORKING


def test_the_priority_table_is_a_total_order(machine: PresenceStateMachine) -> None:
    """Every state ranked, and no two sharing a rank -- otherwise two
    simultaneous conditions would resolve by dict order."""
    assert set(PRIORITY) == set(PresenceState)
    assert len(set(PRIORITY.values())) == len(PRIORITY)


def test_safety_outranks_every_cosmetic_state() -> None:
    """Requirement 9, asserted against the table rather than a scenario."""
    for cosmetic in (PresenceState.HAPPY, PresenceState.SUCCESS, PresenceState.ANNOYED):
        assert PRIORITY[PresenceState.ERROR] > PRIORITY[cosmetic]
        assert PRIORITY[PresenceState.WARNING] > PRIORITY[cosmetic]
        assert PRIORITY[PresenceState.PRIVATE] > PRIORITY[cosmetic]


def test_simultaneous_conditions_resolve_deterministically(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """Three conditions true at once, applied in two different orders, must
    resolve to the same state."""
    a = PresenceStateMachine(Clock())
    b = PresenceStateMachine(Clock())

    events = [
        started(with_tools=True),
        ev(EventType.PERMISSION_REQUIRED, tool="t", level=3, summary="s"),
        ev(EventType.ERROR, code="boom"),
    ]
    for event in events:
        a.on_event(event)
    for event in reversed(events):
        b.on_event(event)

    assert a.state is b.state is PresenceState.ERROR


# ===========================================================================
# DWELL, HYSTERESIS, EXPIRY
# ===========================================================================


def test_success_does_not_disappear_immediately(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """Below ~1.5s a state change reads as a flicker rather than feedback."""
    assert DURATION[PresenceState.SUCCESS] >= 1.5

    machine.on_event(completed())
    assert machine.state is PresenceState.SUCCESS

    clock.advance(DURATION[PresenceState.SUCCESS] - 0.2)
    machine.tick()
    assert machine.state is PresenceState.SUCCESS, "SUCCESS vanished before it was seen"


def test_repeated_thinking_events_do_not_thrash(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """A fast agent loop re-asserts THINKING constantly. Re-asserting a held
    condition must be a no-op, or the avatar strobes."""
    machine.on_event(started())
    baseline = machine.transition_count

    for i in range(500):
        clock.advance(0.01)
        machine.on_event(ev(EventType.AGENT_THINKING, step=i))

    assert machine.state is PresenceState.THINKING
    assert machine.transition_count == baseline, "repeated events caused transitions"


def test_alternating_activity_is_damped_by_dwell(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """THINKING and WORKING alternating faster than the dwell window must not
    produce a transition per event."""
    machine.on_event(started())
    baseline = machine.transition_count

    for _ in range(50):
        clock.advance(0.02)  # well inside MIN_DWELL_S
        machine.on_event(ev(EventType.AGENT_WAITING, reason="tool", tool="read_file"))
        machine.on_event(ev(EventType.AGENT_THINKING, step=1))

    assert machine.transition_count - baseline <= 2


def test_a_transient_state_returns_to_the_underlying_one(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """Requirement 6. A WARNING over ongoing work reverts to the work, not to
    IDLE -- the work never stopped."""
    machine.on_event(started(with_tools=True))
    machine.on_event(ev(EventType.ERROR, code="permission_denied"))
    assert machine.state is PresenceState.WARNING

    clock.advance(DURATION[PresenceState.WARNING] + 0.1)
    machine.tick()
    assert machine.state is PresenceState.WORKING


def test_dwell_does_not_delay_a_higher_priority_state(
    machine: PresenceStateMachine,
) -> None:
    machine.on_event(spoke())
    assert machine.state is PresenceState.LISTENING

    # Immediately, inside the dwell window.
    machine.on_event(ev(EventType.PRIVATE_MODE, Origin.USER, mode="private", capturing=False))
    assert machine.state is PresenceState.PRIVATE


def test_idle_becomes_sleep_after_long_silence(clock: Clock) -> None:
    machine = PresenceStateMachine(clock, idle_to_sleep_s=60.0)
    machine.on_event(completed())
    clock.advance(120.0)
    machine.tick()
    assert machine.state is PresenceState.SLEEP


def test_activity_wakes_from_sleep(clock: Clock) -> None:
    machine = PresenceStateMachine(clock, idle_to_sleep_s=60.0)
    clock.advance(120.0)
    machine.tick()
    assert machine.state is PresenceState.SLEEP

    machine.on_event(spoke())
    assert machine.state is PresenceState.LISTENING


# ===========================================================================
# OCCURRENCE VS STATE (requirements 4 and 10)
# ===========================================================================


def test_task_completed_does_not_become_persistent_state(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """Requirement 4. The occurrence is visible, then it is over."""
    machine.on_event(completed())
    assert machine.state is PresenceState.SUCCESS

    # Well past SUCCESS's lifetime but short of the sleep timeout, so IDLE is
    # the right answer rather than SLEEP.
    clock.advance(60.0)
    machine.tick()
    assert machine.state is PresenceState.IDLE


def test_a_reconnecting_client_is_told_current_state_not_history(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """Requirement 10. Recovery is by being told what is true now."""
    machine.on_event(started(with_tools=True))
    snap = machine.snapshot()

    assert snap["presence_state"] == "working"
    assert "conditions" in snap
    # No event history, no way to replay an occurrence out of it.
    assert "events" not in snap


def test_an_old_success_does_not_survive_into_a_snapshot(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """The specific failure named in the brief: AURA must not appear freshly
    successful to a client that reconnects five minutes later."""
    machine.on_event(completed())
    assert machine.snapshot()["presence_state"] == "success"

    clock.advance(300.0)
    machine.tick()
    assert machine.snapshot()["presence_state"] == "idle"


def test_persistent_state_does_survive_a_reconnect(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """The other half: a condition that is genuinely still true must be
    recoverable, or the overlay comes back visible during private mode."""
    machine.on_event(ev(EventType.PRIVATE_MODE, Origin.USER, mode="private", capturing=False))
    clock.advance(3600.0)
    machine.tick()
    assert machine.snapshot()["presence_state"] == "private"


# ===========================================================================
# UNTRUSTED INPUT
# ===========================================================================


def test_an_observation_cannot_drive_presence(machine: PresenceStateMachine) -> None:
    """Screen content must not be able to change what AURA looks like she is
    doing -- a small but real lie a web page could otherwise tell."""
    machine.on_event(
        Event(
            type=EventType.VLM_OBSERVATION,
            origin=Origin.OBSERVATION,
            payload={"application": "Chrome", "confidence": 0.9, "text": "working"},
        )
    )
    assert machine.state is PresenceState.IDLE
    assert machine.transition_count == 0


def test_the_machine_ignores_its_own_output(machine: PresenceStateMachine) -> None:
    """PRESENCE_STATE is what this machine publishes. Consuming it would be a
    feedback loop."""
    machine.on_event(ev(EventType.PRESENCE_STATE, state="error", previous="idle", cause="x"))
    assert machine.state is PresenceState.IDLE


# ===========================================================================
# PERSONALITY
# ===========================================================================


def test_happy_is_reserved_for_multi_step_work(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """An assistant that emotes constantly is noise; one that emotes rarely
    carries information."""
    machine.on_event(completed(tool_calls=0))
    assert machine.state is PresenceState.SUCCESS

    clock.advance(10.0)
    machine.tick()
    machine.on_event(completed(tool_calls=3))
    assert machine.state is PresenceState.HAPPY


def test_annoyed_is_earned_by_giving_up_not_by_being_asked_twice(
    machine: PresenceStateMachine,
) -> None:
    machine.on_event(completed(tool_calls=1, stop_reason="step_budget"))
    assert machine.state is PresenceState.ANNOYED


def test_personality_states_can_be_switched_off(clock: Clock) -> None:
    machine = PresenceStateMachine(clock, personality=False)
    machine.on_event(completed(tool_calls=1, stop_reason="step_budget"))
    assert machine.state is not PresenceState.ANNOYED

    machine.on_event(completed(tool_calls=5))
    assert machine.state is not PresenceState.HAPPY


# ===========================================================================
# ARCHITECTURE
# ===========================================================================


def test_the_state_machine_has_no_ui_dependency() -> None:
    """The architectural rule. If this fails, the presentation layer has
    stopped being replaceable."""
    tree = ast.parse((SRC / "core" / "presence.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = ("PySide6", "PyQt", "tkinter", "pystray", "PIL")
    leaked = {m for m in imported if any(m.startswith(f) for f in forbidden)}
    assert not leaked, f"the presence state machine imports UI code: {sorted(leaked)}"


def test_no_module_in_core_imports_a_ui_toolkit() -> None:
    """The rule holds for the whole of core, not only this file."""
    offenders: list[str] = []
    for path in (SRC / "core").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "PySide6" in text or "import tkinter" in text:
            offenders.append(path.name)
    assert not offenders, f"core depends on a UI toolkit: {offenders}"


def test_activity_states_are_the_ones_an_interruption_should_end() -> None:
    assert ACTIVITY == {
        PresenceState.THINKING,
        PresenceState.WORKING,
        PresenceState.WAITING,
    }


# ===========================================================================
# THE DRIVER
# ===========================================================================


async def test_the_driver_publishes_presence_state_changes() -> None:
    bus = EventBus()
    driver = PresenceDriver(bus, PresenceStateMachine(Clock()), tick_s=0.01)
    watcher, _ = bus.subscribe()
    await driver.start()

    bus.publish(started())
    await _settle()

    published = [
        e for e in list(watcher._queue) if e.type is EventType.PRESENCE_STATE
    ]
    await driver.stop()

    assert published, "the driver published no presence state"
    assert published[0].payload["state"] == "thinking"
    assert published[0].payload["previous"] == "idle"


async def test_the_driver_survives_a_burst() -> None:
    """Rapid event sequences must not crash it or leave it wedged."""
    bus = EventBus()
    clock = Clock()
    driver = PresenceDriver(bus, PresenceStateMachine(clock), tick_s=0.01)
    await driver.start()

    for i in range(2000):
        bus.publish(ev(EventType.AGENT_THINKING, step=i))
    await _settle()

    state = driver.machine.state
    await driver.stop()
    assert state in (PresenceState.THINKING, PresenceState.IDLE)


async def test_the_driver_stops_cleanly() -> None:
    bus = EventBus()
    driver = PresenceDriver(bus, PresenceStateMachine(Clock()), tick_s=0.01)
    before = bus.subscriber_count
    await driver.start()
    await _settle()
    await driver.stop()

    assert bus.subscriber_count == before, "the driver leaked a subscription"
    await driver.stop()  # idempotent


async def _settle(rounds: int = 12) -> None:
    import asyncio

    for _ in range(rounds):
        await asyncio.sleep(0.02)


# ===========================================================================
# FOUND LIVE (10A.4)
# ===========================================================================


def test_a_new_turn_supersedes_an_unanswered_permission(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """Found on the running stack.

    A refused write_file left WAITING held with nothing to clear it, so it
    outranked THINKING through an entirely unrelated later turn -- AURA looked
    like she was waiting for an approval that no longer existed.
    """
    machine.on_event(
        ev(EventType.PERMISSION_REQUIRED, tool="write_file", level=2, summary="s")
    )
    assert machine.state is PresenceState.WAITING

    clock.advance(MIN_DWELL_S + 0.1)
    machine.on_event(started())
    assert machine.state is PresenceState.THINKING


def test_an_unanswered_permission_does_not_wait_forever(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """Nothing resumes a refused approval until 10C, so WAITING needs a
    ceiling of its own."""
    machine.on_event(
        ev(EventType.PERMISSION_REQUIRED, tool="write_file", level=2, summary="s")
    )
    assert machine.state is PresenceState.WAITING

    clock.advance(DURATION[PresenceState.WAITING] + 1.0)
    machine.tick()
    assert machine.state is PresenceState.IDLE


def test_private_mode_does_not_leak_the_underlying_activity(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    """Found live: the snapshot read conditions ['success', 'private'], which
    tells a client what AURA was doing underneath.

    Requirement 8 is about what can be INFERRED, not only about what is drawn.
    """
    machine.on_event(completed())
    machine.on_event(
        ev(EventType.PRIVATE_MODE, Origin.USER, mode="private", capturing=False)
    )

    snap = machine.snapshot()
    assert snap["presence_state"] == "private"
    assert snap["conditions"] == ["private"], (
        f"private mode leaked {snap['conditions']}"
    )


def test_sleep_also_does_not_leak_activity(
    machine: PresenceStateMachine, clock: Clock
) -> None:
    machine.on_event(started(with_tools=True))
    machine.on_event(ev(EventType.PRIVATE_MODE, Origin.USER, mode="sleep", capturing=False))

    assert machine.snapshot()["conditions"] == ["sleep"]
