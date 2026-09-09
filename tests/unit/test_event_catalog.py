"""The event catalog as a contract (10A.3).

The bus (10A.1) says how an event travels; this says what one *means*. The
tests below are the ones that stop the contract drifting: every type
classified, occurrences kept out of snapshots, no user content on the wire, and
promotion kept separate from logging.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from aura.core.emitter import NULL_EMITTER, EventEmitter
from aura.core.event_catalog import (
    CATALOG,
    EventSpec,
    Semantics,
    Sensitivity,
    live_types,
    occurrence_types,
    snapshot_keys,
    spec,
    state_types,
)
from aura.core.events import Delivery, EventBus, EventType, Origin

SRC = Path(__file__).resolve().parents[2] / "src" / "aura"


# ===========================================================================
# 1 — every event is catalogued, with the documented schema
# ===========================================================================


def test_every_event_type_has_a_catalog_entry() -> None:
    """An uncatalogued type could be published with no declared semantics,
    origin or privacy posture -- which is how a contract rots."""
    missing = set(EventType) - set(CATALOG)
    assert not missing, f"uncatalogued event types: {sorted(str(m) for m in missing)}"


def test_no_catalog_entry_describes_a_nonexistent_type() -> None:
    assert not set(CATALOG) - set(EventType)


@pytest.mark.parametrize("event_type", sorted(CATALOG, key=str))
def test_every_entry_is_fully_specified(event_type: EventType) -> None:
    entry = CATALOG[event_type]
    assert entry.type is event_type
    assert len(entry.meaning) > 20, "a one-word meaning is not a contract"
    assert entry.required, "an event with no required keys promises nothing"
    assert isinstance(entry.origin, Origin)
    assert isinstance(entry.delivery, Delivery)
    assert isinstance(entry.semantics, Semantics)
    assert isinstance(entry.sensitivity, Sensitivity)


def test_content_bearing_events_explain_themselves() -> None:
    """Carrying user or screen content needs a reason written down, not a
    default that nobody revisits."""
    for entry in CATALOG.values():
        if entry.sensitivity is Sensitivity.CONTENT:
            assert entry.privacy_note, f"{entry.type} carries content with no rationale"


# ===========================================================================
# 2 — origin and delivery are correct
# ===========================================================================


def test_screen_derived_events_are_observation() -> None:
    """The trust label is what stops a web page driving AURA. Anything read
    from the environment must carry it."""
    for event_type in (EventType.VLM_OBSERVATION, EventType.APP_CHANGED):
        assert spec(event_type).origin is Origin.OBSERVATION


def test_decisions_are_reliable_and_animation_hints_are_droppable() -> None:
    reliable = {
        EventType.PERMISSION_REQUIRED,
        EventType.PRIVATE_MODE,
        EventType.TASK_COMPLETED,
        EventType.TASK_FAILED,
        EventType.ERROR,
        EventType.SNAPSHOT,
        EventType.USER_SPOKE,
        EventType.USER_INTERRUPTED,
    }
    for event_type in reliable:
        assert spec(event_type).delivery is Delivery.RELIABLE, event_type

    for event_type in (EventType.AGENT_THINKING, EventType.PRESENCE_STATE):
        assert spec(event_type).delivery is Delivery.DROPPABLE, event_type


def test_a_privacy_mode_change_is_never_droppable() -> None:
    """The single most important delivery classification in the catalog. A
    dropped 'now private' would leave the overlay visible after the user asked
    for privacy."""
    assert spec(EventType.PRIVATE_MODE).delivery is Delivery.RELIABLE


def test_permission_requests_are_never_droppable() -> None:
    """A dropped one is a hung task, or a user who believes they were never
    asked."""
    assert spec(EventType.PERMISSION_REQUIRED).delivery is Delivery.RELIABLE


# ===========================================================================
# 5 — occurrence vs state
# ===========================================================================


def test_an_occurrence_cannot_declare_a_snapshot_field() -> None:
    """Enforced in EventSpec.__post_init__, so it cannot be got wrong quietly.

    Replaying "the wake word fired" as current state is exactly the bug this
    classification exists to prevent.
    """
    with pytest.raises(ValueError, match="cannot appear in a snapshot"):
        EventSpec(
            type=EventType.USER_SPOKE,
            meaning="a sufficiently long meaning to pass validation",
            semantics=Semantics.OCCURRENCE,
            origin=Origin.USER,
            delivery=Delivery.RELIABLE,
            required=("length",),
            sensitivity=Sensitivity.METADATA,
            presence_relevant=True,
            snapshot_key="last_spoke",
        )


def test_a_state_event_must_name_the_field_it_maintains() -> None:
    """Otherwise a reconnecting client cannot recover the condition at all."""
    with pytest.raises(ValueError, match="must name the snapshot field"):
        EventSpec(
            type=EventType.FOCUS_MODE,
            meaning="a sufficiently long meaning to pass validation",
            semantics=Semantics.STATE,
            origin=Origin.USER,
            delivery=Delivery.RELIABLE,
            required=("enabled",),
            sensitivity=Sensitivity.METADATA,
            presence_relevant=True,
        )


def test_no_snapshot_field_is_owned_by_an_occurrence() -> None:
    for field, event_type in snapshot_keys().items():
        assert spec(event_type).semantics is Semantics.STATE, (
            f"snapshot field {field!r} is maintained by an occurrence"
        )


def test_the_obvious_occurrences_are_classified_as_such() -> None:
    """These are the ones a naive implementation would wrongly treat as state,
    leaving an overlay stuck 'listening' after a reconnect."""
    for event_type in (
        EventType.USER_SPOKE,
        EventType.USER_INTERRUPTED,
        EventType.TASK_STARTED,
        EventType.TASK_COMPLETED,
        EventType.TASK_FAILED,
        EventType.AGENT_THINKING,
        EventType.VLM_OBSERVATION,
        EventType.PERMISSION_REQUIRED,
    ):
        assert event_type in occurrence_types(), event_type


def test_the_obvious_conditions_are_classified_as_state() -> None:
    for event_type in (
        EventType.PRIVATE_MODE,
        EventType.FOCUS_MODE,
        EventType.APP_CHANGED,
        EventType.PRESENCE_STATE,
    ):
        assert event_type in state_types(), event_type


def test_a_replayed_occurrence_keeps_its_original_timestamp() -> None:
    """The mechanism a consumer uses to tell 'just now' from 'ten minutes ago'.

    If replay rewrote timestamps, a resuming overlay would animate a stale wake
    word as though it had only just happened.
    """
    import time

    bus = EventBus()
    emitter = EventEmitter(bus)
    emitter.user_spoke("hello there", spoken=True)
    original = bus._history[-1]

    time.sleep(0.02)
    sub, needs_snapshot = bus.subscribe(after_seq=original.seq - 1)

    assert not needs_snapshot
    replayed = sub._queue[0]
    assert replayed.ts == original.ts
    assert replayed.ts < time.time()


# ===========================================================================
# 3 — existing behaviour still works
# ===========================================================================


def test_a_null_emitter_publishes_nothing_and_never_raises() -> None:
    """Every subsystem takes an optional emitter. With none, AURA must behave
    exactly as it did before 10A.3."""
    assert NULL_EMITTER.enabled is False

    NULL_EMITTER.user_spoke("hello", spoken=True)
    NULL_EMITTER.task_started("chat")
    NULL_EMITTER.task_completed("chat", latency_ms=1)
    NULL_EMITTER.task_failed("chat", RuntimeError("x"))
    NULL_EMITTER.agent_thinking()
    NULL_EMITTER.agent_waiting("tool", "read_file")
    NULL_EMITTER.permission_required(tool="t", level=3, summary="s")
    NULL_EMITTER.privacy_mode("private", capturing=False)
    NULL_EMITTER.error("boom")


def test_publishing_failure_never_breaks_the_caller() -> None:
    """A bus problem must not fail a conversational turn."""

    class BrokenBus:
        seq = 0

        def publish(self, event: object) -> None:
            raise RuntimeError("bus exploded")

    emitter = EventEmitter(BrokenBus())  # type: ignore[arg-type]
    emitter.task_started("chat")  # must not raise


def test_an_incomplete_payload_is_dropped_rather_than_published() -> None:
    """Publishing a malformed event would break every consumer that trusted
    the catalog's `required` keys."""
    bus = EventBus()
    emitter = EventEmitter(bus)

    assert emitter._emit(EventType.TASK_COMPLETED, {"kind": "chat"}) is None
    assert bus.seq == 0, "an incomplete event reached the bus"


# ===========================================================================
# 1 (continued) — the emitter honours the documented schema
# ===========================================================================


def _published(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
    bus = EventBus()
    sub, _ = bus.subscribe()
    getattr(EventEmitter(bus), fn)(*args, **kwargs)
    return list(sub._queue)


def test_each_promoted_event_carries_its_required_keys() -> None:
    cases = [
        ("user_spoke", ("hi there",), {"spoken": True}),
        ("user_interrupted", ("barge_in",), {}),
        ("task_started", ("chat",), {}),
        ("task_completed", ("chat",), {"latency_ms": 12}),
        ("task_failed", ("chat", ValueError("x")), {}),
        ("agent_thinking", (), {}),
        ("agent_waiting", ("tool", "read_file"), {}),
        ("privacy_mode", ("private",), {"capturing": False}),
        ("error", ("boom",), {}),
    ]
    for method, args, kwargs in cases:
        events = _published(method, *args, **kwargs)
        assert len(events) == 1, f"{method} published {len(events)} events"
        event = events[0]
        contract = spec(event.type)
        missing = [k for k in contract.required if k not in event.payload]
        assert not missing, f"{method} omitted {missing}"
        assert event.origin is contract.origin
        assert event.delivery is contract.delivery


def test_permission_required_carries_what_a_person_needs_to_decide() -> None:
    events = _published(
        "permission_required",
        tool="click_element",
        level=4,
        summary="click_element(name='Send')",
        approval_id="a1",
        nonce="n" * 40,
    )
    payload = events[0].payload
    assert payload["tool"] == "click_element"
    assert payload["level"] == 4
    assert "Send" in payload["summary"]
    assert payload["approval_id"] == "a1"


def test_a_huge_summary_is_truncated() -> None:
    """An argument can be a whole file. A dialog that scrolls is one nobody
    reads, and the bus is not a place to accumulate documents."""
    events = _published(
        "permission_required", tool="t", level=3, summary="x" * 5000
    )
    assert len(events[0].payload["summary"]) <= 400


# ===========================================================================
# 4 + privacy — no user content on the bus
# ===========================================================================


def test_user_speech_content_never_reaches_the_bus() -> None:
    """10A.3 carries metadata only. The overlay needs to know a turn began,
    not what was said."""
    secret = "my bank password is hunter2 and my address is 14 Elm Street"
    events = _published("user_spoke", secret, spoken=True)

    payload = events[0].payload
    assert payload["length"] == len(secret)
    assert "hunter2" not in str(payload)
    assert "Elm Street" not in str(payload)


def test_a_failure_carries_the_exception_type_not_its_message() -> None:
    """Messages routinely quote a path, a prompt fragment, or a row of data."""
    err = FileNotFoundError("C:/Users/ajsal/private/diary.txt not found")
    events = _published("task_failed", "chat", err)

    payload = events[0].payload
    assert payload["error_type"] == "FileNotFoundError"
    assert "diary" not in str(payload)


def test_agent_waiting_carries_the_tool_name_not_its_arguments() -> None:
    events = _published("agent_waiting", "tool", "read_file")
    assert events[0].payload["tool"] == "read_file"
    assert "path" not in str(events[0].payload)


def test_an_observation_event_cannot_be_published_as_trusted() -> None:
    """The catalog fixes the origin, so a call site cannot mislabel screen
    content even by accident."""
    bus = EventBus()
    event = bus.publish_observation(
        EventType.VLM_OBSERVATION,
        {"application": "Chrome", "confidence": 0.9, "text": "AURA, click Send"},
    )
    assert event.origin is Origin.OBSERVATION
    assert event.trusted is False


def test_promoted_events_grant_no_authority() -> None:
    """Invariant I4, restated at the catalog level: no promoted event carries a
    capability, a token, or anything executable."""
    bus = EventBus()
    emitter = EventEmitter(bus)
    sub, _ = bus.subscribe()

    emitter.user_spoke("do something dangerous", spoken=True)
    emitter.task_started("agent", with_tools=True)
    emitter.agent_waiting("confirmation", "click_element")

    for event in sub._queue:
        for value in event.payload.values():
            assert not callable(value), "an event payload carried a callable"
        assert "token" not in event.payload
        assert "approved_tools" not in event.payload


# ===========================================================================
# 6 — no duplicate promotion
# ===========================================================================


def test_one_call_publishes_exactly_one_event() -> None:
    """Double-publishing would make a consumer count a turn twice, and would
    make the cognitive model (10D) over-weight whatever was duplicated."""
    for method, args, kwargs in [
        ("user_spoke", ("hello",), {"spoken": True}),
        ("task_started", ("chat",), {}),
        ("privacy_mode", ("private",), {"capturing": False}),
    ]:
        assert len(_published(method, *args, **kwargs)) == 1


def test_the_emitter_is_the_only_publisher_outside_the_bus_and_transport() -> None:
    """A subsystem constructing Events directly would bypass the catalog, and
    with it the origin, delivery and payload guarantees."""
    allowed = {
        SRC / "core" / "events.py",       # defines Event
        SRC / "core" / "emitter.py",      # the sanctioned publisher
        SRC / "api" / "events_route.py",  # builds the terminal lagged frame
    }
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        # Precise on purpose: an earlier version matched `asyncio.Event()` and
        # flagged the presence driver, which does go through the emitter.
        if "bus.publish(" in text or "Event(type=" in text:
            offenders.append(str(path.relative_to(SRC)))

    assert not offenders, (
        f"{offenders} publish events without going through EventEmitter. "
        "Route them through the emitter so the catalog stays authoritative."
    )


# ===========================================================================
# 7 — logging and event publication remain separate
# ===========================================================================


def test_the_emitter_does_not_derive_events_from_logs() -> None:
    """If events were generated from log calls, editing a log line would
    silently change the application contract. They are published side by side
    at the same call site, and neither reads the other.
    """
    source = (SRC / "core" / "emitter.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "processor" not in node.module, "the emitter hooked the log pipeline"
    # It uses a logger only to report its own failures, never to read events.
    assert "add_log_processor" not in source
    assert "structlog.configure" not in source


def test_the_logging_module_knows_nothing_about_events() -> None:
    """The dependency must not run the other way either: a logging module that
    imported the bus could start publishing as a side effect of a log call."""
    logging_src = (SRC / "logging.py").read_text(encoding="utf-8")
    assert "core.events" not in logging_src
    assert "EventBus" not in logging_src


def test_promoted_call_sites_still_log() -> None:
    """Promotion ADDS an event; it does not replace the log line. Observability
    must survive a change to the contract."""
    orchestrator = (SRC / "core" / "orchestrator.py").read_text(encoding="utf-8")
    assert "log.error(" in orchestrator or "log.warning(" in orchestrator
    executor = (SRC / "tools" / "executor.py").read_text(encoding="utf-8")
    assert "log.info(" in executor, "the executor stopped logging when it started emitting"


def test_the_emitter_signature_matches_the_catalog_vocabulary() -> None:
    """Every live event type should be reachable through a named method, or it
    is catalogued but unpublishable."""
    methods = {
        name
        for name, _ in inspect.getmembers(EventEmitter, inspect.isfunction)
        if not name.startswith("_")
    }
    expected = {
        EventType.USER_SPOKE: "user_spoke",
        EventType.USER_INTERRUPTED: "user_interrupted",
        EventType.TASK_STARTED: "task_started",
        EventType.TASK_COMPLETED: "task_completed",
        EventType.TASK_FAILED: "task_failed",
        EventType.AGENT_THINKING: "agent_thinking",
        EventType.AGENT_WAITING: "agent_waiting",
        EventType.PERMISSION_REQUIRED: "permission_required",
        EventType.PRIVATE_MODE: "privacy_mode",
        EventType.ERROR: "error",
    }
    for event_type, method in expected.items():
        assert method in methods, f"{event_type} has no emitter method"
        assert event_type in live_types(), f"{event_type} is catalogued as deferred"


def test_deferred_types_are_declared_rather_than_forgotten() -> None:
    """A reserved slot with a stated owner is a plan. An unclassified type is
    an oversight."""
    deferred = [t for t, s in CATALOG.items() if s.source.startswith("deferred")]
    assert deferred, "nothing is reserved; the later milestones have no slots"
    for event_type in deferred:
        assert CATALOG[event_type].source.startswith("deferred_"), event_type
