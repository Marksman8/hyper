"""The SPEECH_* producers (10B.1).

10A catalogued these three events and marked them `deferred_10b`; nothing
published them. This closes that, the same shape of gap the 10A.12 producers
closed.

Two properties carry the weight here. The event that legitimately carries prose
must carry AURA's words and not the user's, and the closing event must be
published exactly once however the turn unwinds -- a subtitle whose SPEECH_ENDED
depends on a tidy exit stays on screen after the room goes quiet.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

import pytest

from aura.core.emitter import MAX_SPEECH_CHUNK, EventEmitter
from aura.core.event_catalog import Semantics, spec
from aura.core.events import Delivery, Event, EventBus, EventType, Origin
from aura.core.presence import PresenceState, PresenceStateMachine
from aura.voice.reporter import SpeechReporter


def _bus() -> tuple[EventBus, EventEmitter]:
    bus = EventBus()
    return bus, EventEmitter(bus)


class FakeClient:
    """Stands in for CoreClient. Records, and can be told to fail or block."""

    def __init__(self, *, ok: bool = True, block: threading.Event | None = None) -> None:
        self.published: list[dict[str, Any]] = []
        self._ok = ok
        self._block = block
        self.calls = 0

    def publish_event(self, payload: dict[str, Any]) -> bool:
        self.calls += 1
        if self._block is not None:
            self._block.wait(timeout=5.0)
        self.published.append(payload)
        return self._ok


# ===========================================================================
# The events themselves
# ===========================================================================


def test_the_three_events_are_published() -> None:
    bus, emitter = _bus()
    sub, _ = bus.subscribe()

    emitter.speech_started(length=42)
    emitter.speech_chunk("The build finished.")
    emitter.speech_ended(completed=True)

    types = [e.type for e in sub._queue]
    assert types == [
        EventType.SPEECH_STARTED,
        EventType.SPEECH_CHUNK,
        EventType.SPEECH_ENDED,
    ]


def test_speech_is_core_origin_and_trusted() -> None:
    """These are AURA's own utterances, not something observed or something the
    user did."""
    bus, emitter = _bus()
    sub, _ = bus.subscribe()
    emitter.speech_chunk("Hello.")

    event = next(iter(sub._queue))
    assert event.origin is Origin.CORE
    assert event.trusted is True


def test_speech_is_droppable() -> None:
    """A subtitle that skipped a sentence under load is a cosmetic loss. Making
    these reliable would let a slow consumer lag the bus over cosmetics."""
    bus, emitter = _bus()
    sub, _ = bus.subscribe()
    emitter.speech_started(length=10)

    assert next(iter(sub._queue)).delivery is Delivery.DROPPABLE


def test_speech_events_are_occurrences_and_never_in_the_snapshot() -> None:
    """"AURA said this sentence" is not a state to be recovered on reconnect.
    Replaying it would put a stale subtitle on a reconnecting overlay."""
    for event_type in (
        EventType.SPEECH_STARTED,
        EventType.SPEECH_CHUNK,
        EventType.SPEECH_ENDED,
    ):
        contract = spec(event_type)
        assert contract.semantics is Semantics.OCCURRENCE
        assert contract.snapshot_key is None


def test_the_chunk_carries_aura_words_while_user_speech_carries_a_length() -> None:
    """The asymmetry is the point.

    The user's words are captured input from an always-listening microphone.
    AURA's words are about to be said out loud in the room, and the subtitle
    exists so they can be read when the audio cannot be heard.
    """
    bus, emitter = _bus()
    sub, _ = bus.subscribe()

    emitter.user_spoke("what is my calendar today", spoken=True)
    emitter.speech_chunk("You have three meetings.")

    user_event, aura_event = list(sub._queue)
    assert "calendar" not in str(user_event.payload)
    assert user_event.payload["length"] == 25
    assert aura_event.payload["text"] == "You have three meetings."


def test_a_chunk_is_capped() -> None:
    """SPEECH_CHUNK is the one event that legitimately carries prose, so a cap
    is what stops it becoming "whatever the caller wanted on the bus"."""
    bus, emitter = _bus()
    sub, _ = bus.subscribe()

    emitter.speech_chunk("x" * 50_000)

    assert len(next(iter(sub._queue)).payload["text"]) == MAX_SPEECH_CHUNK


def test_speaking_is_not_a_presence_state() -> None:
    """She is not "speaking" INSTEAD of working. There is no SPEAKING state,
    and the mouth and subtitle are driven directly by these events."""
    machine = PresenceStateMachine()
    for event_type, payload in (
        (EventType.SPEECH_STARTED, {"length": 10}),
        (EventType.SPEECH_CHUNK, {"text": "hello"}),
        (EventType.SPEECH_ENDED, {"completed": True}),
    ):
        machine.on_event(Event(type=event_type, origin=Origin.CORE, payload=payload))

    assert machine.state is PresenceState.IDLE
    assert not hasattr(PresenceState, "SPEAKING")


def test_the_speech_types_are_no_longer_deferred() -> None:
    """A spec left marked `deferred` after its producer exists excludes itself
    from `live_snapshot_keys()`, and so from the completeness check that would
    have caught it. Three 10A specs had drifted that way; this asserts 10B.1's
    do not."""
    from aura.core.event_catalog import live_types

    live = live_types()
    for event_type in (
        EventType.SPEECH_STARTED,
        EventType.SPEECH_CHUNK,
        EventType.SPEECH_ENDED,
        # Fixed here too: built in 10A.4 and 10A.12, still marked deferred.
        EventType.APP_CHANGED,
        EventType.FOCUS_MODE,
        EventType.PRESENCE_STATE,
    ):
        assert event_type in live, f"{event_type} has a producer but is marked deferred"


def test_ended_distinguishes_finishing_from_being_cut_off() -> None:
    """The overlay needs the difference: an interrupted reply should not leave
    its last sentence on screen as though it were the final word."""
    bus, emitter = _bus()
    sub, _ = bus.subscribe()

    emitter.speech_ended(completed=False)

    assert next(iter(sub._queue)).payload["completed"] is False


# ===========================================================================
# The reporter -- off the speech path
# ===========================================================================


def test_the_reporter_does_not_block_the_caller() -> None:
    """The constraint the whole module exists for: a network round trip must
    not land between two spoken sentences."""
    gate = threading.Event()
    client = FakeClient(block=gate)
    reporter = SpeechReporter(client)
    reporter.start()

    started = time.perf_counter()
    for i in range(10):
        reporter.chunk(f"sentence {i}")
    elapsed = time.perf_counter() - started

    gate.set()
    reporter.close()
    assert elapsed < 0.1, f"publishing blocked the speech path for {elapsed:.3f}s"


def test_a_full_queue_drops_the_oldest() -> None:
    """What a subtitle needs is the sentence being spoken NOW; the backlog is
    by definition already behind the audio."""
    gate = threading.Event()
    reporter = SpeechReporter(FakeClient(block=gate), depth=4)
    # Not started: nothing drains, so the queue fills deterministically.

    for i in range(20):
        reporter.chunk(f"sentence {i}")

    assert reporter.dropped > 0
    remaining = [reporter._queue.get_nowait()["text"] for _ in range(4)]
    assert "sentence 19" in remaining, "the newest sentence was dropped"
    assert "sentence 0" not in remaining
    gate.set()


def test_an_unreachable_core_costs_subtitles_and_nothing_else() -> None:
    reporter = SpeechReporter(FakeClient(ok=False))
    reporter.start()

    reporter.started(10)
    reporter.chunk("hello")
    reporter.ended(completed=True)
    reporter.close()

    assert reporter.sent == 0


def test_a_raising_client_does_not_kill_the_thread() -> None:
    """Reporting is a convenience. It must never take down the process that is
    doing the talking."""

    class Exploding:
        def __init__(self) -> None:
            self.calls = 0

        def publish_event(self, payload: dict[str, Any]) -> bool:
            self.calls += 1
            raise RuntimeError("boom")

    client = Exploding()
    reporter = SpeechReporter(client)
    reporter.start()
    for _ in range(5):
        reporter.chunk("hello")
    reporter.close()

    assert client.calls == 5, "the worker died on the first failure"


def test_empty_chunks_are_not_published() -> None:
    """The sentence chunker can legitimately hand over punctuation alone."""
    reporter = SpeechReporter(FakeClient())
    reporter.chunk("   ")
    reporter.chunk("")

    with pytest.raises(queue.Empty):
        reporter._queue.get_nowait()


def test_close_is_safe_without_start() -> None:
    SpeechReporter(FakeClient()).close()


# ===========================================================================
# The bridge
# ===========================================================================


def _bus_of(client: Any) -> EventBus:
    return client._transport.app.state.aura.events


async def test_the_bridge_labels_speech_core_and_user_speech_user(client: Any) -> None:
    """The property is not "clients get USER". It is that **a client cannot
    choose an origin** -- the catalog decides, and the catalog is a local
    version-controlled file.

    That distinction only becomes visible with speech: the same endpoint now
    produces a CORE event and a USER event depending on which catalogued thing
    the caller said happened.
    """
    sub, _ = _bus_of(client).subscribe()

    await client.post("/v1/events/client", json={"type": "speech_chunk", "text": "Hi."})
    await client.post("/v1/events/client", json={"type": "user_spoke", "length": 4})

    # Filtered by type: USER_SPOKE also drives the presence state machine, so
    # the queue carries its PRESENCE_STATE reply too.
    by_type = {e.type: e for e in sub._queue}
    assert by_type[EventType.SPEECH_CHUNK].origin is Origin.CORE
    assert by_type[EventType.USER_SPOKE].origin is Origin.USER


async def test_the_bridge_ignores_an_origin_in_the_request(client: Any) -> None:
    """Sending one is not an error, because the field does not exist -- it is
    dropped before anything looks at it."""
    sub, _ = _bus_of(client).subscribe()

    resp = await client.post(
        "/v1/events/client",
        json={"type": "speech_chunk", "text": "Hi.", "origin": "observation"},
    )

    assert resp.status_code == 200
    assert next(iter(sub._queue)).origin is Origin.CORE


async def test_the_bridge_still_refuses_uncatalogued_types(client: Any) -> None:
    for forbidden in ("tool_started", "permission_required", "private_mode"):
        resp = await client.post("/v1/events/client", json={"type": forbidden})
        assert resp.status_code == 422, f"{forbidden} was accepted"


async def test_the_bridge_caps_chunk_text(client: Any) -> None:
    """Rejected at the schema, so an oversized body is never parsed into an
    event at all."""
    resp = await client.post(
        "/v1/events/client", json={"type": "speech_chunk", "text": "x" * 5_000}
    )
    assert resp.status_code == 422
