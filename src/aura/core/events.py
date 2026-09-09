"""The AURA event bus.

**This is a security contract, not a pub/sub utility.** Phase 10's perception
and computer-use pillars both hang off it, and the thing that keeps them safe
is a property of this module: an event is *data*, and consuming one can never
grant a capability, escalate a permission, or cause a tool to run.

The three enforcement mechanisms, in order of how much they actually buy:

1. **Payloads are frozen JSON-safe values.** `_freeze` walks every payload at
   construction and rejects anything that is not a string, number, bool, None,
   mapping, or sequence of those. A callable, a bound method, an open socket,
   a `ToolCall` -- none of them can travel in an event. This is what makes
   "events cannot carry executable authority" a fact rather than a convention.

2. **The bus imports nothing that can execute.** No `aura.tools`, no
   `aura.core.agent`, no subprocess. There is no code path from `publish()` to
   anything with side effects outside this module. `test_event_bus.py` asserts
   the import set, so a future phase cannot quietly add one.

3. **The bus is one-way.** A `Subscription` yields events. It has no method
   that sends anything back. The authority return path -- approvals -- is a
   different module (`aura.security.approvals`) reached by a different
   transport (authenticated HTTP), and the two do not import each other.

What this module deliberately does *not* try to enforce: publisher honesty. A
publisher that labels screen content `Origin.CORE` has lied, and no amount of
validation here detects that. The mitigation is `publish_observation()`, which
hard-codes the origin so the perception layer cannot get it wrong by accident,
plus the rule that only `aura.vision` may call it.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from aura.logging import get_logger

log = get_logger(__name__)

# How many events are retained for replay to a reconnecting subscriber. A
# client that fell further behind than this gets a SNAPSHOT instead, which is
# correct but more expensive -- so this is sized to cover a brief network or
# repaint stall, not a long absence.
REPLAY_BUFFER = 256

# Per-subscriber queue depth. Droppable events are evicted at this bound; a
# subscriber that hits it with only reliable events pending is disconnected
# rather than silently truncated (see Delivery).
SUBSCRIBER_QUEUE = 128

# Payload limits. A deeply nested or enormous payload is either a bug or an
# attempt to stall every subscriber at once.
MAX_PAYLOAD_DEPTH = 6
MAX_PAYLOAD_KEYS = 64
MAX_STRING_LEN = 8192


class Origin(StrEnum):
    """Where an event came from, and therefore how much to trust it."""

    # AURA's own state. Trusted.
    CORE = "core"
    # A human did something. Trusted.
    USER = "user"
    # Read from a screen, an application, a document, or a model's reading of
    # one. **UNTRUSTED.** A consumer must treat the payload as hostile text:
    # never as an instruction, never as a reason to act, never as evidence for
    # a learned preference.
    OBSERVATION = "observation"


class Delivery(StrEnum):
    """What the bus promises about getting this event to a subscriber."""

    # State hints and animation cues. May be dropped under pressure; a missed
    # repaint is not worth back-pressuring the orchestrator.
    DROPPABLE = "droppable"
    # Decisions. **Never silently dropped.** If a subscriber cannot be given a
    # reliable event, its subscription is closed and it must resync -- because
    # a lost PERMISSION_REQUIRED is a hung task, or a user who believes they
    # were never asked.
    RELIABLE = "reliable"


class EventType(StrEnum):
    """The taxonomy. Closed on purpose: a typo must not invent an event that
    nothing subscribes to and nobody notices is missing."""

    # -- lifecycle / recovery
    SNAPSHOT = "snapshot"

    # -- user
    USER_SPOKE = "user_spoke"
    USER_INTERRUPTED = "user_interrupted"

    # -- task
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"

    # -- agent
    AGENT_THINKING = "agent_thinking"
    AGENT_WAITING = "agent_waiting"

    # -- environment
    APP_CHANGED = "app_changed"
    VLM_OBSERVATION = "vlm_observation"

    # -- authority
    PERMISSION_REQUIRED = "permission_required"

    # -- modes
    PRIVATE_MODE = "private_mode"
    FOCUS_MODE = "focus_mode"

    # -- presentation
    PRESENCE_STATE = "presence_state"
    SPEECH_STARTED = "speech_started"
    SPEECH_CHUNK = "speech_chunk"
    SPEECH_ENDED = "speech_ended"

    # -- system
    RESOURCE_PRESSURE = "resource_pressure"
    ERROR = "error"


class InvalidEventError(ValueError):
    """An event was constructed with something it may not carry."""


class SubscriberLaggedError(RuntimeError):
    """A subscriber fell too far behind to receive a reliable event.

    Raised from the subscription's iterator. The correct response is to
    resubscribe with a snapshot; the subscriber has NOT received everything and
    must not pretend it has.
    """


_SCALARS = (str, int, float, bool, type(None))


def _freeze(value: Any, depth: int = 0) -> Any:
    """Deep-freeze a payload value, rejecting anything not JSON-safe.

    This is the mechanism behind "an event cannot carry executable authority".
    Functions, class instances, file handles and tool objects all fail here, at
    construction, before anything can subscribe to them.
    """
    if depth > MAX_PAYLOAD_DEPTH:
        raise InvalidEventError(f"payload nested deeper than {MAX_PAYLOAD_DEPTH}")

    # bool before int: bool IS an int subclass, and the order matters for the
    # isinstance checks below being readable.
    if isinstance(value, _SCALARS):
        if isinstance(value, str) and len(value) > MAX_STRING_LEN:
            raise InvalidEventError(f"payload string longer than {MAX_STRING_LEN}")
        return value

    if isinstance(value, Mapping):
        if len(value) > MAX_PAYLOAD_KEYS:
            raise InvalidEventError(f"payload mapping wider than {MAX_PAYLOAD_KEYS} keys")
        return MappingProxyType(
            {str(k): _freeze(v, depth + 1) for k, v in value.items()}
        )

    # str is a Sequence, and is handled above.
    if isinstance(value, Sequence):
        if len(value) > MAX_PAYLOAD_KEYS:
            raise InvalidEventError(f"payload sequence longer than {MAX_PAYLOAD_KEYS}")
        return tuple(_freeze(v, depth + 1) for v in value)

    raise InvalidEventError(
        f"payload may not contain {type(value).__name__}. Events carry data, "
        "not objects: anything with behaviour could hand a subscriber authority "
        "it was never granted."
    )


def _coerce(enum: type[StrEnum], value: Any, field_name: str) -> Any:
    """Accept the enum or one of its exact values; reject everything else.

    Strict on purpose. Silently coercing an unrecognised origin to CORE would
    turn a bug into a privilege escalation.
    """
    if isinstance(value, enum):
        return value
    try:
        return enum(value)
    except (ValueError, KeyError) as exc:
        allowed = ", ".join(m.value for m in enum)
        raise InvalidEventError(
            f"{field_name}={value!r} is not valid. Allowed: {allowed}"
        ) from exc


@dataclass(frozen=True, slots=True)
class Event:
    """One thing that happened.

    Immutable, sequenced, origin-labelled. `seq` is 0 until the bus assigns it
    at publish; an event with seq 0 has not been published.
    """

    type: EventType
    origin: Origin
    delivery: Delivery = Delivery.DROPPABLE
    payload: Mapping[str, Any] = field(default_factory=dict)
    seq: int = 0
    ts: float = field(default_factory=time.time)
    session_id: str | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        # Enforced at construction, not at publish: an invalid event must not
        # exist even briefly, or it can be passed around before anyone checks.
        object.__setattr__(self, "type", _coerce(EventType, self.type, "type"))
        object.__setattr__(self, "origin", _coerce(Origin, self.origin, "origin"))
        object.__setattr__(self, "delivery", _coerce(Delivery, self.delivery, "delivery"))
        object.__setattr__(self, "payload", _freeze(dict(self.payload)))

    @property
    def trusted(self) -> bool:
        """False for anything read off a screen or out of a document.

        Consumers that act on content -- the cognitive model, any future
        planner -- must check this. Consumers that only render need not, but
        must escape the text.
        """
        return self.origin is not Origin.OBSERVATION

    def to_wire(self) -> dict[str, Any]:
        """JSON-safe form for the SSE transport (10A.2)."""
        return {
            "type": str(self.type),
            "origin": str(self.origin),
            "delivery": str(self.delivery),
            "seq": self.seq,
            "ts": self.ts,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "payload": _unfreeze(self.payload),
        }


def _unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_unfreeze(v) for v in value]
    return value


# A callable that returns the current state, for SNAPSHOT recovery. Returns a
# plain mapping; it is frozen like any other payload.
SnapshotProvider = Callable[[], Mapping[str, Any]]


@dataclass(slots=True)
class Subscription:
    """One subscriber's view of the stream.

    Yields events in publish order. Raises `SubscriberLaggedError` if it fell
    behind far enough to miss a reliable event -- which is the bus keeping its
    promise, not breaking it.
    """

    id: int
    _queue: deque[Event] = field(default_factory=deque)
    _wake: asyncio.Event = field(default_factory=asyncio.Event)
    _closed: bool = False
    _lagged: bool = False
    last_seq: int = 0
    dropped: int = 0

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def lagged(self) -> bool:
        return self._lagged

    @property
    def pending(self) -> int:
        return len(self._queue)

    @property
    def pending_reliable(self) -> int:
        return sum(1 for e in self._queue if e.delivery is Delivery.RELIABLE)

    def close(self) -> None:
        self._closed = True
        self._wake.set()

    async def __aiter__(self) -> AsyncIterator[Event]:
        while True:
            while self._queue:
                event = self._queue.popleft()
                self.last_seq = event.seq
                yield event
            if self._lagged:
                raise SubscriberLaggedError(
                    f"subscription {self.id} missed a reliable event after seq "
                    f"{self.last_seq}; resubscribe with a snapshot"
                )
            if self._closed:
                return
            self._wake.clear()
            # Re-check: an event may have arrived between the drain and the
            # clear, and waiting then would hang until the next publish.
            if self._queue or self._closed or self._lagged:
                continue
            await self._wake.wait()


class EventBus:
    """One-way, in-process, ordered event distribution.

    **Ordering.** `publish` is serialised by a lock, so sequence numbers are
    strictly monotonic with no gaps and no duplicates, and every subscriber is
    fed in the same order. Two concurrent producers cannot interleave a single
    event's assignment and enqueue.

    **Thread safety.** `publish` may be called from any thread. Waking async
    subscribers is marshalled to their loop with `call_soon_threadsafe`, which
    is why the loop is captured when a subscription is created.
    """

    def __init__(
        self,
        replay_buffer: int = REPLAY_BUFFER,
        queue_size: int = SUBSCRIBER_QUEUE,
    ) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._history: deque[Event] = deque(maxlen=replay_buffer)
        self._subs: dict[int, Subscription] = {}
        self._loops: dict[int, asyncio.AbstractEventLoop | None] = {}
        self._next_id = 0
        self._queue_size = queue_size
        self._closed = False

    # -- publishing ----------------------------------------------------------

    def publish(self, event: Event) -> Event:
        """Assign a sequence number and fan out. Never blocks, never raises for
        a slow subscriber.

        Returns the sealed event, with `seq` set.
        """
        if self._closed:
            raise RuntimeError("event bus is closed")

        with self._lock:
            self._seq += 1
            sealed = replace(event, seq=self._seq)
            self._history.append(sealed)
            targets = list(self._subs.values())
            for sub in targets:
                if self._accept(sub):
                    sub._queue.append(sealed)
            wakes = [(sub, self._loops.get(sub.id)) for sub in targets]

        # Waking happens outside the lock: a subscriber's loop may be busy, and
        # holding the publish lock across that would serialise the whole bus on
        # the slowest consumer.
        for sub, loop in wakes:
            self._wake(sub, loop)
        return sealed

    def _accept(self, sub: Subscription) -> bool:
        """Make room in one subscriber's queue. Called under the lock.

        Returns False when the event must not be queued for this subscriber.

        Droppable events are evicted oldest-first. If nothing droppable
        remains, the subscriber is marked lagged and closed -- it is never
        given a truncated stream while believing it is complete.

        **A lagged subscriber receives nothing further.** An earlier version
        marked it lagged and then appended anyway, so its queue grew without
        bound while it was already doomed -- a slow consumer became a memory
        leak. It has been told it must resync; there is nothing to gain by
        continuing to fill a queue it will never finish draining.
        """
        if sub._lagged or sub._closed:
            return False

        if len(sub._queue) < self._queue_size:
            return True

        for index, queued in enumerate(sub._queue):
            if queued.delivery is Delivery.DROPPABLE:
                del sub._queue[index]
                sub.dropped += 1
                return True

        # Every pending event is reliable. Dropping one would break the
        # guarantee, and blocking would back-pressure the orchestrator, so the
        # subscription is failed instead.
        sub._lagged = True
        sub._closed = True
        log.warning(
            "events.subscriber_lagged",
            subscription=sub.id,
            pending_reliable=len(sub._queue),
        )
        return False

    @staticmethod
    def _wake(sub: Subscription, loop: asyncio.AbstractEventLoop | None) -> None:
        if loop is None or loop.is_closed():
            sub._wake.set()
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            sub._wake.set()
        else:
            loop.call_soon_threadsafe(sub._wake.set)

    def publish_observation(
        self,
        event_type: EventType,
        payload: Mapping[str, Any],
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> Event:
        """Publish untrusted content, with the origin hard-coded.

        The perception layer uses this so that labelling screen content as
        trusted is not something a caller can do by accident. It is the only
        mitigation available for publisher honesty, which the bus cannot verify.
        """
        return self.publish(
            Event(
                type=event_type,
                origin=Origin.OBSERVATION,
                delivery=Delivery.DROPPABLE,
                payload=payload,
                session_id=session_id,
                request_id=request_id,
            )
        )

    # -- subscribing ---------------------------------------------------------

    def subscribe(self, after_seq: int | None = None) -> tuple[Subscription, bool]:
        """Open a subscription.

        `after_seq` is the last sequence number a reconnecting client saw.
        Returns `(subscription, needs_snapshot)`:

        * `after_seq is None`  -- a fresh subscriber; caller should snapshot.
        * `after_seq` still in the replay buffer -- missed events are replayed,
          no snapshot needed.
        * `after_seq` too old -- the gap cannot be filled; caller must snapshot.
        """
        with self._lock:
            sub = Subscription(id=self._next_id)
            self._next_id += 1
            self._subs[sub.id] = sub
            try:
                self._loops[sub.id] = asyncio.get_running_loop()
            except RuntimeError:
                self._loops[sub.id] = None

            if after_seq is None:
                sub.last_seq = self._seq
                return sub, True

            replayable = [e for e in self._history if e.seq > after_seq]
            oldest = self._history[0].seq if self._history else self._seq + 1
            # A gap exists if the first event we still hold is newer than the
            # one immediately after the client's position.
            gap = bool(self._history) and oldest > after_seq + 1
            if gap or after_seq > self._seq:
                sub.last_seq = self._seq
                log.info(
                    "events.replay_gap",
                    subscription=sub.id,
                    client_seq=after_seq,
                    oldest_held=oldest,
                )
                return sub, True

            for event in replayable[-self._queue_size :]:
                sub._queue.append(event)
            sub.last_seq = after_seq
            if sub._queue:
                sub._wake.set()
            return sub, False

    def snapshot_event(
        self, provider: SnapshotProvider
    ) -> tuple[Event, Subscription]:
        """Take a consistent snapshot and open a subscription at that instant.

        **The consistency semantics, stated precisely**, because getting this
        wrong is the classic way a resync silently loses an event:

        The snapshot is taken *while holding the publish lock*, and the
        subscription is created in the same critical section. So the snapshot
        reflects exactly the events with `seq <= N`, and the subscription
        receives exactly the events with `seq > N`. No event falls in the gap,
        and none is delivered twice. This is a consistent cut, not a
        best-effort one.

        The cost is that the provider runs under the lock and must therefore be
        fast and must not publish. That constraint is documented rather than
        enforced, because a provider that deadlocks the bus is a bug that shows
        up immediately and loudly.
        """
        with self._lock:
            state = dict(provider())
            self._seq += 1
            event = Event(
                type=EventType.SNAPSHOT,
                origin=Origin.CORE,
                delivery=Delivery.RELIABLE,
                payload=state,
                seq=self._seq,
            )
            self._history.append(event)

            sub = Subscription(id=self._next_id)
            self._next_id += 1
            self._subs[sub.id] = sub
            try:
                self._loops[sub.id] = asyncio.get_running_loop()
            except RuntimeError:
                self._loops[sub.id] = None
            sub.last_seq = self._seq
            return event, sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            self._subs.pop(sub.id, None)
            loop = self._loops.pop(sub.id, None)
        sub.close()
        self._wake(sub, loop)

    # -- introspection -------------------------------------------------------

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)

    def pending_reliable(self) -> int:
        """Reliable events queued across all subscribers. Used by shutdown."""
        with self._lock:
            return sum(s.pending_reliable for s in self._subs.values())

    # -- shutdown ------------------------------------------------------------

    async def aclose(self, drain_timeout: float = 2.0) -> bool:
        """Stop accepting publishes, then let subscribers drain.

        Returns True if every reliable event was consumed before the timeout.

        Shutdown preserves the reliable guarantee in the only way it can: it
        waits. It does not discard pending reliable events quietly, and if the
        timeout expires it says so in the return value and the log rather than
        closing as though everything was delivered.
        """
        self._closed = True
        deadline = time.monotonic() + drain_timeout

        while time.monotonic() < deadline:
            if self.pending_reliable() == 0:
                break
            await asyncio.sleep(0.01)

        undelivered = self.pending_reliable()
        with self._lock:
            subs = list(self._subs.values())
            loops = dict(self._loops)
            self._subs.clear()
            self._loops.clear()

        for sub in subs:
            sub.close()
            self._wake(sub, loops.get(sub.id))

        if undelivered:
            log.warning("events.shutdown_undelivered", reliable_pending=undelivered)
        else:
            log.info("events.shutdown_clean")
        return undelivered == 0
