"""SSE transport for the event bus, and the approval return path.

Two endpoints that together are the whole 10A.2 contract:

    GET  /v1/events            one-way, authenticated, resumable
    POST /v1/approvals/{id}    the ONLY way authority flows back

Kept in their own module rather than added to `api/app.py` because the
separation is the point: the stream has no way to grant anything, and the
approval endpoint has no way to publish. Someone reading either file should not
find the other's capability nearby.

**Why the stream pumps through a local queue** rather than iterating the
subscription directly: heartbeats need a timeout, and wrapping `anext()` in
`wait_for` cancels the subscription's async generator mid-await, tearing it
down and losing whatever was queued. Cancelling `Queue.get()` is safe, so a
background pump feeds a local queue and the generator races that.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Request

from aura.core.events import (
    Event,
    EventBus,
    EventType,
    Origin,
    SnapshotProvider,
    SubscriberLaggedError,
    Subscription,
)
from aura.logging import get_logger

log = get_logger(__name__)

# Comment frames keep proxies and idle connections alive. SSE comments start
# with ':' and are ignored by every conforming client.
HEARTBEAT_S = 15.0


def frame(event: Event) -> str:
    """One SSE frame.

    The `id:` field carries the sequence number, which is what makes
    `Last-Event-ID` resume work: the client sends back the last id it saw and
    the bus decides between replay and snapshot.
    """
    return (
        f"id: {event.seq}\n"
        f"event: {event.type}\n"
        f"data: {json.dumps(event.to_wire(), separators=(',', ':'))}\n\n"
    )


def parse_last_event_id(raw: str | None) -> int | None:
    """Read the resume position a client claims.

    Junk is treated as "no position", not as an error: a malformed header from
    a confused client should produce a snapshot and a working stream, not a
    400 that leaves the overlay blank.
    """
    if not raw:
        return None
    try:
        value = int(raw.strip())
    except (ValueError, AttributeError):
        log.info("events.bad_last_event_id", raw=raw[:64])
        return None
    return value if value >= 0 else None


def open_stream(
    bus: EventBus, last_event_id: int | None, provider: SnapshotProvider
) -> tuple[Subscription, Event | None]:
    """Subscribe, deciding between replay and snapshot.

    Returns the subscription and, when a resync was needed, the SNAPSHOT event
    to send first.

    The two-step (subscribe, then discard and snapshot) is deliberate: a
    snapshot must be a consistent cut, and `snapshot_event` is the only thing
    that takes one. Discarding the first subscription costs nothing and loses
    nothing, because the snapshot reflects everything published up to its own
    sequence number.
    """
    if last_event_id is None:
        return _snapshot(bus, provider)

    sub, needs_snapshot = bus.subscribe(after_seq=last_event_id)
    if not needs_snapshot:
        return sub, None

    bus.unsubscribe(sub)
    return _snapshot(bus, provider)


def _snapshot(bus: EventBus, provider: SnapshotProvider) -> tuple[Subscription, Event]:
    event, sub = bus.snapshot_event(provider)
    return sub, event


async def _pump(sub: Subscription, queue: asyncio.Queue[tuple[str, Any]]) -> None:
    """Move events off the subscription and onto a cancellable queue."""
    try:
        async for event in sub:
            await queue.put(("event", event))
    except SubscriberLaggedError as exc:
        # Not an error to swallow. The client is missing reliable events and
        # must be told, so it can resubscribe rather than carry on with a hole.
        await queue.put(("lagged", str(exc)))
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("events.pump_failed", error=str(exc))
        await queue.put(("error", str(exc)))
    finally:
        await queue.put(("closed", None))


async def event_stream(
    bus: EventBus,
    sub: Subscription,
    first: Event | None,
    request: Request | None = None,
) -> AsyncIterator[str]:
    """The SSE body.

    Always unsubscribes, on every exit path. A leaked subscription would keep
    receiving events forever and eventually lag, which looks like a bug in the
    bus rather than in the client that walked away.
    """
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    pump = asyncio.create_task(_pump(sub, queue))

    try:
        if first is not None:
            yield frame(first)

        while True:
            try:
                kind, value = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_S)
            except TimeoutError:
                if request is not None and await request.is_disconnected():
                    return
                yield ": keepalive\n\n"
                continue

            if kind == "event":
                yield frame(value)
            elif kind == "lagged":
                # A terminal frame the client can act on. Sent as a real event
                # so a subscriber that only handles events still sees it.
                yield frame(
                    Event(
                        type=EventType.ERROR,
                        origin=Origin.CORE,
                        payload={
                            "code": "subscriber_lagged",
                            "message": str(value),
                            "action": "resubscribe_without_last_event_id",
                        },
                        seq=sub.last_seq,
                    )
                )
                return
            elif kind == "error":
                return
            else:  # closed
                return
    finally:
        pump.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump
        bus.unsubscribe(sub)
        log.info("events.stream_closed", subscription=sub.id, last_seq=sub.last_seq)
