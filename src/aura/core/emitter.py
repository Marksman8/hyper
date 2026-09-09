"""Typed publishers for the catalogued events.

Every subsystem that announces something goes through here rather than
constructing `Event` objects itself. Three reasons, in order of importance:

1. **Origin and delivery come from the catalog, not the call site.** A call
   site cannot accidentally publish screen content as `Origin.CORE`, or make a
   permission request droppable.
2. **Payload schemas live in one file.** The `required` keys in the catalog are
   checked here, so a consumer can rely on them existing.
3. **Publishing never breaks the thing it describes.** Every method swallows
   its own failures. A bus problem must not fail a conversational turn, and an
   emitter that could raise would eventually be wrapped in a try/except at
   every call site anyway -- better to do it once, here.

**A no-op emitter is a valid emitter.** `EventEmitter(None)` publishes nothing,
which is how every existing code path keeps working unchanged when no bus is
wired: the orchestrator does not care whether anyone is listening.

Note what this module does *not* do: it does not read, wrap, or derive from the
logging system. Logs and events are published side by side at the same call
site and neither is generated from the other -- see the header of
`event_catalog.py` for why that separation matters.
"""

from __future__ import annotations

from typing import Any

from aura.core.event_catalog import spec
from aura.core.events import Event, EventBus, EventType
from aura.logging import get_logger

log = get_logger(__name__)

# Approval summaries are shown to a human, so they need enough room to describe
# an action -- but not enough to smuggle a document into the bus.
MAX_SUMMARY = 400


class EventEmitter:
    """Publishes catalogued events. Safe to call with no bus."""

    def __init__(self, bus: EventBus | None = None) -> None:
        self._bus = bus

    @property
    def enabled(self) -> bool:
        return self._bus is not None

    # -- the one place an Event is constructed outside tests -----------------

    def _emit(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> Event | None:
        if self._bus is None:
            return None

        contract = spec(event_type)
        missing = [k for k in contract.required if k not in payload]
        if missing:
            # A programming error, not a runtime condition. Logged loudly and
            # dropped, because publishing a malformed event would break every
            # consumer that trusted the catalog.
            log.error(
                "events.payload_incomplete",
                event_name=str(event_type),
                missing=missing,
            )
            return None

        try:
            return self._bus.publish(
                Event(
                    type=event_type,
                    origin=contract.origin,
                    delivery=contract.delivery,
                    payload=payload,
                    session_id=session_id,
                    request_id=request_id,
                )
            )
        except Exception as exc:
            # Never let announcing something break the thing being announced.
            log.warning(
                "events.publish_failed", event_name=str(event_type), error=str(exc)
            )
            return None

    # -- user ----------------------------------------------------------------

    def user_spoke(self, text: str, *, spoken: bool, session_id: str | None = None) -> None:
        """A turn began from the user.

        Deliberately carries the transcript LENGTH, not the transcript. The
        overlay needs to know a turn started, not what was said.
        """
        self._emit(
            EventType.USER_SPOKE,
            {"length": len(text), "spoken": spoken},
            session_id=session_id,
        )

    def user_interrupted(self, method: str, *, session_id: str | None = None) -> None:
        """Barge-in or a spoken stop command."""
        self._emit(
            EventType.USER_INTERRUPTED, {"method": method}, session_id=session_id
        )

    # -- task ----------------------------------------------------------------

    def task_started(
        self, kind: str, *, with_tools: bool = False, session_id: str | None = None
    ) -> None:
        self._emit(
            EventType.TASK_STARTED,
            {"kind": kind, "with_tools": with_tools},
            session_id=session_id,
        )

    def task_completed(
        self,
        kind: str,
        *,
        latency_ms: int,
        tool_calls: int = 0,
        reply_length: int = 0,
        session_id: str | None = None,
    ) -> None:
        self._emit(
            EventType.TASK_COMPLETED,
            {
                "kind": kind,
                "latency_ms": latency_ms,
                "tool_calls": tool_calls,
                "reply_length": reply_length,
            },
            session_id=session_id,
        )

    def task_failed(
        self, kind: str, error: BaseException, *, session_id: str | None = None
    ) -> None:
        """Carries the exception TYPE, never its message.

        A message routinely quotes a path, a prompt fragment, or a row of data.
        The type is enough for the overlay to show an error state, and for a
        person to go and read the log where the detail actually belongs.
        """
        self._emit(
            EventType.TASK_FAILED,
            {"kind": kind, "error_type": type(error).__name__},
            session_id=session_id,
        )

    # -- agent ---------------------------------------------------------------

    def agent_thinking(self, step: int = 0, *, session_id: str | None = None) -> None:
        self._emit(EventType.AGENT_THINKING, {"step": step}, session_id=session_id)

    def agent_waiting(
        self, reason: str, tool: str = "", *, session_id: str | None = None
    ) -> None:
        """Blocked on a tool or on a human. Tool NAME only, never arguments."""
        self._emit(
            EventType.AGENT_WAITING,
            {"reason": reason, "tool": tool},
            session_id=session_id,
        )

    # -- authority -----------------------------------------------------------

    def permission_required(
        self,
        *,
        tool: str,
        level: int,
        summary: str,
        approval_id: str = "",
        nonce: str = "",
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> Event | None:
        """A tool needs a human.

        The single event that carries content by design: a confirmation the
        user cannot understand is one they will click through. Carrying the
        nonce is safe -- the stream is authenticated, and per invariant I4 the
        event still grants nothing on its own. Redemption goes to the
        authenticated approval endpoint and nowhere else.
        """
        return self._emit(
            EventType.PERMISSION_REQUIRED,
            {
                "tool": tool,
                "level": level,
                "summary": summary[:MAX_SUMMARY],
                "approval_id": approval_id,
                "nonce": nonce,
            },
            session_id=session_id,
            request_id=request_id,
        )

    # -- modes ---------------------------------------------------------------

    def privacy_mode(self, mode: str, *, capturing: bool) -> None:
        """Vision mode changed. Reliable: a dropped 'now private' is the one
        drop that would actually matter."""
        self._emit(EventType.PRIVATE_MODE, {"mode": mode, "capturing": capturing})

    # -- presentation ---------------------------------------------------------

    def presence_state(self, state: str, previous: str, cause: str) -> None:
        """The derived avatar state changed (10A.4).

        Published by `PresenceDriver`, consumed by the presentation layer. The
        state machine itself never touches the bus -- that separation is what
        keeps the renderer replaceable.
        """
        self._emit(
            EventType.PRESENCE_STATE,
            {"state": state, "previous": previous, "cause": cause},
        )

    def focus_mode(self, *, enabled: bool) -> None:
        """Focus mode changed (10A.4 slot, wired in 10A.12).

        STATE, so it reaches a reconnecting client through the snapshot rather
        than by replaying the moment it was toggled.
        """
        self._emit(EventType.FOCUS_MODE, {"enabled": enabled})

    def app_changed(self, application: str) -> None:
        """The foreground application changed.

        `Origin.OBSERVATION`, set by the catalog: this is read from the
        environment, so a consumer must treat it as data. It is used to decide
        where the avatar sits and for nothing else.

        Carries the PROCESS NAME only. A window title quotes document names and
        message subjects, and belongs nowhere near a bus.
        """
        self._emit(EventType.APP_CHANGED, {"application": application})

    # -- system --------------------------------------------------------------

    def error(self, code: str, *, detail: str = "", session_id: str | None = None) -> None:
        self._emit(
            EventType.ERROR,
            {"code": code, "detail": detail[:200]},
            session_id=session_id,
        )


# A shared no-op, so a subsystem with no bus needs no None-checks at all.
NULL_EMITTER = EventEmitter(None)


def emitter_for(bus: EventBus | None) -> EventEmitter:
    return EventEmitter(bus) if bus is not None else NULL_EMITTER
