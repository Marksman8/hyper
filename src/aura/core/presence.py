"""The presence state machine.

**A state-resolution system, not "latest event wins".** The machine holds a set
of *conditions* -- facts that are currently true -- and resolves the visible
state by taking the highest-priority one. Almost every behaviour the design
asks for falls out of that single mechanism rather than needing its own special
case:

| Requirement | Falls out of |
|---|---|
| Priority | Resolution picks the highest-priority active condition |
| Terminal vs current | Occurrences set a condition **with a deadline**;
  state events set one that persists |
| Timeout / expiry | Expired conditions stop being active, so resolution moves on |
| Hysteresis | Re-asserting a held condition is a no-op, so no transition |
| Reconnect | Only persistent conditions are reported; transients are never replayed |

The two cases that genuinely need more than resolution:

**Interruption.** `USER_INTERRUPTED` does not merely add `LISTENING` at a
higher priority -- it *clears* the activity conditions, because an interruption
ends the work. Solving this by priority instead would mean LISTENING outranked
THINKING forever, which is wrong the rest of the time.

**Dwell.** A state must stay visible long enough to be perceived, so a
transition inside the dwell window is deferred. But dwell is **preemptable by
priority**: an ERROR must never wait behind a SUCCESS animation.

**This module knows nothing about rendering.** No Qt, no avatar, no artwork --
it produces `PresenceState` values and transitions. `tests/unit/test_presence.py`
asserts the absence of any UI import, so the presentation layer stays
replaceable.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any

from aura.core.events import Event, EventType, Origin
from aura.logging import get_logger

log = get_logger(__name__)


class PresenceState(StrEnum):
    """What AURA looks like she is doing."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    WORKING = "working"
    WAITING = "waiting"
    SUCCESS = "success"
    CONFUSED = "confused"
    WARNING = "warning"
    ANNOYED = "annoyed"
    HAPPY = "happy"
    SLEEP = "sleep"
    PRIVATE = "private"
    ERROR = "error"


class Priority(IntEnum):
    """Resolution order. Higher wins.

    Two decisions here are deliberate and worth arguing with rather than
    inheriting silently.

    **PRIVATE outranks everything, including ERROR.** Privacy is an explicit
    instruction from the user; an error is information. An overlay that
    reappears to report a problem after the user asked for privacy has broken
    the promise, and the error is not lost -- it is still logged, audited and
    on the bus. The overlay is a convenience; privacy is a commitment.

    **ERROR outranks every cosmetic and personality state.** SUCCESS, HAPPY and
    ANNOYED must never mask a failure, which is what the design means by safety
    taking precedence over personality.
    """

    IDLE = 0
    LISTENING = 10
    THINKING = 20
    WORKING = 30
    HAPPY = 40
    SUCCESS = 45
    ANNOYED = 50
    CONFUSED = 55
    WARNING = 60
    WAITING = 70
    ERROR = 80
    SLEEP = 90
    PRIVATE = 100


PRIORITY: dict[PresenceState, Priority] = {
    PresenceState.IDLE: Priority.IDLE,
    PresenceState.LISTENING: Priority.LISTENING,
    PresenceState.THINKING: Priority.THINKING,
    PresenceState.WORKING: Priority.WORKING,
    PresenceState.HAPPY: Priority.HAPPY,
    PresenceState.SUCCESS: Priority.SUCCESS,
    PresenceState.ANNOYED: Priority.ANNOYED,
    PresenceState.CONFUSED: Priority.CONFUSED,
    PresenceState.WARNING: Priority.WARNING,
    PresenceState.WAITING: Priority.WAITING,
    PresenceState.ERROR: Priority.ERROR,
    PresenceState.SLEEP: Priority.SLEEP,
    PresenceState.PRIVATE: Priority.PRIVATE,
}

# How long a transient condition stays true, in seconds.
#
# SUCCESS is the one with a real perceptual constraint: below roughly 1.5s a
# state change is registered as a flicker rather than as feedback, so 2.5s is
# chosen to be comfortably readable without lingering. The rest are set by how
# much attention the condition deserves -- ERROR longest because it is the one
# a person most needs to notice.
DURATION: dict[PresenceState, float] = {
    PresenceState.SUCCESS: 2.5,
    PresenceState.HAPPY: 3.0,
    PresenceState.CONFUSED: 3.0,
    PresenceState.WARNING: 4.0,
    PresenceState.ANNOYED: 4.0,
    PresenceState.ERROR: 8.0,
    PresenceState.LISTENING: 6.0,
    # A person who has not answered a permission prompt in two minutes is
    # not about to. Until 10C's approve-and-resume flow exists, nothing
    # else would ever clear this.
    PresenceState.WAITING: 120.0,
}

# Minimum time a state stays on screen once shown, unless something with a
# higher priority preempts it. Without this, a fast agent loop strobes between
# THINKING and WORKING several times a second.
MIN_DWELL_S = 0.4

# Silence after which AURA is considered asleep.
IDLE_TO_SLEEP_S = 600.0

# States that mean "work is in progress". An interruption clears these, and
# task completion clears these, because both END the work.
ACTIVITY = frozenset(
    {PresenceState.THINKING, PresenceState.WORKING, PresenceState.WAITING}
)


@dataclass(slots=True)
class Condition:
    """One thing that is currently true.

    `expires_at is None` means it persists until something clears it. That is
    the whole occurrence-vs-state distinction, expressed as data.
    """

    state: PresenceState
    since: float
    expires_at: float | None = None
    cause: str = ""

    def active(self, now: float) -> bool:
        return self.expires_at is None or now < self.expires_at


@dataclass(frozen=True, slots=True)
class Transition:
    """A change worth telling the presentation layer about."""

    previous: PresenceState
    current: PresenceState
    at: float
    cause: str

    def describe(self) -> str:
        return f"{self.previous} -> {self.current} ({self.cause})"


class PresenceStateMachine:
    """Resolves AURA's visible state from the event contract.

    Passive: it is fed events and ticked. It does not subscribe to anything and
    does not publish -- `PresenceDriver` does both, which keeps this class
    testable without a bus and without a clock.
    """

    def __init__(
        self,
        clock: Callable[[], float] | None = None,
        *,
        min_dwell_s: float = MIN_DWELL_S,
        idle_to_sleep_s: float = IDLE_TO_SLEEP_S,
        personality: bool = True,
    ) -> None:
        # Injected so dwell and expiry can be tested deterministically rather
        # than with sleeps.
        self._clock = clock or time.monotonic
        self._min_dwell = min_dwell_s
        self._idle_to_sleep = idle_to_sleep_s
        # When False, HAPPY and ANNOYED are never entered. The design calls
        # these the two states that make an assistant tiresome, so switching
        # them off must be possible without disabling presence entirely.
        self._personality = personality

        self._conditions: dict[PresenceState, Condition] = {}
        self._state = PresenceState.IDLE
        self._entered_at = self._clock()
        self._last_activity = self._clock()
        self._transitions = 0

    # -- introspection -------------------------------------------------------

    @property
    def state(self) -> PresenceState:
        return self._state

    @property
    def transition_count(self) -> int:
        return self._transitions

    def active_conditions(self) -> list[Condition]:
        now = self._clock()
        return sorted(
            (c for c in self._conditions.values() if c.active(now)),
            key=lambda c: PRIORITY[c.state],
            reverse=True,
        )

    # -- conditions ----------------------------------------------------------

    def _hold(self, state: PresenceState, cause: str) -> None:
        """Assert a condition that persists until cleared."""
        existing = self._conditions.get(state)
        if existing is not None and existing.expires_at is None:
            # Already held. Re-asserting must not restart anything, or a
            # repeated event would keep a state alive forever and reset dwell.
            return
        self._conditions[state] = Condition(
            state=state, since=self._clock(), cause=cause
        )

    def _flash(self, state: PresenceState, cause: str, seconds: float | None = None) -> None:
        """Assert a condition that expires on its own.

        This is how an occurrence becomes visible without becoming permanent --
        requirement 4. A TASK_COMPLETED does not make AURA successful forever.
        """
        if not self._personality and state in (PresenceState.HAPPY, PresenceState.ANNOYED):
            return
        now = self._clock()
        duration = seconds if seconds is not None else DURATION.get(state, 2.0)
        self._conditions[state] = Condition(
            state=state, since=now, expires_at=now + duration, cause=cause
        )

    def _clear(self, *states: PresenceState) -> None:
        for state in states:
            self._conditions.pop(state, None)

    def _clear_activity(self) -> None:
        """End the work. Used by both interruption and completion."""
        self._clear(*ACTIVITY)

    # -- the event contract --------------------------------------------------

    def on_event(self, event: Event) -> Transition | None:
        """Fold one event into the conditions, then resolve.

        Returns a transition if the visible state changed.

        **Observations are ignored entirely.** Screen content must not be able
        to drive what AURA looks like she is doing; an untrusted event that
        could set WORKING would be a small but real lie told by a web page.
        `APP_CHANGED` will drive *placement* in the presentation layer, which
        is a different thing from state.
        """
        if event.origin is Origin.OBSERVATION:
            return None
        # PRESENCE_STATE is this machine's own output. Consuming it would be a
        # feedback loop.
        if event.type is EventType.PRESENCE_STATE:
            return None

        self._apply(event)
        return self._resolve(cause=str(event.type))

    def _apply(self, event: Event) -> None:
        payload = event.payload
        kind = event.type

        if kind is not EventType.SNAPSHOT:
            self._last_activity = self._clock()
            self._clear(PresenceState.SLEEP)

        if kind is EventType.USER_SPOKE:
            # A new turn supersedes whatever the last one ended as.
            self._clear(PresenceState.SUCCESS, PresenceState.HAPPY, PresenceState.ERROR)
            self._flash(PresenceState.LISTENING, "user_spoke")

        elif kind is EventType.USER_INTERRUPTED:
            # Interruption ENDS the work -- see the module docstring for why
            # this is a clear rather than a priority.
            self._clear_activity()
            self._clear(PresenceState.SUCCESS, PresenceState.HAPPY)
            self._flash(PresenceState.LISTENING, f"interrupted:{payload.get('method', '')}")

        elif kind is EventType.TASK_STARTED:
            self._clear(PresenceState.SUCCESS, PresenceState.HAPPY, PresenceState.ANNOYED)
            self._clear(PresenceState.LISTENING)
            # A new turn supersedes a permission request nobody answered.
            # Found live: a refused write_file left WAITING held, and it
            # then outranked THINKING through an unrelated later turn.
            self._clear(PresenceState.WAITING)
            self._hold(PresenceState.THINKING, "task_started")
            if payload.get("with_tools"):
                self._hold(PresenceState.WORKING, "task_started:tools")

        elif kind is EventType.AGENT_THINKING:
            # Re-asserted constantly by a fast loop. `_hold` makes that a
            # no-op, which is the hysteresis requirement.
            self._hold(PresenceState.THINKING, "agent_thinking")

        elif kind is EventType.AGENT_WAITING:
            reason = str(payload.get("reason", ""))
            if reason == "confirmation":
                self._flash(PresenceState.WAITING, "awaiting_approval")
            else:
                self._hold(PresenceState.WORKING, f"tool:{payload.get('tool', '')}")

        elif kind is EventType.PERMISSION_REQUIRED:
            self._flash(PresenceState.WAITING, f"permission:{payload.get('tool', '')}")

        elif kind is EventType.TASK_COMPLETED:
            self._clear_activity()
            tool_calls = int(payload.get("tool_calls", 0) or 0)
            stop_reason = str(payload.get("stop_reason", "") or "")
            if stop_reason in ("step_budget", "time_budget"):
                # Gave up rather than finished. The design is explicit that
                # ANNOYED must be earned, not a reaction to being asked twice.
                self._flash(PresenceState.ANNOYED, f"gave_up:{stop_reason}")
            elif tool_calls >= 2:
                # Reserved for genuinely completed multi-step work, so it stays
                # meaningful rather than firing on every reply.
                self._flash(PresenceState.HAPPY, f"multi_step:{tool_calls}")
            else:
                self._flash(PresenceState.SUCCESS, "task_completed")

        elif kind is EventType.TASK_FAILED:
            self._clear_activity()
            self._flash(PresenceState.CONFUSED, f"failed:{payload.get('error_type', '')}")

        elif kind is EventType.ERROR:
            code = str(payload.get("code", ""))
            if code == "permission_denied":
                self._flash(PresenceState.WARNING, code)
            else:
                self._flash(PresenceState.ERROR, code or "error")

        elif kind is EventType.PRIVATE_MODE:
            mode = str(payload.get("mode", ""))
            if mode == "private":
                self._hold(PresenceState.PRIVATE, "private_mode")
                self._clear(PresenceState.SLEEP)
            elif mode == "sleep":
                self._hold(PresenceState.SLEEP, "sleep_mode")
                self._clear(PresenceState.PRIVATE)
            else:
                self._clear(PresenceState.PRIVATE, PresenceState.SLEEP)

    # -- resolution ----------------------------------------------------------

    def tick(self, now: float | None = None) -> Transition | None:
        """Advance time: expire conditions, fall asleep, re-resolve.

        Called on a timer by the driver, and directly by tests with an injected
        clock.
        """
        moment = now if now is not None else self._clock()
        if (
            moment - self._last_activity > self._idle_to_sleep
            and PresenceState.PRIVATE not in self._conditions
        ):
            self._hold(PresenceState.SLEEP, "idle_timeout")
        return self._resolve(cause="tick")

    def _resolve(self, cause: str) -> Transition | None:
        now = self._clock()

        expired = [s for s, c in self._conditions.items() if not c.active(now)]
        for state in expired:
            del self._conditions[state]

        candidate = PresenceState.IDLE
        candidate_cause = "default"
        best = -1
        for condition in self._conditions.values():
            rank = PRIORITY[condition.state]
            if rank > best:
                best, candidate, candidate_cause = rank, condition.state, condition.cause

        if candidate is self._state:
            return None

        # Dwell, with two escapes.
        #
        # It may be preempted by priority -- a SUCCESS animation may hold the
        # screen for a moment, but may never hold up an ERROR.
        #
        # And it does NOT apply once the current state's condition has gone.
        # Dwell exists to stop flicker between *competing* conditions; holding
        # a state that is no longer true is not damping, it is lying. Waking
        # from SLEEP found this: the sleep condition was cleared but dwell kept
        # showing it, so AURA stayed asleep through being spoken to.
        still_true = self._state in self._conditions
        held_for = now - self._entered_at
        if (
            still_true
            and held_for < self._min_dwell
            and PRIORITY[candidate] <= PRIORITY[self._state]
        ):
            return None

        previous = self._state
        self._state = candidate
        self._entered_at = now
        self._transitions += 1
        transition = Transition(
            previous=previous,
            current=candidate,
            at=now,
            cause=candidate_cause or cause,
        )
        log.info(
            "presence.transition",
            previous=str(previous),
            current=str(candidate),
            cause=transition.cause,
        )
        return transition

    # -- snapshot (requirement 10) -------------------------------------------

    def snapshot(self) -> dict[str, object]:
        """Current state, for a reconnecting client.

        Reports the **resolved state**, never the events that produced it. A
        client recovers by being told what is true now, not by replaying
        occurrences -- which is what stops a five-minute-old TASK_COMPLETED
        making AURA look freshly successful.

        Transient conditions that are still genuinely live are reported,
        because they *are* current; ones that have expired are already gone.
        """
        now = self._clock()
        # In PRIVATE, report the state and nothing else. Found live: the
        # condition list read ['success', 'private'], which tells a client what
        # AURA was doing underneath -- undermining the very suppression PRIVATE
        # exists to provide. Requirement 8 is about what can be INFERRED, not
        # only about what is drawn.
        if self._state in (PresenceState.PRIVATE, PresenceState.SLEEP):
            return {
                "presence_state": str(self._state),
                "since": round(now - self._entered_at, 3),
                "conditions": [str(self._state)],
            }
        return {
            "presence_state": str(self._state),
            "since": round(now - self._entered_at, 3),
            "conditions": [
                str(c.state) for c in self._conditions.values() if c.active(now)
            ],
        }

    def reset(self) -> None:
        """Back to IDLE with nothing held. For shutdown and for tests."""
        self._conditions.clear()
        self._state = PresenceState.IDLE
        self._entered_at = self._clock()


class PresenceDriver:
    """Feeds the machine from the bus and publishes what it decides.

    The seam the architecture rule asks for:

        Event Bus -> PresenceStateMachine -> PRESENCE_STATE -> presentation

    The presentation layer subscribes to `PRESENCE_STATE` like any other
    consumer. It never talks to the machine, and the machine never talks to it,
    so swapping sprites for Live2D for a terminal readout changes nothing here.

    Runs as one background task: consume events, tick for expiry, publish on
    change.
    """

    def __init__(
        self,
        bus: Any,
        machine: PresenceStateMachine | None = None,
        *,
        tick_s: float = 0.25,
    ) -> None:
        self._bus = bus
        self.machine = machine or PresenceStateMachine()
        # Fast enough that a 2.5s SUCCESS ends visibly on time, slow enough to
        # be invisible in a CPU graph. Expiry granularity, not animation rate --
        # animation is the presentation layer's business.
        self._tick_s = tick_s
        self._task: asyncio.Task[None] | None = None
        self._sub: Any = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        """Subscribe, then start the loop.

        Subscribing here rather than inside the task matters: `create_task`
        returns before the coroutine has run, so a subscription made inside it
        would miss anything published in that window. A presence layer that
        drops the first event of a session is exactly the kind of bug that
        looks like flakiness.
        """
        if self._task is not None:
            return
        self._stop.clear()
        self._sub, _ = self._bus.subscribe()
        self._task = asyncio.create_task(self._run(), name="presence-driver")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def _publish(self, transition: Transition | None) -> None:
        if transition is None:
            return
        from aura.core.emitter import EventEmitter

        EventEmitter(self._bus).presence_state(
            state=str(transition.current),
            previous=str(transition.previous),
            cause=transition.cause,
        )

    async def _run(self) -> None:
        sub = self._sub
        pump = asyncio.create_task(self._consume(sub))
        try:
            while not self._stop.is_set():
                await asyncio.sleep(self._tick_s)
                self._publish(self.machine.tick())
        except asyncio.CancelledError:
            raise
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump
            self._bus.unsubscribe(sub)
            self._sub = None

    async def _consume(self, sub: Any) -> None:
        try:
            async for event in sub:
                self._publish(self.machine.on_event(event))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A lagged subscription, most likely. Presence is cosmetic: losing
            # it must never take down the process that produces the events.
            log.warning("presence.driver_stopped", error=str(exc))
