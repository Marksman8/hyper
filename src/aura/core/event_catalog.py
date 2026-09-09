"""What each event *means*.

10A.1 built the mechanism; this is the contract. `events.py` knows how to
deliver an event and nothing about what one signifies, which is deliberate --
the delivery machinery should not need changing when the vocabulary grows.

**This is not a log-to-event mapping.** AURA's structured logs are
observability: they exist to explain a failure at 2am, they change freely, and
nothing depends on their shape. The bus is an application contract that the
presence overlay, and later the cognitive model, are built against. Deriving
one from the other would mean that editing a log line silently changes the
contract, which is exactly backwards. So an event and a log line are published
*separately*, at the same call site, and neither is generated from the other.

Two distinctions do most of the work here.

**Occurrence vs state.** `wake.detected` happened at an instant and is over.
`vision.mode_changed` announces a condition that persists. The difference
matters at reconnect: replaying an occurrence must never make a client believe
it just happened, and only state may appear in a SNAPSHOT. Get this wrong and
an overlay shows "listening" because it resumed a stream containing a
five-minute-old wake word.

**Content vs metadata.** In 10A.3 the bus carries *no user content*: no
transcripts, no prompts, no replies, no screen text. Only lengths, ids, tool
names and modes. The presence layer does not need content to animate, and a bus
that accumulates transcripts becomes the most sensitive thing in the process
for no benefit. `PERMISSION_REQUIRED.summary` is the single deliberate
exception -- informed consent is impossible without saying what is being
approved.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from aura.core.events import Delivery, EventType, Origin


class Semantics(StrEnum):
    """Does this event describe a moment, or a condition?"""

    # Something happened, at a point in time, and is over. Must NOT be
    # reconstructed as current state after a reconnect.
    OCCURRENCE = "occurrence"
    # A condition changed and now persists. Belongs in SNAPSHOT.
    STATE = "state"


class Sensitivity(StrEnum):
    """How much the payload reveals."""

    # Metadata only: counts, ids, durations, tool names, modes.
    METADATA = "metadata"
    # Describes something the user did or is doing, without quoting them.
    ACTIVITY = "activity"
    # Contains user- or screen-derived content. Requires a reason.
    CONTENT = "content"


@dataclass(frozen=True, slots=True)
class EventSpec:
    """The full contract for one event type."""

    type: EventType
    meaning: str
    semantics: Semantics
    origin: Origin
    delivery: Delivery
    # Keys the payload must carry. Enforced by the emitter and by tests, so a
    # consumer can rely on them existing.
    required: tuple[str, ...]
    sensitivity: Sensitivity
    # Should the presence layer react at all? 10A.4 decides *how*; this only
    # says whether it is presentation-relevant, so 10A.4 has a starting point
    # rather than a guess.
    presence_relevant: bool
    # STATE events contribute a field to SNAPSHOT. Occurrences never do.
    snapshot_key: str | None = None
    # Where it is published from. "core" events are emitted in-process;
    # "client" events arrive over the restricted inbound endpoint because they
    # happen in the voice process (see §10A.3 notes in the contract doc).
    source: str = "core"
    privacy_note: str = ""

    def __post_init__(self) -> None:
        if self.semantics is Semantics.OCCURRENCE and self.snapshot_key:
            raise ValueError(
                f"{self.type}: an occurrence cannot appear in a snapshot. "
                "Replaying it as current state is precisely the bug this "
                "classification exists to prevent."
            )
        if self.semantics is Semantics.STATE and not self.snapshot_key:
            raise ValueError(
                f"{self.type}: a state event must name the snapshot field it "
                "sets, or a reconnecting client cannot recover it."
            )


CATALOG: dict[EventType, EventSpec] = {
    # -- lifecycle ----------------------------------------------------------
    EventType.SNAPSHOT: EventSpec(
        type=EventType.SNAPSHOT,
        meaning="Full current state, sent once when a client connects or resyncs.",
        semantics=Semantics.STATE,
        origin=Origin.CORE,
        delivery=Delivery.RELIABLE,
        required=("version", "seq_at_snapshot"),
        sensitivity=Sensitivity.METADATA,
        presence_relevant=True,
        snapshot_key="_self",
        privacy_note="Aggregates other state events; carries no content itself.",
    ),
    # -- user ---------------------------------------------------------------
    EventType.USER_SPOKE: EventSpec(
        type=EventType.USER_SPOKE,
        meaning="The user finished saying something and a turn is starting.",
        semantics=Semantics.OCCURRENCE,
        origin=Origin.USER,
        delivery=Delivery.RELIABLE,
        required=("length", "spoken"),
        sensitivity=Sensitivity.ACTIVITY,
        presence_relevant=True,
        privacy_note=(
            "Carries the transcript LENGTH, never the transcript. The overlay "
            "needs to know a turn began, not what was said."
        ),
    ),
    EventType.USER_INTERRUPTED: EventSpec(
        type=EventType.USER_INTERRUPTED,
        meaning="The user cut AURA off mid-answer, by barge-in or a stop command.",
        semantics=Semantics.OCCURRENCE,
        origin=Origin.USER,
        delivery=Delivery.RELIABLE,
        required=("method",),
        sensitivity=Sensitivity.ACTIVITY,
        presence_relevant=True,
        source="client",
        privacy_note="`method` is 'barge_in' or 'stop_command'. No audio, no text.",
    ),
    # -- task ---------------------------------------------------------------
    EventType.TASK_STARTED: EventSpec(
        type=EventType.TASK_STARTED,
        meaning="A conversational or agent turn began.",
        semantics=Semantics.OCCURRENCE,
        origin=Origin.CORE,
        delivery=Delivery.RELIABLE,
        required=("kind", "with_tools"),
        sensitivity=Sensitivity.METADATA,
        presence_relevant=True,
        privacy_note="`kind` is chat|stream|agent. The prompt is not carried.",
    ),
    EventType.TASK_COMPLETED: EventSpec(
        type=EventType.TASK_COMPLETED,
        meaning="A turn finished and produced an answer.",
        semantics=Semantics.OCCURRENCE,
        origin=Origin.CORE,
        delivery=Delivery.RELIABLE,
        required=("kind", "latency_ms", "tool_calls"),
        sensitivity=Sensitivity.METADATA,
        presence_relevant=True,
        privacy_note="Reply length only; the reply itself belongs to 10B's SPEECH_*.",
    ),
    EventType.TASK_FAILED: EventSpec(
        type=EventType.TASK_FAILED,
        meaning="A turn ended without an answer.",
        semantics=Semantics.OCCURRENCE,
        origin=Origin.CORE,
        delivery=Delivery.RELIABLE,
        required=("kind", "error_type"),
        sensitivity=Sensitivity.METADATA,
        presence_relevant=True,
        privacy_note=(
            "Exception TYPE, not its message: a message can quote a path or a "
            "prompt fragment."
        ),
    ),
    # -- agent --------------------------------------------------------------
    EventType.AGENT_THINKING: EventSpec(
        type=EventType.AGENT_THINKING,
        meaning="The model is generating; no tool is running.",
        semantics=Semantics.OCCURRENCE,
        origin=Origin.CORE,
        delivery=Delivery.DROPPABLE,
        required=("step",),
        sensitivity=Sensitivity.METADATA,
        presence_relevant=True,
        privacy_note="Droppable: a missed animation cue costs nothing.",
    ),
    EventType.AGENT_WAITING: EventSpec(
        type=EventType.AGENT_WAITING,
        meaning="The agent is blocked -- a tool is running, or a human is needed.",
        semantics=Semantics.OCCURRENCE,
        origin=Origin.CORE,
        delivery=Delivery.RELIABLE,
        required=("reason", "tool"),
        sensitivity=Sensitivity.METADATA,
        presence_relevant=True,
        privacy_note="Tool NAME only, never its arguments.",
    ),
    # -- environment --------------------------------------------------------
    EventType.APP_CHANGED: EventSpec(
        type=EventType.APP_CHANGED,
        meaning="The foreground window changed.",
        semantics=Semantics.STATE,
        origin=Origin.OBSERVATION,
        delivery=Delivery.DROPPABLE,
        required=("application",),
        sensitivity=Sensitivity.ACTIVITY,
        presence_relevant=True,
        snapshot_key="foreground_application",
        source="core",
        privacy_note=(
            "UNTRUSTED. Window titles quote document names and message "
            "subjects, so only the process name is carried, never the title."
        ),
    ),
    EventType.VLM_OBSERVATION: EventSpec(
        type=EventType.VLM_OBSERVATION,
        meaning="Something was read off the screen.",
        semantics=Semantics.OCCURRENCE,
        origin=Origin.OBSERVATION,
        delivery=Delivery.DROPPABLE,
        required=("application", "confidence"),
        sensitivity=Sensitivity.CONTENT,
        presence_relevant=False,
        source="deferred_10c",
        privacy_note=(
            "UNTRUSTED and the highest-risk payload on the bus. Never an "
            "instruction; never evidence for a learned preference (a 10D "
            "constraint). Only published when the privacy gate already allowed "
            "the capture."
        ),
    ),
    # -- authority ----------------------------------------------------------
    EventType.PERMISSION_REQUIRED: EventSpec(
        type=EventType.PERMISSION_REQUIRED,
        meaning="A tool needs human approval before it can run.",
        semantics=Semantics.OCCURRENCE,
        origin=Origin.CORE,
        delivery=Delivery.RELIABLE,
        required=("tool", "level", "summary"),
        sensitivity=Sensitivity.CONTENT,
        presence_relevant=True,
        privacy_note=(
            "The ONE deliberate content exception: informed consent is "
            "impossible without saying what is being approved. Carries the "
            "approval nonce, which is safe because the stream is already "
            "authenticated -- and the event still grants nothing by itself."
        ),
    ),
    # -- modes --------------------------------------------------------------
    EventType.PRIVATE_MODE: EventSpec(
        type=EventType.PRIVATE_MODE,
        meaning="The vision privacy mode changed.",
        semantics=Semantics.STATE,
        origin=Origin.USER,
        delivery=Delivery.RELIABLE,
        required=("mode", "capturing"),
        sensitivity=Sensitivity.METADATA,
        presence_relevant=True,
        snapshot_key="vision_mode",
        privacy_note=(
            "Reliable because the overlay hiding is a privacy behaviour. A "
            "dropped 'now private' is the one drop that actually matters."
        ),
    ),
    EventType.FOCUS_MODE: EventSpec(
        type=EventType.FOCUS_MODE,
        meaning="Focus mode was turned on or off.",
        semantics=Semantics.STATE,
        origin=Origin.USER,
        delivery=Delivery.RELIABLE,
        required=("enabled",),
        sensitivity=Sensitivity.METADATA,
        presence_relevant=True,
        snapshot_key="focus_mode",
        source="core",
    ),
    # -- presentation -------------------------------------------------------
    EventType.PRESENCE_STATE: EventSpec(
        type=EventType.PRESENCE_STATE,
        meaning="The derived avatar state changed.",
        semantics=Semantics.STATE,
        origin=Origin.CORE,
        delivery=Delivery.DROPPABLE,
        required=("state",),
        sensitivity=Sensitivity.METADATA,
        presence_relevant=True,
        snapshot_key="presence_state",
        source="core",
        privacy_note="10A.4 owns the state machine; this reserves the contract slot.",
    ),
    EventType.SPEECH_STARTED: EventSpec(
        type=EventType.SPEECH_STARTED,
        meaning="AURA began speaking a reply; the subtitle should appear.",
        semantics=Semantics.OCCURRENCE,
        origin=Origin.CORE,
        delivery=Delivery.DROPPABLE,
        required=("length",),
        sensitivity=Sensitivity.METADATA,
        presence_relevant=True,
        source="client",
    ),
    EventType.SPEECH_CHUNK: EventSpec(
        type=EventType.SPEECH_CHUNK,
        meaning="A sentence of AURA's reply, for the subtitle.",
        semantics=Semantics.OCCURRENCE,
        origin=Origin.CORE,
        delivery=Delivery.DROPPABLE,
        required=("text",),
        sensitivity=Sensitivity.CONTENT,
        presence_relevant=True,
        source="client",
        privacy_note=(
            "Carries AURA's own words, which is the point -- the user asked to "
            "be able to read what she says when audio is not audible. It is "
            "output, not captured input."
        ),
    ),
    EventType.SPEECH_ENDED: EventSpec(
        type=EventType.SPEECH_ENDED,
        meaning="AURA finished speaking, or was cut off mid-reply.",
        semantics=Semantics.OCCURRENCE,
        origin=Origin.CORE,
        delivery=Delivery.DROPPABLE,
        required=("completed",),
        sensitivity=Sensitivity.METADATA,
        presence_relevant=True,
        source="client",
    ),
    # -- system -------------------------------------------------------------
    EventType.RESOURCE_PRESSURE: EventSpec(
        type=EventType.RESOURCE_PRESSURE,
        meaning="VRAM or RAM is constrained; subsystems should back off.",
        semantics=Semantics.STATE,
        origin=Origin.CORE,
        delivery=Delivery.RELIABLE,
        required=("level",),
        sensitivity=Sensitivity.METADATA,
        presence_relevant=True,
        snapshot_key="resource_pressure",
        source="deferred_10c",
    ),
    EventType.ERROR: EventSpec(
        type=EventType.ERROR,
        meaning="Something went wrong that a person should know about.",
        semantics=Semantics.OCCURRENCE,
        origin=Origin.CORE,
        delivery=Delivery.RELIABLE,
        required=("code",),
        sensitivity=Sensitivity.METADATA,
        presence_relevant=True,
        privacy_note="Codes and types, not messages, for the same reason as TASK_FAILED.",
    ),
}


def spec(event_type: EventType) -> EventSpec:
    """The contract for one event type. Raises for an uncatalogued type."""
    try:
        return CATALOG[event_type]
    except KeyError as exc:  # pragma: no cover - the catalog test prevents this
        raise KeyError(
            f"{event_type} has no catalog entry. Every event must declare its "
            "semantics, origin, delivery and privacy posture before it can be "
            "published."
        ) from exc


def state_types() -> tuple[EventType, ...]:
    """Types that describe a persisting condition, and may reach a SNAPSHOT."""
    return tuple(
        t for t, s in CATALOG.items() if s.semantics is Semantics.STATE
    )


def occurrence_types() -> tuple[EventType, ...]:
    """Types that describe a moment, and must never be replayed as state."""
    return tuple(
        t for t, s in CATALOG.items() if s.semantics is Semantics.OCCURRENCE
    )


def snapshot_keys() -> dict[str, EventType]:
    """Snapshot field -> the state event that maintains it.

    Lets a reviewer see at a glance that every snapshot field has an owner, and
    that no field is maintained by an occurrence.
    """
    return {
        s.snapshot_key: t
        for t, s in CATALOG.items()
        if s.snapshot_key and s.snapshot_key != "_self"
    }


def live_snapshot_keys() -> dict[str, EventType]:
    """Snapshot fields that must be populated *today*.

    A deferred slot (`RESOURCE_PRESSURE` for 10C) legitimately declares a field
    with no producer behind it yet -- that is a reserved contract, not an
    oversight. This is the subset a running AURA is obliged to carry, and a
    test asserts the snapshot actually contains all of them.
    """
    return {
        field: event_type
        for field, event_type in snapshot_keys().items()
        if not CATALOG[event_type].source.startswith("deferred")
    }


def live_types() -> tuple[EventType, ...]:
    """Types that actually have a producer. The rest are reserved slots.

    Keeping this honest is load-bearing rather than tidy. `live_snapshot_keys()`
    is derived from it, and that is what the snapshot-completeness test asserts
    against -- so a spec left marked `deferred` after its producer was written
    quietly excludes itself from the check that would have caught it. That
    happened: `APP_CHANGED`, `FOCUS_MODE` and `PRESENCE_STATE` still said
    `deferred_10a4` after 10A.4 and 10A.12 built them, which is exactly the
    class of contract violation the test exists to find.

    Two entries remain deferred, both genuinely: `VLM_OBSERVATION` and
    `RESOURCE_PRESSURE` wait on 10C.
    """
    return tuple(t for t, s in CATALOG.items() if not s.source.startswith("deferred"))
