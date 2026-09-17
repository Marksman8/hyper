"""HTTP request/response models.

Kept separate from `aura.core.types` on purpose. Core types are AURA's internal
vocabulary and are free to change; these are a published contract that other
applications (spec section 24) will depend on. Conflating them means every
internal refactor becomes a breaking API change.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    """Text to speak aloud.

    Capped well below what the synthesiser could chew through: a spoken reply
    is a sentence or two, and a client asking for a novel is either confused or
    trying to make Core spend minutes in a worker thread.
    """

    text: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    """AURA-native chat request."""

    message: str = Field(min_length=1, max_length=100_000)
    session_id: str | None = None
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0, le=32_768)
    # The reply will be spoken, so ask the model for prose rather than
    # markdown. A client that renders text should leave this false.
    spoken: bool = False
    # Phase 7. False answers as the unadapted persona -- so adaptation can
    # be switched off and therefore blamed, and so the A/B harness can get
    # both arms from one running AURA.
    personalize: bool = True


class AgentRequest(BaseModel):
    """Ask AURA to do something, using tools if it needs them."""

    message: str = Field(min_length=1, max_length=100_000)
    session_id: str | None = None
    # Approval *tokens* a human authorises for this task, supplied up front:
    # `name:fingerprint`, as returned in a previous response's
    # `pending_confirmations`. A bare tool name grants nothing (audit 01
    # finding 2) -- an approval names one call with one set of arguments, not a
    # capability held for the rest of the task. The model cannot add to this
    # list; anything else it asks for comes back as a refusal it has to explain
    # rather than route around.
    approve: list[str] = Field(default_factory=list)
    max_steps: int | None = Field(default=None, ge=1, le=50)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class AgentStepOut(BaseModel):
    index: int
    kind: str
    content: str = ""
    tools: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: int = 0


class AgentResponse(BaseModel):
    session_id: str
    content: str
    model: str
    provider: str
    usage: UsageOut
    latency_ms: int
    tool_calls_made: int = 0
    steps: list[AgentStepOut] = Field(default_factory=list)
    # True when the loop ran out of budget rather than finishing, so a caller
    # can tell "done" from "gave up".
    stopped_early: bool = False
    stop_reason: str = "complete"
    # Approval tokens for the calls that were refused pending a human, bound to
    # the arguments the model actually asked for. Pass them back verbatim in
    # `approve` to authorise exactly those calls.
    pending_confirmations: list[str] = Field(default_factory=list)


class UsageOut(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    session_id: str
    content: str
    model: str
    provider: str
    usage: UsageOut
    finish_reason: str
    latency_ms: int


class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None


class OpenAIChatRequest(BaseModel):
    """OpenAI-compatible request, so existing clients can point at AURA.

    Note the semantic difference from a plain model server: AURA prepends its
    own system prompt and conversation memory. Send `session_id` (an AURA
    extension) to continue a stored conversation; omit it for a stateless call.
    """

    model: str | None = None
    messages: list[OpenAIMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0, le=32_768)
    session_id: str | None = None


class SessionOut(BaseModel):
    id: str
    title: str | None
    created_at: float
    updated_at: float


class TurnOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: float
    model: str | None = None
    latency_ms: int = 0
    error: str | None = None


class FactIn(BaseModel):
    key: str = Field(min_length=1, max_length=200)
    value: str = Field(min_length=1, max_length=5_000)
    permanence: Literal["session", "temporary", "long_term", "permanent"] = "long_term"


class FactOut(BaseModel):
    key: str
    value: str
    confidence: float
    evidence_count: int
    source: str
    permanence: str
    last_confirmed_at: float


class EvidenceOut(BaseModel):
    """One reason a fact is believed. The point of Phase 6: a justification
    the user can read, disagree with, and act on."""

    quote: str
    source: str
    session_id: str | None
    turn_id: str | None
    created_at: float


class FactExplanation(BaseModel):
    fact: FactOut
    # A sentence a person can check, e.g. "Inferred from 2 separate occasions."
    reason: str
    believed: bool
    evidence: list[EvidenceOut]


class TurnMatchOut(BaseModel):
    session_id: str
    turn_id: str
    role: str
    content: str
    created_at: float
    score: float


class ProfileDimension(BaseModel):
    dimension: str
    value: str
    from_fact: str
    source: str
    confidence: float
    drives_parameters: bool


class ProfileOut(BaseModel):
    """What AURA is adapting, and why. Silent adaptation is a debugging
    nightmare: "AURA is being weird today" has to be answerable."""

    enabled: bool
    dimensions: list[ProfileDimension]
    # The exact instruction lines added to the system prompt.
    directives: list[str]
    other_facts: list[str]


class ProviderStatus(BaseModel):
    name: str
    kind: str
    endpoint: str | None
    model: str | None
    available: bool
    detail: str
    latency_ms: int


class ResourceStatus(BaseModel):
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    process_ram_mb: float
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None


class StatusResponse(BaseModel):
    name: str
    version: str
    uptime_s: float
    emergency_stop: bool
    provider: ProviderStatus
    memory: dict[str, int]
    resources: ResourceStatus


class ToolSpecOut(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    level: int
    level_label: str
    timeout_s: float
    reversible: bool
    # True when this tool cannot run without an explicit human confirmation
    # under the current policy. Clients render the approval prompt from this.
    needs_confirmation: bool


class ToolCallRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Must be set deliberately by whoever is approving. The executor treats an
    # unset flag as a refusal, never as consent.
    confirmed: bool = False
    session_id: str | None = None


class ToolCallResponse(BaseModel):
    call_id: str
    name: str
    ok: bool
    content: str = ""
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = 0
    decision: str = "allow"


class FocusModeRequest(BaseModel):
    """Turn focus mode on or off (Phase 10A.12).

    Focus reduces intrusion without reducing capability: AURA still listens,
    still works, still answers. She is quieter about it. That is why it is not
    a presence STATE -- she is not "in focus mode" instead of thinking, she is
    thinking quietly.
    """

    enabled: bool


class ClientEvent(BaseModel):
    """An occurrence that happened in a client process (Phase 10A.3, 10B.1).

    The voice loop runs as a SEPARATE PROCESS -- audio devices and the GIL put
    it there -- so wake-word detection, barge-in and speech playback cannot
    reach Core's in-process bus directly. This is the narrow bridge, and it is
    narrow on purpose:

    * only the types listed below are accepted
    * **origin comes from the event catalog, never from the request**
    * delivery comes from the catalog, not from the request
    * every payload is bounded

    **On origin, and why 10B.1 changed how this is phrased.** The rule used to
    be "forced to USER". That was the right behaviour for the two events that
    existed, and the wrong statement of the property. The property is that *a
    client cannot choose an origin* -- the catalog decides, and the catalog is
    a version-controlled local file, the same arrangement as the workflow
    manifest in ADR-0004.

    That distinction matters as soon as speech arrives. `USER_SPOKE` is a user
    action and is catalogued `Origin.USER`. `SPEECH_CHUNK` is AURA's own
    utterance and is catalogued `Origin.CORE`, because the voice loop is not a
    third-party client -- it is AURA's mouth, running out-of-process for
    hardware reasons. Labelling her own words `USER` would be a lie the
    presence layer then acts on: `Origin.USER` means a human did something, and
    it resets idle timers on that basis.

    What that concedes: a caller holding the API token can now publish a
    CORE-origin speech event, i.e. put words in a subtitle that AURA never
    said. Bounded by invariant I4 -- an event cannot cause a tool to run -- and
    dominated by the fact that the same token can make AURA genuinely say
    anything through /v1/chat.
    """

    type: Literal[
        "user_spoke",
        "user_interrupted",
        "speech_started",
        "speech_chunk",
        "speech_ended",
    ]
    # Length rather than text, for the USER events: the bus carries no
    # transcripts (see event_catalog.py). A client that sent one would have it
    # ignored. `text` below is the deliberate exception, and it runs the other
    # way -- it is AURA's output, not the user's captured input.
    length: int = Field(default=0, ge=0, le=100_000)
    method: str = Field(default="", max_length=32)
    spoken: bool = True
    # SPEECH_CHUNK only. Capped here as well as in the emitter: the schema is
    # what stops an oversized body being parsed at all.
    text: str = Field(default="", max_length=1_000)
    # SPEECH_ENDED only. Whether she finished or was cut off.
    completed: bool = True
    # SPEECH_CHUNK only (10B.3). One digit per 20 ms of audio, driving the
    # avatar's mouth. Constrained to digits by the pattern rather than merely
    # capped: an envelope is a fixed alphabet, so anything else is a caller
    # bug and there is no reason to let it reach a consumer.
    envelope: str = Field(default="", max_length=400, pattern=r"^[0-9]*$")


class ApprovalRedeem(BaseModel):
    """Grant or deny a pending permission request (Phase 10A.2).

    The nonce and request_id together bind this to the one request that asked.
    Without them an approval could be forged, replayed, or moved onto a
    different action -- the confused-deputy attack this whole path exists to
    prevent.
    """

    nonce: str = Field(min_length=8, max_length=256)
    request_id: str = Field(min_length=1, max_length=128)
    # False records an explicit refusal, which also burns the approval: a user
    # who said no must not have it replayed into a yes.
    granted: bool = True


class EStopRequest(BaseModel):
    engaged: bool


class VisionModeRequest(BaseModel):
    """Change what AURA may look at.

    Deliberately an HTTP endpoint and not a tool. Turning vision *on* is a
    human decision; see aura/tools/vision.py for why the model is only ever
    given the narrowing half.
    """

    mode: Literal["private", "assist", "active", "sleep"]


class VisionAppRequest(BaseModel):
    app: str = Field(min_length=1, max_length=200)


class ErrorResponse(BaseModel):
    error: dict[str, Any]
