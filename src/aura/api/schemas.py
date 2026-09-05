"""HTTP request/response models.

Kept separate from `aura.core.types` on purpose. Core types are AURA's internal
vocabulary and are free to change; these are a published contract that other
applications (spec section 24) will depend on. Conflating them means every
internal refactor becomes a breaking API change.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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


class AgentRequest(BaseModel):
    """Ask AURA to do something, using tools if it needs them."""

    message: str = Field(min_length=1, max_length=100_000)
    session_id: str | None = None
    # Tools a human authorises for this task, named up front. The model cannot
    # add to this list; anything else it asks for comes back as a refusal it
    # has to explain rather than route around.
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


class EStopRequest(BaseModel):
    engaged: bool


class ErrorResponse(BaseModel):
    error: dict[str, Any]
