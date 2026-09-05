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


class EStopRequest(BaseModel):
    engaged: bool


class ErrorResponse(BaseModel):
    error: dict[str, Any]
