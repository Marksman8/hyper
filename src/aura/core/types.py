"""Core domain types.

These are AURA's own vocabulary. They deliberately do NOT mirror any vendor's
API schema: the OpenAI-shaped wire format is an implementation detail of one
provider, translated at the provider boundary. If these types ever grow a field
that exists only because llama.cpp has it, the seam has leaked.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """Who produced a message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    # Reserved for Phase 3/4. Present now so persisted turns never need a
    # schema migration when tools arrive.
    TOOL = "tool"


@dataclass(slots=True, frozen=True)
class ToolInvocation:
    """A tool call the model asked for.

    Deliberately *not* `aura.tools.ToolCall`. This is what a model produced —
    a request, unvalidated and unapproved. The agent loop converts it into a
    `tools.ToolCall` only after deciding it may run. Keeping the two types
    apart is what stops `aura.core` depending on `aura.tools`, and stops a
    model-produced request from being mistaken for an authorised one.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True, frozen=True)
class Message:
    """One message in a conversation."""

    role: Role
    content: str
    name: str | None = None
    # Set on an assistant message that requested tools.
    tool_calls: tuple[ToolInvocation, ...] = ()
    # Set on a TOOL message, linking the result back to the request.
    tool_call_id: str | None = None
    # Free-form provenance: which tool produced it, which monitor it was read
    # from, which memory it was retrieved from. Never sent to the model.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        """Minimal representation for a chat-completions style API."""
        msg: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            # Arguments go back as a JSON *string*, which is what the wire
            # format uses and what the model produced in the first place.
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


@dataclass(slots=True, frozen=True)
class GenerationParams:
    """Per-request sampling controls.

    Separate from Message so a caller can vary sampling without rebuilding the
    conversation, and so the agent loop (Phase 4) can use low temperature for
    tool selection and higher for prose within the same turn.
    """

    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024
    stop: tuple[str, ...] = ()
    seed: int | None = None

    def merged(
        self,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: tuple[str, ...] | None = None,
        seed: int | None = None,
    ) -> GenerationParams:
        """Return a copy with only the non-None overrides applied.

        Explicit keywords rather than **kwargs so a typo in a caller is a type
        error rather than a silently ignored parameter.
        """
        return GenerationParams(
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            stop=self.stop if stop is None else stop,
            seed=self.seed if seed is None else seed,
        )


@dataclass(slots=True, frozen=True)
class Usage:
    """Token accounting. Zeros mean the provider did not report it."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True, frozen=True)
class ChatResult:
    """A completed non-streaming generation."""

    content: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"
    latency_ms: int = 0
    # Populated when the model asked for tools instead of answering.
    tool_calls: tuple[ToolInvocation, ...] = ()
    # Chain-of-thought, when the model produces it as a separate channel.
    # Modelled as its own field rather than folded into `content` because the
    # two have different lifetimes: the answer is replayed into later prompts,
    # the reasoning is not. It is kept because it is the best available
    # evidence for "why did AURA do this?" (spec section 25) and will be what
    # the Phase 4 agent loop inspects when a plan goes wrong.
    reasoning: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ChatChunk:
    """One increment of a streaming generation.

    `delta` is the new answer text only. `reasoning_delta` carries thinking
    tokens, which most callers should not render. The terminal chunk has
    `done=True`, may carry an empty delta, and is the only chunk guaranteed to
    have usage populated.
    """

    delta: str
    done: bool = False
    finish_reason: str | None = None
    usage: Usage | None = None
    reasoning_delta: str = ""


@dataclass(slots=True, frozen=True)
class ProviderHealth:
    """Result of probing a provider."""

    available: bool
    detail: str = ""
    model: str | None = None
    latency_ms: int = 0


@dataclass(slots=True, frozen=True)
class ProviderInfo:
    """Static description of a provider, for /v1/status and the UI."""

    name: str
    kind: str
    endpoint: str | None = None
    model: str | None = None
    supports_streaming: bool = True
    supports_embeddings: bool = False
    supports_tools: bool = False


@dataclass(slots=True)
class Turn:
    """A persisted conversation turn."""

    session_id: str
    role: Role
    content: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    model: str | None = None
    provider: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    finish_reason: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> Message:
        return Message(role=self.role, content=self.content)


@dataclass(slots=True)
class Session:
    """A conversation."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    title: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class Permanence(StrEnum):
    """How long a remembered fact should survive (spec section 16).

    Encoded now, in v0.1, because the distinction between "today, be brief" and
    "I generally prefer brevity" cannot be reconstructed after the fact.
    """

    SESSION = "session"
    TEMPORARY = "temporary"
    LONG_TERM = "long_term"
    PERMANENT = "permanent"


@dataclass(slots=True)
class Fact:
    """A single remembered item about the user or the world (spec section 15)."""

    scope: str
    key: str
    value: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    confidence: float = 1.0
    evidence_count: int = 1
    source: str = "explicit"
    permanence: Permanence = Permanence.LONG_TERM
    created_at: float = field(default_factory=time.time)
    last_confirmed_at: float = field(default_factory=time.time)
