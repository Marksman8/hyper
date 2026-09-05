"""The LLM provider interface.

This is the single most important file in AURA (spec section 32). Everything
above it -- orchestrator, agent loop, memory, learning -- is written against
this ABC and nothing else. Everything below it is swappable.

Invariant, enforced by tests/unit/test_seam.py:
    No module outside aura.models.providers may import an inference library or
    name a concrete model file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any

from aura.core.types import (
    ChatChunk,
    ChatResult,
    GenerationParams,
    Message,
    ProviderHealth,
    ProviderInfo,
    Role,
)


class LLMProvider(ABC):
    """Abstract language-model backend.

    Implementations must be safe to share across concurrent requests. They own
    their own connection pooling and are closed exactly once at shutdown.

    Only `chat`, `stream`, `info` and `health` are required. `generate` has a
    default implementation in terms of `chat`, and `embed` raises by default
    because plenty of useful backends do not serve embeddings.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in logs, audit rows and persisted turns."""

    @abstractmethod
    def info(self) -> ProviderInfo:
        """Static capability description. Must not perform I/O."""

    @abstractmethod
    async def health(self) -> ProviderHealth:
        """Probe the backend. Must never raise; report failure in the result.

        Callers use this for status display and readiness gating, where an
        exception would be strictly less useful than a structured 'no'.
        """

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Message],
        params: GenerationParams | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> ChatResult:
        """Generate a complete response.

        `tools` is a list of definitions in the wire format the backend
        expects. Passing schemas rather than objects is what keeps this
        interface independent of `aura.tools` — a provider needs to know what
        a tool looks like on the wire, not what it does.

        Raises:
            ProviderUnavailableError: backend unreachable.
            ProviderTimeoutError: backend too slow.
            ProviderError: anything else the backend reported.
        """

    @abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Generate a response incrementally.

        Not `async def`: implementations are async generators, so calling this
        returns the iterator directly rather than a coroutine wrapping it. The
        final chunk must have `done=True`.
        """

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        params: GenerationParams | None = None,
    ) -> str:
        """Single-prompt convenience wrapper. Returns text only."""
        messages: list[Message] = []
        if system:
            messages.append(Message(role=Role.SYSTEM, content=system))
        messages.append(Message(role=Role.USER, content=prompt))
        result = await self.chat(messages, params)
        return result.content

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input.

        Default: unsupported. Phase 6 requires this; v0.1 does not.
        """
        raise NotImplementedError(
            self.name + " does not provide embeddings. "
            "Point llm.base_url at a server started with an embedding model, "
            "or configure a dedicated embedding provider."
        )

    async def close(self) -> None:
        """Release resources. Idempotent."""
        return None

    async def __aenter__(self) -> LLMProvider:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
