"""A deterministic, dependency-free provider.

Two jobs:

1. Prove the seam. Every layer above `LLMProvider` must work identically with a
   backend that has no GPU, no weights and no network. If a test needs a real
   model to pass, something above the seam has reached below it.
2. Let AURA boot before any weights are downloaded, so the API, memory, auth
   and client can be exercised on a machine that is still fetching 3 GB.

Set `llm.provider: echo` in config to use it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from aura.core.types import (
    ChatChunk,
    ChatResult,
    GenerationParams,
    Message,
    ProviderHealth,
    ProviderInfo,
    Role,
    Usage,
)
from aura.models.base import LLMProvider


class EchoProvider(LLMProvider):
    """Returns a canned, deterministic response derived from the input."""

    def __init__(self, chunk_delay_s: float = 0.0, prefix: str = "[echo] ") -> None:
        self._delay = chunk_delay_s
        self._prefix = prefix

    @property
    def name(self) -> str:
        return "echo"

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.name,
            kind="in_process",
            endpoint=None,
            model="echo-v1",
            supports_streaming=True,
            supports_embeddings=False,
            supports_tools=False,
        )

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            available=True, detail="echo provider is always available", model="echo-v1"
        )

    def _reply(self, messages: Sequence[Message]) -> str:
        last_user = next((m.content for m in reversed(messages) if m.role is Role.USER), "")
        return self._prefix + last_user

    async def chat(
        self,
        messages: Sequence[Message],
        params: GenerationParams | None = None,
    ) -> ChatResult:
        text = self._reply(messages)
        return ChatResult(
            content=text,
            model="echo-v1",
            provider=self.name,
            # Rough word-count stand-in; enough for the accounting path to be
            # exercised end to end without pretending to be a tokenizer.
            usage=Usage(
                prompt_tokens=sum(len(m.content.split()) for m in messages),
                completion_tokens=len(text.split()),
            ),
            finish_reason="stop",
            latency_ms=0,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        text = self._reply(messages)
        words = text.split(" ")
        for i, word in enumerate(words):
            if self._delay:
                await asyncio.sleep(self._delay)
            yield ChatChunk(delta=word if i == 0 else " " + word)
        yield ChatChunk(
            delta="",
            done=True,
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=sum(len(m.content.split()) for m in messages),
                completion_tokens=len(words),
            ),
        )
