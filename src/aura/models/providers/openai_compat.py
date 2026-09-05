"""Provider for any OpenAI-compatible chat-completions server.

Verified targets: llama.cpp `llama-server` (v0.1 default), Ollama, LM Studio,
vLLM. The protocol, not the library, is the portability layer -- so upgrading
AURA's brain to a bigger model, a different engine, or a machine on the LAN is
a change to `llm.base_url` in config.

This module is the ONLY place in AURA allowed to know what a
`chat/completions` payload looks like.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from aura.core.types import (
    ChatChunk,
    ChatResult,
    GenerationParams,
    Message,
    ProviderHealth,
    ProviderInfo,
    Usage,
)
from aura.errors import ProviderError, ProviderTimeoutError, ProviderUnavailableError
from aura.logging import get_logger
from aura.models.base import LLMProvider

log = get_logger(__name__)

_SSE_DONE = "[DONE]"


class OpenAICompatProvider(LLMProvider):
    """Talks OpenAI chat-completions over HTTP."""

    def __init__(
        self,
        base_url: str,
        model: str = "default",
        api_key: str = "",
        timeout_s: float = 180.0,
        connect_retries: int = 2,
        default_params: GenerationParams | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._connect_retries = connect_retries
        self._defaults = default_params or GenerationParams()

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key

        # Generous read timeout (a 4B model can take a while for 1k tokens on a
        # laptop GPU) but a short connect timeout, so "server is not running"
        # is reported in under a second instead of after three minutes.
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=2.0),
        )

    @property
    def name(self) -> str:
        return "openai_compat"

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.name,
            kind="http",
            endpoint=self._base_url,
            model=self._model,
            supports_streaming=True,
            supports_embeddings=True,
            supports_tools=True,
        )

    # -- payload construction ------------------------------------------------

    def _payload(
        self,
        messages: Sequence[Message],
        params: GenerationParams | None,
        stream: bool,
    ) -> dict[str, Any]:
        p = params or self._defaults
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_wire() for m in messages],
            "temperature": p.temperature,
            "top_p": p.top_p,
            "max_tokens": p.max_tokens,
            "stream": stream,
        }
        if p.stop:
            payload["stop"] = list(p.stop)
        if p.seed is not None:
            payload["seed"] = p.seed
        if stream:
            # llama.cpp and vLLM both honour this; servers that do not simply
            # omit usage, which ChatChunk already treats as optional.
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _usage(raw: dict[str, Any]) -> Usage:
        u = raw.get("usage") or {}
        return Usage(
            prompt_tokens=int(u.get("prompt_tokens") or 0),
            completion_tokens=int(u.get("completion_tokens") or 0),
        )

    # -- error translation ---------------------------------------------------

    def _wrap(self, exc: Exception) -> ProviderError:
        """Turn transport failures into AURA errors with actionable text."""
        if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout):
            return ProviderUnavailableError(
                "No model server at " + self._base_url + ". "
                "Start one with scripts/start_model_server.ps1, or point "
                "llm.base_url at a running server."
            )
        if isinstance(exc, httpx.ReadTimeout | httpx.WriteTimeout | httpx.PoolTimeout):
            return ProviderTimeoutError(
                "Model server at " + self._base_url + " timed out. "
                "The model may be loading, or max_tokens may be too high."
            )
        return ProviderError("Model server transport error: " + repr(exc))

    @staticmethod
    def _check_status(resp: httpx.Response, body: str) -> None:
        if resp.status_code < 400:
            return
        detail = body.strip()[:500]
        try:
            parsed = json.loads(body)
            err = parsed.get("error")
            if isinstance(err, dict):
                detail = str(err.get("message", detail))
            elif isinstance(err, str):
                detail = err
        except (json.JSONDecodeError, AttributeError):
            pass
        raise ProviderError("Model server returned " + str(resp.status_code) + ": " + detail)

    # -- LLMProvider ---------------------------------------------------------

    async def health(self) -> ProviderHealth:
        started = time.perf_counter()
        try:
            resp = await self._client.get("/models", timeout=httpx.Timeout(5.0, connect=1.5))
        except httpx.HTTPError as exc:
            return ProviderHealth(available=False, detail=str(self._wrap(exc)), latency_ms=0)
        latency = int((time.perf_counter() - started) * 1000)
        if resp.status_code >= 400:
            return ProviderHealth(
                available=False,
                detail="/models returned " + str(resp.status_code),
                latency_ms=latency,
            )
        model = self._model
        try:
            data = resp.json().get("data") or []
            if data and isinstance(data[0], dict):
                model = str(data[0].get("id", model))
        except (ValueError, AttributeError, IndexError):
            pass
        return ProviderHealth(available=True, detail="ok", model=model, latency_ms=latency)

    async def chat(
        self,
        messages: Sequence[Message],
        params: GenerationParams | None = None,
    ) -> ChatResult:
        payload = self._payload(messages, params, stream=False)
        started = time.perf_counter()
        last: Exception | None = None

        for attempt in range(self._connect_retries + 1):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                self._check_status(resp, resp.text)
                raw = resp.json()
                break
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                # Only connection failures are retried. A 400 or a timeout
                # mid-generation means retrying just burns the user's time.
                last = exc
                if attempt < self._connect_retries:
                    log.debug("provider.connect_retry", attempt=attempt + 1)
                    continue
                raise self._wrap(exc) from exc
            except httpx.HTTPError as exc:
                raise self._wrap(exc) from exc
        else:  # pragma: no cover - loop always breaks or raises
            raise self._wrap(last or RuntimeError("unreachable"))

        latency_ms = int((time.perf_counter() - started) * 1000)
        choices = raw.get("choices") or []
        if not choices:
            raise ProviderError("Model server returned no choices.")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        # Reasoning models (Qwen3.x, DeepSeek-R1, gpt-oss) return thinking in a
        # separate field. An empty `content` with a populated `reasoning_content`
        # and finish_reason="length" means the token budget was spent thinking
        # before the model reached an answer.
        reasoning = message.get("reasoning_content") or ""

        return ChatResult(
            content=content,
            model=str(raw.get("model") or self._model),
            provider=self.name,
            usage=self._usage(raw),
            finish_reason=str(choice.get("finish_reason") or "stop"),
            latency_ms=latency_ms,
            reasoning=reasoning,
            raw=raw,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        payload = self._payload(messages, params, stream=True)
        finish_reason: str | None = None
        usage: Usage | None = None

        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    self._check_status(resp, body)

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == _SSE_DONE:
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        log.warning("provider.bad_sse_frame", frame=data[:200])
                        continue

                    if event.get("usage"):
                        usage = self._usage(event)

                    for choice in event.get("choices") or []:
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])
                        delta_obj = choice.get("delta") or {}
                        delta = delta_obj.get("content")
                        thought = delta_obj.get("reasoning_content")
                        # Emitted as separate chunks so a caller can render the
                        # answer while ignoring, logging, or separately
                        # displaying the thinking.
                        if thought:
                            yield ChatChunk(delta="", reasoning_delta=thought)
                        if delta:
                            yield ChatChunk(delta=delta)
        except httpx.HTTPError as exc:
            raise self._wrap(exc) from exc

        yield ChatChunk(delta="", done=True, finish_reason=finish_reason or "stop", usage=usage)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            resp = await self._client.post(
                "/embeddings", json={"model": self._model, "input": list(texts)}
            )
            self._check_status(resp, resp.text)
            raw = resp.json()
        except httpx.HTTPError as exc:
            raise self._wrap(exc) from exc

        items = sorted(raw.get("data") or [], key=lambda d: d.get("index", 0))
        return [[float(x) for x in item["embedding"]] for item in items]

    async def close(self) -> None:
        await self._client.aclose()
