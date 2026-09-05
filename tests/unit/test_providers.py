"""Provider contract tests.

The OpenAI-compatible provider is tested against mocked HTTP rather than a real
llama-server, so the suite runs on a machine with no GPU and no weights. The
integration test in tests/integration/ covers the real thing.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from aura.core.types import GenerationParams, Message, Role
from aura.errors import ProviderError, ProviderTimeoutError, ProviderUnavailableError
from aura.models.providers.echo import EchoProvider
from aura.models.providers.openai_compat import OpenAICompatProvider

BASE = "http://127.0.0.1:8081/v1"
MESSAGES = [Message(role=Role.USER, content="hello there")]


def _completion(text: str = "hi") -> dict[str, object]:
    return {
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }


# -- echo --------------------------------------------------------------------


async def test_echo_chat_is_deterministic() -> None:
    result = await EchoProvider().chat(MESSAGES)
    assert result.content == "[echo] hello there"
    assert result.provider == "echo"


async def test_echo_stream_reassembles_to_the_same_text() -> None:
    provider = EchoProvider()
    chunks = [c async for c in provider.stream(MESSAGES)]
    assert chunks[-1].done is True
    assert "".join(c.delta for c in chunks) == "[echo] hello there"


async def test_echo_generate_wraps_chat() -> None:
    assert await EchoProvider().generate("ping") == "[echo] ping"


async def test_embeddings_unsupported_by_default() -> None:
    with pytest.raises(NotImplementedError):
        await EchoProvider().embed(["x"])


# -- openai-compatible -------------------------------------------------------


@respx.mock
async def test_chat_parses_a_normal_response() -> None:
    respx.post(BASE + "/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("hi"))
    )
    provider = OpenAICompatProvider(BASE)
    result = await provider.chat(MESSAGES)
    assert result.content == "hi"
    assert result.model == "test-model"
    assert result.usage.prompt_tokens == 7
    assert result.usage.total_tokens == 10
    await provider.close()


@respx.mock
async def test_generation_params_reach_the_wire() -> None:
    route = respx.post(BASE + "/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion())
    )
    provider = OpenAICompatProvider(BASE)
    await provider.chat(MESSAGES, GenerationParams(temperature=0.1, max_tokens=64, stop=("</s>",)))
    sent = route.calls[0].request
    body = sent.read().decode()
    assert '"temperature":0.1' in body.replace(" ", "")
    assert '"max_tokens":64' in body.replace(" ", "")
    assert "</s>" in body
    await provider.close()


@respx.mock
async def test_streaming_yields_deltas_then_a_done_chunk() -> None:
    sse = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(BASE + "/chat/completions").mock(
        return_value=httpx.Response(200, text=sse, headers={"Content-Type": "text/event-stream"})
    )
    provider = OpenAICompatProvider(BASE)
    chunks = [c async for c in provider.stream(MESSAGES)]

    assert "".join(c.delta for c in chunks) == "Hello"
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 2
    await provider.close()


@respx.mock
async def test_malformed_sse_frames_are_skipped_not_fatal() -> None:
    sse = 'data: {not json}\n\ndata: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
    respx.post(BASE + "/chat/completions").mock(return_value=httpx.Response(200, text=sse))
    provider = OpenAICompatProvider(BASE)
    chunks = [c async for c in provider.stream(MESSAGES)]
    assert "".join(c.delta for c in chunks) == "ok"
    await provider.close()


@respx.mock
async def test_connection_refused_maps_to_provider_unavailable() -> None:
    respx.post(BASE + "/chat/completions").mock(side_effect=httpx.ConnectError("refused"))
    provider = OpenAICompatProvider(BASE, connect_retries=0)
    with pytest.raises(ProviderUnavailableError, match="No model server"):
        await provider.chat(MESSAGES)
    await provider.close()


@respx.mock
async def test_read_timeout_maps_to_provider_timeout() -> None:
    respx.post(BASE + "/chat/completions").mock(side_effect=httpx.ReadTimeout("slow"))
    provider = OpenAICompatProvider(BASE)
    with pytest.raises(ProviderTimeoutError):
        await provider.chat(MESSAGES)
    await provider.close()


@respx.mock
async def test_server_error_message_is_surfaced() -> None:
    respx.post(BASE + "/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "context overflow"}})
    )
    provider = OpenAICompatProvider(BASE)
    with pytest.raises(ProviderError, match="context overflow"):
        await provider.chat(MESSAGES)
    await provider.close()


@respx.mock
async def test_connect_failures_are_retried() -> None:
    route = respx.post(BASE + "/chat/completions").mock(
        side_effect=[
            httpx.ConnectError("refused"),
            httpx.Response(200, json=_completion("recovered")),
        ]
    )
    provider = OpenAICompatProvider(BASE, connect_retries=1)
    result = await provider.chat(MESSAGES)
    assert result.content == "recovered"
    assert route.call_count == 2
    await provider.close()


@respx.mock
async def test_health_reports_down_without_raising() -> None:
    respx.get(BASE + "/models").mock(side_effect=httpx.ConnectError("refused"))
    provider = OpenAICompatProvider(BASE)
    health = await provider.health()
    assert health.available is False
    assert "No model server" in health.detail
    await provider.close()


@respx.mock
async def test_health_reports_the_loaded_model() -> None:
    respx.get(BASE + "/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "Qwen3-4B"}]})
    )
    provider = OpenAICompatProvider(BASE)
    health = await provider.health()
    assert health.available is True
    assert health.model == "Qwen3-4B"
    await provider.close()


@respx.mock
async def test_embeddings_preserve_input_order() -> None:
    respx.post(BASE + "/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            },
        )
    )
    provider = OpenAICompatProvider(BASE)
    assert await provider.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    await provider.close()
