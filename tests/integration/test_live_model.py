"""Tests that need a real llama-server with real weights.

Skipped unless one is reachable, so `uv run pytest` stays fast and green on a
machine with no GPU. Run them deliberately after starting the model server:

    uv run pytest -m integration -v

These are the only tests permitted to depend on an actual model. Everything
else must pass against the echo provider -- see tests/unit/test_seam.py.
"""

from __future__ import annotations

import httpx
import pytest

from aura.config import get_settings
from aura.core.types import GenerationParams, Message, Role
from aura.models.providers.openai_compat import OpenAICompatProvider

pytestmark = pytest.mark.integration


def _server_is_up(base_url: str) -> bool:
    try:
        return httpx.get(base_url + "/models", timeout=2.0).status_code < 400
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="module")
def base_url() -> str:
    url = get_settings().llm.base_url
    if not _server_is_up(url):
        pytest.skip(f"No model server at {url}. Start scripts/start_model_server.ps1 first.")
    return url


@pytest.fixture
async def provider(base_url: str):  # type: ignore[no-untyped-def]
    p = OpenAICompatProvider(base_url, timeout_s=120.0)
    yield p
    await p.close()


async def test_health_reports_the_loaded_model(provider: OpenAICompatProvider) -> None:
    health = await provider.health()
    assert health.available
    assert health.model


async def test_it_answers_a_factual_question(provider: OpenAICompatProvider) -> None:
    """A weak sanity check on model quality, not a benchmark.

    Real evaluation is Phase 8 (spec section 22). This only catches a model
    that loaded but is producing garbage -- wrong chat template, broken
    quantisation, corrupted download.
    """
    result = await provider.chat(
        [
            Message(role=Role.SYSTEM, content="Answer with a single word."),
            Message(role=Role.USER, content="What is the capital of France?"),
        ],
        GenerationParams(temperature=0.0, max_tokens=16),
    )
    assert "paris" in result.content.lower()
    assert result.usage.completion_tokens > 0


async def test_streaming_produces_multiple_chunks(
    provider: OpenAICompatProvider,
) -> None:
    chunks = [
        c
        async for c in provider.stream(
            [Message(role=Role.USER, content="Count from 1 to 10.")],
            GenerationParams(temperature=0.0, max_tokens=64),
        )
    ]
    text = "".join(c.delta for c in chunks)
    assert len([c for c in chunks if c.delta]) > 1, "server did not actually stream"
    assert chunks[-1].done
    assert "10" in text


async def test_generation_is_reasonably_fast(provider: OpenAICompatProvider) -> None:
    """Guards against an accidental CPU-only build.

    On an RTX 5060 with a 4B Q4_K_M model, ~64 tokens should take well under
    10 s. If this fails, layers are probably not offloaded to the GPU -- check
    the llama-server startup log for 'offloaded N/N layers'.
    """
    result = await provider.chat(
        [Message(role=Role.USER, content="Write two sentences about the sea.")],
        GenerationParams(temperature=0.0, max_tokens=64),
    )
    assert result.latency_ms < 10_000, (
        f"{result.latency_ms} ms for {result.usage.completion_tokens} tokens "
        "- is the model running on CPU?"
    )
