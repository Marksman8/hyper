"""The memory HTTP surface.

The exit criterion is user-facing, so it is tested at the surface the user
actually reaches: `why` must return the words behind a belief, and deletion
must leave nothing that could re-derive it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from aura.api.app import create_app
from aura.config import (
    ApiConfig,
    LLMConfig,
    LoggingConfig,
    MemoryConfig,
    SecurityConfig,
    Settings,
)
from aura.core.types import Evidence, Fact, Role, Turn
from aura.memory.store import MemoryStore

TEST_TOKEN = "test-token-not-a-real-secret"


@pytest.fixture
def api_settings(tmp_path: Path) -> Settings:
    return Settings(
        api=ApiConfig(host="127.0.0.1", port=8760),
        llm=LLMConfig(provider="echo"),
        memory=MemoryConfig(
            db_path=str(tmp_path / "aura.db"),
            # Off: these tests drive the store directly and must not have a
            # background extractor racing them.
            learn_from_conversation=False,
        ),
        security=SecurityConfig(api_token=TEST_TOKEN, token_file=str(tmp_path / "token")),
        logging=LoggingConfig(level="WARNING", file=str(tmp_path / "aura.log")),
    )


@pytest_asyncio.fixture
async def client(api_settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(api_settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://aura.test",
            headers={"Authorization": "Bearer " + TEST_TOKEN},
        ) as c:
            yield c


def _store(client: AsyncClient) -> MemoryStore:
    return client._transport.app.state.aura.store  # type: ignore[attr-defined,union-attr]


async def _seed_fact(client: AsyncClient) -> Fact:
    store = _store(client)
    fact = await store.upsert_fact(Fact(scope="user", key="response_length", value="short"))
    await store.add_evidence(
        Evidence(
            fact_id=fact.id,
            quote="keep your answers short",
            turn_id="t1",
            session_id="s1",
            source="inferred",
        )
    )
    return fact


# -- why (the exit criterion) -----------------------------------------------


async def test_why_returns_the_users_own_words(client: AsyncClient) -> None:
    await _seed_fact(client)
    body = (await client.get("/v1/memory/response_length/why")).json()

    assert body["fact"]["key"] == "response_length"
    assert [e["quote"] for e in body["evidence"]] == ["keep your answers short"]
    assert body["reason"], "a belief with no explanation is the thing this phase fixes"


async def test_a_single_inference_is_reported_as_not_yet_believed(
    client: AsyncClient,
) -> None:
    await _seed_fact(client)
    body = (await client.get("/v1/memory/response_length/why")).json()

    assert body["believed"] is False
    assert body["fact"]["confidence"] < 0.5
    assert "not acting on it" in body["reason"]


async def test_corroboration_flips_it_to_believed(client: AsyncClient) -> None:
    fact = await _seed_fact(client)
    await _store(client).add_evidence(
        Evidence(fact_id=fact.id, quote="still too long", turn_id="t2", source="inferred")
    )

    body = (await client.get("/v1/memory/response_length/why")).json()
    assert body["believed"] is True
    assert len(body["evidence"]) == 2
    assert "2 separate" in body["reason"]


async def test_why_on_an_unknown_key_is_a_404_not_a_500(client: AsyncClient) -> None:
    assert (await client.get("/v1/memory/nothing_here/why")).status_code == 404


# -- deletion (spec section 23) ---------------------------------------------


async def test_deleting_a_fact_removes_its_justification_too(client: AsyncClient) -> None:
    fact = await _seed_fact(client)
    assert (await client.delete("/v1/memory/response_length")).json()["deleted"] is True

    assert (await client.get("/v1/memory/response_length/why")).status_code == 404
    assert await _store(client).evidence_for(fact.id) == []


# -- search -----------------------------------------------------------------


async def test_search_finds_a_past_turn(client: AsyncClient) -> None:
    store = _store(client)
    session = await store.create_session()
    await store.add_turn(
        Turn(
            session_id=session.id,
            role=Role.USER,
            content="the barge in threshold was measured at 0.024",
        )
    )

    hits = (await client.get("/v1/memory/search", params={"q": "barge threshold"})).json()
    assert hits
    assert "barge in threshold" in hits[0]["content"]
    assert hits[0]["session_id"] == session.id


async def test_search_with_punctuation_does_not_error(client: AsyncClient) -> None:
    """A user's question must never be parsed as FTS5 query syntax."""
    resp = await client.get("/v1/memory/search", params={"q": 'what about "stop"?'})
    assert resp.status_code == 200
    assert resp.json() == []


async def test_memory_endpoints_require_authentication(api_settings: Settings) -> None:
    app = create_app(api_settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://aura.test"
        ) as anon:
            assert (await anon.get("/v1/memory")).status_code == 401
            assert (await anon.get("/v1/memory/x/why")).status_code == 401
            assert (await anon.get("/v1/memory/search", params={"q": "x"})).status_code == 401
