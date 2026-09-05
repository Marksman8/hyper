"""Persistence: the thing that has to survive a restart."""

from __future__ import annotations

import time

import pytest

from aura.core.types import Fact, Permanence, Role, Turn
from aura.errors import NotFoundError
from aura.memory.store import MemoryStore, build_history


async def test_session_roundtrip(store: MemoryStore) -> None:
    session = await store.create_session(title="first")
    fetched = await store.get_session(session.id)
    assert fetched.id == session.id
    assert fetched.title == "first"


async def test_missing_session_raises_not_found(store: MemoryStore) -> None:
    with pytest.raises(NotFoundError):
        await store.get_session("does-not-exist")


async def test_ensure_session_creates_when_none_given(store: MemoryStore) -> None:
    session = await store.ensure_session(None)
    assert (await store.get_session(session.id)).id == session.id


async def test_turns_return_in_chronological_order(store: MemoryStore) -> None:
    session = await store.create_session()
    for i in range(5):
        await store.add_turn(Turn(session_id=session.id, role=Role.USER, content=f"m{i}"))
    turns = await store.recent_turns(session.id, 10)
    assert [t.content for t in turns] == ["m0", "m1", "m2", "m3", "m4"]


async def test_recent_turns_keeps_the_newest(store: MemoryStore) -> None:
    session = await store.create_session()
    for i in range(10):
        await store.add_turn(Turn(session_id=session.id, role=Role.USER, content=f"m{i}"))
    turns = await store.recent_turns(session.id, 3)
    assert [t.content for t in turns] == ["m7", "m8", "m9"]


async def test_turns_created_in_the_same_millisecond_still_order(
    store: MemoryStore,
) -> None:
    """Timestamps can collide on Windows; rowid must break the tie."""
    session = await store.create_session()
    stamp = time.time()
    for i in range(4):
        await store.add_turn(
            Turn(
                session_id=session.id,
                role=Role.USER,
                content=f"m{i}",
                created_at=stamp,
            )
        )
    turns = await store.recent_turns(session.id, 10)
    assert [t.content for t in turns] == ["m0", "m1", "m2", "m3"]


async def test_deleting_a_session_cascades_to_turns(store: MemoryStore) -> None:
    session = await store.create_session()
    await store.add_turn(Turn(session_id=session.id, role=Role.USER, content="x"))
    assert await store.delete_session(session.id)
    assert (await store.stats())["turns"] == 0


async def test_repeating_a_fact_reinforces_it(store: MemoryStore) -> None:
    """Spec section 15: evidence accumulates, rows do not duplicate."""
    for _ in range(3):
        await store.upsert_fact(Fact(scope="user", key="style", value="concise", confidence=0.8))
    facts = await store.list_facts("user")
    assert len(facts) == 1
    assert facts[0].evidence_count == 3
    assert facts[0].confidence > 0.8


async def test_forget_expired_clears_session_and_stale_temporary_facts(
    store: MemoryStore,
) -> None:
    """Spec section 16: 'today, be brief' must not become permanent."""
    now = time.time()
    await store.upsert_fact(Fact(scope="user", key="a", value="1", permanence=Permanence.SESSION))
    await store.upsert_fact(
        Fact(
            scope="user",
            key="b",
            value="2",
            permanence=Permanence.TEMPORARY,
            last_confirmed_at=now - 48 * 3600,
        )
    )
    await store.upsert_fact(Fact(scope="user", key="c", value="3", permanence=Permanence.TEMPORARY))
    await store.upsert_fact(Fact(scope="user", key="d", value="4", permanence=Permanence.LONG_TERM))

    removed = await store.forget_expired(now)
    assert removed == 2
    assert {f.key for f in await store.list_facts("user")} == {"c", "d"}


async def test_delete_fact(store: MemoryStore) -> None:
    await store.upsert_fact(Fact(scope="user", key="k", value="v"))
    assert await store.delete_fact("user", "k")
    assert not await store.delete_fact("user", "k")


async def test_audit_is_recorded_and_readable(store: MemoryStore) -> None:
    await store.append_audit(actor="user", action="chat", detail={"n": 1})
    rows = await store.recent_audit()
    assert rows[0]["action"] == "chat"
    assert rows[0]["detail"] == {"n": 1}


def test_build_history_respects_the_budget() -> None:
    turns = [Turn(session_id="s", role=Role.USER, content="x" * 100) for _ in range(10)]
    kept = build_history(turns, char_budget=350)
    assert len(kept) == 3
    # The most recent turns are the ones that survive.
    assert kept[-1] is turns[-1]


def test_build_history_always_keeps_at_least_one_turn() -> None:
    """A single oversized message must not produce an empty prompt."""
    turns = [Turn(session_id="s", role=Role.USER, content="x" * 10_000)]
    assert len(build_history(turns, char_budget=100)) == 1
