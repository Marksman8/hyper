"""AURA reaching her own memory.

`search_turns` has worked since Phase 6 and `/v1/memory/search` exposed it to
the *user*. These cover the part that was missing: a tool, so the model can
answer "what did we talk about yesterday" instead of having no way to look.
"""

from __future__ import annotations

import asyncio
import time

from aura.config import Settings
from aura.core.types import Role, Turn
from aura.memory.store import MemoryStore
from aura.security.permissions import Decision, PermissionLevel, PermissionPolicy
from aura.tools.executor import ToolExecutor
from aura.tools.factory import build_context, build_registry
from aura.tools.memory import SearchMemoryTool, memory_tools
from aura.tools.types import ToolCall


async def _seed(store: MemoryStore) -> str:
    session = await store.create_session()
    yesterday = time.time() - 86400
    await store.add_turn(
        Turn(
            session_id=session.id,
            role=Role.USER,
            content="Let's use Piper for the voice rather than the browser synthesiser.",
            created_at=yesterday,
        )
    )
    await store.add_turn(
        Turn(
            session_id=session.id,
            role=Role.ASSISTANT,
            content="Agreed. Piper keeps the same voice on every machine.",
            created_at=yesterday,
        )
    )
    return session.id


async def test_it_finds_an_exchange_from_another_session(store: MemoryStore) -> None:
    """The whole point: a previous conversation, not the current one."""
    await _seed(store)
    tool = SearchMemoryTool(store)
    result = await asyncio.to_thread(
        tool.run,
        ToolCall(name="search_memory", arguments={"query": "piper voice"}),
        None,  # type: ignore[arg-type]
    )

    assert result.ok
    assert "Piper" in result.content
    assert result.data["matches"] >= 1


async def test_it_dates_each_match(store: MemoryStore) -> None:
    """A recalled exchange with no date is worse than none -- the model cannot
    tell last night's decision from one three weeks stale."""
    await _seed(store)
    tool = SearchMemoryTool(store)
    result = await asyncio.to_thread(
        tool.run,
        ToolCall(name="search_memory", arguments={"query": "piper"}),
        None,  # type: ignore[arg-type]
    )

    assert "yesterday" in result.content
    assert time.strftime("%Y-") in result.content, "the absolute date is missing"


async def test_a_miss_says_so_and_says_why(store: MemoryStore) -> None:
    """FTS5 matches words, not meaning. A model that does not know that will
    conclude the memory is empty rather than rephrasing."""
    await _seed(store)
    tool = SearchMemoryTool(store)
    result = await asyncio.to_thread(
        tool.run,
        ToolCall(name="search_memory", arguments={"query": "quantum tunnelling"}),
        None,  # type: ignore[arg-type]
    )

    assert result.ok, "an empty result is not a failure"
    assert result.data["matches"] == 0
    assert "words rather than meaning" in result.content


async def test_punctuation_does_not_become_a_500(store: MemoryStore) -> None:
    """FTS5 MATCH treats bare punctuation as a syntax error; the store already
    guards it, and this asserts the tool inherits that rather than raising."""
    await _seed(store)
    tool = SearchMemoryTool(store)
    result = await asyncio.to_thread(
        tool.run,
        ToolCall(name="search_memory", arguments={"query": "?!*"}),
        None,  # type: ignore[arg-type]
    )
    assert result.ok


async def test_the_limit_is_capped(store: MemoryStore) -> None:
    """A model asking for 5000 matches would bury the conversation it is
    recalling them into."""
    await _seed(store)
    tool = SearchMemoryTool(store)
    result = await asyncio.to_thread(
        tool.run,
        ToolCall(name="search_memory", arguments={"query": "piper", "limit": 5000}),
        None,  # type: ignore[arg-type]
    )
    assert result.ok
    assert result.data["matches"] <= 20


def test_an_empty_query_is_refused_before_it_runs(store: MemoryStore) -> None:
    assert SearchMemoryTool(store).validate({"query": "   "}) is not None


# -- permission and wiring ---------------------------------------------------


def test_reading_history_is_level_read_not_observe(store: MemoryStore) -> None:
    """Level 0 is AURA's own state. Conversation history is the user's data --
    everything they have ever said to her -- so it sits where `read_file` sits."""
    spec = SearchMemoryTool(store).spec
    assert spec.level is PermissionLevel.READ
    assert PermissionPolicy().evaluate(spec.level) is Decision.ALLOW


def test_no_store_means_no_tool() -> None:
    """Offering a search that cannot search is worse than offering nothing: the
    model cannot tell a broken tool from an empty memory."""
    assert memory_tools(None) == []


def test_the_registry_offers_it_only_with_a_store(settings: Settings) -> None:
    assert "search_memory" not in build_registry(settings).names()


def test_the_registry_offers_it_with_a_store(settings: Settings, store: MemoryStore) -> None:
    assert "search_memory" in build_registry(settings, store=store).names()


async def test_it_runs_through_the_executor_without_confirmation(
    settings: Settings, store: MemoryStore
) -> None:
    """Recall must not prompt. A tool that asks permission to remember would
    make her unusable as an assistant."""
    await _seed(store)
    ex = ToolExecutor(build_registry(settings, store=store), PermissionPolicy())
    try:
        result = await ex.execute(
            ToolCall(name="search_memory", arguments={"query": "piper"}),
            build_context(settings),
        )
    finally:
        ex.close()

    assert result.ok, result.error
    assert result.decision == "allow"


def test_she_cannot_write_to_her_own_memory(settings: Settings, store: MemoryStore) -> None:
    """Memory poisoning is the reason. Facts are proposed under Phase 6's
    evidence rules and deleted by the user; a persuasive conversation must not
    be able to rewrite what AURA believes."""
    names = build_registry(settings, store=store).names()
    for forbidden in ("remember", "forget", "write_memory", "update_fact", "delete_fact"):
        assert forbidden not in names
