"""Orchestrator behaviour: what actually gets sent to the model.

These tests use a recording provider so they can assert on the assembled
prompt, which is the part of a turn that no API-level test can see.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest

from aura.config import Settings
from aura.core.orchestrator import Orchestrator
from aura.core.types import (
    ChatChunk,
    ChatResult,
    GenerationParams,
    Message,
    ProviderHealth,
    ProviderInfo,
    Role,
    Turn,
)
from aura.errors import EmergencyStopError
from aura.memory.store import MemoryStore
from aura.models.base import LLMProvider
from aura.security.audit import AuditLog


class RecordingProvider(LLMProvider):
    """Captures the exact messages each call received."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    @property
    def name(self) -> str:
        return "recording"

    def info(self) -> ProviderInfo:
        return ProviderInfo(name=self.name, kind="test", model="recording-v1")

    async def health(self) -> ProviderHealth:
        return ProviderHealth(available=True)

    async def chat(
        self, messages: Sequence[Message], params: GenerationParams | None = None
    ) -> ChatResult:
        self.calls.append(list(messages))
        return ChatResult(content="ok", model="recording-v1", provider=self.name)

    async def stream(
        self, messages: Sequence[Message], params: GenerationParams | None = None
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append(list(messages))
        yield ChatChunk(delta="ok")
        yield ChatChunk(delta="", done=True, finish_reason="stop")


@pytest.fixture
def recorder() -> RecordingProvider:
    return RecordingProvider()


@pytest.fixture
def orch(
    settings: Settings, store: MemoryStore, recorder: RecordingProvider
) -> Orchestrator:
    return Orchestrator(settings, recorder, store, AuditLog(store))


async def test_system_prompt_is_always_first(
    orch: Orchestrator, recorder: RecordingProvider
) -> None:
    await orch.chat("hello")
    sent = recorder.calls[0]
    assert sent[0].role is Role.SYSTEM
    assert "AURA" in sent[0].content


async def test_long_term_facts_are_injected_into_the_system_prompt(
    orch: Orchestrator, recorder: RecordingProvider
) -> None:
    """Spec section 15: preferences reach the model via prompt, not training."""
    await orch.remember("preferred_language", "Python", permanence="long_term")
    await orch.chat("what should I use?")

    system = recorder.calls[0][0].content
    assert "preferred_language" in system
    assert "Python" in system


async def test_temporary_facts_do_not_leak_into_the_prompt(
    orch: Orchestrator, recorder: RecordingProvider
) -> None:
    """Spec section 16: 'today, be brief' must not become a standing rule."""
    await orch.remember("tone_today", "very terse", permanence="temporary")
    await orch.chat("hi")

    system = recorder.calls[0][0].content
    assert "tone_today" not in system
    assert "very terse" not in system


async def test_history_is_replayed_in_order(
    orch: Orchestrator, recorder: RecordingProvider
) -> None:
    result = await orch.chat("first")
    await orch.chat("second", session_id=result.session_id)

    roles = [m.role for m in recorder.calls[1]]
    contents = [m.content for m in recorder.calls[1]]
    assert roles == [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.USER]
    assert contents[1:] == ["first", "ok", "second"]


async def test_history_is_trimmed_to_the_configured_turn_count(
    settings: Settings, store: MemoryStore, recorder: RecordingProvider
) -> None:
    """history_turns is 6 in the test fixture."""
    orch = Orchestrator(settings, recorder, store, AuditLog(store))
    session = await store.create_session()
    for i in range(20):
        await store.add_turn(
            Turn(session_id=session.id, role=Role.USER, content=f"old-{i}")
        )

    await orch.chat("new", session_id=session.id)
    replayed = [m.content for m in recorder.calls[0] if m.role is Role.USER]
    # The window is the 6 newest turns, which includes the message just sent.
    assert len(replayed) == 6
    assert replayed[0] == "old-15"
    assert replayed[-1] == "new"
    assert "old-0" not in replayed


async def test_emergency_stop_prevents_any_provider_call(
    orch: Orchestrator, recorder: RecordingProvider
) -> None:
    orch.set_emergency_stop(True)
    with pytest.raises(EmergencyStopError):
        await orch.chat("hello")
    assert recorder.calls == [], "the model must not be reached at all"


async def test_emergency_stop_also_blocks_streaming(
    orch: Orchestrator, recorder: RecordingProvider
) -> None:
    orch.set_emergency_stop(True)
    with pytest.raises(EmergencyStopError):
        async for _ in orch.stream("hello"):
            pass
    assert recorder.calls == []


async def test_streamed_turns_are_persisted_with_usage(
    orch: Orchestrator, store: MemoryStore
) -> None:
    chunks = [c async for c in orch.stream("hello")]
    assert "".join(c.delta for c in chunks) == "ok"

    sessions = await store.list_sessions()
    turns = await store.recent_turns(sessions[0].id, 10)
    assert [t.role for t in turns] == [Role.USER, Role.ASSISTANT]
    assert turns[-1].content == "ok"
    assert turns[-1].provider == "recording"


async def test_a_failing_provider_still_records_the_user_turn(
    settings: Settings, store: MemoryStore
) -> None:
    """A failed turn must not vanish from history."""

    class Broken(RecordingProvider):
        async def chat(
            self, messages: Sequence[Message], params: GenerationParams | None = None
        ) -> ChatResult:
            raise RuntimeError("model exploded")

    orch = Orchestrator(settings, Broken(), store, AuditLog(store))
    with pytest.raises(RuntimeError):
        await orch.chat("this will fail")

    sessions = await store.list_sessions()
    turns = await store.recent_turns(sessions[0].id, 10)
    assert [t.content for t in turns] == ["this will fail"]

    audit = await store.recent_audit()
    assert any(r["decision"] == "error" for r in audit)
