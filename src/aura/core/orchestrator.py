"""The Orchestrator: owner of a conversational turn.

In v0.1 a turn is one step:

    ensure session -> load history -> assemble prompt -> generate ->
    persist -> audit

In Phase 4 the middle of that becomes a loop (plan, select tool, execute,
observe, evaluate, correct, verify). Everything around it -- session handling,
prompt assembly, persistence, audit, emergency stop -- stays exactly as it is
here. That is why this is a class with a single-step `_generate` rather than a
function: the loop replaces one method, not the file.

The orchestrator knows nothing about which model is in use. It holds an
`LLMProvider` and never inspects its type.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from aura.config import Settings
from aura.core.types import (
    ChatChunk,
    Fact,
    GenerationParams,
    Message,
    Permanence,
    Role,
    Session,
    Turn,
    Usage,
)
from aura.errors import EmergencyStopError
from aura.logging import bind_session_id, get_logger
from aura.memory.store import MemoryStore, build_history
from aura.models.base import LLMProvider
from aura.security.audit import AuditLog
from aura.security.permissions import Decision, PermissionLevel

log = get_logger(__name__)


@dataclass(slots=True)
class TurnResult:
    """Outcome of a completed turn."""

    session_id: str
    content: str
    model: str
    provider: str
    usage: Usage
    finish_reason: str
    latency_ms: int


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        provider: LLMProvider,
        store: MemoryStore,
        audit: AuditLog,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._store = store
        self._audit = audit
        # Runtime-flippable, distinct from the config default, so the emergency
        # stop can be engaged without restarting AURA.
        self._estop = settings.app.emergency_stop

    @property
    def provider(self) -> LLMProvider:
        """The active model backend. Read-only; swapping is a restart."""
        return self._provider

    # -- emergency stop (spec sections 13, 23) -------------------------------

    @property
    def emergency_stop(self) -> bool:
        return self._estop

    def set_emergency_stop(self, engaged: bool) -> None:
        self._estop = engaged
        log.warning("orchestrator.emergency_stop", engaged=engaged)

    def _guard(self) -> None:
        if self._estop:
            raise EmergencyStopError(
                "Emergency stop is engaged. No model calls will be made. "
                'Release it with POST /v1/control/estop {"engaged": false}.'
            )

    # -- prompt assembly -----------------------------------------------------

    async def _system_messages(self, *, spoken: bool = False) -> list[Message]:
        """System prompt plus whatever AURA durably knows about the user.

        Facts are injected as text rather than fine-tuned in (spec section 15:
        do not retrain the LLM for every preference). Only long-lived,
        well-evidenced facts are included, so a one-off "be brief today" never
        leaks into every future conversation.
        """
        parts = [self._settings.persona.system_prompt.strip()]
        if spoken:
            parts.append(self._settings.persona.voice_system_prompt.strip())

        facts = await self._store.list_facts(scope="user")
        relevant = [
            f
            for f in facts
            if f.confidence >= 0.5 and f.permanence.value in ("long_term", "permanent")
        ]
        if relevant:
            lines = ["What you know about this user:"]
            lines += ["- " + f.key + ": " + f.value for f in relevant[:20]]
            parts.append("\n".join(lines))

        return [Message(role=Role.SYSTEM, content="\n\n".join(parts))]

    async def _assemble(self, session_id: str, *, spoken: bool = False) -> list[Message]:
        """Build the prompt from the system message plus stored history.

        The current user message is NOT appended here: _open_turn has already
        persisted it, so it is the last row `recent_turns` returns. Appending
        it again sent it to the model twice.
        """
        history = await self._store.recent_turns(session_id, self._settings.memory.history_turns)
        history = build_history(history, self._settings.memory.history_char_budget)

        messages = await self._system_messages(spoken=spoken)
        messages += [t.to_message() for t in history]
        return messages

    # -- turn execution ------------------------------------------------------

    async def _open_turn(
        self, session_id: str | None, user_text: str, *, spoken: bool = False
    ) -> tuple[Session, list[Message]]:
        """Shared prologue for streaming and non-streaming turns."""
        self._guard()
        session = await self._store.ensure_session(session_id)
        bind_session_id(session.id)

        await self._store.add_turn(Turn(session_id=session.id, role=Role.USER, content=user_text))
        messages = await self._assemble(session.id, spoken=spoken)
        return session, messages

    async def chat(
        self,
        user_text: str,
        *,
        session_id: str | None = None,
        params: GenerationParams | None = None,
        spoken: bool = False,
    ) -> TurnResult:
        session, messages = await self._open_turn(session_id, user_text, spoken=spoken)
        started = time.perf_counter()

        try:
            result = await self._provider.chat(messages, params)
        except Exception as exc:
            await self._record_failure(session.id, exc)
            raise

        latency_ms = result.latency_ms or int((time.perf_counter() - started) * 1000)
        await self._store.add_turn(
            Turn(
                session_id=session.id,
                role=Role.ASSISTANT,
                content=result.content,
                model=result.model,
                provider=result.provider,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                latency_ms=latency_ms,
                finish_reason=result.finish_reason,
                # Stored, never replayed. Turn.to_message() sends only
                # `content` back to the model, which is what reasoning models
                # expect: thinking from a previous turn is not context.
                metadata={"reasoning": result.reasoning} if result.reasoning else {},
            )
        )
        await self._audit.record(
            actor="user",
            action="chat",
            level=PermissionLevel.OBSERVE,
            session_id=session.id,
            target=result.model,
            detail={
                "provider": result.provider,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "latency_ms": latency_ms,
                "history_messages": len(messages),
            },
        )
        log.info(
            "turn.complete",
            model=result.model,
            latency_ms=latency_ms,
            completion_tokens=result.usage.completion_tokens,
        )

        return TurnResult(
            session_id=session.id,
            content=result.content,
            model=result.model,
            provider=result.provider,
            usage=result.usage,
            finish_reason=result.finish_reason,
            latency_ms=latency_ms,
        )

    async def stream(
        self,
        user_text: str,
        *,
        session_id: str | None = None,
        params: GenerationParams | None = None,
        spoken: bool = False,
    ) -> AsyncIterator[ChatChunk]:
        """Stream a turn, persisting the assistant message once complete.

        Partial output is persisted on failure too: a half-finished answer that
        the user saw and reacted to is part of the conversation, and silently
        dropping it desynchronises AURA's memory from the user's.
        """
        session, messages = await self._open_turn(session_id, user_text, spoken=spoken)
        started = time.perf_counter()
        collected: list[str] = []
        reasoning: list[str] = []
        usage = Usage()
        finish_reason = "stop"

        try:
            async for chunk in self._provider.stream(messages, params):
                if chunk.delta:
                    collected.append(chunk.delta)
                if chunk.reasoning_delta:
                    reasoning.append(chunk.reasoning_delta)
                if chunk.done:
                    finish_reason = chunk.finish_reason or "stop"
                    if chunk.usage is not None:
                        usage = chunk.usage
                yield chunk
        except asyncio.CancelledError:
            # Client disconnected. Keep what was already generated: the user
            # may have read it, so it is part of the conversation.
            await self._persist_assistant(
                session.id,
                "".join(collected),
                usage,
                "cancelled",
                started,
                reasoning="".join(reasoning),
            )
            raise
        except Exception as exc:
            await self._persist_assistant(
                session.id,
                "".join(collected),
                usage,
                "error",
                started,
                error=str(exc),
                reasoning="".join(reasoning),
            )
            await self._record_failure(session.id, exc)
            raise
        else:
            await self._persist_assistant(
                session.id,
                "".join(collected),
                usage,
                finish_reason,
                started,
                reasoning="".join(reasoning),
            )

    async def _persist_assistant(
        self,
        session_id: str,
        content: str,
        usage: Usage,
        finish_reason: str,
        started: float,
        error: str | None = None,
        reasoning: str = "",
    ) -> None:
        latency_ms = int((time.perf_counter() - started) * 1000)
        info = self._provider.info()
        await self._store.add_turn(
            Turn(
                session_id=session_id,
                role=Role.ASSISTANT,
                content=content,
                model=info.model or "unknown",
                provider=self._provider.name,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                error=error,
                metadata={"reasoning": reasoning} if reasoning else {},
            )
        )
        if error is None:
            await self._audit.record(
                actor="user",
                action="chat_stream",
                session_id=session_id,
                target=info.model,
                detail={
                    "provider": self._provider.name,
                    "latency_ms": latency_ms,
                    "completion_tokens": usage.completion_tokens,
                },
            )

    async def _record_failure(self, session_id: str, exc: Exception) -> None:
        log.error("turn.failed", error=str(exc), error_type=type(exc).__name__)
        # Audit is best-effort here: the request already failed, and raising an
        # audit error would mask the original cause.
        try:
            await self._audit.record(
                actor="user",
                action="chat",
                session_id=session_id,
                decision=Decision.ERROR,
                detail={"error": str(exc), "error_type": type(exc).__name__},
            )
        except Exception as audit_exc:
            log.error("turn.audit_failed", error=str(audit_exc))

    # -- explicit memory commands (spec section 16) --------------------------

    async def remember(self, key: str, value: str, permanence: str = "long_term") -> Fact:
        fact = Fact(
            scope="user",
            key=key,
            value=value,
            source="explicit",
            permanence=Permanence(permanence),
        )
        await self._store.upsert_fact(fact)
        await self._audit.record(
            actor="user", action="memory.remember", target=key, detail={"value": value}
        )
        return fact

    async def forget(self, key: str) -> bool:
        deleted = await self._store.delete_fact("user", key)
        await self._audit.record(actor="user", action="memory.forget", target=key)
        return deleted

    async def known_facts(self) -> Sequence[Fact]:
        return await self._store.list_facts(scope="user")
