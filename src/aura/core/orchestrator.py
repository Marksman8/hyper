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
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from aura.config import Settings
from aura.core.agent import AgentLoop, AgentOutcome
from aura.core.types import (
    ChatChunk,
    Evidence,
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
from aura.logging import bind_session_id, current_session_id, get_logger
from aura.memory.store import MemoryStore, build_history
from aura.models.base import LLMProvider
from aura.personalization import UserProfile, apply_parameters, build_profile, directives
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
    # Present only for a tool-using turn. Carries the step trace so a caller
    # can show what happened rather than only the final sentence.
    agent: Any = None


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        provider: LLMProvider,
        store: MemoryStore,
        audit: AuditLog,
        tools: Any = None,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._store = store
        self._audit = audit
        # An `aura.core.agent.ToolRunner`, or None when tools are off. Typed
        # loosely on purpose: the orchestrator does not import aura.tools.
        self._tools = tools
        # Runtime-flippable, distinct from the config default, so the emergency
        # stop can be engaged without restarting AURA.
        self._estop = settings.app.emergency_stop
        # Phase 6. Built lazily so a provider that cannot chat never blocks boot.
        self._extractor: Any = None
        if settings.memory.learn_from_conversation:
            from aura.memory.extraction import FactExtractor

            self._extractor = FactExtractor(provider, store)
        self._learning: set[asyncio.Task[None]] = set()
        self._last_user_turn_id: str | None = None

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

    async def _system_messages(
        self, *, spoken: bool = False, with_tools: bool = False, personalize: bool = True
    ) -> list[Message]:
        """System prompt plus whatever AURA durably knows about the user.

        Facts are injected as text rather than fine-tuned in (spec section 15:
        do not retrain the LLM for every preference). Only long-lived,
        well-evidenced facts are included, so a one-off "be brief today" never
        leaks into every future conversation.
        """
        parts = [self._settings.persona.system_prompt.strip()]
        if with_tools:
            parts.append(self._settings.persona.agent_system_prompt.strip())
        if spoken:
            parts.append(self._settings.persona.voice_system_prompt.strip())

        profile = await self.user_profile()

        # Facts AURA has no machinery for are still context: a closed
        # vocabulary that dropped `editor: neovim` would make AURA worse at
        # knowing the user in exchange for being better at formatting.
        if profile.other_facts:
            lines = ["What you know about this user:"]
            lines += ["- " + f.key + ": " + f.value for f in profile.other_facts[:20]]
            parts.append("\n".join(lines))

        # Directives go LAST, and are phrased as instructions rather than as
        # facts. "Keep answers brief" is acted on; "response_length: short" is
        # trivia the model must first interpret, competing with everything
        # above it for attention.
        if personalize and self._settings.personalization.enabled:
            adaptations = directives(profile)
            if adaptations:
                parts.append("How this user likes you to answer:\n" + "\n".join(adaptations))

        return [Message(role=Role.SYSTEM, content="\n\n".join(parts))]

    async def user_profile(self) -> UserProfile:
        """What AURA should do differently for this user, derived fresh.

        Not cached: a preference deleted mid-conversation must stop applying
        immediately (spec section 23), and a cached profile would keep it alive
        until the next restart.
        """
        facts = await self._store.list_facts(scope="user")
        return build_profile(facts, floor=self._settings.memory.injection_floor)

    async def _personalise(
        self, params: GenerationParams | None
    ) -> GenerationParams | None:
        """Apply the sampling changes the profile justifies.

        Only explicit preferences have any effect here; see
        personalization/profile.py for why prompts and parameters are given
        different evidence requirements.
        """
        cfg = self._settings.personalization
        if not cfg.enabled or not cfg.adapt_parameters:
            return params

        # None means "use the backend's configured defaults". Adapting starts
        # from those rather than from a fresh object, so personalization tunes
        # the configuration instead of quietly replacing it.
        base = params if params is not None else self._provider.default_params
        adapted = apply_parameters(await self.user_profile(), base)
        # Nothing changed: hand back None so the provider keeps its own path.
        if params is None and adapted is base:
            return None
        return adapted

    async def _assemble(
        self,
        session_id: str,
        *,
        spoken: bool = False,
        with_tools: bool = False,
        personalize: bool = True,
    ) -> list[Message]:
        """Build the prompt from the system message plus stored history.

        The current user message is NOT appended here: _open_turn has already
        persisted it, so it is the last row `recent_turns` returns. Appending
        it again sent it to the model twice.
        """
        history = await self._store.recent_turns(session_id, self._settings.memory.history_turns)
        history = build_history(history, self._settings.memory.history_char_budget)

        messages = await self._system_messages(
            spoken=spoken, with_tools=with_tools, personalize=personalize
        )
        messages += [t.to_message() for t in history]
        return messages

    # -- turn execution ------------------------------------------------------

    async def _open_turn(
        self,
        session_id: str | None,
        user_text: str,
        *,
        spoken: bool = False,
        with_tools: bool = False,
        personalize: bool = True,
    ) -> tuple[Session, list[Message]]:
        """Shared prologue for streaming and non-streaming turns."""
        self._guard()
        session = await self._store.ensure_session(session_id)
        bind_session_id(session.id)

        turn = await self._store.add_turn(
            Turn(session_id=session.id, role=Role.USER, content=user_text)
        )
        # Remembered so extraction can cite the exact turn a belief came from.
        # Evidence that cannot point at a turn cannot be deduplicated, and
        # section 16's "separate occasions" rule depends on that.
        self._last_user_turn_id = turn.id
        messages = await self._assemble(
            session.id, spoken=spoken, with_tools=with_tools, personalize=personalize
        )
        return session, messages

    # -- learning (Phase 6) ---------------------------------------------------

    def _learn_later(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """Schedule fact extraction without making the user wait for it.

        Extraction is a second LLM call. Awaiting it here would add its latency
        to every reply to power something the user cannot see happen, which is
        a bad trade -- so it runs as a background task.

        The task is held in a set because asyncio keeps only a weak reference:
        an unreferenced task can be garbage-collected mid-flight, which shows up
        as learning that works until the machine is busy.
        """
        if self._extractor is None or not self._settings.memory.learn_from_conversation:
            return
        if not user_text.strip() or not assistant_text.strip():
            return

        turn_id = self._last_user_turn_id

        async def _run() -> None:
            try:
                await self._extractor.learn(
                    user_text, assistant_text, session_id=session_id, turn_id=turn_id
                )
            except Exception as exc:
                log.warning("memory.learn_task_failed", error=str(exc))

        task = asyncio.create_task(_run())
        self._learning.add(task)
        task.add_done_callback(self._learning.discard)

    async def drain_learning(self) -> None:
        """Wait for outstanding extraction. For shutdown and for tests."""
        while self._learning:
            await asyncio.gather(*tuple(self._learning), return_exceptions=True)

    async def chat(
        self,
        user_text: str,
        *,
        session_id: str | None = None,
        params: GenerationParams | None = None,
        spoken: bool = False,
        personalize: bool = True,
    ) -> TurnResult:
        """`personalize=False` answers as the unadapted persona.

        Exists so adaptation can be switched off and therefore blamed -- and
        because the A/B harness needs both sides of the comparison from one
        running AURA.
        """
        session, messages = await self._open_turn(
            session_id, user_text, spoken=spoken, personalize=personalize
        )
        if personalize:
            params = await self._personalise(params)
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
        self._learn_later(session.id, user_text, result.content)
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

    async def act(
        self,
        user_text: str,
        *,
        session_id: str | None = None,
        params: GenerationParams | None = None,
        approve: Sequence[str] = (),
        spoken: bool = False,
    ) -> TurnResult:
        """Run a turn that may use tools (spec section 7).

        `approve` names tools a human has authorised for this task, ahead of
        time. Nothing the model produces can add to that list — an approval is
        something a person grants, and the loop has no way to grant one to
        itself.

        Falls back to a plain turn when tools are unavailable, so a caller does
        not have to check first.
        """
        if self._tools is None or not self._settings.agent.enabled:
            return await self.chat(user_text, session_id=session_id, params=params, spoken=spoken)

        session, messages = await self._open_turn(
            session_id, user_text, spoken=spoken, with_tools=True
        )
        cfg = self._settings.agent
        loop = AgentLoop(
            provider=self._provider,
            tools=self._tools,
            max_steps=cfg.max_steps,
            max_seconds=cfg.max_seconds,
            approved_tools=approve,
        )
        # Lower temperature than conversation: picking a tool and its arguments
        # is a decision, and creativity there produces malformed calls.
        tool_params = (params or self._default_params()).merged(temperature=cfg.tool_temperature)

        try:
            outcome: AgentOutcome = await loop.run(
                messages, tool_params, should_stop=lambda: self._estop
            )
        except Exception as exc:
            await self._record_failure(session.id, exc)
            raise

        await self._store.add_turn(
            Turn(
                session_id=session.id,
                role=Role.ASSISTANT,
                content=outcome.content,
                model=outcome.model,
                provider=outcome.provider,
                prompt_tokens=outcome.usage.prompt_tokens,
                completion_tokens=outcome.usage.completion_tokens,
                latency_ms=outcome.latency_ms,
                finish_reason=outcome.stop_reason,
                # The trace is the answer to "why did AURA do this?", so it is
                # stored with the turn rather than only logged.
                metadata={
                    "agent": True,
                    "tool_calls": outcome.tool_calls_made,
                    "steps": [
                        {"kind": st.kind, "tools": [c["name"] for c in st.tool_calls]}
                        for st in outcome.steps
                    ],
                    "stopped_early": outcome.stopped_early,
                    "pending_confirmations": outcome.pending_confirmations,
                },
            )
        )
        await self._audit.record(
            actor="user",
            action="agent.turn",
            level=PermissionLevel.OBSERVE,
            session_id=session.id,
            target=outcome.model,
            detail={
                "tool_calls": outcome.tool_calls_made,
                "steps": len(outcome.steps),
                "stop_reason": outcome.stop_reason,
                "approved": list(approve),
                "latency_ms": outcome.latency_ms,
            },
        )
        log.info(
            "agent.turn_complete",
            tool_calls=outcome.tool_calls_made,
            steps=len(outcome.steps),
            stop_reason=outcome.stop_reason,
            latency_ms=outcome.latency_ms,
        )

        return TurnResult(
            session_id=session.id,
            content=outcome.content,
            model=outcome.model,
            provider=outcome.provider,
            usage=outcome.usage,
            finish_reason=outcome.stop_reason,
            latency_ms=outcome.latency_ms,
            agent=outcome,
        )

    def _default_params(self) -> GenerationParams:
        cfg = self._settings.llm
        return GenerationParams(
            temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_tokens
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
        params = await self._personalise(params)
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
            # Only on the success path. A cancelled or failed generation is a
            # half-finished exchange, and inferring durable preferences from
            # one would be learning from something that did not happen.
            self._learn_later(session.id, user_text, "".join(collected))

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
        stored = await self._store.upsert_fact(fact)
        # Explicit evidence, not just a fact row. Without this the most common
        # case -- the user saying "remember that ..." -- would have nothing to
        # show under `aura memory why`, and confidence would be a number with
        # no justification behind it. "You told AURA this directly" is a
        # perfectly good reason; it just has to be recorded as one.
        await self._store.add_evidence(
            Evidence(
                fact_id=stored.id,
                quote=value,
                session_id=current_session_id(),
                # A distinct turn id per statement, so saying it again on
                # another occasion counts as separate corroboration rather
                # than collapsing into one.
                turn_id=self._last_user_turn_id or f"remember:{uuid.uuid4().hex[:12]}",
                source="explicit",
            )
        )
        await self._audit.record(
            actor="user", action="memory.remember", target=key, detail={"value": value}
        )
        # Re-read: adding evidence re-derives confidence, so the object built
        # before that call is already stale. Returning it would have the API
        # report 1.0 for a fact stored at 0.95.
        return await self._store.get_fact("user", key) or stored

    async def forget(self, key: str) -> bool:
        deleted = await self._store.delete_fact("user", key)
        await self._audit.record(actor="user", action="memory.forget", target=key)
        return deleted

    async def known_facts(self) -> Sequence[Fact]:
        return await self._store.list_facts(scope="user")
