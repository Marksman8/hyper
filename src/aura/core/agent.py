"""The agent loop.

    goal -> plan -> select tool -> execute -> observe -> evaluate
         -> correct -> continue -> verify -> complete

In practice that is one loop: ask the model, run whatever tools it asked for,
give it the results, ask again. Planning, evaluation and correction are things
the model does *inside* that loop rather than separate stages AURA orchestrates
— an explicit plan/execute/verify state machine sounds more rigorous and mostly
produces a rigid agent that cannot revise a plan after the first surprise.

What AURA owns instead is everything that must not be left to the model:

- **A step budget and a wall clock.** A loop that decides when to stop is a
  loop that sometimes does not.
- **The permission gate.** Every call goes through the Phase 3 executor, so a
  model asking to write a file gets the same refusal a human caller would.
  Nothing here can approve anything.
- **The emergency stop**, checked every iteration rather than once at entry.
- **A trace.** Every step is recorded, so "why did AURA do this?" has an
  answer that is not a guess.

Failure is fed back rather than raised. A tool that errors, a call that is
refused, an argument that does not validate — each becomes an observation the
model reads and can work around. That is the difference between an agent that
recovers and one that stops at the first obstacle.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from aura.core.types import (
    ChatResult,
    GenerationParams,
    Message,
    Role,
    ToolInvocation,
    Usage,
)
from aura.errors import EmergencyStopError
from aura.logging import get_logger
from aura.models.base import LLMProvider

log = get_logger(__name__)


class ToolRunner(Protocol):
    """What the loop needs from the tool layer.

    A Protocol rather than an import of `ToolExecutor`, so `aura.core` does not
    depend on `aura.tools` and the loop can be tested with a stub that never
    touches a filesystem.
    """

    async def run(
        self, name: str, arguments: dict[str, Any], *, confirmed: bool
    ) -> tuple[bool, str, str]:
        """Execute one call. Returns (ok, content_or_error, decision)."""
        ...

    def wire_specs(self) -> list[dict[str, Any]]:
        """Tool definitions in the format the model expects."""
        ...

    def needs_confirmation(self, name: str) -> bool:
        """Whether this tool is above the autonomous ceiling."""
        ...


@dataclass(slots=True)
class AgentStep:
    """One iteration, for the trace."""

    index: int
    kind: str  # "tool" | "answer" | "stopped"
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0


@dataclass(slots=True)
class AgentOutcome:
    """What the loop produced."""

    content: str
    steps: list[AgentStep] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    provider: str = ""
    tool_calls_made: int = 0
    # Set when the loop ran out of budget rather than finishing. The caller can
    # tell the difference between "done" and "gave up", which matters: an
    # answer produced after hitting the ceiling may be half a task.
    stopped_early: bool = False
    stop_reason: str = "complete"
    # Tools the model wanted but could not use without a human. Surfaced so a
    # UI can offer the approval rather than the user having to guess.
    pending_confirmations: list[str] = field(default_factory=list)
    latency_ms: int = 0


class AgentLoop:
    """Runs a turn that may use tools."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRunner,
        max_steps: int = 8,
        max_seconds: float = 180.0,
        approved_tools: Sequence[str] = (),
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._max_steps = max_steps
        self._max_seconds = max_seconds
        # Named by a human, per task, ahead of time. The model cannot add to
        # this set — that is the whole point. An empty set means every
        # level-2-and-above call comes back as a refusal the model must
        # explain rather than route around.
        self._approved = frozenset(approved_tools)

    async def run(
        self,
        messages: list[Message],
        params: GenerationParams | None = None,
        should_stop: Any = None,
    ) -> AgentOutcome:
        """Drive the loop until the model answers, or the budget runs out.

        `messages` is mutated as the conversation grows, so the caller can
        persist the intermediate tool traffic if it wants to.
        """
        started = time.perf_counter()
        specs = self._tools.wire_specs()
        steps: list[AgentStep] = []
        usage = Usage()
        pending: list[str] = []
        last: ChatResult | None = None

        for index in range(self._max_steps):
            # Checked every iteration, not once at entry: a loop that can run
            # for minutes must be stoppable during those minutes.
            if should_stop is not None and should_stop():
                raise EmergencyStopError("Emergency stop engaged during a tool-using turn.")

            elapsed = time.perf_counter() - started
            if elapsed > self._max_seconds:
                log.warning("agent.time_budget_exhausted", elapsed_s=round(elapsed, 1))
                steps.append(AgentStep(index=index, kind="stopped", content="time budget"))
                return self._finish(
                    steps, usage, last, pending, started, "time_budget", stopped=True
                )

            step_started = time.perf_counter()
            result = await self._provider.chat(messages, params, tools=specs or None)
            last = result
            usage = Usage(
                prompt_tokens=usage.prompt_tokens + result.usage.prompt_tokens,
                completion_tokens=usage.completion_tokens + result.usage.completion_tokens,
            )
            step_ms = int((time.perf_counter() - step_started) * 1000)

            if not result.tool_calls:
                steps.append(
                    AgentStep(
                        index=index, kind="answer", content=result.content, latency_ms=step_ms
                    )
                )
                log.info("agent.answered", steps=len(steps), tool_calls=_count_tools(steps))
                return self._finish(steps, usage, last, pending, started, "complete")

            # The assistant turn that asked for tools has to go back verbatim,
            # or the tool results that follow reference calls the model cannot
            # see and the conversation stops making sense.
            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=result.content,
                    tool_calls=result.tool_calls,
                )
            )

            step = AgentStep(index=index, kind="tool", latency_ms=step_ms)
            for invocation in result.tool_calls:
                observation, ok, decision = await self._invoke(invocation)
                if decision == "confirm" and invocation.name not in pending:
                    pending.append(invocation.name)
                step.tool_calls.append(
                    {
                        "name": invocation.name,
                        "arguments": invocation.arguments,
                        "ok": ok,
                        "decision": decision,
                    }
                )
                messages.append(
                    Message(
                        role=Role.TOOL,
                        content=observation,
                        tool_call_id=invocation.id,
                        name=invocation.name,
                    )
                )
            steps.append(step)

        # Out of steps. The model is told, so its final answer can acknowledge
        # that the task is unfinished rather than inventing a conclusion.
        log.warning("agent.step_budget_exhausted", max_steps=self._max_steps)
        messages.append(
            Message(
                role=Role.USER,
                content=(
                    f"You have used all {self._max_steps} available tool steps. "
                    "Answer now with what you have found, and say plainly what "
                    "is still unresolved. Do not request more tools."
                ),
            )
        )
        final = await self._provider.chat(messages, params, tools=None)
        usage = Usage(
            prompt_tokens=usage.prompt_tokens + final.usage.prompt_tokens,
            completion_tokens=usage.completion_tokens + final.usage.completion_tokens,
        )
        steps.append(AgentStep(index=self._max_steps, kind="answer", content=final.content))
        return self._finish(steps, usage, final, pending, started, "step_budget", stopped=True)

    # -- one tool call -------------------------------------------------------

    async def _invoke(self, invocation: ToolInvocation) -> tuple[str, bool, str]:
        """Run one call and turn the outcome into an observation."""
        confirmed = invocation.name in self._approved
        log.info(
            "agent.tool_call",
            tool=invocation.name,
            arguments=invocation.arguments,
            pre_approved=confirmed,
        )
        ok, body, decision = await self._tools.run(
            invocation.name, invocation.arguments, confirmed=confirmed
        )

        if ok:
            return body, True, decision

        # Refusals and errors are both fed back as text. The model gets to see
        # exactly why, which is what lets it try a different path instead of
        # repeating the same call.
        if decision == "confirm":
            return (
                f"REFUSED: {body} This action needs the user's approval, which "
                "has not been given for this task. Do not retry it — explain to "
                "the user what you would like to do and why.",
                False,
                decision,
            )
        if decision == "deny":
            return (
                f"DENIED: {body} Do not retry this call; find another approach "
                "or tell the user it is not possible.",
                False,
                decision,
            )
        return f"ERROR: {body}", False, decision

    def _finish(
        self,
        steps: list[AgentStep],
        usage: Usage,
        last: ChatResult | None,
        pending: list[str],
        started: float,
        stop_reason: str,
        stopped: bool = False,
    ) -> AgentOutcome:
        content = ""
        for step in reversed(steps):
            if step.kind == "answer" and step.content:
                content = step.content
                break
        return AgentOutcome(
            content=content,
            steps=steps,
            usage=usage,
            model=last.model if last else "",
            provider=last.provider if last else "",
            tool_calls_made=_count_tools(steps),
            stopped_early=stopped,
            stop_reason=stop_reason,
            pending_confirmations=pending,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def _count_tools(steps: Sequence[AgentStep]) -> int:
    return sum(len(s.tool_calls) for s in steps)
