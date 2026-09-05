"""Adapter between the agent loop and the tool executor.

The loop declares what it needs as a Protocol (`aura.core.agent.ToolRunner`)
and this satisfies it. The indirection is what keeps `aura.core` from importing
`aura.tools`: the loop knows it can run a named thing and get back a string,
and nothing more.

It is also the point where a model-produced `ToolInvocation` becomes an
authorised `ToolCall`. That conversion is a single, obvious line, so the
question "who decided this was allowed to run?" has one place to look.
"""

from __future__ import annotations

from typing import Any

from aura.logging import get_logger
from aura.security.permissions import Decision
from aura.tools.executor import ToolExecutor
from aura.tools.types import ToolCall, ToolContext

log = get_logger(__name__)

# What a tool may return before it starts crowding out the conversation. A
# directory listing or a long file can otherwise consume the whole context in
# one step and leave no room for the model to reason about it.
MAX_OBSERVATION_CHARS = 8_000


class ExecutorToolRunner:
    """Runs agent tool calls through the Phase 3 executor."""

    def __init__(self, executor: ToolExecutor, ctx: ToolContext) -> None:
        self._executor = executor
        self._ctx = ctx

    def wire_specs(self) -> list[dict[str, Any]]:
        """Every registered tool, in the model's format.

        Tools above the ceiling are still advertised, deliberately: the model
        should know that writing a file is a thing it could ask for, so it can
        propose it and let the user approve. Hiding it produces an assistant
        that claims it cannot do something it can.
        """
        return [spec.to_wire() for spec in self._executor.registry.specs()]

    def needs_confirmation(self, name: str) -> bool:
        try:
            spec = self._executor.registry.get(name).spec
        except KeyError:
            return False
        return self._executor.policy.evaluate(spec.level) is Decision.CONFIRM

    async def run(
        self, name: str, arguments: dict[str, Any], *, confirmed: bool
    ) -> tuple[bool, str, str]:
        # The one place a model's request becomes an authorised call. `confirmed`
        # comes from the human-supplied approval list in the agent loop, never
        # from anything the model produced.
        call = ToolCall(name=name, arguments=arguments, confirmed=confirmed)
        result = await self._executor.execute(call, self._ctx)
        body = result.content if result.ok else (result.error or "unknown error")
        return result.ok, _clip(body), result.decision


def _clip(text: str) -> str:
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text
    # Keep both ends: the head says what the thing is, the tail usually carries
    # the error or the conclusion.
    half = MAX_OBSERVATION_CHARS // 2
    return (
        f"{text[:half]}\n\n[... {len(text):,} characters total, middle omitted ...]\n\n"
        f"{text[-half:]}"
    )
