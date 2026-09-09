"""The chokepoint every tool call passes through.

Validation, permission, timeout and audit live here rather than in the tools,
for one reason: a tool author cannot forget to do them. Every check that lives
inside a tool is a check the next tool might omit.

Order is deliberate:

1. **Resolve** the tool. An unknown name is a caller bug, not a policy event.
2. **Validate** arguments. A malformed call is refused on its own terms rather
   than prompting a human to approve something incoherent.
3. **Permission.** Below the autonomous ceiling it runs; above it, an explicit
   confirmation is required and its absence is a refusal, never an assumption.
4. **Execute** with a hard timeout, in a worker thread so a wedged tool cannot
   take the caller with it.
5. **Audit**, unconditionally — including denials, timeouts and crashes. The
   record of what was refused is the more interesting half.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from aura.core.emitter import NULL_EMITTER, EventEmitter
from aura.logging import get_logger
from aura.security.audit import AuditLog
from aura.security.permissions import Decision, PermissionPolicy
from aura.tools.base import Tool
from aura.tools.registry import ToolNotFoundError, ToolRegistry
from aura.tools.types import ToolCall, ToolContext, ToolResult, ToolSpec
from aura.tools.validate import validate_arguments

log = get_logger(__name__)


class ToolExecutor:
    """Runs tool calls under policy."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: PermissionPolicy,
        audit: AuditLog | None = None,
        max_workers: int = 4,
        events: EventEmitter | None = None,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._audit = audit
        # Phase 10A.3. Announces that a human is needed. It does NOT issue an
        # approval or wait for one -- the executor still refuses immediately,
        # exactly as before. Wiring the issue-approve-resume loop changes the
        # agent's control flow and belongs to 10C.
        self._events = events or NULL_EMITTER
        # Tools are blocking and CPU- or IO-bound. A small pool bounds how many
        # can be in flight, and gives the timeout something to abandon.
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aura-tool")

    @property
    def registry(self) -> ToolRegistry:
        """The tools that exist. Read-only."""
        return self._registry

    @property
    def policy(self) -> PermissionPolicy:
        """The policy in force. Read-only: nothing may widen it at runtime."""
        return self._policy

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # -- the one entry point -------------------------------------------------

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()

        try:
            tool = self._registry.get(call.name)
        except ToolNotFoundError as exc:
            result = ToolResult.failure(call, str(exc), decision="deny")
            await self._record(call, result, level=None)
            return result

        spec = tool.spec

        # -- validate --------------------------------------------------------
        schema_error = validate_arguments(spec.parameters, call.arguments)
        if schema_error is not None:
            result = ToolResult.failure(call, f"Invalid arguments: {schema_error}", decision="deny")
            await self._record(call, result, spec.level)
            return result

        tool_error = tool.validate(call.arguments)
        if tool_error is not None:
            result = ToolResult.failure(call, f"Invalid arguments: {tool_error}", decision="deny")
            await self._record(call, result, spec.level)
            return result

        # -- permission ------------------------------------------------------
        decision = self._policy.evaluate(spec.level)
        if decision is Decision.DENY:
            result = ToolResult.failure(
                call,
                f"{spec.name} requires permission level {spec.level.label} "
                f"({int(spec.level)}), which is denied by policy.",
                decision="deny",
            )
            await self._record(call, result, spec.level)
            return result

        if decision is Decision.CONFIRM and not call.confirmed:
            # Absence of approval is a refusal. Treating an unset flag as
            # consent is how a confirmation gate quietly stops existing.
            result = ToolResult.failure(
                call,
                f"{spec.name} is permission level {spec.level.label} "
                f"({int(spec.level)}) and needs explicit confirmation. "
                "Re-send the call with confirmed=true once a human has approved it.",
                decision="confirm",
            )
            self._events.permission_required(
                tool=spec.name,
                level=int(spec.level),
                summary=_describe(call, spec),
                session_id=ctx.session_id,
            )
            self._events.agent_waiting("confirmation", spec.name, session_id=ctx.session_id)
            await self._record(call, result, spec.level)
            return result

        # -- execute ---------------------------------------------------------
        result = await self._run_guarded(tool, call, ctx, started)
        await self._record(call, result, spec.level)
        return result

    async def _run_guarded(
        self, tool: Tool, call: ToolCall, ctx: ToolContext, started: float
    ) -> ToolResult:
        import asyncio

        spec = tool.spec
        loop = asyncio.get_running_loop()
        future = self._pool.submit(tool.run, call, ctx)

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: future.result(spec.timeout_s)),
                timeout=spec.timeout_s + 5.0,
            )
        except (FutureTimeout, TimeoutError):
            # The worker thread may still be running: Python cannot kill one.
            # Tools that can genuinely hang run subprocesses precisely so the
            # timeout has something it can terminate.
            log.warning("tools.timeout", tool=spec.name, timeout_s=spec.timeout_s)
            return ToolResult.failure(
                call,
                f"{spec.name} exceeded its {spec.timeout_s:.0f}s timeout.",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:
            # A tool that raises is a bug, not an expected failure. Convert it
            # so the audit record still happens and, in Phase 4, so the model
            # sees an observation rather than the loop dying.
            log.error(
                "tools.crashed", tool=spec.name, error=str(exc), error_type=type(exc).__name__
            )
            return ToolResult.failure(
                call,
                f"{spec.name} failed: {type(exc).__name__}: {exc}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        duration_ms = int((time.perf_counter() - started) * 1000)
        # Tools are not required to time themselves.
        if not result.duration_ms:
            result = ToolResult(
                call_id=result.call_id,
                name=result.name,
                ok=result.ok,
                content=result.content,
                error=result.error,
                data=result.data,
                duration_ms=duration_ms,
                decision=result.decision,
            )
        return result

    # -- audit ---------------------------------------------------------------

    async def _record(self, call: ToolCall, result: ToolResult, level: Any = None) -> None:
        log.info(
            "tools.executed",
            tool=call.name,
            ok=result.ok,
            decision=result.decision,
            duration_ms=result.duration_ms,
            error=result.error,
        )
        if self._audit is None:
            return
        from aura.security.permissions import PermissionLevel

        await self._audit.record(
            actor="tool",
            action=f"tool.{call.name}",
            level=level if level is not None else PermissionLevel.OBSERVE,
            decision=Decision(result.decision),
            target=str(call.arguments.get("path") or call.arguments.get("command") or ""),
            detail={
                "call_id": call.id,
                "ok": result.ok,
                "duration_ms": result.duration_ms,
                "error": result.error,
                "confirmed": call.confirmed,
            },
        )


def _describe(call: ToolCall, spec: ToolSpec) -> str:
    """A one-line summary a person can actually decide on.

    "click_element" is not a decision; "click_element(name='Send')" is. The
    arguments are what make consent informed, so they are included here even
    though no other event carries tool arguments -- see the PERMISSION_REQUIRED
    entry in event_catalog.py for why this is the one deliberate exception.

    Values are truncated: an argument can be a whole file's contents, and a
    confirmation dialog that scrolls is one nobody reads.
    """
    parts: list[str] = []
    for key, value in sorted(call.arguments.items()):
        text = str(value)
        if len(text) > 60:
            text = text[:57] + "..."
        parts.append(f"{key}={text!r}")
    args = ", ".join(parts)
    return f"{spec.name}({args})" if args else spec.name
