"""Tool domain types.

A tool is a value carrying its own schema, permission level and limits, rather
than a bare function. That makes the tool surface data, which is what lets the
registry filter it per session, the executor enforce it uniformly, and the
Phase 4 agent loop hand the model exactly the definition that will actually
run — no drift between what the model is told and what executes.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from aura.security.permissions import PermissionLevel


@dataclass(slots=True, frozen=True)
class ToolSpec:
    """Everything the system and the model need to know about one tool."""

    name: str
    description: str
    # JSON Schema for the arguments. This is what gets sent to the model in
    # Phase 4, so it is the single definition of the tool's interface.
    parameters: dict[str, Any]
    level: PermissionLevel
    timeout_s: float = 30.0
    # Whether the effect can be undone. Informational in Phase 3; the agent
    # loop uses it to decide what needs verifying before it moves on.
    reversible: bool = True
    # Tools that only observe can be retried freely after a failure. Ones that
    # write cannot, because a timeout does not prove nothing happened.
    idempotent: bool = True

    def to_wire(self) -> dict[str, Any]:
        """The shape an OpenAI-compatible model expects in `tools`."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True, frozen=True)
class ToolCall:
    """A request to run one tool."""

    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # Set when a human has approved a call above the autonomous ceiling. The
    # executor refuses anything that needs confirmation without it, so an
    # approval cannot be implied by a caller simply not thinking about it.
    confirmed: bool = False


@dataclass(slots=True, frozen=True)
class ToolResult:
    """The outcome of a tool call.

    Failure is a value, never an exception. The Phase 4 loop reads `error` and
    adapts; an exception crossing this boundary would end the task instead, and
    would skip the audit record on the way out.
    """

    call_id: str
    name: str
    ok: bool
    # Human- and model-readable result. Kept as text because that is what goes
    # back into the conversation as an observation.
    content: str = ""
    error: str | None = None
    # Machine-readable extras: paths touched, exit codes, byte counts.
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    # "allow" when it ran, "confirm" when it needed approval it did not have,
    # "deny" when policy forbade it outright.
    decision: str = "allow"

    @classmethod
    def success(
        cls,
        call: ToolCall,
        content: str,
        data: dict[str, Any] | None = None,
        duration_ms: int = 0,
    ) -> ToolResult:
        return cls(
            call_id=call.id,
            name=call.name,
            ok=True,
            content=content,
            data=data or {},
            duration_ms=duration_ms,
        )

    @classmethod
    def failure(
        cls,
        call: ToolCall,
        error: str,
        decision: str = "allow",
        data: dict[str, Any] | None = None,
        duration_ms: int = 0,
    ) -> ToolResult:
        return cls(
            call_id=call.id,
            name=call.name,
            ok=False,
            error=error,
            data=data or {},
            duration_ms=duration_ms,
            decision=decision,
        )


@dataclass(slots=True)
class ToolContext:
    """Ambient state one call runs against.

    Passed explicitly rather than read from globals, so a test can run a tool
    against a temporary workspace without touching process state.
    """

    workspace_roots: tuple[str, ...] = ()
    deny_patterns: tuple[str, ...] = ()
    session_id: str | None = None
    actor: str = "user"
    started_at: float = field(default_factory=time.time)
    # Where a subprocess tool runs. Defaults to the first workspace root.
    working_dir: str | None = None
    # The kill switch, as a question a tool can ask while it waits (audit 01,
    # finding 4). `ToolExecutor` binds this to its own stop before handing the
    # context to a tool, so a tool that blocks for a minute can notice.
    #
    # Optional because a tool built directly in a test has no switch, and
    # because most tools are short enough that polling buys nothing. A tool
    # that can run for longer than a person will wait should use it.
    should_stop: Callable[[], bool] | None = None
