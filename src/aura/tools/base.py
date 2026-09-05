"""The Tool interface.

Same discipline as `LLMProvider`: everything above this depends on the ABC, so
a tool can be added, replaced or withheld without the executor, the API or the
Phase 4 agent loop knowing anything about it.

A tool implementation is responsible for exactly one thing — doing the work.
Permission checks, timeouts, argument validation and audit belong to the
executor, so that they cannot be forgotten by a tool author.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from aura.tools.types import ToolCall, ToolContext, ToolResult, ToolSpec


class Tool(ABC):
    """One thing AURA can do."""

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Name, schema, permission level and limits. Must not perform I/O."""

    @abstractmethod
    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        """Do the work.

        Called only after the executor has validated arguments and cleared
        permission, and always inside a timeout it does not have to enforce
        itself.

        Implementations should return `ToolResult.failure(...)` for expected
        problems — a missing file, a non-zero exit code. Raising is reserved
        for genuine bugs; the executor catches those and converts them, so a
        crash still produces an audit record.
        """

    @property
    def name(self) -> str:
        return self.spec.name

    def validate(self, arguments: dict[str, Any]) -> str | None:
        """Optional checks the JSON schema cannot express.

        Return an error message to refuse the call, or None to proceed. Runs
        after schema validation, before the permission check — a malformed call
        should be rejected on its own terms rather than prompting a human to
        approve something incoherent.
        """
        return None
