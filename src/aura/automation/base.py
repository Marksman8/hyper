"""The automation provider seam.

ADR-0004 D10: AURA depends on an abstract automation interface, not on n8n.
Webhooks, node types and execution ids stay inside a provider, so replacing n8n
with Node-RED, Windmill, or direct API calls is a provider swap -- the same
discipline as `LLMProvider`, `OCRProvider` and `AvatarBackend`.

**A provider executes. It never decides what it is allowed to execute.** The
permission level arrives with the call, having come from the local manifest,
and a provider has no way to influence it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AutomationResult:
    """What a workflow returned.

    **`data` is untrusted** (ADR-0004 D6). A workflow that reads mail returns
    content from the internet, which has the same standing as screen text: it
    is data, never an instruction, and it may not escalate a permission.
    """

    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: int = 0


class AutomationProvider(ABC):
    """Runs a named automation. Knows nothing about permissions."""

    @abstractmethod
    def name(self) -> str:
        """Stable identifier for logs and audit rows."""

    @abstractmethod
    def run(
        self, endpoint: str, arguments: dict[str, Any], timeout_s: float
    ) -> AutomationResult:
        """Execute one automation.

        Blocking; the executor already runs tools in a bounded thread pool.
        Must not raise for an ordinary failure -- an unreachable backend is a
        result, not an exception (ADR-0004 D9).
        """

    def health(self) -> bool:
        """Best-effort reachability. Never raises."""
        return True
