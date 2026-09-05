"""The tool catalogue.

Holds the tools that exist and answers two questions: what is available, and
what may this caller see. The second matters — a restricted session should not
merely be refused when it calls a tool, it should never be told the tool
exists. Offering the model a capability and then denying it wastes a turn and
teaches it nothing.
"""

from __future__ import annotations

from collections.abc import Iterator

from aura.logging import get_logger
from aura.security.permissions import PermissionLevel
from aura.tools.base import Tool
from aura.tools.types import ToolSpec

log = get_logger(__name__)


class ToolNotFoundError(KeyError):
    """No tool is registered under that name."""


class ToolRegistry:
    """Name -> Tool, with filtering."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"A tool named {name!r} is already registered.")
        self._tools[name] = tool
        log.debug("tools.registered", tool=name, level=int(tool.spec.level))

    def register_all(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(
                f"No tool named {name!r}. Available: {sorted(self._tools)}"
            ) from exc

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def specs(self, max_level: PermissionLevel | None = None) -> list[ToolSpec]:
        """Specs, optionally hiding anything above `max_level`.

        This is what Phase 4 sends to the model, so the filter is the
        difference between "AURA cannot delete files" and "AURA knows it could
        try and will be told no".
        """
        out = [t.spec for t in self._tools.values()]
        if max_level is not None:
            out = [s for s in out if s.level <= max_level]
        return sorted(out, key=lambda s: s.name)

    def names(self) -> list[str]:
        return sorted(self._tools)
