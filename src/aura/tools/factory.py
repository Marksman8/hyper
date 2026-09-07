"""Assemble the tool layer from config.

One place decides which tools exist and under what policy, so a caller cannot
accidentally construct an executor with a permissive policy or an unguarded
workspace.
"""

from __future__ import annotations

from aura.config import Settings
from aura.logging import get_logger
from aura.security.audit import AuditLog
from aura.security.permissions import PermissionLevel, PermissionPolicy
from aura.tools.applications import application_tools
from aura.tools.executor import ToolExecutor
from aura.tools.filesystem import filesystem_tools
from aura.tools.python_exec import python_tools
from aura.tools.registry import ToolRegistry
from aura.tools.types import ToolContext

log = get_logger(__name__)


def build_registry(settings: Settings) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_all(filesystem_tools())
    # Windows only; returns nothing elsewhere, so the model is never offered a
    # tool that cannot work on this machine.
    registry.register_all(application_tools())
    if settings.tools.python_enabled:
        registry.register_all(python_tools())
    log.info("tools.registry_built", tools=registry.names())
    return registry


def build_policy(settings: Settings) -> PermissionPolicy:
    cfg = settings.tools
    return PermissionPolicy(
        max_autonomous=PermissionLevel(cfg.max_autonomous_level),
        max_with_confirmation=PermissionLevel(cfg.max_confirmable_level),
    )


def build_context(settings: Settings, session_id: str | None = None) -> ToolContext:
    roots = settings.tools.resolved_roots()
    return ToolContext(
        workspace_roots=roots,
        deny_patterns=tuple(settings.tools.deny_patterns),
        session_id=session_id,
        working_dir=roots[0] if roots else None,
    )


def build_executor(settings: Settings, audit: AuditLog | None = None) -> ToolExecutor:
    return ToolExecutor(
        registry=build_registry(settings),
        policy=build_policy(settings),
        audit=audit,
    )
