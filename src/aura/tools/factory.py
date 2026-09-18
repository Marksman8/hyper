"""Assemble the tool layer from config.

One place decides which tools exist and under what policy, so a caller cannot
accidentally construct an executor with a permissive policy or an unguarded
workspace.
"""

from __future__ import annotations

from typing import Any

from aura.automation.factory import build_provider, load_workflows
from aura.config import Settings
from aura.logging import get_logger
from aura.security.audit import AuditLog
from aura.security.estop import EmergencyStop, build_estop
from aura.security.permissions import PermissionLevel, PermissionPolicy
from aura.tools.applications import application_tools
from aura.tools.automation import automation_tools
from aura.tools.executor import ToolExecutor
from aura.tools.filesystem import filesystem_tools
from aura.tools.memory import memory_tools
from aura.tools.python_exec import python_tools
from aura.tools.registry import ToolRegistry
from aura.tools.types import ToolContext
from aura.tools.vision import vision_tools

log = get_logger(__name__)


def build_registry(settings: Settings, store: Any = None) -> ToolRegistry:
    """Assemble the tools.

    `store` is optional because the CLI and several tests build a registry with
    no database. Where one is supplied, AURA gains `search_memory` and can
    finally reach conversations from before the current session -- which were
    always on disk and always indexed, just not reachable by her.
    """
    registry = ToolRegistry()
    registry.register_all(filesystem_tools())
    registry.register_all(memory_tools(store))
    # Windows only; returns nothing elsewhere, so the model is never offered a
    # tool that cannot work on this machine.
    registry.register_all(application_tools())
    if settings.tools.python_enabled:
        registry.register_all(python_tools())
    # Registered whatever the vision mode. The tools are the thing that
    # explains a refusal; withholding them in private mode would leave the
    # model unable to say why it cannot look.
    registry.register_all(vision_tools(settings))
    # Declared workflows (ADR-0004). The manifest is read whether or not
    # automation is enabled, so a broken manifest fails here at startup rather
    # than halfway through an agent task. Nothing is registered when the
    # provider is off.
    registry.register_all(automation_tools(load_workflows(settings), build_provider(settings)))
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


def build_executor(
    settings: Settings,
    audit: AuditLog | None = None,
    events: Any = None,
    estop: EmergencyStop | None = None,
    store: Any = None,
) -> ToolExecutor:
    """Assemble the executor.

    The kill switch is built here when one is not supplied, so an executor
    constructed through the ordinary path always has one. An executor with no
    stop is an executor nothing can halt.
    """
    return ToolExecutor(
        registry=build_registry(settings, store=store),
        policy=build_policy(settings),
        audit=audit,
        events=events,
        estop=estop or build_estop(settings),
    )
