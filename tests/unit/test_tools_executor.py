"""The executor: validation, permission, timeout and audit.

These test the chokepoint rather than the tools. The property that matters is
that nothing gets past it — an unknown tool, a malformed call, an unconfirmed
level-2 operation and a crashing tool must all produce a structured result and
an audit record, never an exception and never a silent success.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from aura.config import Settings, ToolsConfig
from aura.memory.store import MemoryStore
from aura.security.audit import AuditLog
from aura.security.permissions import PermissionLevel, PermissionPolicy
from aura.tools.base import Tool
from aura.tools.executor import ToolExecutor
from aura.tools.factory import build_context, build_registry
from aura.tools.registry import ToolRegistry
from aura.tools.types import ToolCall, ToolContext, ToolResult, ToolSpec


class SleepyTool(Tool):
    """Blocks for longer than its own timeout."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="sleepy",
            description="Sleeps.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            level=PermissionLevel.OBSERVE,
            timeout_s=0.2,
        )

    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        time.sleep(5)
        return ToolResult.success(call, "should never be seen")


class CrashingTool(Tool):
    """Raises, as a buggy tool would."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="crasher",
            description="Raises.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            level=PermissionLevel.OBSERVE,
            timeout_s=5.0,
        )

    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        raise RuntimeError("something went badly wrong")


class CriticalTool(Tool):
    """Above anything a confirmation can reach."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="nuke",
            description="Level 4.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            level=PermissionLevel.CRITICAL,
            timeout_s=5.0,
        )

    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:  # pragma: no cover
        raise AssertionError("a level 4 tool must never run under the default policy")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "notes.txt").write_text("hello")
    return root


@pytest.fixture
def settings(workspace: Path) -> Settings:
    return Settings(tools=ToolsConfig(workspace_roots=[str(workspace)]))


@pytest.fixture
def ctx(settings: Settings) -> ToolContext:
    return build_context(settings)


@pytest.fixture
async def store(tmp_path: Path) -> Any:
    s = MemoryStore(tmp_path / "audit.db")
    s.connect()
    yield s
    s.close()


@pytest.fixture
def executor(settings: Settings) -> Any:
    registry = build_registry(settings)
    registry.register_all([SleepyTool(), CrashingTool(), CriticalTool()])
    ex = ToolExecutor(registry=registry, policy=PermissionPolicy())
    yield ex
    ex.close()


# -- the refusals -----------------------------------------------------------


async def test_unknown_tool_is_refused(executor: ToolExecutor, ctx: ToolContext) -> None:
    result = await executor.execute(ToolCall(name="nope", arguments={}), ctx)
    assert not result.ok
    assert result.decision == "deny"
    assert "No tool named" in (result.error or "")


async def test_missing_required_argument_is_refused(
    executor: ToolExecutor, ctx: ToolContext
) -> None:
    result = await executor.execute(ToolCall(name="read_file", arguments={}), ctx)
    assert not result.ok
    assert "missing required argument 'path'" in (result.error or "")


async def test_unknown_argument_is_refused(
    executor: ToolExecutor, ctx: ToolContext, workspace: Path
) -> None:
    """A typo must be reported, not silently defaulted."""
    result = await executor.execute(
        ToolCall(name="read_file", arguments={"paht": str(workspace / "notes.txt")}), ctx
    )
    assert not result.ok
    assert "unknown argument" in (result.error or "")


async def test_wrong_argument_type_is_refused(executor: ToolExecutor, ctx: ToolContext) -> None:
    result = await executor.execute(ToolCall(name="read_file", arguments={"path": 42}), ctx)
    assert not result.ok
    assert "must be string" in (result.error or "")


# -- permission -------------------------------------------------------------


async def test_read_runs_without_confirmation(
    executor: ToolExecutor, ctx: ToolContext, workspace: Path
) -> None:
    """Level 1 is below the autonomous ceiling."""
    result = await executor.execute(
        ToolCall(name="read_file", arguments={"path": str(workspace / "notes.txt")}), ctx
    )
    assert result.ok
    assert result.content == "hello"


async def test_write_is_refused_without_confirmation(
    executor: ToolExecutor, ctx: ToolContext, workspace: Path
) -> None:
    """The whole point of the gate: an unset flag is a refusal, not consent."""
    target = workspace / "created.txt"
    result = await executor.execute(
        ToolCall(name="write_file", arguments={"path": str(target), "content": "x"}),
        ctx,
    )
    assert not result.ok
    assert result.decision == "confirm"
    assert not target.exists(), "the file was written despite the refusal"


async def test_write_runs_once_confirmed(
    executor: ToolExecutor, ctx: ToolContext, workspace: Path
) -> None:
    target = workspace / "created.txt"
    result = await executor.execute(
        ToolCall(
            name="write_file",
            arguments={"path": str(target), "content": "written"},
            confirmed=True,
        ),
        ctx,
    )
    assert result.ok, result.error
    assert target.read_text() == "written"


async def test_level_four_is_denied_even_when_confirmed(
    executor: ToolExecutor, ctx: ToolContext
) -> None:
    """Confirmation raises the ceiling to level 3, not past it."""
    result = await executor.execute(ToolCall(name="nuke", arguments={}, confirmed=True), ctx)
    assert not result.ok
    assert result.decision == "deny"


# -- containment ------------------------------------------------------------


async def test_reading_outside_the_workspace_is_denied(
    executor: ToolExecutor, ctx: ToolContext, tmp_path: Path
) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("not yours")
    result = await executor.execute(
        ToolCall(name="read_file", arguments={"path": str(outside)}), ctx
    )
    assert not result.ok
    assert result.decision == "deny"
    assert "outside the workspace" in (result.error or "")


# -- failure handling -------------------------------------------------------


async def test_a_crashing_tool_becomes_a_result_not_an_exception(
    executor: ToolExecutor, ctx: ToolContext
) -> None:
    """A raise must not escape: the loop would end and the audit would be skipped."""
    result = await executor.execute(ToolCall(name="crasher", arguments={}), ctx)
    assert not result.ok
    assert "RuntimeError" in (result.error or "")
    assert "something went badly wrong" in (result.error or "")


async def test_a_slow_tool_hits_its_timeout(executor: ToolExecutor, ctx: ToolContext) -> None:
    started = time.perf_counter()
    result = await executor.execute(ToolCall(name="sleepy", arguments={}), ctx)
    elapsed = time.perf_counter() - started
    assert not result.ok
    assert "timeout" in (result.error or "").lower()
    assert elapsed < 4.0, "the executor waited for the tool instead of timing out"


# -- audit ------------------------------------------------------------------


async def test_successful_calls_are_audited(
    settings: Settings, ctx: ToolContext, store: Any, workspace: Path
) -> None:
    ex = ToolExecutor(build_registry(settings), PermissionPolicy(), AuditLog(store))
    try:
        await ex.execute(
            ToolCall(name="read_file", arguments={"path": str(workspace / "notes.txt")}),
            ctx,
        )
    finally:
        ex.close()

    rows = await store.recent_audit()
    assert any(r["action"] == "tool.read_file" and r["decision"] == "allow" for r in rows)


async def test_refusals_are_audited_too(
    settings: Settings, ctx: ToolContext, store: Any, workspace: Path
) -> None:
    """The record of what was refused is the more interesting half."""
    ex = ToolExecutor(build_registry(settings), PermissionPolicy(), AuditLog(store))
    try:
        await ex.execute(
            ToolCall(
                name="write_file",
                arguments={"path": str(workspace / "x.txt"), "content": "x"},
            ),
            ctx,
        )
    finally:
        ex.close()

    rows = await store.recent_audit()
    row = next(r for r in rows if r["action"] == "tool.write_file")
    assert row["decision"] == "confirm"
    assert row["detail"]["confirmed"] is False


# -- registry ---------------------------------------------------------------


def test_registry_hides_tools_above_a_level(settings: Settings) -> None:
    """A restricted session should not be told a tool exists."""
    registry: ToolRegistry = build_registry(settings)
    read_only = {s.name for s in registry.specs(max_level=PermissionLevel.READ)}
    assert "read_file" in read_only
    assert "write_file" not in read_only
    assert "run_python" not in read_only


def test_registering_a_duplicate_name_is_an_error(settings: Settings) -> None:
    registry = build_registry(settings)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_all([SleepyTool(), SleepyTool()])


def test_specs_render_to_the_model_wire_format(settings: Settings) -> None:
    """Phase 4 sends exactly this, so it must match what actually runs."""
    registry = build_registry(settings)
    wire = registry.get("read_file").spec.to_wire()
    assert wire["type"] == "function"
    assert wire["function"]["name"] == "read_file"
    assert "path" in wire["function"]["parameters"]["properties"]
