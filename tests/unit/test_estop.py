"""The emergency stop (Stage 1 finding, critical).

The audit demonstrated the hole live: with the stop engaged, `POST /v1/chat`
was refused and `POST /v1/tools/read_file` returned the file. The switch stopped
AURA thinking and left her acting.

These tests exist so that cannot come back. The important ones are not that the
switch works — they are that it covers **every tool**, that it survives a
restart, and that AURA has no route to it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aura.errors import EmergencyStopError
from aura.security.estop import DENY_PATTERN, EmergencyStop, build_estop
from aura.security.permissions import PermissionLevel, PermissionPolicy
from aura.tools.executor import ToolExecutor
from aura.tools.factory import build_registry
from aura.tools.types import ToolCall, ToolContext
from aura.tools.workspace import DEFAULT_DENY_PATTERNS

SRC = Path(__file__).resolve().parents[2] / "src" / "aura"


@pytest.fixture
def stop(tmp_path: Path) -> EmergencyStop:
    return EmergencyStop(tmp_path / "emergency_stop")


# ===========================================================================
# The switch itself
# ===========================================================================


def test_a_fresh_switch_is_clear(stop: EmergencyStop) -> None:
    assert stop.engaged is False


def test_engaging_creates_the_file(stop: EmergencyStop) -> None:
    stop.engage("test")

    assert stop.engaged is True
    assert stop.path.is_file()


def test_the_file_says_how_to_release_it(stop: EmergencyStop) -> None:
    """Someone finding this file in a directory should not have to read the
    source to work out what it is."""
    stop.engage("test")

    body = stop.path.read_text(encoding="utf-8")
    assert "Delete this file" in body


def test_releasing_removes_it(stop: EmergencyStop) -> None:
    stop.engage("test")
    stop.release("test")

    assert stop.engaged is False


def test_release_is_idempotent(stop: EmergencyStop) -> None:
    stop.release("test")
    stop.release("test")


def test_state_survives_a_new_object(tmp_path: Path) -> None:
    """The property the old boolean did not have.

    Engaging the stop and restarting AURA released it -- and the situation
    where a kill switch matters most is exactly the one where a process is most
    likely to restart.
    """
    EmergencyStop(tmp_path / "emergency_stop").engage("test")

    assert EmergencyStop(tmp_path / "emergency_stop").engaged is True


def test_the_file_is_the_switch(tmp_path: Path) -> None:
    """It works without AURA. Someone can engage the stop from a file manager
    while she is wedged, unresponsive, or not running at all."""
    stop = EmergencyStop(tmp_path / "emergency_stop")
    assert stop.engaged is False

    (tmp_path / "emergency_stop").write_text("stop", encoding="utf-8")

    assert stop.engaged is True


def test_the_state_is_never_cached(stop: EmergencyStop) -> None:
    """A stop that takes effect at the next cache expiry is a stop that does
    not take effect."""
    assert stop.engaged is False
    stop.path.write_text("x", encoding="utf-8")
    assert stop.engaged is True
    stop.path.unlink()
    assert stop.engaged is False


def test_an_unreadable_switch_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Guessing wrong in the other direction means a stopped AURA keeps
    acting."""
    stop = EmergencyStop(tmp_path / "emergency_stop")

    def boom(self: Path) -> bool:
        raise OSError("cannot stat")

    monkeypatch.setattr(Path, "exists", boom)

    assert stop.engaged is True


def test_a_config_stop_cannot_be_released_at_runtime(tmp_path: Path) -> None:
    """Someone who wrote it in a config file meant it."""
    stop = EmergencyStop(tmp_path / "emergency_stop", engaged_by_config=True)

    assert stop.engaged is True
    with pytest.raises(EmergencyStopError, match="configuration"):
        stop.release("test")
    assert stop.engaged is True


def test_require_clear_raises_when_engaged(stop: EmergencyStop) -> None:
    stop.require_clear("Generation")  # clear: no raise

    stop.engage("test")
    with pytest.raises(EmergencyStopError, match="Generation"):
        stop.require_clear("Generation")


# ===========================================================================
# The breach the audit found
# ===========================================================================


def _executor(settings: object, stop: EmergencyStop) -> ToolExecutor:
    return ToolExecutor(
        registry=build_registry(settings),  # type: ignore[arg-type]
        policy=PermissionPolicy(
            max_autonomous=PermissionLevel.MUTATE,
            max_with_confirmation=PermissionLevel.CRITICAL,
        ),
        estop=stop,
    )


@pytest.mark.asyncio
async def test_an_engaged_stop_blocks_tool_execution(
    settings: object, stop: EmergencyStop
) -> None:
    """The exact case the audit demonstrated: chat refused, read_file ran."""
    executor = _executor(settings, stop)
    stop.engage("test")

    result = await executor.execute(
        ToolCall(name="read_file", arguments={"path": "README.md"}), ToolContext()
    )

    assert result.ok is False
    assert result.decision == "deny"
    assert "Emergency stop" in (result.error or "")
    executor.close()


@pytest.mark.asyncio
async def test_the_stop_covers_every_registered_tool(
    settings: object, stop: EmergencyStop
) -> None:
    """Not just the one that was tested. A kill switch with exceptions is a
    list of things that still happen after you press it."""
    executor = _executor(settings, stop)
    stop.engage("test")

    for spec in executor.registry.specs():
        result = await executor.execute(
            ToolCall(name=spec.name, arguments={}, confirmed=True), ToolContext()
        )
        assert result.ok is False, f"{spec.name} ran with the stop engaged"
        assert "Emergency stop" in (result.error or ""), f"{spec.name} failed for another reason"
    executor.close()


@pytest.mark.asyncio
async def test_the_stop_is_checked_before_the_tool_is_resolved(
    settings: object, stop: EmergencyStop
) -> None:
    """"May anything happen at all" does not depend on which tool was asked
    for -- so an unknown name gets the stop's answer, not a lookup error."""
    executor = _executor(settings, stop)
    stop.engage("test")

    result = await executor.execute(
        ToolCall(name="no_such_tool", arguments={}), ToolContext()
    )

    assert "Emergency stop" in (result.error or "")
    executor.close()


@pytest.mark.asyncio
async def test_confirmation_does_not_override_the_stop(
    settings: object, stop: EmergencyStop
) -> None:
    """An approval granted before the stop was pressed is not a licence to
    proceed after it."""
    executor = _executor(settings, stop)
    stop.engage("test")

    result = await executor.execute(
        ToolCall(name="write_file", arguments={"path": "x", "content": "y"}, confirmed=True),
        ToolContext(),
    )

    assert result.ok is False
    assert "Emergency stop" in (result.error or "")
    executor.close()


@pytest.mark.asyncio
async def test_releasing_restores_execution(settings: object, stop: EmergencyStop) -> None:
    executor = _executor(settings, stop)
    stop.engage("test")
    listing = ToolCall(name="list_directory", arguments={"path": "."})
    assert (await executor.execute(listing, ToolContext())).ok is False

    stop.release("test")
    result = await executor.execute(
        ToolCall(name="list_directory", arguments={"path": "."}), ToolContext()
    )

    assert result.error is None or "Emergency stop" not in result.error
    executor.close()


@pytest.mark.asyncio
async def test_a_blocked_call_is_audited(settings: object, stop: EmergencyStop) -> None:
    """The record of what was refused is the more interesting half."""
    recorded: list[dict] = []

    class Recorder:
        async def record(self, **kwargs: object) -> None:
            recorded.append(kwargs)  # type: ignore[arg-type]

    executor = ToolExecutor(
        registry=build_registry(settings),  # type: ignore[arg-type]
        policy=PermissionPolicy(),
        audit=Recorder(),  # type: ignore[arg-type]
        estop=stop,
    )
    stop.engage("test")

    await executor.execute(ToolCall(name="read_file", arguments={"path": "x"}), ToolContext())

    assert recorded and recorded[0]["action"] == "tool.read_file"
    assert str(recorded[0]["decision"]) == "deny"
    executor.close()


# ===========================================================================
# AURA has no route to it
# ===========================================================================


def test_the_switch_is_denied_to_file_tools() -> None:
    """A switch AURA can delete is a switch AURA can disable."""
    assert DENY_PATTERN in DEFAULT_DENY_PATTERNS
    assert any("var/secrets" in pattern for pattern in DEFAULT_DENY_PATTERNS)


def test_no_tool_imports_the_switch() -> None:
    """Nothing in aura.tools may reach the stop except the executor, which
    only reads it. A tool that could construct one could engage or release it.
    """
    offenders: list[str] = []
    for path in (SRC / "tools").rglob("*.py"):
        if path.name == "executor.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = getattr(node, "module", "") or ""
            names = [a.name for a in getattr(node, "names", []) if hasattr(a, "name")]
            if "estop" in module or any("estop" in n for n in names):
                offenders.append(str(path.relative_to(SRC)))
    # factory.py legitimately builds one to hand to the executor.
    offenders = [o for o in offenders if not o.endswith("factory.py")]
    assert not offenders, f"tool modules reaching the kill switch: {offenders}"


def test_the_executor_only_reads_the_switch() -> None:
    """It may ask whether the stop is engaged. It may not engage or release
    one -- that authority belongs to the human."""
    source = (SRC / "tools" / "executor.py").read_text(encoding="utf-8")

    assert "estop.engaged" in source
    for mutation in (".engage(", ".release("):
        assert mutation not in source, f"executor.py calls {mutation}"


def test_the_switch_follows_the_secrets_directory(tmp_path: Path) -> None:
    """So a test that engages it cannot leave the working tree stopped -- the
    mistake that made the first version of this feature poison its own suite.
    """
    from aura.config import SecurityConfig, Settings

    settings = Settings(security=SecurityConfig(token_file=str(tmp_path / "token")))

    assert build_estop(settings).path.parent == tmp_path
