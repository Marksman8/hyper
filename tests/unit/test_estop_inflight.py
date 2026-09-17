"""The stop reaching work that is already running (audit 01, finding 4).

The stop was checked at `ToolExecutor.execute()` entry and between agent steps.
Both are checks on *starting* something. A `run_python` snippet with a 120 s
budget that began one second before the stop was engaged ran to completion, and
the executor's own comment explains why nothing could be done about it after the
fact: "The worker thread may still be running: Python cannot kill one."

A subprocess, unlike a thread, can be killed. These assert that it is.
"""

from __future__ import annotations

import time

from aura.tools.python_exec import RunPythonTool
from aura.tools.types import ToolCall, ToolContext


def _ctx(tmp_path, should_stop=None) -> ToolContext:  # type: ignore[no-untyped-def]
    return ToolContext(
        workspace_roots=(str(tmp_path),),
        working_dir=str(tmp_path),
        should_stop=should_stop,
    )


def test_a_running_snippet_is_killed_when_the_stop_engages(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The whole point: the child dies, and it dies quickly."""
    engaged = {"at": time.monotonic() + 0.5}

    tool = RunPythonTool()
    started = time.perf_counter()
    result = tool.run(
        ToolCall(name="run_python", arguments={"code": "import time; time.sleep(30)"}),
        _ctx(tmp_path, should_stop=lambda: time.monotonic() >= engaged["at"]),
    )
    elapsed = time.perf_counter() - started

    assert not result.ok
    assert "emergency stop" in (result.error or "").lower()
    # Generous, because process teardown on Windows is not instant. The
    # property under test is "seconds, not the full 30".
    assert elapsed < 8.0, f"took {elapsed:.1f}s; the stop did not reach the child"


def test_a_snippet_finishes_normally_when_the_stop_stays_clear(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The polling must not break the ordinary path, or cost it latency."""
    tool = RunPythonTool()
    result = tool.run(
        ToolCall(name="run_python", arguments={"code": "print(6 * 7)"}),
        _ctx(tmp_path, should_stop=lambda: False),
    )
    assert result.ok, result.error
    assert "42" in result.content


def test_no_stop_supplied_still_works(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`should_stop` is optional: a tool built in a test has no switch."""
    tool = RunPythonTool()
    result = tool.run(
        ToolCall(name="run_python", arguments={"code": "print('hi')"}), _ctx(tmp_path)
    )
    assert result.ok, result.error
    assert "hi" in result.content


def test_the_inner_timeout_still_fires(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Polling replaced `subprocess.run(timeout=)`, so its job moves here."""
    tool = RunPythonTool()
    result = tool.run(
        ToolCall(
            name="run_python",
            arguments={"code": "import time; time.sleep(30)", "timeout_s": 1},
        ),
        _ctx(tmp_path, should_stop=lambda: False),
    )
    assert not result.ok
    assert "exceeded" in (result.error or "").lower()
