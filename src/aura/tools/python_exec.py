"""Run Python code.

**In a subprocess, never `exec()` in-process.** In-process execution would
share AURA's memory, its open database handle and its API tokens with whatever
the model wrote. A subprocess shares none of that, and — the practical part —
can actually be killed when it runs long, which a thread cannot.

Honest about what this is not. A subprocess with a timeout and a working
directory is **isolation, not a sandbox**:

- it runs as the user, with the user's permissions
- it can reach the network
- it can read anything the user can read, workspace or not

Real sandboxing on Windows means a container, or a job object with a restricted
token — its own project, and not this phase.

**Until then it sits at LEVEL 4, which the default policy denies outright.**

It sat at LEVEL 2 behind a confirmation until security audit 01 finding 3, which
observed what that combination actually permits. `estop.py` states as one of the
three properties making the kill switch a switch that "AURA cannot reach it…
`tools/workspace.py` already denies [`var/secrets/`] to every file tool". True,
and this is not a file tool. It never calls `Workspace.resolve()` and does not
need to: `Path("var/secrets/emergency_stop").unlink()` is four words and passes
no guard. Nor does the deny list stop it reading `.env`, the API token, or
`~/.ssh/` — the docstring above says as much and the deny list exists to prevent
exactly those reads.

Worse in combination with finding 2: a confirmation is granted per tool *name*
and survives a task restart, so one approval of `run_python` for anything at all
left arbitrary code execution available for the rest of that task.

LEVEL 4 does not make the tool safe. It makes it unreachable without a human
deliberately raising `tools.max_with_confirmation`, which is the honest standing
for a capability the architecture cannot bound. The remedies that would earn a
lower level are listed in `docs/SECURITY-AUDIT-01.md` §3.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

from aura.logging import get_logger
from aura.security.permissions import PermissionLevel
from aura.tools.base import Tool
from aura.tools.types import ToolCall, ToolContext, ToolResult, ToolSpec

log = get_logger(__name__)

# How often the poll loop asks whether the stop has been engaged. Short enough
# that "stop" feels immediate, long enough that a snippet doing real work is not
# competing with the loop for the GIL.
_POLL_INTERVAL_S = 0.2

MAX_OUTPUT_CHARS = 20_000
MAX_CODE_CHARS = 100_000


class RunPythonTool(Tool):
    """Execute a Python snippet and return what it printed."""

    def __init__(self, interpreter: str | None = None) -> None:
        # The venv interpreter by default, so the snippet sees the same
        # packages AURA has rather than whatever `python` resolves to.
        self._interpreter = interpreter or sys.executable

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="run_python",
            description=(
                "Execute a Python snippet in a separate process and return its "
                "stdout, stderr and exit code. Use print() to return values. "
                "The process runs as the current user with network access and "
                "is not sandboxed, so it is denied unless a human has raised "
                "the permission ceiling for it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source to execute.",
                        "maxLength": MAX_CODE_CHARS,
                    },
                    "timeout_s": {
                        "type": "number",
                        "description": "Seconds to allow, 1-120. Default 30.",
                        "minimum": 1,
                        "maximum": 120,
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            # LEVEL 4, not 2. See the module docstring and audit 01 finding 3:
            # this tool is outside the workspace guard, so it can reach the
            # secrets directory and the kill switch inside it.
            level=PermissionLevel.CRITICAL,
            # Generous, because the tool enforces its own inner timeout on the
            # subprocess; this is the outer backstop.
            timeout_s=130.0,
            reversible=False,
            idempotent=False,
        )

    def validate(self, arguments: dict[str, object]) -> str | None:
        code = arguments.get("code")
        if isinstance(code, str) and not code.strip():
            return "'code' is empty"
        return None

    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        code: str = call.arguments["code"]
        timeout_s = float(call.arguments.get("timeout_s") or 30.0)

        cwd = ctx.working_dir or (ctx.workspace_roots[0] if ctx.workspace_roots else None)

        # A file rather than `python -c`: tracebacks then carry real line
        # numbers, which is the difference between a model fixing its own bug
        # and guessing at it.
        with tempfile.TemporaryDirectory(prefix="aura-py-") as tmp:
            script = Path(tmp) / "snippet.py"
            script.write_text(code, encoding="utf-8")

            try:
                proc = subprocess.Popen(  # noqa: S603 - fixed interpreter, no shell
                    [self._interpreter, str(script)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd,
                    # No shell, ever: the snippet is a file argument, so shell
                    # metacharacters in the code cannot become shell syntax.
                    shell=False,
                )
            except OSError as exc:
                return ToolResult.failure(call, f"Could not start Python: {exc}")

            # `Popen` and a poll loop rather than `subprocess.run(timeout=)`,
            # so the emergency stop can reach a snippet that is already
            # running. Audit 01 finding 4: the stop was only ever checked
            # before a tool started, and a 120 s budget is a long time to keep
            # acting after a person has said stop.
            stop = ctx.should_stop
            deadline = time.monotonic() + timeout_s
            stopped = False
            while True:
                try:
                    stdout_raw, stderr_raw = proc.communicate(timeout=_POLL_INTERVAL_S)
                    break
                except subprocess.TimeoutExpired:
                    pass

                if stop is not None and stop():
                    stopped = True
                    _terminate(proc)
                    break
                if time.monotonic() >= deadline:
                    _terminate(proc)
                    log.warning("tools.python_timeout", timeout_s=timeout_s)
                    return ToolResult.failure(
                        call,
                        f"Execution exceeded {timeout_s:.0f}s and was terminated.",
                        data={"timeout_s": timeout_s},
                        duration_ms=int((time.perf_counter() - started) * 1000),
                    )

            if stopped:
                log.warning("tools.python_killed_by_estop")
                return ToolResult.failure(
                    call,
                    "The emergency stop was engaged while this snippet was "
                    "running, so the process was terminated. Any work it had "
                    "already done is not undone.",
                    decision="deny",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )

        stdout = _clip(stdout_raw)
        stderr = _clip(stderr_raw)
        duration_ms = int((time.perf_counter() - started) * 1000)

        parts: list[str] = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"[stderr]\n{stderr}")
        if not parts:
            parts.append("(no output)")
        body = "\n".join(parts)

        data = {"exit_code": proc.returncode, "stdout": stdout, "stderr": stderr}

        if proc.returncode != 0:
            # A non-zero exit is an expected outcome, not a tool failure: the
            # snippet ran and the model needs the traceback to fix it. Reported
            # as failure so the agent loop notices, with the output attached.
            return ToolResult.failure(
                call,
                f"Exited with code {proc.returncode}.\n{body}",
                data=data,
                duration_ms=duration_ms,
            )

        return ToolResult.success(call, body, data=data, duration_ms=duration_ms)


def _terminate(proc: subprocess.Popen[str]) -> None:
    """Stop the child, escalating if it does not go quietly.

    A subprocess is the reason this tool can be stopped at all: the executor's
    worker thread cannot be killed, and a snippet that ignores SIGTERM would
    otherwise outlive the decision to stop it.
    """
    proc.terminate()
    try:
        proc.communicate(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:  # pragma: no cover - the OS gave up
            log.error("tools.python_would_not_die", pid=proc.pid)


def _clip(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    # Keep the tail as well as the head: a traceback ends with the part that
    # matters, and a loop that prints a lot would otherwise hide its own error.
    head = text[: MAX_OUTPUT_CHARS // 2]
    tail = text[-MAX_OUTPUT_CHARS // 2 :]
    return f"{head}\n\n[... {len(text):,} chars total, middle omitted ...]\n\n{tail}"


def python_tools(interpreter: str | None = None) -> list[Tool]:
    return [RunPythonTool(interpreter=interpreter)]
