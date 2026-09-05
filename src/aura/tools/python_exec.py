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
token — its own project, and not this phase. Until then this tool sits at
LEVEL 2 behind an explicit confirmation, and this docstring says what it does
not protect against rather than implying safety it does not have.
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
                "The process runs as the current user with network access, so "
                "it requires explicit confirmation before it runs."
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
            level=PermissionLevel.ACT,
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
                proc = subprocess.run(  # noqa: S603 - fixed interpreter, no shell
                    [self._interpreter, str(script)],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    cwd=cwd,
                    # No shell, ever: the snippet is a file argument, so shell
                    # metacharacters in the code cannot become shell syntax.
                    shell=False,
                )
            except subprocess.TimeoutExpired:
                log.warning("tools.python_timeout", timeout_s=timeout_s)
                return ToolResult.failure(
                    call,
                    f"Execution exceeded {timeout_s:.0f}s and was terminated.",
                    data={"timeout_s": timeout_s},
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            except OSError as exc:
                return ToolResult.failure(call, f"Could not start Python: {exc}")

        stdout = _clip(proc.stdout)
        stderr = _clip(proc.stderr)
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
