"""Filesystem tools.

Every path goes through `Workspace.resolve` before anything touches disk. No
tool here opens a path it was handed directly, and none of them catch
`PathNotAllowedError` and continue — a refusal is the end of the call.

Reading is LEVEL 1 (the user's data). Writing is LEVEL 2 (a reversible side
effect), which under the default policy means it does not run without an
explicit human confirmation.
"""

from __future__ import annotations

import time
from pathlib import Path

from aura.logging import get_logger
from aura.security.permissions import PermissionLevel
from aura.tools.base import Tool
from aura.tools.types import ToolCall, ToolContext, ToolResult, ToolSpec
from aura.tools.workspace import PathNotAllowedError, Workspace

log = get_logger(__name__)

# A model asking to read a file does not know how big it is. Truncate rather
# than refuse: half a large file is usually enough to answer the question, and
# refusing wastes a turn.
MAX_READ_BYTES = 200_000
MAX_LISTING = 500


def _workspace(ctx: ToolContext) -> Workspace:
    return Workspace.from_config(ctx.workspace_roots, ctx.deny_patterns)


class ReadFileTool(Tool):
    """Read a text file from the workspace."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_file",
            description=(
                "Read the contents of a text file. Returns up to "
                f"{MAX_READ_BYTES:,} bytes; longer files are truncated and say so. "
                "Only files inside the configured workspace can be read, and "
                "credentials are never readable."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, absolute or relative to the workspace.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            level=PermissionLevel.READ,
            timeout_s=10.0,
            reversible=True,
            idempotent=True,
        )

    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        try:
            path = _workspace(ctx).resolve(call.arguments["path"], must_exist=True)
        except PathNotAllowedError as exc:
            return ToolResult.failure(call, str(exc), decision="deny")

        if path.is_dir():
            return ToolResult.failure(call, f"{path} is a directory. Use list_directory instead.")

        try:
            raw = path.read_bytes()
        except OSError as exc:
            return ToolResult.failure(call, f"Could not read {path}: {exc}")

        truncated = len(raw) > MAX_READ_BYTES
        # errors="replace" rather than refusing: a file with one bad byte is
        # still worth reading, and a model can work around a replacement char.
        text = raw[:MAX_READ_BYTES].decode("utf-8", errors="replace")
        if truncated:
            text += f"\n\n[truncated: {len(raw):,} bytes total]"

        return ToolResult.success(
            call,
            text,
            data={"path": str(path), "bytes": len(raw), "truncated": truncated},
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


class ListDirectoryTool(Tool):
    """List the contents of a directory."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list_directory",
            description=(
                "List files and subdirectories. Returns name, type and size. "
                "Only directories inside the configured workspace can be listed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to list."},
                    "pattern": {
                        "type": "string",
                        "description": "Optional glob filter, e.g. '*.py'.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            level=PermissionLevel.READ,
            timeout_s=10.0,
        )

    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        try:
            path = _workspace(ctx).resolve(call.arguments["path"], must_exist=True)
        except PathNotAllowedError as exc:
            return ToolResult.failure(call, str(exc), decision="deny")

        if not path.is_dir():
            return ToolResult.failure(call, f"{path} is not a directory.")

        pattern = call.arguments.get("pattern") or "*"
        try:
            entries = sorted(path.glob(pattern), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as exc:
            return ToolResult.failure(call, f"Could not list {path}: {exc}")

        truncated = len(entries) > MAX_LISTING
        lines: list[str] = []
        for entry in entries[:MAX_LISTING]:
            try:
                if entry.is_dir():
                    lines.append(f"{entry.name}/")
                else:
                    lines.append(f"{entry.name}  ({entry.stat().st_size:,} bytes)")
            except OSError:
                # A file can vanish between glob and stat. Skip it rather than
                # failing the whole listing.
                continue

        if truncated:
            lines.append(f"[truncated: {len(entries):,} entries total]")

        return ToolResult.success(
            call,
            "\n".join(lines) if lines else "(empty)",
            data={"path": str(path), "count": len(entries), "truncated": truncated},
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


class FileInfoTool(Tool):
    """Check whether a path exists and what it is."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_info",
            description=(
                "Check whether a path exists, whether it is a file or directory, "
                "and its size and modification time."
            ),
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to inspect."}},
                "required": ["path"],
                "additionalProperties": False,
            },
            level=PermissionLevel.READ,
            timeout_s=5.0,
        )

    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        try:
            path = _workspace(ctx).resolve(call.arguments["path"])
        except PathNotAllowedError as exc:
            return ToolResult.failure(call, str(exc), decision="deny")

        if not path.exists():
            return ToolResult.success(
                call,
                f"{path} does not exist.",
                data={"path": str(path), "exists": False},
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        stat = path.stat()
        kind = "directory" if path.is_dir() else "file"
        modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        return ToolResult.success(
            call,
            f"{path} is a {kind}, {stat.st_size:,} bytes, modified {modified}.",
            data={
                "path": str(path),
                "exists": True,
                "kind": kind,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            },
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


class WriteFileTool(Tool):
    """Write a text file inside the workspace."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="write_file",
            description=(
                "Write text to a file, creating it if needed. Overwrites the "
                "whole file. Requires explicit confirmation before it runs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to write."},
                    "content": {"type": "string", "description": "Full contents to write."},
                    "create_parents": {
                        "type": "boolean",
                        "description": "Create missing parent directories. Default false.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            level=PermissionLevel.ACT,
            timeout_s=15.0,
            # An overwrite loses what was there. The backup below makes it
            # recoverable, which is why this is ACT and not MUTATE.
            reversible=True,
            # Re-running is safe in content terms but writes again; the agent
            # loop should not retry blindly on a timeout.
            idempotent=False,
        )

    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        try:
            path = _workspace(ctx).resolve(call.arguments["path"])
        except PathNotAllowedError as exc:
            return ToolResult.failure(call, str(exc), decision="deny")

        content: str = call.arguments["content"]
        create_parents = bool(call.arguments.get("create_parents", False))

        if path.is_dir():
            return ToolResult.failure(call, f"{path} is a directory.")

        if not path.parent.exists():
            if not create_parents:
                return ToolResult.failure(
                    call,
                    f"{path.parent} does not exist. Pass create_parents=true to create it.",
                )
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                return ToolResult.failure(call, f"Could not create {path.parent}: {exc}")

        # Keep the previous contents so an overwrite can be undone. Spec
        # section 8 asks for rollback where possible, and for a file write it
        # is possible and cheap.
        backup: Path | None = None
        existed = path.exists()
        if existed:
            backup = path.with_suffix(path.suffix + ".aura-bak")
            try:
                backup.write_bytes(path.read_bytes())
            except OSError as exc:
                return ToolResult.failure(
                    call, f"Could not back up {path} before overwriting: {exc}"
                )

        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return ToolResult.failure(call, f"Could not write {path}: {exc}")

        verb = "Overwrote" if existed else "Created"
        note = f" Previous contents saved to {backup.name}." if backup else ""
        return ToolResult.success(
            call,
            f"{verb} {path} ({len(content):,} characters).{note}",
            data={
                "path": str(path),
                "bytes": len(content.encode("utf-8")),
                "existed": existed,
                "backup": str(backup) if backup else None,
            },
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


def filesystem_tools() -> list[Tool]:
    return [ReadFileTool(), ListDirectoryTool(), FileInfoTool(), WriteFileTool()]
