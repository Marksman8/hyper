"""Letting AURA reach her own memory.

**The gap this closes.** `MemoryStore.search_turns` has existed since Phase 6:
BM25 over every turn ever stored, across every session, already tested. The
`/v1/memory/search` route exposes it, and `aura memory search` uses it. The one
thing that could not reach it was **AURA**.

So asked what was discussed yesterday she had nothing to go on, because
`_assemble` loads `recent_turns(session_id)` -- the *current* session only. Every
earlier conversation was on disk, indexed, searchable by the user, and invisible
to her. That is not a missing database. It is a missing tool.

**Reading is LEVEL 1, not LEVEL 0.** Level 0 is "AURA's own state" -- uptime,
what model is loaded. Conversation history is the *user's* data: it contains
whatever they have ever said to her, which is exactly the class of thing
`read_file` is level 1 for. The same permission, for the same reason.

**What it deliberately does not do.** It does not write, delete, or change
confidence in anything. Facts are proposed by the extraction pipeline under
Phase 6's evidence rules and removed by the user; a tool that let the model
edit its own memory would let a persuasive conversation rewrite what AURA
believes, which is the memory-poisoning path the evidence design exists to
close.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aura.logging import get_logger
from aura.security.permissions import PermissionLevel
from aura.tools.base import Tool
from aura.tools.types import ToolCall, ToolContext, ToolResult, ToolSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aura.memory.store import MemoryStore

log = get_logger(__name__)

MAX_RESULTS = 20
# Enough to recognise the exchange, short enough that eight matches do not
# crowd out the conversation they are being recalled into.
SNIPPET_CHARS = 320


def _when(ts: float) -> str:
    """A date a person and a model both read the same way.

    Relative on its own ("2 days ago") is ambiguous once it lands in a prompt
    the model re-reads later, so both forms are given.
    """
    now = time.time()
    days = int((now - ts) // 86400)
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
    if days <= 0:
        return f"{stamp} (today)"
    if days == 1:
        return f"{stamp} (yesterday)"
    return f"{stamp} ({days} days ago)"


class SearchMemoryTool(Tool):
    """Find earlier conversations by what was said in them."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_memory",
            description=(
                "Search everything the user has ever said to you, and everything "
                "you have replied, across all past conversations. Use this "
                "whenever they refer to something from an earlier session -- "
                "'what did we decide yesterday', 'the thing I mentioned last "
                "week', 'that file we looked at'. Returns matching exchanges "
                "with their dates. Searches words, not meaning, so use the "
                "words they would have used."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Words to look for in past conversations.",
                        "maxLength": 500,
                    },
                    "limit": {
                        "type": "integer",
                        "description": f"How many exchanges to return, 1-{MAX_RESULTS}. Default 8.",
                        "minimum": 1,
                        "maximum": MAX_RESULTS,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            level=PermissionLevel.READ,
            timeout_s=10.0,
            reversible=True,
            idempotent=True,
        )

    def validate(self, arguments: dict[str, object]) -> str | None:
        query = arguments.get("query")
        if isinstance(query, str) and not query.strip():
            return "'query' is empty"
        return None

    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        import asyncio

        query: str = call.arguments["query"]
        limit = int(call.arguments.get("limit") or 8)
        limit = max(1, min(MAX_RESULTS, limit))

        # The store's API is async and `Tool.run` is sync, called on a worker
        # thread by the executor. There is no running loop on that thread, so a
        # private one is correct here rather than a workaround.
        matches = asyncio.run(self._store.search_turns(query, limit=limit))

        if not matches:
            return ToolResult.success(
                call,
                f"Nothing in past conversations matches {query!r}. "
                "This searches words rather than meaning, so a different "
                "phrasing may find it.",
                data={"query": query, "matches": 0},
            )

        lines: list[str] = [f"{len(matches)} earlier exchange(s) matching {query!r}:"]
        for m in matches:
            turn = m.turn
            body = (turn.content or "").strip().replace("\n", " ")
            if len(body) > SNIPPET_CHARS:
                body = body[: SNIPPET_CHARS - 1] + "…"
            speaker = "the user" if str(turn.role) == "user" else "you"
            lines.append(f"- [{_when(turn.created_at)}] {speaker}: {body}")

        log.info("tools.memory_searched", query=query, matches=len(matches))
        return ToolResult.success(
            call,
            "\n".join(lines),
            data={
                "query": query,
                "matches": len(matches),
                "sessions": sorted({m.session_id for m in matches}),
            },
        )


def memory_tools(store: MemoryStore | None) -> list[Tool]:
    """Nothing without a store.

    The CLI and several tests build a registry with no database at all. Offering
    a search tool that cannot search would be worse than not offering one: the
    model would call it, get an error, and have no way to tell a broken tool
    from an empty memory.
    """
    if store is None:
        return []
    return [SearchMemoryTool(store)]
