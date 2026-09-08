"""Terminal view of what AURA knows, and why.

The Phase 6 exit criterion lives here: `aura memory why <key>` has to show the
actual words that produced a belief. A confidence score on its own is not a
justification -- the user cannot check it, disagree with it, or find the
conversation it came from.

A client of the HTTP API like every other CLI in this repo, so the facts shown
are the ones the running core would actually use.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aura.clients.cli import _base_url, _headers, _unreachable

console = Console()


async def _request(method: str, path: str, **params: Any) -> Any:
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.request(
                method, _base_url() + path, params=params or None, headers=_headers()
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        console.print(f"[red]{exc.response.status_code}[/red] {exc.response.text}")
        return None
    except httpx.HTTPError as exc:
        _unreachable(exc)
        return None


def _when(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _confidence(value: float, believed: bool) -> str:
    colour = "green" if believed else "yellow"
    return f"[{colour}]{value:.2f}[/]"


async def run_list(floor: float = 0.5) -> None:
    facts = await _request("GET", "/v1/memory")
    if facts is None:
        return
    if not facts:
        console.print("[dim]AURA has not learned anything about you yet.[/dim]")
        return

    table = Table(box=None, padding=(0, 2, 0, 0))
    table.add_column("key", style="bold")
    table.add_column("value")
    table.add_column("conf", justify="right")
    table.add_column("seen", justify="right")
    table.add_column("source")

    for f in facts:
        believed = f["confidence"] >= floor
        table.add_row(
            f["key"],
            f["value"],
            _confidence(f["confidence"], believed),
            str(f["evidence_count"]),
            f["source"],
        )

    console.print(table)
    console.print(
        "\n[dim]Green is acted on. Yellow is noticed but not yet believed -- it needs "
        "corroboration on a separate occasion.\nRun [/dim][bold]aura memory why <key>"
        "[/bold][dim] to see the evidence.[/dim]"
    )


async def run_why(key: str) -> None:
    """The exit criterion: show the words behind a belief."""
    payload = await _request("GET", f"/v1/memory/{key}/why")
    if payload is None:
        console.print(f"[yellow]AURA knows nothing under '{key}'.[/yellow]")
        return

    fact = payload["fact"]
    believed = payload["believed"]
    head = (
        f"[bold]{fact['key']}[/bold]: {fact['value']}\n\n"
        f"Confidence {_confidence(fact['confidence'], believed)} — {payload['reason']}\n"
    )
    head += (
        "[green]This is used in conversation.[/green]"
        if believed
        else "[yellow]Not used yet: below the threshold AURA acts on.[/yellow]"
    )

    evidence = payload.get("evidence") or []
    if evidence:
        head += "\n\n[bold]Because you said:[/bold]"
        for e in evidence:
            marker = "[green]said directly[/green]" if e["source"] == "explicit" else "inferred"
            head += f'\n  · "{e["quote"]}"\n    [dim]{_when(e["created_at"])} · {marker}[/dim]'
    else:
        head += "\n\n[red]No evidence recorded.[/red]"

    console.print(Panel(head, border_style="green" if believed else "yellow"))


async def run_search(query: str, limit: int = 8) -> None:
    hits = await _request("GET", "/v1/memory/search", q=query, limit=limit)
    if hits is None:
        return
    if not hits:
        console.print(f"[dim]Nothing in past conversations matches '{query}'.[/dim]")
        return

    for h in hits:
        text = h["content"]
        if len(text) > 300:
            text = text[:300] + "…"
        console.print(
            Panel(
                text,
                title=f"[dim]{h['role']} · {_when(h['created_at'])}[/dim]",
                subtitle=f"[dim]session {h['session_id'][:8]}[/dim]",
                border_style="blue",
            )
        )


async def run_forget(key: str) -> None:
    """Deletion is complete: the fact and every piece of evidence behind it."""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.delete(_base_url() + f"/v1/memory/{key}", headers=_headers())
            resp.raise_for_status()
            gone = bool(resp.json().get("deleted"))
    except httpx.HTTPError as exc:
        _unreachable(exc)
        return

    if gone:
        console.print(f"[green]Forgotten:[/green] {key} (and the evidence behind it).")
    else:
        console.print(f"[yellow]AURA knew nothing under '{key}'.[/yellow]")
