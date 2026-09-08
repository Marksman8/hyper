"""Seeing what AURA adapts, and testing whether it helps.

`aura profile` exists because silent adaptation is a debugging nightmare.
Personalization changes behaviour every turn without being asked, so when
something reads oddly the user has to be able to see what is being applied and
trace it back to something they said. Otherwise "AURA is being weird today" is
unfalsifiable.

`aura ab` is the honest version of the phase exit criterion. It is an
instrument, not a verdict: it shows two answers blind, records which was
picked, and reports counts without pretending twelve judgements prove anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aura.clients.cli import _base_url, _headers, _unreachable
from aura.personalization.ab import ABLog, present, resolve

console = Console()


async def _get(path: str) -> Any:
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(_base_url() + path, headers=_headers())
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        _unreachable(exc)
        return None


async def _ask(message: str, personalize: bool) -> str | None:
    """One non-streaming turn, with adaptation on or off."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0)) as client:
            resp = await client.post(
                _base_url() + "/v1/chat",
                json={
                    "message": message,
                    "personalize": personalize,
                    # No session id: each arm must not see the other's turn, or
                    # the second answer is conditioned on the first.
                    "session_id": None,
                },
                headers=_headers(),
            )
            resp.raise_for_status()
            return str(resp.json().get("content", ""))
    except httpx.HTTPError as exc:
        _unreachable(exc)
        return None


async def run_profile() -> None:
    payload = await _get("/v1/profile")
    if payload is None:
        return

    if not payload["enabled"]:
        console.print("[yellow]Personalization is switched off in config.[/yellow]\n")

    dimensions = payload.get("dimensions") or []
    if not dimensions:
        console.print(
            "[dim]AURA is not adapting anything yet. It needs a preference it "
            "believes -- see[/dim] [bold]aura memory list[/bold][dim].[/dim]"
        )
    else:
        table = Table(box=None, padding=(0, 2, 0, 0))
        table.add_column("adapts", style="bold")
        table.add_column("to")
        table.add_column("because of fact")
        table.add_column("conf", justify="right")
        table.add_column("changes sampling?")
        for d in dimensions:
            table.add_row(
                d["dimension"],
                d["value"],
                d["from_fact"],
                f"{d['confidence']:.2f}",
                "[green]yes[/green]" if d["drives_parameters"] else "[dim]no (inferred)[/dim]",
            )
        console.print(table)

    directives = payload.get("directives") or []
    if directives:
        console.print(
            Panel(
                "\n".join("· " + d for d in directives),
                title="[bold]Added to the system prompt[/bold]",
                border_style="blue",
            )
        )

    others = payload.get("other_facts") or []
    if others:
        console.print("\n[bold]Known, but not driving behaviour:[/bold]")
        for fact in others:
            console.print("  · " + fact)

    console.print(
        "\n[dim]Only preferences you stated outright change sampling. Inferred "
        "ones shape the writing but never the token budget --\n"
        "a wrong directive is a differently-toned answer; a wrong limit is a "
        "truncated one.[/dim]"
    )


async def run_ab(prompt: str, log_path: Path) -> None:
    """One blind comparison."""
    console.print("[dim]Generating both answers…[/dim]")
    adapted = await _ask(prompt, personalize=True)
    if adapted is None:
        return
    baseline = await _ask(prompt, personalize=False)
    if baseline is None:
        return

    if adapted.strip() == baseline.strip():
        console.print(
            "[yellow]Both answers are identical.[/yellow] Nothing is being adapted for "
            "this prompt, so there is nothing to judge.\nCheck [bold]aura profile[/bold]."
        )
        return

    comparison, first, second = present(prompt, adapted, baseline)
    console.print(Panel(first, title="[bold]A[/bold]", border_style="white"))
    console.print(Panel(second, title="[bold]B[/bold]", border_style="white"))

    choice = console.input("Which is better? [bold]A[/bold] / [bold]B[/bold] / [dim]tie[/dim]: ")
    if choice.strip().upper() not in ("A", "B", "TIE"):
        console.print("[dim]Not recorded.[/dim]")
        return

    log = ABLog(log_path)
    log.record(resolve(comparison, choice))
    console.print("\n" + log.tally().summary())


def show_tally(log_path: Path) -> None:
    tally = ABLog(log_path).tally()
    console.print(tally.summary())
    if tally.judged:
        console.print(
            f"[dim]The adapted answer was shown first {tally.adapted_shown_first} "
            f"of {tally.judged} times.[/dim]"
        )
