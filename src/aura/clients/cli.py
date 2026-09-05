"""Terminal client for AURA Core.

Deliberately a *client*, talking to the same authenticated HTTP API any other
application would use, rather than importing the orchestrator in-process. Two
reasons: it exercises the real API surface on every use, and it keeps AURA Core
running as one long-lived process holding one warm model, which is the whole
point of the two-process design.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from aura.config import get_settings
from aura.security.auth import load_or_create_token

console = Console()


def _base_url() -> str:
    cfg = get_settings()
    return "http://" + cfg.api.host + ":" + str(cfg.api.port)


def _headers() -> dict[str, str]:
    cfg = get_settings()
    token = load_or_create_token(cfg.security.resolved_token_file(), cfg.security.api_token)
    return {"Authorization": "Bearer " + token}


def _unreachable(exc: Exception) -> None:
    console.print(
        Panel(
            "Cannot reach AURA Core at " + _base_url() + "\n\n"
            "Start it with:  [bold]aura serve[/bold]\n\n"
            "[dim]" + str(exc) + "[/dim]",
            title="[red]Core offline[/red]",
            border_style="red",
        )
    )


async def run_status() -> None:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(_base_url() + "/v1/status", headers=_headers())
    except httpx.HTTPError as exc:
        _unreachable(exc)
        return

    if resp.status_code >= 400:
        console.print("[red]" + str(resp.status_code) + "[/red] " + resp.text)
        return

    data = resp.json()
    p, r = data["provider"], data["resources"]

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Version", data["version"])
    table.add_row("Uptime", str(data["uptime_s"]) + " s")
    table.add_row(
        "Emergency stop",
        "[red]ENGAGED[/red]" if data["emergency_stop"] else "[green]off[/green]",
    )
    table.add_row(
        "Model",
        ("[green]" + str(p["model"]) + "[/green]" if p["available"] else "[red]unavailable[/red]"),
    )
    table.add_row("Endpoint", str(p["endpoint"] or p["kind"]))
    if not p["available"]:
        table.add_row("Detail", "[yellow]" + p["detail"] + "[/yellow]")
    table.add_row(
        "RAM", f"{r['ram_used_gb']:.1f} / {r['ram_total_gb']:.1f} GB  ({r['ram_percent']:.0f}%)"
    )
    if r.get("vram_total_gb"):
        table.add_row("VRAM", f"{r['vram_used_gb']:.1f} / {r['vram_total_gb']:.1f} GB")
    table.add_row("AURA process", f"{r['process_ram_mb']:.0f} MB")
    m = data["memory"]
    table.add_row(
        "Stored",
        f"{m['sessions']} sessions, {m['turns']} turns, {m['facts']} facts",
    )

    console.print(Panel(table, title="[bold]AURA[/bold]", border_style="cyan"))


async def run_chat(
    one_shot: str | None = None,
    session_id: str | None = None,
    stream: bool = True,
) -> None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=3.0)) as client:
        if one_shot:
            await _send(client, one_shot, session_id, stream)
            return

        console.print(
            Panel(
                "[bold]AURA[/bold]  ready\n[dim]/exit  /new  /status  /memory  /session[/dim]",
                border_style="cyan",
            )
        )
        while True:
            try:
                text = console.input("[bold cyan]you >[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye[/dim]")
                return
            if not text:
                continue
            if text in ("/exit", "/quit"):
                console.print("[dim]bye[/dim]")
                return
            if text == "/new":
                session_id = None
                console.print("[dim]started a new session[/dim]")
                continue
            if text == "/session":
                console.print("[dim]" + (session_id or "(none yet)") + "[/dim]")
                continue
            if text == "/status":
                await run_status()
                continue
            if text == "/memory":
                await _show_memory(client)
                continue

            session_id = await _send(client, text, session_id, stream) or session_id


async def _send(
    client: httpx.AsyncClient, text: str, session_id: str | None, stream: bool
) -> str | None:
    payload: dict[str, Any] = {"message": text, "stream": stream}
    if session_id:
        payload["session_id"] = session_id

    try:
        if not stream:
            resp = await client.post(_base_url() + "/v1/chat", json=payload, headers=_headers())
            if resp.status_code >= 400:
                _show_error(resp.json())
                return session_id
            data = resp.json()
            console.print("[bold magenta]aura>[/bold magenta] ", end="")
            console.print(Markdown(data["content"]))
            console.print(
                f"[dim]{data['model']}  {data['latency_ms']} ms  "
                f"{data['usage']['completion_tokens']} tok[/dim]\n"
            )
            return str(data["session_id"])

        return await _send_streaming(client, payload, session_id)
    except httpx.HTTPError as exc:
        _unreachable(exc)
        return session_id


async def _send_streaming(
    client: httpx.AsyncClient, payload: dict[str, Any], session_id: str | None
) -> str | None:
    collected: list[str] = []
    new_session = session_id

    async with client.stream(
        "POST", _base_url() + "/v1/chat", json=payload, headers=_headers()
    ) as resp:
        if resp.status_code >= 400:
            body = (await resp.aread()).decode("utf-8", errors="replace")
            try:
                _show_error(json.loads(body))
            except json.JSONDecodeError:
                console.print("[red]" + str(resp.status_code) + "[/red] " + body[:500])
            return session_id

        console.print("[bold magenta]aura>[/bold magenta]")
        # Re-rendering markdown on every token is too slow to keep up with a
        # local model, so stream plain text and let the caller re-read it as
        # markdown in the scrollback if they want.
        with Live(console=console, refresh_per_second=12, transient=False) as live:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    frame = json.loads(data)
                except json.JSONDecodeError:
                    continue

                kind = frame.get("type")
                if kind == "error":
                    live.update("")
                    _show_error(frame)
                    return session_id
                if kind == "delta":
                    if frame.get("session_id"):
                        new_session = str(frame["session_id"])
                    collected.append(frame["content"])
                    live.update("".join(collected))

    console.print()
    return new_session


async def _show_memory(client: httpx.AsyncClient) -> None:
    try:
        resp = await client.get(_base_url() + "/v1/memory", headers=_headers())
    except httpx.HTTPError as exc:
        _unreachable(exc)
        return
    facts = resp.json()
    if not facts:
        console.print("[dim]AURA has not stored anything about you yet.[/dim]")
        return
    table = Table("key", "value", "conf", "seen", "permanence")
    for f in facts:
        table.add_row(
            f["key"],
            f["value"],
            f"{f['confidence']:.2f}",
            str(f["evidence_count"]),
            f["permanence"],
        )
    console.print(table)


def _show_error(payload: dict[str, Any]) -> None:
    err = payload.get("error", payload)
    console.print(
        Panel(
            str(err.get("message", err)),
            title="[red]" + str(err.get("code", "error")) + "[/red]",
            border_style="red",
        )
    )
