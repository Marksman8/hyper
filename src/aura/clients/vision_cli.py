"""Terminal controls for what AURA may look at.

**A client, not a local mutation.** The privacy policy lives in the AURA Core
process, because that is where captures happen. A CLI that changed its own
in-process copy would print "vision off" and leave the running server looking
at your screen — a silent failure of the one promise this subsystem makes. So
every command here goes over the same authenticated HTTP API the desktop shell
uses.
"""

from __future__ import annotations

from typing import Any

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aura.clients.cli import _base_url, _headers, _unreachable

console = Console()


async def _request(
    method: str, path: str, payload: dict[str, Any] | None = None, timeout_s: float = 15.0
) -> Any:
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.request(
                method, _base_url() + path, json=payload, headers=_headers()
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        console.print(f"[red]{exc.response.status_code}[/red] {exc.response.text}")
        return None
    except httpx.HTTPError as exc:
        _unreachable(exc)
        return None


def _render(status: dict[str, Any] | None) -> None:
    if status is None:
        return

    mode = str(status.get("mode", "unknown"))
    colour = {"private": "green", "sleep": "green", "assist": "yellow", "active": "red"}
    explain = {
        "private": "Vision is OFF. No screenshots are taken.",
        "sleep": "AURA is asleep; vision is paused.",
        "assist": "AURA looks only when you ask.",
        "active": "AURA may look when you ask and on events.",
    }

    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_row("Mode", f"[{colour.get(mode, 'white')}]{mode}[/]")
    table.add_row("", explain.get(mode, ""))
    if status.get("denied_apps"):
        table.add_row("Never captured", ", ".join(status["denied_apps"]))
    if status.get("allowed_apps"):
        table.add_row("Restricted to", ", ".join(status["allowed_apps"]))
    if status.get("denied_monitors"):
        table.add_row("Excluded screens", ", ".join(str(m) for m in status["denied_monitors"]))
    table.add_row(
        "Credential windows",
        f"{status.get('deny_title_patterns', 0)} patterns always blocked",
    )
    ocr = status.get("ocr") or {}
    table.add_row("OCR", f"{ocr.get('name', '?')} ({'loaded' if ocr.get('loaded') else 'idle'})")

    console.print(Panel(table, title="[bold]Vision[/bold]", border_style="blue"))


async def run_status() -> None:
    _render(await _request("GET", "/v1/vision"))


async def run_set_mode(mode: str) -> None:
    _render(await _request("POST", "/v1/vision/mode", {"mode": mode}))


async def run_deny(app: str) -> None:
    _render(await _request("POST", "/v1/vision/deny", {"app": app}))


async def run_allow(app: str) -> None:
    _render(await _request("POST", "/v1/vision/allow", {"app": app}))


async def run_read(monitor: int) -> None:
    """Capture and OCR once, through the tool the model would use.

    Going through /v1/tools rather than a private path means this exercises the
    permission gate and the privacy gate exactly as a real call would, so a
    successful `aura vision read` is evidence about the real system.
    """
    payload = {"arguments": {"monitor": monitor}, "confirmed": True}
    # Generous, because the first read of a session also loads the OCR model
    # (~4s measured). A tighter client timeout would report 'core offline'
    # for a server that is working perfectly.
    result = await _request("POST", "/v1/tools/read_screen", payload, timeout_s=90.0)
    if result is None:
        return

    if not result.get("ok"):
        console.print(
            Panel(
                str(result.get("error") or "Refused."),
                title="[yellow]Refused[/yellow]",
                border_style="yellow",
            )
        )
        return

    data = result.get("data") or {}
    console.print(
        Panel(
            result.get("content", ""),
            title=f"[green]Monitor {monitor}[/green] · {data.get('lines', 0)} lines · "
            f"{data.get('ocr_ms', 0)} ms",
            border_style="green",
        )
    )


async def run_monitors() -> None:
    result = await _request("POST", "/v1/tools/list_monitors", {"arguments": {}})
    if result is None:
        return
    console.print(result.get("content") or result.get("error") or "No displays found.")
