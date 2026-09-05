"""Terminal client for the agent loop.

Exists because the alternative was a curl command, and on Windows `curl` is an
alias for `Invoke-WebRequest`, which takes `-Headers` as a hashtable rather
than repeated `-H` flags. Every example written in bash syntax fails there with
a parameter-binding error that says nothing about the actual problem.

So: `aura do "..."`, which works the same in PowerShell, cmd and bash.

It shows the tool calls as they are reported rather than only the final answer,
because the interesting part of an agent turn is usually what it decided to
look at.
"""

from __future__ import annotations

from typing import Any

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from aura.config import get_settings
from aura.security.auth import load_or_create_token

console = Console()

_DECISION_STYLE = {
    "allow": "green",
    "confirm": "yellow",
    "deny": "red",
    "error": "red",
}


def _base_url() -> str:
    cfg = get_settings()
    return f"http://{cfg.api.host}:{cfg.api.port}"


def _headers() -> dict[str, str]:
    cfg = get_settings()
    token = load_or_create_token(cfg.security.resolved_token_file(), cfg.security.api_token)
    return {"Authorization": f"Bearer {token}"}


async def run_agent(
    message: str,
    approve: list[str],
    session_id: str | None = None,
    show_trace: bool = True,
) -> None:
    """Send one task and print what AURA did about it."""
    payload: dict[str, Any] = {"message": message, "approve": approve}
    if session_id:
        payload["session_id"] = session_id

    if approve:
        console.print(f"[dim]pre-approved for this task: {', '.join(approve)}[/dim]")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=3.0)) as client:
            with console.status("[dim]thinking...[/dim]", spinner="dots"):
                resp = await client.post(
                    _base_url() + "/v1/agent", json=payload, headers=_headers()
                )
    except httpx.HTTPError as exc:
        console.print(
            Panel(
                f"Cannot reach AURA Core at {_base_url()}\n\n"
                "Start it with:  [bold]aura serve[/bold]\n\n"
                f"[dim]{exc}[/dim]",
                title="[red]Core offline[/red]",
                border_style="red",
            )
        )
        return

    if resp.status_code >= 400:
        _show_error(resp)
        return

    data = resp.json()

    if show_trace:
        for step in data.get("steps", []):
            for call in step.get("tools", []):
                decision = "allow" if call["ok"] else call["decision"]
                style = _DECISION_STYLE.get(decision, "white")
                args = ", ".join(f"{k}={_short(v)}" for k, v in call["arguments"].items())
                console.print(
                    f"  [{style}]*[/{style}] [bold]{call['name']}[/bold]({args}) "
                    f"[{style}]{decision}[/{style}]"
                )

    console.print()
    console.print(Markdown(data["content"] or "_(no answer)_"))
    console.print()

    bits = [
        data["model"],
        f"{data['tool_calls_made']} tool call{'s' if data['tool_calls_made'] != 1 else ''}",
        f"{data['latency_ms']} ms",
    ]
    if data.get("stopped_early"):
        bits.append(f"[yellow]stopped early: {data['stop_reason']}[/yellow]")
    console.print(f"[dim]{'  |  '.join(bits)}[/dim]")

    pending = data.get("pending_confirmations") or []
    if pending:
        # The model wanted to do something and was refused. Say how to allow it,
        # rather than leaving the user to work out that an approval exists.
        flags = " ".join(f"--approve {name}" for name in pending)
        console.print(
            Panel(
                f"AURA wanted to use: [bold]{', '.join(pending)}[/bold]\n\n"
                f'Re-run with approval:\n  [bold]aura do {flags} "{message}"[/bold]',
                title="[yellow]Needs your approval[/yellow]",
                border_style="yellow",
            )
        )


def _short(value: Any, limit: int = 40) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _show_error(resp: httpx.Response) -> None:
    try:
        err = resp.json().get("error", {})
        message = err.get("message", resp.text)
        code = err.get("code", str(resp.status_code))
    except ValueError:
        message, code = resp.text[:500], str(resp.status_code)
    console.print(Panel(str(message), title=f"[red]{code}[/red]", border_style="red"))
