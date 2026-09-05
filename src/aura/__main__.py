"""AURA command line.

aura serve      run the core (API + orchestrator + memory)
aura chat       interactive client against a running core
aura status     one-shot status report
aura doctor     check this machine can actually run AURA
"""

from __future__ import annotations

import asyncio
import sys

import typer
from rich.console import Console

from aura import __version__
from aura.config import get_model_catalogue, get_settings, models_dir
from aura.logging import configure_logging, get_logger

app = typer.Typer(
    name="aura",
    help="AURA - local-first personal AI runtime.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
log = get_logger(__name__)


@app.command()
def serve(
    host: str = typer.Option("", help="Override api.host (loopback only)."),
    port: int = typer.Option(0, help="Override api.port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
) -> None:
    """Run AURA Core."""
    import uvicorn

    cfg = get_settings()
    configure_logging(
        level=cfg.logging.level,
        fmt=cfg.logging.format,
        log_file=cfg.logging.resolved_file(),
        redact_keys=tuple(cfg.logging.redact_keys),
    )

    bind_host = host or cfg.api.host
    bind_port = port or cfg.api.port

    if reload:
        # --reload needs an import string so the reloader can re-import.
        uvicorn.run(
            "aura.api.app:create_app",
            factory=True,
            host=bind_host,
            port=bind_port,
            reload=True,
            log_config=None,
        )
    else:
        from aura.api.app import create_app

        uvicorn.run(create_app(cfg), host=bind_host, port=bind_port, log_config=None)


@app.command()
def chat(
    message: str = typer.Argument("", help="Send one message and exit."),
    session: str = typer.Option("", "--session", "-s", help="Continue a session id."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Wait for the full reply."),
) -> None:
    """Talk to a running AURA Core."""
    from aura.clients.cli import run_chat

    asyncio.run(
        run_chat(
            one_shot=message or None,
            session_id=session or None,
            stream=not no_stream,
        )
    )


@app.command()
def status() -> None:
    """Print AURA Core status."""
    from aura.clients.cli import run_status

    asyncio.run(run_status())


@app.command()
def doctor() -> None:
    """Check that this machine can run AURA, and say what is missing."""
    from aura.runtime.resources import snapshot

    cfg = get_settings()
    res = snapshot()
    problems: list[str] = []

    console.print(f"[bold]AURA {__version__}[/bold]  doctor\n")

    py = sys.version_info
    py_ok = (3, 12) <= (py.major, py.minor) < (3, 13)
    _row("Python", f"{py.major}.{py.minor}.{py.micro}", py_ok)
    if not py_ok:
        problems.append(
            "Python 3.12 is required. Phase 2+ dependencies (torch, whisper) "
            "have no 3.13/3.14 wheels. Run: uv sync"
        )

    _row("RAM", f"{res.ram_used_gb:.1f} / {res.ram_total_gb:.1f} GB used", res.ram_percent < 90)
    if res.ram_total_gb < 15:
        problems.append("Under 16 GB RAM. Expect to run the 4B model only.")

    if res.vram_total_gb is not None:
        free = res.vram_total_gb - (res.vram_used_gb or 0)
        _row("VRAM", f"{free:.1f} GB free of {res.vram_total_gb:.1f} GB", free > 3.5)
        if free < 3.5:
            problems.append(
                f"Only {free:.1f} GB VRAM free; the default model needs ~3.5 GB. "
                "Close GPU-heavy applications."
            )
    else:
        _row("GPU", "no nvidia-smi found", False)
        problems.append("No NVIDIA GPU detected. AURA will fall back to slow CPU inference.")

    cat = get_model_catalogue()
    entry = cat.active_entry()
    weights = models_dir() / entry.filename
    _row("Model", f"{entry.display_name} ({entry.quantization})", weights.is_file())
    if not weights.is_file():
        problems.append(f"Weights missing at {weights}. Run: python scripts/download_model.py")

    _row("Provider", f"{cfg.llm.provider} -> {cfg.llm.base_url}", True)

    console.print()
    if problems:
        console.print("[bold yellow]Action required:[/bold yellow]")
        for p in problems:
            console.print(f"  - {p}")
        raise typer.Exit(1)
    console.print("[bold green]All checks passed.[/bold green]")


def _row(label: str, value: str, ok: bool) -> None:
    mark = "[green]OK [/green]" if ok else "[red]!! [/red]"
    console.print(f"  {mark} {label:<10} {value}")


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
