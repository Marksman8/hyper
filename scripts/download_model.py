"""Download the active model's GGUF weights from Hugging Face.

Uses a plain HTTPS range-resumable download rather than `huggingface_hub`, so
AURA gains no dependency for a task that is one GET request. Public repos only;
if a gated model is ever needed, set HF_TOKEN and it is sent as a bearer token.

    uv run python scripts/download_model.py
    uv run python scripts/download_model.py --model qwen3-8b
    uv run python scripts/download_model.py --list
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aura.config import ModelEntry, get_model_catalogue, models_dir

console = Console()
HF_BASE = "https://huggingface.co"


def _url(entry: ModelEntry) -> str:
    return f"{HF_BASE}/{entry.repo_id}/resolve/main/{entry.filename}?download=true"


def _list_catalogue() -> None:
    cat = get_model_catalogue()
    table = Table("key", "model", "quant", "VRAM", "status")
    for key, entry in cat.catalogue.items():
        path = models_dir() / entry.filename
        status = "[green]downloaded[/green]" if path.is_file() else "[dim]missing[/dim]"
        if key == cat.active:
            key = f"[bold]{key}[/bold] *"
        table.add_row(key, entry.display_name, entry.quantization, f"{entry.vram_gb} GB", status)
    console.print(table)
    console.print("[dim]* active (set `active:` in config/models.yaml)[/dim]")


def download(entry: ModelEntry, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    resume_from = part.stat().st_size if part.is_file() else 0

    headers: dict[str, str] = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
        console.print(f"[dim]resuming from {resume_from / 1e9:.2f} GB[/dim]")

    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=300.0)) as client:
        with client.stream("GET", _url(entry), headers=headers) as resp:
            if resp.status_code == 416:
                # Range beyond EOF: the .part file is already complete.
                part.replace(dest)
                console.print("[green]already complete[/green]")
                return
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0)) + resume_from

            with (
                open(part, "ab" if resume_from else "wb") as fh,
                Progress(
                    TextColumn("[cyan]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress,
            ):
                task = progress.add_task(entry.filename, total=total, completed=resume_from)
                for block in resp.iter_bytes(chunk_size=1024 * 1024):
                    fh.write(block)
                    progress.update(task, advance=len(block))

    part.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download AURA model weights.")
    parser.add_argument("--model", help="Catalogue key (default: the active model).")
    parser.add_argument("--list", action="store_true", help="Show the catalogue and exit.")
    args = parser.parse_args()

    if args.list:
        _list_catalogue()
        return 0

    cat = get_model_catalogue()
    key = args.model or cat.active
    if key not in cat.catalogue:
        console.print(f"[red]Unknown model {key!r}.[/red] Known: {sorted(cat.catalogue)}")
        return 1

    entry = cat.catalogue[key]
    dest = models_dir() / entry.filename

    if dest.is_file():
        console.print(f"[green]Already present:[/green] {dest}")
        return 0

    console.print(
        f"[bold]{entry.display_name}[/bold]  ({entry.quantization}, ~{entry.vram_gb} GB VRAM)"
    )
    console.print(f"[dim]{entry.repo_id}[/dim]\n")

    try:
        download(entry, dest)
    except httpx.HTTPStatusError as exc:
        console.print(f"[red]HTTP {exc.response.status_code}[/red] fetching {exc.request.url}")
        if exc.response.status_code in (401, 403):
            # Hugging Face returns 401 for "does not exist" as well as for
            # "gated" and "private" - it deliberately does not distinguish, to
            # avoid leaking the existence of private repos. So check the
            # obvious cause first.
            console.print(
                "[yellow]A 401 from Hugging Face means the repo is gated, private, "
                "or does not exist under that name - it does not distinguish.[/yellow]"
            )
            console.print(f"  Check it resolves:  {HF_BASE}/{entry.repo_id}")
            console.print("  If it is gated:     accept the licence, then set HF_TOKEN.")
            console.print("  If it is a typo:    fix config/models.yaml.")
        if exc.response.status_code == 404:
            console.print(
                f"[yellow]Repo exists but {entry.filename!r} is not in it. "
                f"See https://huggingface.co/{entry.repo_id}/tree/main[/yellow]"
            )
        return 1
    except httpx.HTTPError as exc:
        console.print(f"[red]Download failed:[/red] {exc}")
        console.print("[dim]Re-run to resume from where it stopped.[/dim]")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Re-run to resume.[/yellow]")
        return 130

    console.print(f"\n[bold green]Saved[/bold green] {dest}")
    console.print("Next: scripts/start_model_server.ps1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
