"""Download the active vision-language model (10C.2).

    uv run python scripts/download_vlm.py
    uv run python scripts/download_vlm.py --model smolvlm2-2.2b
    uv run python scripts/download_vlm.py --list

**A VLM is two files.** The weights, and an `mmproj` vision projector that
turns pixels into tokens the model can attend to. Both are fetched here and
both are required — a VLM GGUF loaded without its projector starts happily,
answers questions, and cannot see, which reads as a bad model rather than a
missing file. That failure is worth spending a few lines to make impossible.

Same plain-HTTPS resumable download as `download_model.py`: no dependency on
`huggingface_hub` for what is two GET requests.
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

from aura.config import VisionModelEntry, get_model_catalogue, models_dir

console = Console()
HF_BASE = "https://huggingface.co"

# Where VLM weights live: beside the language model, but in their own
# directory so "what is taking up my disk" has an obvious answer.
def vision_models_dir() -> Path:
    return models_dir() / "vision"


def _url(repo_id: str, filename: str) -> str:
    return f"{HF_BASE}/{repo_id}/resolve/main/{filename}?download=true"


def _list_catalogue() -> None:
    cat = get_model_catalogue()
    table = Table("key", "model", "VRAM", "weights", "projector")
    for key, entry in cat.vision_catalogue.items():
        out = vision_models_dir()
        weights = "[green]yes[/green]" if (out / entry.filename).is_file() else "[dim]no[/dim]"
        proj = "[green]yes[/green]" if (out / entry.mmproj).is_file() else "[dim]no[/dim]"
        label = f"[bold]{key}[/bold] *" if key == cat.active_vision else key
        table.add_row(label, entry.display_name, f"{entry.vram_gb} GB", weights, proj)
    console.print(table)
    console.print("[dim]* active (set `active_vision:` in config/models.yaml)[/dim]")
    console.print(
        "\n[yellow]These do not run alongside the language model.[/yellow] "
        "3.3 GB of Qwen3.5 plus ~2.9 GB of vision does not fit on an 8 GB card "
        "that is also driving the display. 10C.3 evicts one to load the other."
    )


def download(repo_id: str, filename: str, dest: Path) -> None:
    """Resumable fetch of one file."""
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
        with client.stream("GET", _url(repo_id, filename), headers=headers) as resp:
            if resp.status_code == 416:
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
                task = progress.add_task(filename, total=total, completed=resume_from)
                for block in resp.iter_bytes(chunk_size=1024 * 1024):
                    fh.write(block)
                    progress.update(task, advance=len(block))

    part.replace(dest)


def _fetch_pair(entry: VisionModelEntry) -> int:
    out = vision_models_dir()
    wanted = [
        ("weights", entry.filename),
        # Second, and never skipped independently: the projector is the half
        # that makes it a vision model at all.
        ("projector", entry.mmproj),
    ]

    for label, filename in wanted:
        dest = out / filename
        if dest.is_file():
            console.print(f"[green]{label} already present:[/green] {filename}")
            continue
        console.print(f"[bold]{label}[/bold]  {filename}")
        try:
            download(entry.repo_id, filename, dest)
        except httpx.HTTPStatusError as exc:
            console.print(f"[red]HTTP {exc.response.status_code}[/red] for {exc.request.url}")
            if exc.response.status_code in (401, 403):
                console.print("[dim]Gated repo? Set HF_TOKEN and retry.[/dim]")
            return 1
        except httpx.HTTPError as exc:
            console.print(f"[red]Download failed:[/red] {exc}")
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Download an AURA vision model.")
    parser.add_argument("--model", help="Catalogue key (default: active_vision).")
    parser.add_argument("--list", action="store_true", help="Show the catalogue and exit.")
    args = parser.parse_args()

    if args.list:
        _list_catalogue()
        return 0

    cat = get_model_catalogue()
    key = args.model or cat.active_vision
    if key not in cat.vision_catalogue:
        console.print(f"[red]Unknown vision model {key!r}.[/red]")
        console.print(f"Known: {sorted(cat.vision_catalogue)}")
        return 1

    entry = cat.vision_catalogue[key]
    console.print(
        f"[bold]{entry.display_name}[/bold]  ({entry.quantization}, ~{entry.vram_gb} GB VRAM)"
    )
    console.print(f"[dim]{entry.repo_id}[/dim]\n")

    status = _fetch_pair(entry)
    if status:
        return status

    out = vision_models_dir()
    total = sum((out / f).stat().st_size for f in (entry.filename, entry.mmproj))
    console.print(f"\n[bold green]Ready.[/bold green] {total / 1e9:.2f} GB in {out}")
    console.print("Start it with: [cyan]scripts/start_vlm_server.ps1[/cyan]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
