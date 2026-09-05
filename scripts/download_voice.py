"""Download the Piper TTS voice named in config.

Piper voices live in the rhasspy/piper-voices repo as an .onnx plus an
.onnx.json config; both are needed. Same plain-HTTPS approach as
download_model.py -- no extra dependency for two GET requests.

    uv run python scripts/download_voice.py
    uv run python scripts/download_voice.py --voice en_GB-jenny_dioco-medium
    uv run python scripts/download_voice.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aura.config import get_settings, voice_models_dir

console = Console()
REPO = "rhasspy/piper-voices"
BASE = f"https://huggingface.co/{REPO}/resolve/main"

# Curated English voices. Piper publishes ~100; these are the ones worth
# defaulting to, with the female voices AURA's spec asks for listed first.
KNOWN: dict[str, tuple[str, str]] = {
    "en_US-amy-medium": ("en/en_US/amy/medium", "female, US, natural - AURA default"),
    "en_US-kristin-medium": ("en/en_US/kristin/medium", "female, US, warmer"),
    "en_US-hfc_female-medium": ("en/en_US/hfc_female/medium", "female, US, clear"),
    "en_GB-jenny_dioco-medium": ("en/en_GB/jenny_dioco/medium", "female, UK"),
    "en_US-lessac-medium": ("en/en_US/lessac/medium", "neutral, US, crisp"),
    "en_US-amy-low": ("en/en_US/amy/low", "female, US, smallest and fastest"),
}


def _list() -> None:
    table = Table("voice", "description", "status")
    active = get_settings().voice.tts.voice
    for name, (_, desc) in KNOWN.items():
        have = (voice_models_dir() / f"{name}.onnx").is_file()
        label = f"[bold]{name}[/bold] *" if name == active else name
        table.add_row(label, desc, "[green]downloaded[/green]" if have else "[dim]missing[/dim]")
    console.print(table)
    console.print("[dim]* active (set voice.tts.voice in config/default.yaml)[/dim]")


def _fetch(client: httpx.Client, url: str, dest: Path) -> None:
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(dest, "wb") as fh:
            for block in resp.iter_bytes(chunk_size=256 * 1024):
                fh.write(block)
                written += len(block)
        console.print(f"  [green]saved[/green] {dest.name} ({written / 1e6:.1f} MB, of {total})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a Piper TTS voice.")
    parser.add_argument("--voice", help="Voice name (default: the configured one).")
    parser.add_argument("--list", action="store_true", help="Show known voices and exit.")
    args = parser.parse_args()

    if args.list:
        _list()
        return 0

    name = args.voice or get_settings().voice.tts.voice
    if name not in KNOWN:
        console.print(f"[yellow]{name!r} is not in the curated list.[/yellow]")
        console.print(f"Known: {sorted(KNOWN)}")
        console.print(f"Any voice from https://huggingface.co/{REPO} works; add it to KNOWN.")
        return 1

    subdir, desc = KNOWN[name]
    out_dir = voice_models_dir()
    onnx = out_dir / f"{name}.onnx"
    conf = out_dir / f"{name}.onnx.json"

    if onnx.is_file() and conf.is_file():
        console.print(f"[green]Already present:[/green] {onnx}")
        return 0

    console.print(f"[bold]{name}[/bold]  ({desc})")
    try:
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=300.0)) as c:
            if not onnx.is_file():
                _fetch(c, f"{BASE}/{subdir}/{name}.onnx", onnx)
            if not conf.is_file():
                _fetch(c, f"{BASE}/{subdir}/{name}.onnx.json", conf)
    except httpx.HTTPStatusError as exc:
        console.print(f"[red]HTTP {exc.response.status_code}[/red] fetching {exc.request.url}")
        return 1
    except httpx.HTTPError as exc:
        console.print(f"[red]Download failed:[/red] {exc}")
        return 1

    console.print('\n[bold green]Ready.[/bold green] Try: uv run aura say "Hello, I am AURA."')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
