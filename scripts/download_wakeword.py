"""Download openWakeWord's pretrained models.

A few MB, fetched once, so a voice session never stalls on the network.

    uv run python scripts/download_wakeword.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aura.voice.wake.openwakeword import PRETRAINED, ensure_pretrained_models

console = Console()


def main() -> int:
    console.print("[bold]openWakeWord[/bold] pretrained models")
    console.print(f"[dim]available phrases: {', '.join(PRETRAINED)}[/dim]\n")
    try:
        ensure_pretrained_models()
    except Exception as exc:
        console.print(f"[red]Download failed:[/red] {exc}")
        return 1
    console.print("[bold green]Done.[/bold green]")
    console.print(
        '[dim]"aura" is not a pretrained phrase - see docs/architecture/01-phase2-voice.md[/dim]'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
