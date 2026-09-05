"""Put the Silero VAD model where AURA expects it.

Silero VAD is ~2 MB of ONNX that decides, frame by frame, whether the
microphone is hearing speech. AURA uses it for endpointing (when did the user
stop talking) and barge-in (did they start talking over the reply).

faster-whisper already ships a copy, so the fast path is a file copy with no
network. The download is the fallback.

    uv run python scripts/download_vad.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import httpx
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aura.voice.vad import vad_model_path

console = Console()

# Upstream, if faster-whisper is not installed or moves its assets.
UPSTREAM = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"


def _bundled_copy() -> Path | None:
    """faster-whisper's bundled asset, if present."""
    try:
        import faster_whisper
    except ImportError:
        return None
    assets = Path(faster_whisper.__file__).parent / "assets"
    for name in ("silero_vad_v6.onnx", "silero_vad.onnx", "silero_encoder_v5.onnx"):
        candidate = assets / name
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    dest = vad_model_path()
    if dest.is_file():
        console.print(f"[green]Already present:[/green] {dest}")
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)

    bundled = _bundled_copy()
    if bundled is not None:
        shutil.copy2(bundled, dest)
        console.print(f"[green]Copied[/green] {bundled.name} -> {dest}")
        return 0

    console.print("[dim]not bundled locally; downloading...[/dim]")
    try:
        resp = httpx.get(UPSTREAM, follow_redirects=True, timeout=60.0)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    except httpx.HTTPError as exc:
        console.print(f"[red]Download failed:[/red] {exc}")
        return 1

    console.print(f"[green]Saved[/green] {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
