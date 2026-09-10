"""Render the micro-expression catalogue to fixed WAV files (10B.6).

    uv run python scripts/render_micro_expressions.py
    uv run python scripts/render_micro_expressions.py --keep-voices

**Why pre-render at all.** Piper voices are per-language, so AURA's own
`en_US-amy-medium` cannot pronounce 了解 — it would read the characters as
nothing, or mangle them. Speaking Japanese at runtime would mean loading a
second TTS model, which is RAM this machine does not have spare. Fixed WAVs
give correct pronunciation, zero runtime cost, and one property neither
alternative offers: **every sound AURA can make in another language is a file
you can listen to before she ever says it.**

This is a BUILD step. It downloads two extra Piper voices, renders eight short
clips, and then the voices are no longer needed — the default is to delete
them, because 130 MB of models to produce 200 KB of audio is not a thing to
keep lying around.

Both voices are female speakers, deliberately: `hi_fi_captain` ships a female
speaker id and SIWIS is female. A micro-expression in a male voice ahead of
AURA's own would sound like a second person in the room.

Clip loudness is matched to AURA's own measured speaking level, so an
interjection sits inside the reply rather than sounding spliced in.

**Japanese needs a phonemizer that is not a project dependency.** Piper hands
Japanese to `pyopenjtalk`, which builds from source and needs cmake and a C++
toolchain. `pyopenjtalk` itself has no Windows wheel; `pyopenjtalk-plus` does:

    uv pip install pyopenjtalk-plus

It is deliberately not in pyproject.toml. The clips are committed, so nobody
needs it unless they are changing the catalogue — and adding a compiler
dependency to every checkout to support one build script would be the wrong
trade. `uv sync` will remove it again, which is correct.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import wave
from pathlib import Path

import httpx
import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aura.config import PROJECT_ROOT, get_settings, voice_models_dir
from aura.voice.personality import CATALOGUE

console = Console()
BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
BUILD_DIR = PROJECT_ROOT / "var" / "build" / "piper-voices"
OUT_DIR = PROJECT_ROOT / "assets" / "micro"

# One voice per language, and the speaker within it. Female in both, to sit
# beside AURA's own voice rather than beside a different person.
VOICES: dict[str, tuple[str, str, int]] = {
    # language: (voice name, repo subdirectory, speaker id)
    "ja": ("ja_JA-hi_fi_captain-medium", "ja/ja_JA/hi_fi_captain/medium", 0),
    "fr": ("fr_FR-siwis-medium", "fr/fr_FR/siwis/medium", 0),
}


def _fetch(client: httpx.Client, url: str, dest: Path) -> None:
    if dest.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for block in resp.iter_bytes(chunk_size=256 * 1024):
                fh.write(block)
    console.print(f"  downloaded {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")


def _reference_level() -> float:
    """AURA's own speaking level, so the clips can be matched to it.

    Without this a clip lands noticeably louder or quieter than the sentence
    after it, which reads as a recording spliced in rather than as her saying
    something.
    """
    from aura.voice.factory import build_tts

    tts = build_tts(get_settings())
    chunk = next(iter(tts.synthesize("Understood, I will take a look at that.")))
    audio = np.asarray(chunk.audio, dtype=np.float32)
    return float(np.sqrt(np.mean(audio**2)))


def _write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes((pcm * 32767).astype("<i2").tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-voices",
        action="store_true",
        help="Do not delete the downloaded language voices afterwards.",
    )
    args = parser.parse_args()

    languages = sorted({expression.language for expression in CATALOGUE})
    missing = [lang for lang in languages if lang not in VOICES]
    if missing:
        console.print(f"[red]No voice configured for: {missing}[/red]")
        return 1

    target_rate = 22_050
    console.print("[bold]Measuring AURA's own speaking level[/bold]")
    try:
        reference = _reference_level()
    except Exception as exc:
        console.print(f"[yellow]Could not measure ({exc}); clips will not be matched.[/yellow]")
        reference = 0.0
    else:
        console.print(f"  reference RMS {reference:.4f}\n")

    from piper import PiperVoice, SynthesisConfig

    rendered: list[tuple[str, str, float, int]] = []

    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=600.0)) as client:
        for lang in languages:
            name, subdir, speaker = VOICES[lang]
            console.print(f"[bold]{lang}[/bold] — {name}")
            onnx = BUILD_DIR / f"{name}.onnx"
            conf = BUILD_DIR / f"{name}.onnx.json"
            _fetch(client, f"{BASE}/{subdir}/{name}.onnx", onnx)
            _fetch(client, f"{BASE}/{subdir}/{name}.onnx.json", conf)

            voice = PiperVoice.load(onnx, config_path=conf)
            rate = int(voice.config.sample_rate)
            syn = SynthesisConfig(speaker_id=speaker, normalize_audio=True)

            for expression in CATALOGUE:
                if expression.language != lang:
                    continue
                pieces = [
                    np.asarray(c.audio_int16_array, dtype=np.float32) / 32768.0
                    for c in voice.synthesize(expression.text, syn_config=syn)
                ]
                audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
                if audio.size == 0:
                    console.print(f"  [red]{expression.id}: produced no audio[/red]")
                    continue

                # Match AURA's level, so a clip does not jump out beside her.
                if reference > 0:
                    level = float(np.sqrt(np.mean(audio**2)))
                    if level > 1e-6:
                        audio = audio * min(4.0, reference / level)

                if rate != target_rate:
                    # Linear resample. Adequate for a half-second interjection;
                    # anything fancier would be a dependency for no audible gain.
                    n = int(len(audio) * target_rate / rate)
                    audio = np.interp(
                        np.linspace(0, len(audio), n, endpoint=False),
                        np.arange(len(audio)),
                        audio,
                    ).astype(np.float32)

                out = OUT_DIR / expression.clip
                _write_wav(out, audio, target_rate)
                rendered.append(
                    (expression.id, expression.gloss, len(audio) / target_rate, out.stat().st_size)
                )
                console.print(f"  [green]rendered[/green] {expression.clip}")
            console.print()

    table = Table("clip", "gloss", "seconds", "bytes")
    for cid, gloss, seconds, size in rendered:
        table.add_row(cid, gloss, f"{seconds:.2f}", f"{size:,}")
    console.print(table)
    total = sum(size for _, _, _, size in rendered)
    console.print(f"\n[bold]{len(rendered)} clips, {total / 1024:.0f} KB total[/bold]")
    console.print(f"Written to {OUT_DIR.relative_to(PROJECT_ROOT)}")

    if not args.keep_voices and BUILD_DIR.exists():
        freed = sum(f.stat().st_size for f in BUILD_DIR.rglob("*") if f.is_file())
        shutil.rmtree(BUILD_DIR)
        console.print(f"[dim]Removed the build voices ({freed / 1e6:.0f} MB).[/dim]")

    console.print(
        f"\n[dim]AURA's own voice ({voice_models_dir().name}) was not touched.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
