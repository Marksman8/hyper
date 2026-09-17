"""Synthesise one utterance and exit.

    python -m aura.voice.synth  < text-on-stdin  > wav-on-stdout

**Why a process and not a function call.** `/v1/speech` exists so a typed
conversation can be heard, and the obvious implementation -- Core imports
`build_tts` and keeps a voice resident -- was built, measured, and thrown away.
It worked, and it cost: Core went from 69 MB to 232 MB with the voice loaded and
only came back to 115 MB after eviction, because onnxruntime does not return its
arenas to the OS. `tests/unit/test_voice_seam.py` forbids exactly that import,
and the measurement agreed with the test.

A subprocess keeps every property that mattered:

* **Core never imports `aura.voice`**, so the two-process design holds and the
  seam test stays honest rather than being widened to fit.
* **The memory genuinely goes away.** Not evicted, not released-in-principle --
  the process exits.
* **It is AURA's own voice**, not whatever the browser's `speechSynthesis`
  happens to be, which would make her sound like a different person on every
  machine the console is opened on.

The cost is a model load per utterance, ~1.5 s, because nothing stays warm.
That is the trade: latency for a resident set. On a machine whose binding
constraint is 15.7 GB of RAM rather than CPU, it is the right way round, and the
same reasoning `run_python` uses for spawning rather than executing in-process.

stdout is binary and must stay that way -- a single stray `print` corrupts the
WAV -- so `main` grabs the real stdout handle and then points `sys.stdout` at
stderr before anything else runs. A test asserts stdout begins with b"RIFF",
because the first version did not do this and every file it produced started
with a log line.
"""

from __future__ import annotations

import io
import struct
import sys


def _wav_bytes(samples: list[float], sample_rate: int) -> bytes:
    """Wrap float samples in a 16-bit PCM WAV container.

    Values are clamped rather than cast: a float above 1.0 wraps to a loud
    negative, which is heard as a click rather than as clipping.
    """
    frames = bytearray()
    for value in samples:
        clamped = -1.0 if value < -1.0 else (1.0 if value > 1.0 else value)
        frames += struct.pack("<h", int(clamped * 32767.0))

    data = bytes(frames)
    out = io.BytesIO()
    out.write(b"RIFF")
    out.write(struct.pack("<I", 36 + len(data)))
    out.write(b"WAVEfmt ")
    out.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    out.write(b"data")
    out.write(struct.pack("<I", len(data)))
    out.write(data)
    return out.getvalue()


def main() -> int:
    # stdout is the WAV and nothing else. AURA configures structured logging on
    # import and it writes to stdout, so the first version of this emitted a log
    # line where the RIFF header should have been and every file was corrupt --
    # caught by a test asserting stdout starts with b"RIFF".
    #
    # Swapping the handle is deliberately cruder than reconfiguring the logger:
    # it holds no matter what any library decides to print, and the module
    # docstring's promise should be enforced rather than trusted.
    wav_out = sys.stdout.buffer
    sys.stdout = sys.stderr

    text = sys.stdin.buffer.read().decode("utf-8", errors="replace").strip()
    if not text:
        print("no text on stdin", file=sys.stderr)
        return 2

    from aura.config import get_settings
    from aura.voice.factory import build_tts
    from aura.voice.tts.normalize import for_speech

    # Strip the markup before synthesising. The first version of this module
    # did not, and AURA read a bulleted answer out as "star star Structure star
    # star" -- the exact failure `normalize.py` was written for and says so in
    # its own docstring. `aura voice` has always called this; the console path
    # simply skipped it.
    speakable = for_speech(text)
    if not speakable:
        print("nothing speakable", file=sys.stderr)
        return 3

    tts = build_tts(get_settings())
    tts.load()

    chunks = list(tts.synthesize(speakable))
    if not chunks:
        # Punctuation, an emoji, anything that normalises to nothing speakable.
        # A distinct code so the caller can say so rather than returning a
        # zero-length WAV that plays as silence.
        print("nothing speakable", file=sys.stderr)
        return 3

    rate = chunks[0].sample_rate
    samples = [s for chunk in chunks for s in chunk.audio]
    wav_out.write(_wav_bytes(samples, rate))
    wav_out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
