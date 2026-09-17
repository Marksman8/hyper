"""Speech for text clients (Phase 10B, extension).

**Why this exists.** Voice is a *client*, not a subsystem: `aura voice` does its
own STT, calls Core over the same HTTP API as anything else, and speaks the
reply itself. That shape is right and is not changing. But it leaves a gap --
someone typing in the console gets text and nothing else, so hearing AURA
requires being able to talk to her, which is exactly the situation where a
person cannot.

**How, and the version that was thrown away.** The obvious implementation was
for Core to ask `build_tts` for a provider and keep it resident, loading on
demand and evicting when idle. That was built and measured:

    Core idle            72 MB
    voice loaded        232 MB
    after eviction      115 MB

The 43 MB that never came back is onnxruntime's arena, which Python releasing
the model does not return to the OS. `tests/unit/test_voice_seam.py` forbids
`aura.api` importing `aura.voice` for precisely this reason, and the measurement
agreed with the test rather than excusing it.

So synthesis happens in a **short-lived subprocess** -- `python -m
aura.voice.synth`, text in on stdin, WAV out on stdout, exit. Core never imports
the voice package, stays 69 MB, and still speaks in AURA's own voice rather than
whatever `speechSynthesis` the browser happens to ship.

The cost is a model load per utterance, around 1.5 s, because nothing stays
warm. On a machine whose binding constraint is 15.7 GB of RAM rather than CPU
that is the right way round -- and it is the same reasoning `run_python` uses
for spawning rather than executing in-process.
"""

from __future__ import annotations

import struct
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

from aura.errors import AuraError
from aura.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aura.config import Settings

log = get_logger(__name__)

# A spoken reply is a sentence or two. Anything much longer is a mistake or an
# attempt to make Core spend minutes in a subprocess, and both are refused
# rather than served.
MAX_CHARS = 4000

# Model load is ~1.5 s and synthesis runs near 0.4x real time, so this ceiling
# is generous for the longest permitted text and still bounded.
TIMEOUT_S = 90.0

# `aura.voice.synth` exits with this when the text normalises to nothing
# speakable -- punctuation, an emoji. Distinct from a failure so the caller can
# say so rather than returning a WAV that plays as silence.
NOTHING_SPEAKABLE = 3


class SpeechUnavailableError(AuraError):
    """No usable voice. Not a crash: a machine with no voice installed is a
    supported configuration and the console simply does not speak."""

    http_status = 503
    code = "speech_unavailable"


class SpeechService:
    """Spawns one synthesis per utterance.

    Holds no model and no state beyond counters, which is the entire point:
    there is nothing here to evict, because nothing stays resident.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.utterances = 0
        self.last_synth_ms = 0

    def synthesize_wav(self, text: str) -> tuple[bytes, float]:
        """Return (wav_bytes, audio_seconds). Blocking; call it in a thread."""
        text = text.strip()
        if not text:
            raise SpeechUnavailableError("Nothing to speak.")
        if len(text) > MAX_CHARS:
            raise SpeechUnavailableError(
                f"That is {len(text)} characters; the limit is {MAX_CHARS}."
            )

        started = time.perf_counter()
        try:
            done = subprocess.run(
                [sys.executable, "-m", "aura.voice.synth"],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=TIMEOUT_S,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SpeechUnavailableError(
                f"Synthesis exceeded {TIMEOUT_S:.0f}s and was terminated."
            ) from exc
        except OSError as exc:
            raise SpeechUnavailableError(f"Could not start the voice: {exc}") from exc

        if done.returncode == NOTHING_SPEAKABLE:
            raise SpeechUnavailableError("Nothing speakable in that text.")
        if done.returncode != 0 or not done.stdout:
            detail = (done.stderr or b"").decode("utf-8", errors="replace").strip()
            # A missing piper install or a missing voice model lands here, and
            # is reported as unavailable rather than 500: nothing is broken,
            # the feature is simply not installed.
            raise SpeechUnavailableError(
                "The voice did not produce audio. "
                + (detail[-400:] if detail else f"Exit code {done.returncode}.")
            )

        wav = done.stdout
        seconds = duration_s(wav)
        self.utterances += 1
        self.last_synth_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "speech.synthesized",
            chars=len(text),
            audio_s=round(seconds, 2),
            synth_ms=self.last_synth_ms,
            bytes=len(wav),
        )
        return wav, seconds

    def close(self) -> None:
        """Nothing to release. Kept so callers need not special-case it."""


def duration_s(wav: bytes) -> float:
    """Seconds of audio, read back out of the header the child wrote."""
    if len(wav) < 44:
        return 0.0
    rate = struct.unpack("<I", wav[24:28])[0]
    size = struct.unpack("<I", wav[40:44])[0]
    return (size / (rate * 2)) if rate else 0.0


def speech_status(service: SpeechService | None) -> dict[str, Any]:
    """What the console shows.

    There is deliberately no `loaded` field: nothing stays resident, so the
    question does not arise. `resident: false` is a claim about the design, not
    a reading of a timer.
    """
    if service is None:
        return {"available": True, "resident": False, "utterances": 0, "last_synth_ms": 0}
    return {
        "available": True,
        "resident": False,
        "utterances": service.utterances,
        "last_synth_ms": service.last_synth_ms,
    }
