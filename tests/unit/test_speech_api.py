"""Speech for text clients.

The route exists so a typed conversation can be heard. The properties worth
holding: Core holds no model, a machine with no voice installed gets a clear
refusal rather than a stack trace, and the WAV the child writes is a WAV.
"""

from __future__ import annotations

import struct
import subprocess
import sys

import pytest

from aura.api.speech_route import (
    MAX_CHARS,
    NOTHING_SPEAKABLE,
    SpeechService,
    SpeechUnavailableError,
    duration_s,
    speech_status,
)
from aura.voice.synth import _wav_bytes


def _service() -> SpeechService:
    return SpeechService(settings=None)  # type: ignore[arg-type]


# -- the container ----------------------------------------------------------


def test_the_wav_header_says_what_the_payload_is() -> None:
    """A malformed header plays as silence, which is indistinguishable from a
    broken voice and much harder to diagnose."""
    wav = _wav_bytes([0.0, 1.0, -1.0], 16000)

    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert struct.unpack("<I", wav[4:8])[0] == len(wav) - 8

    channels, rate, _, _, bits = struct.unpack("<HHIIH", wav[22:36])
    assert channels == 1
    assert rate == 16000
    assert bits == 16
    assert struct.unpack("<I", wav[40:44])[0] == 3 * 2


def test_samples_are_clamped_rather_than_wrapped() -> None:
    """A float above 1.0 cast without clamping wraps to a loud negative, heard
    as a click rather than as clipping."""
    wav = _wav_bytes([2.0, -2.0], 16000)
    left, right = struct.unpack("<hh", wav[44:48])
    assert left == 32767
    assert right == -32767


def test_duration_is_read_back_out_of_the_header() -> None:
    """Core never sees the samples -- only the bytes the child wrote -- so the
    length it reports has to come from the header."""
    wav = _wav_bytes([0.0] * 22050, 22050)
    assert duration_s(wav) == pytest.approx(1.0, abs=0.01)


def test_a_truncated_payload_reports_zero_rather_than_crashing() -> None:
    assert duration_s(b"RIFF") == 0.0


# -- refusals, before anything is spawned -----------------------------------


def test_empty_text_is_refused() -> None:
    with pytest.raises(SpeechUnavailableError):
        _service().synthesize_wav("   ")


def test_an_over_long_request_is_refused() -> None:
    """Otherwise a caller can park a subprocess for the full timeout."""
    with pytest.raises(SpeechUnavailableError, match="limit"):
        _service().synthesize_wav("a" * (MAX_CHARS + 1))


def test_nothing_is_resident_between_utterances() -> None:
    """The whole reason for the subprocess. There is no model here to evict."""
    svc = _service()
    assert svc.utterances == 0
    status = speech_status(svc)
    assert status["resident"] is False
    assert "loaded" not in status, "a `loaded` field would imply something stays"


# -- the child --------------------------------------------------------------


def test_the_child_refuses_empty_input() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "aura.voice.synth"],
        input=b"   ",
        capture_output=True,
        timeout=60,
    )
    assert done.returncode == 2
    assert not done.stdout


def test_the_child_writes_nothing_but_wav_to_stdout() -> None:
    """stdout is binary. One stray print and every WAV is corrupt.

    Needs a real voice installed; skipped where there is none, because a
    machine without one is a supported configuration.
    """
    done = subprocess.run(
        [sys.executable, "-m", "aura.voice.synth"],
        input=b"Testing one two.",
        capture_output=True,
        timeout=120,
    )
    if done.returncode != 0:
        pytest.skip("no voice installed on this machine")

    assert done.stdout[:4] == b"RIFF", "something wrote to stdout before the WAV"
    assert done.stdout[8:12] == b"WAVE"
    assert struct.unpack("<I", done.stdout[4:8])[0] == len(done.stdout) - 8
    assert duration_s(done.stdout) > 0.1


def test_unspeakable_text_gets_its_own_exit_code() -> None:
    """So the caller can say "nothing speakable" rather than returning a WAV
    that plays as silence."""
    done = subprocess.run(
        [sys.executable, "-m", "aura.voice.synth"],
        input=b"...",
        capture_output=True,
        timeout=120,
    )
    if done.returncode not in (0, NOTHING_SPEAKABLE):
        pytest.skip("no voice installed on this machine")
    if done.returncode == NOTHING_SPEAKABLE:
        assert not done.stdout
