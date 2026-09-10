"""The lip-sync envelope (10B.3).

Amplitude, not phonemes: RMS per 20 ms frame, quantised to one digit, carried
on `SPEECH_CHUNK` beside the sentence it belongs to.

The exit criterion for this milestone is *"mouth openness correlates with audio
amplitude"*, so the tests that matter are the ones that measure that against
signals whose amplitude is known by construction -- a ramp, a burst, silence --
rather than against a recording nobody can reason about.
"""

from __future__ import annotations

import numpy as np
import pytest

from aura.core.envelope import FRAME_MS, LEVELS, MAX_FRAMES, duration_ms, openness
from aura.voice.envelope import SILENCE_FLOOR, envelope

RATE = 22_050


def _tone(seconds: float, amplitude: float = 0.5, rate: int = RATE) -> np.ndarray:
    """A sine at a fixed amplitude, so RMS is known in advance."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# ===========================================================================
# Measurement
# ===========================================================================


def test_one_digit_per_frame() -> None:
    audio = _tone(1.0)
    result = envelope(audio, RATE)

    assert len(result) == pytest.approx(1000 / FRAME_MS, abs=1)
    assert result.isdigit()


def test_a_steady_tone_gives_a_steady_mouth() -> None:
    """Constant amplitude in, constant level out. If this drifts, the mouth
    flutters through a held note."""
    result = envelope(_tone(0.5), RATE)

    assert len(set(result)) == 1
    assert result[0] == str(LEVELS)


def test_openness_tracks_amplitude() -> None:
    """The exit criterion, stated directly: louder audio opens the mouth
    further, measured on a signal whose amplitude is known by construction."""
    loud = _tone(0.2, amplitude=1.0)
    quiet = _tone(0.2, amplitude=0.05)
    # One clip: half quiet, half loud. Normalisation is per-sentence, so the
    # comparison has to happen inside a single envelope.
    combined = np.concatenate([quiet, loud])

    result = envelope(combined, RATE)
    first_half = result[: len(result) // 2]
    second_half = result[len(result) // 2 :]

    assert max(first_half) < min(second_half)


def test_a_ramp_produces_a_monotonic_envelope() -> None:
    seconds = 1.0
    t = np.linspace(0, seconds, int(RATE * seconds), endpoint=False, dtype=np.float32)
    ramp = (np.linspace(0, 1, len(t), dtype=np.float32) * np.sin(2 * np.pi * 220 * t)).astype(
        np.float32
    )

    values = [int(c) for c in envelope(ramp, RATE)]

    # Allow a little jitter from the sine crossing frame boundaries, but the
    # trend must be upward throughout.
    assert values[0] < values[len(values) // 2] < values[-1]


def test_silence_keeps_the_mouth_shut() -> None:
    """Per-sentence normalisation would otherwise stretch a noise floor into a
    full open-and-close -- a mouth flapping at nothing."""
    near_silence = (np.random.default_rng(0).normal(0, SILENCE_FLOOR / 10, RATE)).astype(
        np.float32
    )

    result = envelope(near_silence, RATE)

    assert set(result) == {"0"}


def test_a_quiet_sentence_still_opens_the_mouth() -> None:
    """The other half of that trade. Absolute normalisation would make quiet
    speech read as a rendering fault rather than as quiet speech."""
    result = envelope(_tone(0.3, amplitude=0.08), RATE)

    assert result.strip("0"), "a quiet but audible sentence moved nothing"
    assert max(result) == str(LEVELS)


def test_empty_and_degenerate_input_is_not_an_error() -> None:
    assert envelope(np.array([], dtype=np.float32), RATE) == ""
    assert envelope(_tone(1.0), 0) == ""
    # Shorter than one frame: no whole frames, so nothing to report.
    assert envelope(np.zeros(5, dtype=np.float32), RATE) == ""


def test_a_trailing_partial_frame_is_dropped() -> None:
    """It would have a lower RMS purely because it is short, putting a
    spurious dip at the end of every sentence."""
    frame = int(RATE * FRAME_MS / 1000)
    audio = _tone(0.2)
    ragged = np.concatenate([audio, audio[: frame // 2]])

    assert envelope(ragged, RATE) == envelope(audio[: len(ragged) // frame * frame], RATE)


def test_a_long_sentence_is_capped() -> None:
    assert len(envelope(_tone(30.0), RATE)) == MAX_FRAMES


# ===========================================================================
# Reading -- the consumer's half, which needs no audio stack
# ===========================================================================


def test_openness_maps_digits_to_zero_one() -> None:
    assert openness("0", 0) == 0.0
    assert openness(str(LEVELS), 0) == 1.0
    assert 0.0 < openness("4", 0) < 1.0


def test_running_past_the_end_closes_the_mouth() -> None:
    """A consumer that overruns should close, not hold the last shape open."""
    assert openness("999", 3) == 0.0
    assert openness("999", -1) == 0.0
    assert openness("", 0) == 0.0


def test_duration_is_frames_times_frame_length() -> None:
    assert duration_ms("12345") == 5 * FRAME_MS


def test_the_format_module_has_no_dependencies() -> None:
    """The point of splitting it out: the presence process renders a mouth
    without importing the voice package, whose own types module says outright
    that nothing outside aura.voice imports it."""
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "src" / "aura" / "core" / "envelope.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    imported = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
    ]
    assert not imported, "aura/core/envelope.py grew an import"
