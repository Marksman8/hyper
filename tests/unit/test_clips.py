"""The pre-rendered micro-expression clips (10B.6).

The clips are committed audio, so unlike most assets they can be asserted on
directly: every phrase in the catalogue has a file, every file is the right
format, and none of them is long enough to be an interruption rather than an
interjection.

The property that matters most is the negative one. A catalogue entry with no
audio must make that expression *unavailable*, not produce a silent pause in
the middle of a reply -- a bug that leaves no trace in a log and is completely
obvious in a room.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from aura.voice.clips import CLIPS_DIR, ClipLibrary, load_wav
from aura.voice.personality import CATALOGUE, Moment, PersonalityEngine

RATE = 22_050


def _write(path: Path, seconds: float = 0.4, rate: int = RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    pcm = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(pcm.tobytes())


# ===========================================================================
# The committed clips
# ===========================================================================


def test_every_catalogue_entry_has_a_clip() -> None:
    """If this fails, run scripts/render_micro_expressions.py."""
    missing = [e.id for e in CATALOGUE if not (CLIPS_DIR / e.clip).is_file()]

    assert not missing, f"no audio for {missing}; run the render script"


def test_the_clips_match_auras_own_sample_rate() -> None:
    """A clip at a different rate would play at the wrong pitch and speed when
    queued into the same speaker as her reply."""
    for expression in CATALOGUE:
        _, rate = load_wav(CLIPS_DIR / expression.clip)
        assert rate == RATE, f"{expression.clip} is {rate} Hz"


def test_clips_are_short_enough_to_be_an_interjection() -> None:
    """Over about a second and a half it stops being a micro-expression and
    starts being AURA saying something in a language you may not speak."""
    for expression in CATALOGUE:
        audio, rate = load_wav(CLIPS_DIR / expression.clip)
        seconds = len(audio) / rate
        assert 0.15 < seconds < 1.5, f"{expression.clip} is {seconds:.2f}s"


def test_clips_are_audible_and_not_clipped() -> None:
    for expression in CATALOGUE:
        audio, _ = load_wav(CLIPS_DIR / expression.clip)
        peak = float(np.abs(audio).max())
        assert 0.05 < peak <= 1.0, f"{expression.clip} peaks at {peak:.3f}"


def test_clip_levels_are_consistent_with_each_other() -> None:
    """They were normalised against AURA's own speaking level at render time.
    One clip much louder than the rest would jump out of a reply."""
    levels = [
        float(np.sqrt(np.mean(load_wav(CLIPS_DIR / e.clip)[0] ** 2))) for e in CATALOGUE
    ]

    assert max(levels) / min(levels) < 4.0, f"levels vary too much: {levels}"


def test_the_whole_set_is_small() -> None:
    """Committed audio. It should stay the sort of thing nobody notices in a
    checkout."""
    total = sum((CLIPS_DIR / e.clip).stat().st_size for e in CATALOGUE)

    assert total < 2_000_000, f"{total / 1e6:.1f} MB of clips"


# ===========================================================================
# The library
# ===========================================================================


def test_a_missing_clip_makes_its_expression_unavailable(tmp_path: Path) -> None:
    """The important negative. Adding a phrase and forgetting to render it must
    not put a silent gap in the middle of a reply."""
    _write(tmp_path / CATALOGUE[0].clip)
    library = ClipLibrary(tmp_path)
    library.load(CATALOGUE)

    assert library.available == {CATALOGUE[0].id}
    assert len(library.missing) == len(CATALOGUE) - 1


def test_the_engine_only_offers_what_has_audio(tmp_path: Path) -> None:
    _write(tmp_path / CATALOGUE[0].clip)
    library = ClipLibrary(tmp_path)
    library.load(CATALOGUE)

    engine = PersonalityEngine(languages=("ja", "fr"))
    engine.restrict_to(library.available)

    for moment in Moment:
        for candidate in engine.candidates(moment):
            assert candidate.id in library.available


def test_no_clips_at_all_means_the_engine_says_nothing(tmp_path: Path) -> None:
    """The state an unrendered checkout is in. It must be silent, not broken."""
    library = ClipLibrary(tmp_path)
    library.load(CATALOGUE)

    engine = PersonalityEngine(languages=("ja", "fr"))
    engine.restrict_to(library.available)

    assert all(engine.select(moment) is None for moment in Moment)


def test_an_unreadable_clip_is_skipped_not_fatal(tmp_path: Path) -> None:
    """A corrupt file makes one phrase unavailable -- the same outcome as never
    having rendered it. It is not worth taking the voice process down for."""
    bad = tmp_path / CATALOGUE[0].clip
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"this is not a wav")
    _write(tmp_path / CATALOGUE[1].clip)

    library = ClipLibrary(tmp_path)
    library.load(CATALOGUE)

    assert CATALOGUE[0].id not in library.available
    assert CATALOGUE[1].id in library.available


def test_an_eight_bit_wav_is_refused(tmp_path: Path) -> None:
    """The renderer writes 16-bit. Anything else means the file did not come
    from it, and reading it as 16-bit would produce noise."""
    path = tmp_path / "odd.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(1)
        fh.setframerate(RATE)
        fh.writeframes(b"\x80" * 100)

    with pytest.raises(ValueError, match="16-bit"):
        load_wav(path)


def test_loading_is_idempotent_and_cached(tmp_path: Path) -> None:
    _write(tmp_path / CATALOGUE[0].clip)
    library = ClipLibrary(tmp_path)
    library.load(CATALOGUE)

    first = library.get(CATALOGUE[0].id)
    second = library.get(CATALOGUE[0].id)

    assert first is not None
    assert first is second, "the clip was re-read rather than held"


# ===========================================================================
# The wiring
# ===========================================================================


def test_the_gloss_reaches_the_subtitle() -> None:
    """What makes it comprehensible with the sound off, which is the case this
    milestone exists for."""
    expression = CATALOGUE[0]
    line = expression.subtitle()

    assert expression.text in line
    assert expression.gloss in line


def test_a_micro_expression_only_leads_a_reply() -> None:
    """Asserted against the call site: an interjection between two sentences
    reads as an interruption rather than as an acknowledgement."""
    source = (
        Path(__file__).resolve().parents[2] / "src" / "aura" / "voice" / "session.py"
    ).read_text(encoding="utf-8")

    assert "if first:\n                        self._maybe_micro_expression()" in source


def test_the_render_script_names_a_female_voice_per_language() -> None:
    """A micro-expression in a male voice ahead of AURA's own would sound like
    a second person in the room."""
    from importlib.util import module_from_spec, spec_from_file_location

    path = (
        Path(__file__).resolve().parents[2] / "scripts" / "render_micro_expressions.py"
    )
    spec = spec_from_file_location("render_micro", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    languages = {e.language for e in CATALOGUE}
    assert set(module.VOICES) >= languages
    # hi_fi_captain is multi-speaker; speaker 0 is the female one.
    assert module.VOICES["ja"][2] == 0
