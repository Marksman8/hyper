"""Tests that need the real Piper voice and its weights.

Skipped unless the configured voice is actually downloaded, so `uv run pytest`
stays fast and green on a machine with no voice extra. Run them deliberately:

    uv run pytest -m integration -v

The unit tests measure the envelope against signals whose amplitude is known by
construction -- a ramp, a two-level clip, silence. Those prove the arithmetic.
**This file proves the assumption the whole 10B.3 design rests on**, which no
synthetic tone can: that Piper hands back a sentence's audio in one piece, so
the envelope is in hand at the moment SPEECH_CHUNK is published, and lip sync
therefore costs no latency and needs no second event.

If Piper ever starts streaming a sentence in fragments, the design still works
but degrades -- the mouth would finish early -- and that is worth finding out
here rather than by watching an avatar stop moving mid-word.
"""

from __future__ import annotations

import numpy as np
import pytest

from aura.config import get_settings, voice_models_dir
from aura.core.envelope import FRAME_MS, LEVELS

pytestmark = pytest.mark.integration

SENTENCE = "All eighteen checks passed, so nothing needs your attention."


@pytest.fixture(scope="module")
def tts() -> object:
    """The configured Piper voice, or skip."""
    cfg = get_settings()
    name = cfg.voice.tts.voice
    if not (voice_models_dir() / f"{name}.onnx").is_file():
        pytest.skip(f"voice {name!r} is not downloaded; scripts/download_voice.py")
    pytest.importorskip("piper", reason="the voice extra is not installed")

    from aura.voice.factory import build_tts

    return build_tts(cfg)


def test_the_configured_voice_is_female(tts: object) -> None:
    """10B.4. AURA's default voice is female, and that is a config value rather
    than anything engineered -- this asserts the config has not drifted."""
    name = get_settings().voice.tts.voice
    female = ("amy", "kristin", "hfc_female", "jenny", "alba", "libritts")

    assert any(marker in name.lower() for marker in female), (
        f"{name!r} is not one of the voices recorded as female in "
        "scripts/download_voice.py"
    )
    assert tts.info().model == name  # type: ignore[attr-defined]


def test_piper_yields_a_sentence_in_one_piece(tts: object) -> None:
    """The assumption 10B.3 is built on.

    SPEECH_CHUNK is published at the first audio chunk. If that chunk holds the
    whole sentence, its envelope is available right then for free. If Piper
    ever fragments a sentence, this fails and the design needs revisiting.
    """
    chunks = list(tts.synthesize(SENTENCE))  # type: ignore[attr-defined]

    assert len(chunks) == 1, (
        f"Piper returned {len(chunks)} chunks for one sentence. The envelope in "
        "session.py covers only the first, so the mouth would stop early."
    )


def test_the_envelope_covers_the_whole_utterance(tts: object) -> None:
    from aura.voice.envelope import envelope

    chunk = next(iter(tts.synthesize(SENTENCE)))  # type: ignore[attr-defined]
    result = envelope(chunk.audio, chunk.sample_rate)

    expected = int(chunk.duration_s * 1000 / FRAME_MS)
    assert len(result) == pytest.approx(expected, abs=1)


def test_real_speech_produces_a_varied_envelope(tts: object) -> None:
    """Real speech is bursts and gaps, and the mouth should show both.

    A flat envelope would mean the normalisation is wrong -- either everything
    clipped to maximum or everything crushed to nothing -- and a mouth held at
    one position for a whole sentence looks more broken than one that never
    moves at all.
    """
    from aura.voice.envelope import envelope

    chunk = next(iter(tts.synthesize(SENTENCE)))  # type: ignore[attr-defined]
    values = np.array([int(c) for c in envelope(chunk.audio, chunk.sample_rate)])

    assert values.max() == LEVELS, "nothing reached a fully open mouth"
    assert (values == 0).sum() >= 5, "no closed frames -- the gaps between words vanished"
    assert values.std() > 1.5, f"envelope is too flat (std {values.std():.2f})"


def test_silence_between_words_closes_the_mouth(tts: object) -> None:
    """The visible consequence of the above: she should not talk through the
    pauses."""
    from aura.core.envelope import openness
    from aura.voice.envelope import envelope

    chunk = next(iter(tts.synthesize(SENTENCE)))  # type: ignore[attr-defined]
    result = envelope(chunk.audio, chunk.sample_rate)

    closed = [i for i in range(len(result)) if openness(result, i) == 0.0]
    assert closed, "the mouth never closed during a sentence with pauses in it"
    # And the very start is silence, before the first phoneme lands.
    assert openness(result, 0) == 0.0
