"""Micro-expressions (10B.5).

Almost every test here is about the engine declining to do anything. That is
the correct emphasis: a personality layer is judged by how often it stays quiet,
and the brief's warning was specific -- foreign words must not leak into every
response.

The randomness is injected, so "roughly one in eight" is measured on a seeded
sequence rather than asserted in a comment.
"""

from __future__ import annotations

import itertools
import random

import pytest

from aura.voice.personality import (
    CATALOGUE,
    MIN_GAP_S,
    Frequency,
    MicroExpression,
    Moment,
    PersonalityEngine,
)


class Clock:
    """A hand-wound clock, so cooldowns are tested without sleeping."""

    def __init__(self) -> None:
        self.t = 1_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _engine(**kwargs: object) -> tuple[PersonalityEngine, Clock]:
    clock = Clock()
    defaults: dict[str, object] = {
        "frequency": Frequency.MEDIUM,
        "languages": ("ja", "fr"),
        "rng": random.Random(7),
        "clock": clock,
    }
    defaults.update(kwargs)
    return PersonalityEngine(**defaults), clock  # type: ignore[arg-type]


def _always(**kwargs: object) -> tuple[PersonalityEngine, Clock]:
    """An engine whose dice always come up hit, so the OTHER rules are what is
    being measured."""

    class Certain(random.Random):
        def random(self) -> float:
            return 0.0

    return _engine(rng=Certain(), **kwargs)


# ===========================================================================
# Switching it off means off
# ===========================================================================


def test_disabled_never_fires() -> None:
    engine, _ = _always(enabled=False)

    assert all(engine.select(Moment.ACKNOWLEDGING) is None for _ in range(200))


def test_frequency_off_never_fires() -> None:
    """Two separate switches, because "off" and "never" are different
    questions a user might ask, and both must actually silence it."""
    engine, _ = _always(frequency=Frequency.OFF)

    assert all(engine.select(Moment.ACKNOWLEDGING) is None for _ in range(200))


def test_off_is_not_probabilistic() -> None:
    """Switching the feature off must not be a weighting that a lucky roll can
    get past."""
    engine, _ = _engine(frequency=Frequency.OFF, rng=random.Random(1))

    for _ in range(1_000):
        assert engine.select(Moment.SUCCEEDING) is None
    assert engine.fired == 0


# ===========================================================================
# Restraint
# ===========================================================================


def test_low_frequency_is_roughly_one_in_eight() -> None:
    """Measured, on eligible moments with the cooldown removed so the
    frequency is the only thing being tested."""
    engine, _ = _engine(
        frequency=Frequency.LOW, rng=random.Random(11), min_gap_s=0.0
    )

    hits = sum(engine.select(Moment.ACKNOWLEDGING) is not None for _ in range(4_000))

    assert 0.09 < hits / 4_000 < 0.16, f"fired {hits / 40:.1f}% of the time"


def test_medium_is_chattier_but_not_a_coin_flip() -> None:
    """A coin flip reads as constant, because a listener remembers the hits."""
    engine, _ = _engine(
        frequency=Frequency.MEDIUM, rng=random.Random(11), min_gap_s=0.0
    )

    hits = sum(engine.select(Moment.ACKNOWLEDGING) is not None for _ in range(4_000))

    assert 0.20 < hits / 4_000 < 0.31


def test_the_same_expression_never_runs_twice() -> None:
    engine, clock = _always(min_gap_s=0.0)

    seen = []
    for _ in range(40):
        choice = engine.select(Moment.ACKNOWLEDGING)
        if choice is not None:
            seen.append(choice.id)
        clock.advance(1.0)

    assert len(seen) > 5, "nothing fired; the test measured nothing"
    assert all(a != b for a, b in itertools.pairwise(seen))


def test_a_lone_candidate_stays_silent_rather_than_repeating() -> None:
    """With one phrase left for a moment, silence is the better failure."""
    only = MicroExpression(
        id="solo",
        language="ja",
        text="はい",
        gloss="yes",
        clip="ja/hai.wav",
        moments=(Moment.ACKNOWLEDGING,),
    )
    engine, clock = _always(catalogue=(only,), min_gap_s=0.0)

    first = engine.select(Moment.ACKNOWLEDGING)
    clock.advance(1.0)
    second = engine.select(Moment.ACKNOWLEDGING)

    assert first is not None
    assert second is None


def test_the_minimum_gap_is_absolute() -> None:
    """Frequency governs the long-run average; this governs a burst of short
    turns, which is what a person actually notices."""
    engine, clock = _always()

    assert engine.select(Moment.ACKNOWLEDGING) is not None
    clock.advance(MIN_GAP_S - 1)
    assert engine.select(Moment.ACKNOWLEDGING) is None
    assert engine.select(Moment.THINKING) is None, "a different moment bypassed the gap"

    clock.advance(2.0)
    assert engine.select(Moment.ACKNOWLEDGING) is not None


def test_three_quick_turns_produce_at_most_one() -> None:
    """The scenario the brief warned about, stated as a test."""
    engine, clock = _always()

    fired = 0
    for _ in range(3):
        fired += engine.select(Moment.ACKNOWLEDGING) is not None
        clock.advance(7.0)

    assert fired == 1


# ===========================================================================
# Fit
# ===========================================================================


def test_an_expression_only_appears_at_a_moment_it_fits() -> None:
    """よし means "good". It has no business appearing when something failed."""
    engine, clock = _always(min_gap_s=0.0)

    for _ in range(60):
        choice = engine.select(Moment.FAILING)
        if choice is not None:
            assert Moment.FAILING in choice.moments
        clock.advance(1.0)


def test_an_unconfigured_language_is_never_heard() -> None:
    engine, clock = _always(languages=("fr",), min_gap_s=0.0)

    for _ in range(60):
        choice = engine.select(Moment.ACKNOWLEDGING)
        if choice is not None:
            assert choice.language == "fr"
        clock.advance(1.0)


def test_a_moment_with_no_candidates_is_silent() -> None:
    engine, _ = _always(languages=("de",))

    assert engine.select(Moment.ACKNOWLEDGING) is None
    assert engine.considered == 0, "a moment with nothing to say counted as considered"


def test_moments_are_not_presence_states() -> None:
    """Deliberately a separate, smaller vocabulary. Coupling them would make
    every new visual state a personality question."""
    assert {str(m) for m in Moment} == {
        "acknowledging",
        "thinking",
        "succeeding",
        "failing",
    }


# ===========================================================================
# The catalogue itself
# ===========================================================================


def test_every_expression_has_a_gloss() -> None:
    """An untranslated phrase on screen is worse than no phrase, for exactly
    the person this milestone serves -- the one who cannot hear the audio."""
    for expression in CATALOGUE:
        assert expression.gloss, f"{expression.id} has no gloss"
        assert expression.text
        assert expression.clip.endswith(".wav")


def test_the_catalogue_is_small_enough_to_read() -> None:
    """The point of a fixed set: one screen tells you everything AURA can say
    in another language. If this fails, ask whether it should."""
    assert len(CATALOGUE) <= 40


def test_ids_are_unique() -> None:
    ids = [expression.id for expression in CATALOGUE]
    assert len(ids) == len(set(ids))


def test_the_subtitle_carries_both_phrase_and_gloss() -> None:
    expression = CATALOGUE[0]
    line = expression.subtitle()

    assert expression.text in line
    assert expression.gloss in line


def test_clips_are_declared_per_language() -> None:
    """10B.6 renders these to fixed WAVs. A clip filed under the wrong
    language would be spoken by the wrong voice."""
    for expression in CATALOGUE:
        assert expression.clip.startswith(f"{expression.language}/")


# ===========================================================================
# The model does not write these
# ===========================================================================


def test_selection_takes_no_text_from_anywhere() -> None:
    """The safety property, asserted against the interface.

    `select()` receives a Moment and nothing else. There is no parameter a
    model's output could travel through, so "the model chose a foreign phrase"
    is not a thing that can happen -- it is unexpressible rather than
    discouraged.
    """
    import inspect

    signature = inspect.signature(PersonalityEngine.select)

    assert list(signature.parameters) == ["self", "moment"]


def test_every_phrase_comes_from_the_catalogue() -> None:
    engine, clock = _always(min_gap_s=0.0)
    known = {expression.id for expression in CATALOGUE}

    for _ in range(100):
        choice = engine.select(Moment.SUCCEEDING)
        if choice is not None:
            assert choice.id in known
        clock.advance(1.0)


def test_build_engine_tolerates_a_config_without_the_section() -> None:
    """An older config file must not crash the voice process."""
    from aura.voice.personality import build_engine

    engine = build_engine(object())

    assert engine.select(Moment.ACKNOWLEDGING) is None


def test_build_engine_reads_the_real_config() -> None:
    from aura.config import get_settings
    from aura.voice.personality import build_engine

    engine = build_engine(get_settings())

    assert isinstance(engine, PersonalityEngine)
    assert engine.candidates(Moment.ACKNOWLEDGING), "the default config can say nothing"


@pytest.mark.parametrize("moment", list(Moment))
def test_every_moment_has_something_to_say_in_japanese(moment: Moment) -> None:
    """The default language set must cover every moment, or some moments are
    silently unreachable."""
    engine = PersonalityEngine(languages=("ja",))

    assert engine.candidates(moment), f"nothing in ja fits {moment}"
