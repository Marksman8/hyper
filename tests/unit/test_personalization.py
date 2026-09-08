"""Acting on preferences, and the limits on doing so.

The safety property under test: an *inferred* preference may change how AURA
writes, but never what it is mechanically able to say. A wrong directive gives
a differently-toned answer; a wrong max_tokens truncates mid-sentence, and the
user cannot connect a broken reply to something they said weeks ago.
"""

from __future__ import annotations

from aura.core.types import Fact, GenerationParams, Permanence
from aura.personalization.profile import (
    MIN_MAX_TOKENS,
    apply_parameters,
    build_profile,
    directives,
)


def _fact(
    key: str,
    value: str,
    *,
    source: str = "explicit",
    confidence: float = 0.95,
    permanence: Permanence = Permanence.LONG_TERM,
) -> Fact:
    return Fact(
        scope="user",
        key=key,
        value=value,
        source=source,
        confidence=confidence,
        permanence=permanence,
    )


# -- reading preferences ----------------------------------------------------


def test_a_brevity_preference_becomes_a_dimension() -> None:
    profile = build_profile([_fact("response_length", "prefers short, direct answers")])
    assert profile.get("response_length") == "short"


def test_dimensions_are_matched_by_meaning_not_exact_key() -> None:
    """The model that proposed the fact will not have used our word for it."""
    for key in ("verbosity", "answer_length", "brevity"):
        profile = build_profile([_fact(key, "keep it concise")])
        assert profile.get("response_length") == "short", key


def test_a_detail_preference_is_read() -> None:
    profile = build_profile([_fact("response_length", "wants thorough, in depth answers")])
    assert profile.get("response_length") == "detailed"


def test_formality_and_language_are_read() -> None:
    profile = build_profile(
        [_fact("tone", "casual and friendly"), _fact("language", "British English")]
    )
    assert profile.get("formality") == "casual"
    assert profile.get("language") == "British English"


def test_unknown_facts_are_kept_as_context_not_discarded() -> None:
    """A closed vocabulary that silently dropped `my_editor: neovim` would make
    AURA worse at knowing the user in exchange for being better at formatting."""
    profile = build_profile([_fact("editor", "neovim"), _fact("timezone", "Asia/Kolkata")])

    assert not profile.dimensions
    assert {f.key for f in profile.other_facts} == {"editor", "timezone"}


def test_a_fact_with_no_recognisable_value_stays_a_plain_fact() -> None:
    """The key looks like a dimension but the text says nothing we can act on."""
    profile = build_profile([_fact("tone", "depends on the day")])
    assert "formality" not in profile.dimensions
    assert [f.key for f in profile.other_facts] == ["tone"]


# -- what does not reach the profile ----------------------------------------


def test_unbelieved_facts_are_ignored() -> None:
    """Phase 6 decided what counts as believed; this phase does not get a
    second, looser opinion."""
    profile = build_profile([_fact("response_length", "short", source="inferred", confidence=0.3)])
    assert profile.is_empty


def test_a_temporary_preference_does_not_become_permanent_behaviour() -> None:
    """'Be brief today' must not become how AURA always writes."""
    profile = build_profile(
        [_fact("response_length", "short", permanence=Permanence.TEMPORARY)]
    )
    assert profile.is_empty


def test_an_explicit_statement_outranks_an_inference() -> None:
    """Even a better-corroborated one: the user saying so beats AURA noticing."""
    profile = build_profile(
        [
            _fact("verbosity", "detailed", source="inferred", confidence=0.9),
            _fact("response_length", "short", source="explicit", confidence=0.95),
        ]
    )
    assert profile.get("response_length") == "short"
    assert profile.dimensions["response_length"].source == "explicit"


# -- directives -------------------------------------------------------------


def test_a_preference_becomes_an_instruction_not_a_fact() -> None:
    """The whole point of the phase.

    `response_length: short` is trivia the model has to interpret. "Keep
    answers brief" is something it acts on.
    """
    lines = directives(build_profile([_fact("response_length", "short and direct")]))
    assert lines
    text = " ".join(lines).lower()
    assert "brief" in text
    assert "response_length" not in text


def test_an_empty_profile_contributes_nothing() -> None:
    assert directives(build_profile([])) == []


def test_a_neutral_preference_adds_no_noise() -> None:
    """Asking for 'normal' length is asking for the default, so it should not
    spend prompt tokens saying so."""
    assert directives(build_profile([_fact("response_length", "normal")])) == []


def test_a_language_preference_becomes_an_instruction() -> None:
    lines = directives(build_profile([_fact("language", "Spanish")]))
    assert any("Spanish" in line for line in lines)


# -- the parameter asymmetry (the safety property) --------------------------


def test_an_explicit_brevity_preference_lowers_max_tokens() -> None:
    profile = build_profile([_fact("response_length", "short", source="explicit")])
    adjusted = apply_parameters(profile, GenerationParams(max_tokens=4096))
    assert adjusted.max_tokens < 4096


def test_an_inferred_preference_never_touches_sampling() -> None:
    """The load-bearing test of this module.

    A well-corroborated inference still only changes how AURA writes. It must
    not change what AURA is mechanically able to say.
    """
    profile = build_profile(
        [_fact("response_length", "short", source="inferred", confidence=0.89)]
    )
    original = GenerationParams(max_tokens=4096)

    assert apply_parameters(profile, original).max_tokens == 4096
    # ...but it DOES still change the prompt.
    assert directives(profile), "an inferred preference should still shape the writing"


def test_brevity_never_truncates_below_the_floor() -> None:
    """A preference for brevity is a preference, not permission to cut answers
    off mid-sentence."""
    profile = build_profile([_fact("response_length", "as short as possible")])
    adjusted = apply_parameters(profile, GenerationParams(max_tokens=64))
    assert adjusted.max_tokens >= MIN_MAX_TOKENS


def test_a_detail_preference_raises_the_ceiling() -> None:
    profile = build_profile([_fact("response_length", "thorough and detailed")])
    adjusted = apply_parameters(profile, GenerationParams(max_tokens=256))
    assert adjusted.max_tokens > 256


def test_other_sampling_settings_are_left_alone() -> None:
    """Length is the only preference with an honest mechanical equivalent.
    Temperature is not a thing the user expressed an opinion about."""
    profile = build_profile([_fact("response_length", "short")])
    original = GenerationParams(temperature=0.35, top_p=0.8, max_tokens=4096, seed=7)
    adjusted = apply_parameters(profile, original)

    assert adjusted.temperature == 0.35
    assert adjusted.top_p == 0.8
    assert adjusted.seed == 7


def test_applying_parameters_does_not_mutate_the_original() -> None:
    """GenerationParams is shared across a turn; mutating it would leak one
    request's personalization into the next."""
    profile = build_profile([_fact("response_length", "short")])
    original = GenerationParams(max_tokens=4096)
    apply_parameters(profile, original)
    assert original.max_tokens == 4096


def test_an_empty_profile_leaves_parameters_untouched() -> None:
    original = GenerationParams(max_tokens=1024)
    assert apply_parameters(build_profile([]), original) is original


# -- visibility -------------------------------------------------------------


def test_the_profile_explains_where_each_adaptation_came_from() -> None:
    """Silent adaptation is a debugging nightmare: 'AURA is being weird today'
    has to be answerable."""
    profile = build_profile([_fact("response_length", "short", source="explicit")])
    described = profile.describe()

    assert len(described) == 1
    entry = described[0]
    assert entry["dimension"] == "response_length"
    assert entry["value"] == "short"
    assert entry["from_fact"] == "response_length"
    assert entry["drives_parameters"] is True


def test_an_inferred_dimension_reports_that_it_does_not_drive_parameters() -> None:
    profile = build_profile(
        [_fact("verbosity", "short", source="inferred", confidence=0.7)]
    )
    assert profile.describe()[0]["drives_parameters"] is False


def test_a_programming_language_is_not_a_reply_language() -> None:
    """Found by an existing test, not by design.

    `preferred_language: Python` is a perfectly reasonable fact, and treating
    the language dimension as free text turned it into the instruction "Reply
    in Python" -- which AURA would then have followed every turn.
    """
    profile = build_profile([_fact("preferred_language", "Python")])

    assert "language" not in profile.dimensions
    assert [f.key for f in profile.other_facts] == ["preferred_language"]
    assert not any("Python" in line for line in directives(profile))


def test_an_unknown_language_falls_through_to_a_plain_fact() -> None:
    """Being incomplete is the cheaper failure: the information stays in the
    prompt, it just does not drive machinery."""
    profile = build_profile([_fact("language", "Esperanto")])
    assert "language" not in profile.dimensions
    assert [f.key for f in profile.other_facts] == ["language"]


def test_a_recognised_natural_language_still_works() -> None:
    for text, expected in (
        # The qualifier survives: "British" is the part the user cared about,
        # so the list validates the value rather than rewriting it.
        ("British English", "British English"),
        ("reply in Malayalam please", "reply in Malayalam please"),
        ("Japanese", "Japanese"),
    ):
        profile = build_profile([_fact("language", text)])
        assert profile.get("language") == expected, text
