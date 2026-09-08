"""The confidence rule (spec section 16).

The property that matters: one offhand remark must never become a standing
belief about the user. That is the difference between personalisation and
surveillance, and it is enforced by a number - the injection floor - so it is
worth testing against that floor directly rather than against a vibe.
"""

from __future__ import annotations

from dataclasses import dataclass

from aura.memory.beliefs import (
    EXPLICIT_CEILING,
    INFERRED_CEILING,
    assess,
    independent_evidence,
)

# The orchestrator injects facts at confidence >= 0.5. Hard-coded here on
# purpose: if someone lowers that floor, these tests should be what objects.
INJECTION_FLOOR = 0.5


@dataclass
class E:
    """A stand-in for a stored evidence row."""

    source: str = "inferred"
    turn_id: str | None = None
    session_id: str | None = None


# -- the rule ---------------------------------------------------------------


def test_no_evidence_is_no_belief() -> None:
    belief = assess([])
    assert belief.confidence == 0.0
    assert belief.independent_count == 0


def test_one_inference_is_not_believed() -> None:
    """The load-bearing test of this module.

    A single remark sits BELOW the injection floor, so it never reaches the
    system prompt and never shapes a reply.
    """
    belief = assess([E(turn_id="t1")])
    assert belief.confidence < INJECTION_FLOOR
    assert not belief.has_explicit


def test_two_independent_inferences_cross_the_floor() -> None:
    """'Repeated evidence' is two separate occasions, not two mentions."""
    belief = assess([E(turn_id="t1"), E(turn_id="t2")])
    assert belief.confidence >= INJECTION_FLOOR
    assert belief.independent_count == 2


def test_one_explicit_statement_is_believed_immediately() -> None:
    """The user saying so is not a hypothesis to be corroborated."""
    belief = assess([E(source="explicit", turn_id="t1")])
    assert belief.confidence >= 0.9
    assert belief.has_explicit


def test_inference_never_reaches_explicit_certainty() -> None:
    many = [E(turn_id=f"t{i}") for i in range(50)]
    assert assess(many).confidence <= INFERRED_CEILING


def test_explicit_never_reaches_absolute_certainty() -> None:
    """People misspeak, change their minds, and are sarcastic."""
    many = [E(source="explicit", turn_id=f"t{i}") for i in range(50)]
    assert assess(many).confidence <= EXPLICIT_CEILING
    assert assess(many).confidence < 1.0


def test_confidence_rises_with_corroboration() -> None:
    scores = [
        assess([E(turn_id=f"t{i}") for i in range(n)]).confidence for n in range(1, 6)
    ]
    assert scores == sorted(scores)
    assert len(set(scores)) == len(scores), "corroboration must actually move the number"


def test_explicit_evidence_dominates_a_mixed_set() -> None:
    """One clear statement outweighs an inference that disagreed with it."""
    belief = assess([E(turn_id="t1"), E(source="explicit", turn_id="t2")])
    assert belief.has_explicit
    assert belief.confidence >= 0.9


# -- independence -----------------------------------------------------------


def test_same_turn_counts_once() -> None:
    """A model restating an inference twice in one reply must not thereby
    talk itself into believing it."""
    assert independent_evidence([E(turn_id="t1"), E(turn_id="t1"), E(turn_id="t1")]) == 1
    assert assess([E(turn_id="t1")] * 3).confidence < INJECTION_FLOOR


def test_different_turns_count_separately() -> None:
    assert independent_evidence([E(turn_id="t1"), E(turn_id="t2")]) == 2


def test_evidence_without_a_turn_falls_back_to_the_session() -> None:
    assert independent_evidence([E(session_id="s1"), E(session_id="s1")]) == 1
    assert independent_evidence([E(session_id="s1"), E(session_id="s2")]) == 2


def test_evidence_with_no_provenance_counts_individually() -> None:
    """Nothing to deduplicate against, so it cannot be assumed duplicated."""
    assert independent_evidence([E(), E(), E()]) == 3


def test_explanation_is_a_sentence_a_person_can_check() -> None:
    assert "directly" in assess([E(source="explicit", turn_id="t1")]).explain()
    assert "not acting on it" in assess([E(turn_id="t1")]).explain()
    assert "2 separate" in assess([E(turn_id="t1"), E(turn_id="t2")]).explain()
