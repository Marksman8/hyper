"""How confident AURA is, and why.

Spec §16: an inferred preference starts at low confidence and needs repeated
evidence; only explicit statements start high. This module is that rule, kept
here rather than in SQL so it can be read, tested and argued with.

The old behaviour was `confidence = MIN(1.0, confidence + 0.05)` inside the
upsert. That is a counter dressed as a belief: it treats "the user said to
remember this" and "the model thought it noticed something" as the same
evidence, and it cannot be recomputed or explained after the fact. Here,
confidence is **derived from the evidence rows** — so deleting a piece of
evidence lowers the belief, which is what a justification means.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# A stated preference is taken close to at face value, but never at 1.0:
# people misspeak, change their minds, and say things sarcastically.
EXPLICIT_BASE = 0.95
EXPLICIT_CEILING = 0.99

# An inference from one remark. Deliberately below the 0.5 injection floor, so
# a single offhand comment never becomes a standing belief about the user --
# the failure mode that makes personalisation feel like surveillance.
INFERRED_BASE = 0.3
INFERRED_CEILING = 0.9

# How much of the remaining gap each new corroboration closes. 0.55 puts the
# second independent observation at 0.57 -- just over the injection floor --
# so "twice, on separate occasions" is what it takes to be believed.
GAIN = 0.55


@dataclass(frozen=True, slots=True)
class Belief:
    """A confidence value and the reasoning that produced it."""

    confidence: float
    independent_count: int
    has_explicit: bool

    def explain(self) -> str:
        """One sentence a person can check."""
        if self.has_explicit:
            return f"You told AURA this directly ({self.independent_count}x)."
        if self.independent_count <= 1:
            return "Inferred from one remark, so AURA is not acting on it yet."
        return f"Inferred from {self.independent_count} separate occasions."


def independent_evidence(evidence: Sequence[object]) -> int:
    """Count corroborations that are genuinely separate occasions.

    Evidence items are anything with `turn_id` and `session_id` attributes.

    The rule that matters: two observations from the *same turn* count once. A
    model that restates an inference twice in one reply must not thereby talk
    itself into believing it. Evidence with no turn recorded is counted
    individually, since there is nothing to deduplicate it against.
    """
    seen: set[str] = set()
    anonymous = 0
    for item in evidence:
        turn = getattr(item, "turn_id", None)
        session = getattr(item, "session_id", None)
        if turn:
            seen.add(str(turn))
        elif session:
            seen.add("session:" + str(session))
        else:
            anonymous += 1
    return len(seen) + anonymous


def assess(evidence: Sequence[object]) -> Belief:
    """Derive a confidence from the evidence behind a fact."""
    if not evidence:
        # No evidence left. The caller decides whether to delete the fact; this
        # function will not pretend a belief with nothing behind it is credible.
        return Belief(confidence=0.0, independent_count=0, has_explicit=False)

    has_explicit = any(getattr(item, "source", "inferred") == "explicit" for item in evidence)
    count = max(1, independent_evidence(evidence))

    if has_explicit:
        # One clear statement is enough; repeats add a little, not much.
        base, ceiling = EXPLICIT_BASE, EXPLICIT_CEILING
    else:
        base, ceiling = INFERRED_BASE, INFERRED_CEILING

    # Each corroboration closes GAIN of the remaining distance to the ceiling.
    confidence = ceiling - (ceiling - base) * (GAIN ** (count - 1))
    return Belief(
        confidence=round(min(ceiling, confidence), 4),
        independent_count=count,
        has_explicit=has_explicit,
    )
