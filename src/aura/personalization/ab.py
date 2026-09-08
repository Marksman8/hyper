"""Blind A/B between the adapted persona and the baseline.

The roadmap's exit criterion for Phase 7 is "a blind A/B shows a preference for
the adapted behaviour". A single user rating a handful of prompts **cannot**
produce a statistically significant result, and this module will not pretend
otherwise: 13-7 out of 20 is a coin, not evidence.

What it does instead is be a trustworthy instrument:

- Both answers come from the same running AURA, one with personalization on and
  one with it off, so nothing but the adaptation differs.
- The pair is shown in a **random order** with no labels, and which side was
  which is recorded, so position bias can be checked afterwards rather than
  assumed absent.
- The tally is stored with the prompts, so a later run is comparable and the
  raw data survives for someone to disagree with.

It reports counts. It does not compute a p-value on n=12, and it does not tell
the user they have proven anything.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aura.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class Comparison:
    """One blind judgement."""

    prompt: str
    adapted: str
    baseline: str
    # Which side the adapted answer was shown as: "A" or "B". Recorded so a
    # tally that is really just "people pick the first one" is detectable.
    adapted_shown_as: str
    # "adapted", "baseline", or "tie". None until the user answers.
    picked: str | None = None
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class Tally:
    """Results so far. Deliberately just counts."""

    adapted: int = 0
    baseline: int = 0
    ties: int = 0
    # How often the adapted answer happened to be shown first, and how often
    # the user picked whichever was shown first, regardless of which it was.
    adapted_shown_first: int = 0
    picked_first_shown: int = 0

    @property
    def judged(self) -> int:
        return self.adapted + self.baseline + self.ties

    def summary(self) -> str:
        """A sentence that does not overclaim."""
        if self.judged == 0:
            return "No comparisons recorded yet."

        head = (
            f"{self.judged} comparisons: adapted {self.adapted}, "
            f"baseline {self.baseline}, ties {self.ties}."
        )
        if self.judged < 20:
            return head + " Far too few to mean anything; keep going."

        bias = self.picked_first_shown / self.judged
        note = ""
        if bias >= 0.7:
            note = (
                f" Warning: the first-shown answer was picked {bias:.0%} of the time, "
                "so this tally may be measuring position bias rather than quality."
            )
        lead = abs(self.adapted - self.baseline)
        if lead <= self.judged**0.5:
            return head + " The difference is within what chance would produce." + note
        return head + note


class ABLog:
    """Stores comparisons as JSON lines, so the raw data outlives the summary."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def record(self, comparison: Comparison) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(comparison), ensure_ascii=False) + "\n")
        log.info("ab.recorded", picked=comparison.picked)

    def load(self) -> list[Comparison]:
        if not self._path.exists():
            return []
        out: list[Comparison] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(Comparison(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                # A corrupt line loses one judgement, not the whole record.
                continue
        return out

    def tally(self) -> Tally:
        result = Tally()
        for c in self.load():
            if c.picked is None:
                continue
            if c.picked == "adapted":
                result.adapted += 1
            elif c.picked == "baseline":
                result.baseline += 1
            else:
                result.ties += 1

            if c.adapted_shown_as == "A":
                result.adapted_shown_first += 1
            picked_first = (c.picked == "adapted" and c.adapted_shown_as == "A") or (
                c.picked == "baseline" and c.adapted_shown_as == "B"
            )
            if picked_first:
                result.picked_first_shown += 1
        return result


def present(
    prompt: str, adapted: str, baseline: str, rng: random.Random | None = None
) -> tuple[Comparison, str, str]:
    """Lay a pair out blind.

    Returns the comparison plus the two answers in display order. The caller
    shows them as A and B and never learns which is which until it resolves
    the choice through `resolve`.
    """
    chooser = rng or random.SystemRandom()
    adapted_first = chooser.random() < 0.5
    comparison = Comparison(
        prompt=prompt,
        adapted=adapted,
        baseline=baseline,
        adapted_shown_as="A" if adapted_first else "B",
    )
    if adapted_first:
        return comparison, adapted, baseline
    return comparison, baseline, adapted


def resolve(comparison: Comparison, chosen_side: str) -> Comparison:
    """Translate the user's "A" or "B" back into which arm they picked."""
    side = chosen_side.strip().upper()
    if side == "TIE":
        comparison.picked = "tie"
    elif side in ("A", "B"):
        comparison.picked = "adapted" if side == comparison.adapted_shown_as else "baseline"
    return comparison
