"""Coordinate ascent over the configuration space.

One knob at a time, one step at a time. Slow and explainable: when a candidate
wins there is no question about which change caused it, and a person approving
it is reading a single line of diff rather than a vector.

Not genetic algorithms, and the reasoning is the roadmap's own. GA needs a
cheap, low-noise fitness function and a large evaluation budget. One evaluation
here is 63 model calls and about a minute; a population of 50 over 20
generations is 1,000 evaluations -- most of a day, mostly fitting noise.

The search proposes and measures. It never applies anything: deploy is a
separate, human-gated act in deploy.py.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from aura.eval.compare import compare
from aura.eval.types import Report, Task
from aura.evolution.candidate import Candidate, propose
from aura.evolution.space import KNOBS, Knob, current_value
from aura.logging import get_logger

log = get_logger(__name__)

# An async callable that scores one configuration. Injected rather than
# imported so the search can be tested without a model server.
Evaluator = Callable[[dict[str, float]], Any]


class HillClimber:
    """Proposes neighbours and keeps the ones that measurably win."""

    def __init__(
        self,
        settings: Any,
        evaluate: Evaluator,
        knobs: Sequence[Knob] = KNOBS,
    ) -> None:
        self._settings = settings
        self._evaluate = evaluate
        self._knobs = list(knobs)

    async def step(
        self,
        baseline: Report,
        tasks: Sequence[Task],
        knob: Knob,
        on_candidate: Callable[[Candidate], None] | None = None,
    ) -> list[Candidate]:
        """Evaluate both neighbours of one knob.

        Both directions, always. Trying only the direction that happened to
        help last time is how a search talks itself into a local optimum and
        then reports confidence about it.
        """
        start = current_value(self._settings, knob.path)
        results: list[Candidate] = []

        for value in knob.neighbours(start):
            candidate = propose(self._settings, knob.path, value, start)
            log.info("evolution.evaluating", id=candidate.id, path=knob.path, value=value)

            report = await self._evaluate(candidate.overrides)
            verdict = compare(baseline, report)

            candidate.score = report.score
            candidate.baseline_score = baseline.score
            candidate.verdict = verdict.verdict
            candidate.reason = verdict.reason
            results.append(candidate)

            log.info(
                "evolution.evaluated",
                id=candidate.id,
                verdict=verdict.verdict,
                score=report.score,
                baseline=baseline.score,
            )
            if on_candidate:
                on_candidate(candidate)

        return results

    async def run(
        self,
        baseline: Report,
        tasks: Sequence[Task],
        knobs: Sequence[Knob] | None = None,
        on_candidate: Callable[[Candidate], None] | None = None,
    ) -> list[Candidate]:
        """Sweep the given knobs once.

        Deliberately a single sweep rather than looping until nothing improves.
        An unattended search that keeps going has to decide for itself when to
        stop, and every candidate it finds still needs a person to approve it --
        so the honest unit of work is "one sweep, then come and look".
        """
        targets = list(knobs) if knobs is not None else self._knobs
        found: list[Candidate] = []
        for knob in targets:
            found.extend(await self.step(baseline, tasks, knob, on_candidate))
        return found


def best_of(candidates: Sequence[Candidate]) -> Candidate | None:
    """The winner, if there is one.

    Only candidates the comparison called 'better' are eligible. A candidate
    with the highest raw score but an 'indistinguishable' verdict is not a
    winner -- that is the whole point of Phase 8's noise floor, and picking on
    score here would quietly undo it.
    """
    winners = [c for c in candidates if c.is_deployable and c.score is not None]
    if not winners:
        return None
    return max(winners, key=lambda c: c.score or 0.0)
