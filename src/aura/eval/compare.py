"""Comparing a candidate against the incumbent.

This module's most important job is **refusing to declare a winner**.

Phase 9 hill-climbs on this signal. Hill-climbing on noise produces a confident
sequence of changes that improve nothing, and because each step "measured
better" nobody notices for weeks. So a difference smaller than the measured
spread is reported as *indistinguishable* -- which is a real result, not a
failure to produce one.

The second job is refusing to average safety away. A candidate that gains two
tool-selection passes and loses a safety task has not improved, whatever the
headline says.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aura.eval.types import Category, Report, TaskResult
from aura.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class TaskDelta:
    task_id: str
    category: str
    incumbent: float
    candidate: float

    @property
    def change(self) -> float:
        return self.candidate - self.incumbent


@dataclass(slots=True)
class Comparison:
    """The verdict, and everything needed to disagree with it."""

    incumbent_score: float
    candidate_score: float
    noise_floor: float
    # "better" | "worse" | "indistinguishable" | "rejected_on_safety"
    verdict: str
    reason: str
    regressions: list[TaskDelta] = field(default_factory=list)
    improvements: list[TaskDelta] = field(default_factory=list)
    safety_regressions: list[TaskDelta] = field(default_factory=list)
    incumbent_latency_ms: float = 0.0
    candidate_latency_ms: float = 0.0

    @property
    def change(self) -> float:
        return self.candidate_score - self.incumbent_score

    @property
    def accepted(self) -> bool:
        """Phase 9 asks this before deploying anything."""
        return self.verdict == "better"

    def summary(self) -> str:
        head = (
            f"incumbent {self.incumbent_score:.1%} -> candidate "
            f"{self.candidate_score:.1%} ({self.change:+.1%})"
        )
        return f"{head}\n{self.reason}"


def _noise_floor(incumbent: Report, candidate: Report) -> float:
    """How large a difference has to be before it means anything.

    Combines the per-task spread of both runs with the resolution the sample
    size actually supports. With three repeats of twelve tasks, one task
    flipping moves the headline by ~2.8 points, so anything smaller than that
    is not a measurement -- it is rounding.
    """
    spread = math.hypot(incumbent.spread, candidate.spread)
    runs = max(1, min(incumbent.total_runs, candidate.total_runs))
    resolution = 1.0 / runs
    # Half the combined spread: the spread is across tasks, and a headline is
    # their mean, so using it whole would set the bar absurdly high.
    return max(resolution, spread / 2.0)


def _deltas(incumbent: Report, candidate: Report) -> list[TaskDelta]:
    before = {r.task_id: r for r in incumbent.results}
    out: list[TaskDelta] = []
    for result in candidate.results:
        previous = before.get(result.task_id)
        if previous is None:
            # A task the incumbent never ran cannot be compared. Silently
            # including it would let a candidate "win" by adding easy tasks.
            log.info("eval.task_not_in_baseline", task=result.task_id)
            continue
        out.append(
            TaskDelta(
                task_id=result.task_id,
                category=str(result.category),
                incumbent=previous.pass_rate,
                candidate=result.pass_rate,
            )
        )
    return out


def compare(incumbent: Report, candidate: Report) -> Comparison:
    """Judge a candidate. Conservative by construction."""
    deltas = _deltas(incumbent, candidate)
    floor = _noise_floor(incumbent, candidate)

    regressions = sorted((d for d in deltas if d.change < 0), key=lambda d: d.change)
    improvements = sorted((d for d in deltas if d.change > 0), key=lambda d: -d.change)
    safety = [d for d in regressions if d.category == Category.SAFETY.value]

    result = Comparison(
        incumbent_score=incumbent.score,
        candidate_score=candidate.score,
        noise_floor=floor,
        verdict="indistinguishable",
        reason="",
        regressions=regressions,
        improvements=improvements,
        safety_regressions=safety,
        incumbent_latency_ms=incumbent.median_latency_ms(),
        candidate_latency_ms=candidate.median_latency_ms(),
    )

    # Safety first, and not tradeable. Checked before the headline so a
    # candidate cannot buy a security regression with unrelated wins.
    if safety:
        names = ", ".join(d.task_id for d in safety)
        result.verdict = "rejected_on_safety"
        result.reason = (
            f"Rejected: safety regressed on {names}. This is not traded against "
            "gains elsewhere, however large they are."
        )
        return result

    change = result.change
    if abs(change) < floor:
        result.verdict = "indistinguishable"
        result.reason = (
            f"The {abs(change):.1%} difference is inside the {floor:.1%} noise floor "
            f"for this sample ({candidate.total_runs} runs). Not a result. "
            "More repeats or more tasks would be needed to tell these apart."
        )
        return result

    if change > 0:
        result.verdict = "better"
        result.reason = f"Improved by {change:.1%}, above the {floor:.1%} noise floor."
        if regressions:
            # An improvement that broke things is still worth naming. A net
            # gain can hide a task that went from always passing to never.
            lost = ", ".join(d.task_id for d in regressions)
            result.reason += f" Still regressed on: {lost}."
    else:
        result.verdict = "worse"
        result.reason = (
            f"Regressed by {abs(change):.1%}, beyond the {floor:.1%} noise floor. "
            f"Worst: {', '.join(d.task_id for d in regressions[:3])}."
        )
    return result


# -- persistence -------------------------------------------------------------


def save_report(report: Report, path: Path) -> None:
    """Write a report as JSON so a later run can compare against it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")
    log.info("eval.report_saved", path=str(path), score=report.score)


def load_report(path: Path) -> Report:
    """Read a saved report back.

    Rebuilt field by field rather than by splatting the JSON in, because a
    report written by an older build may be missing keys, and a baseline that
    fails to load is a baseline nobody keeps.
    """
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    report = Report(
        label=str(payload.get("label", "")),
        model=str(payload.get("model", "")),
        provider=str(payload.get("provider", "")),
        repeats=int(payload.get("repeats", 1)),
        started_at=float(payload.get("started_at", 0.0)),
        duration_s=float(payload.get("duration_s", 0.0)),
        config=payload.get("config") or {},
    )
    from aura.eval.types import Attempt, CheckResult

    for raw in payload.get("results") or []:
        try:
            category = Category(str(raw.get("category", "general")))
        except ValueError:
            category = Category.GENERAL
        result = TaskResult(task_id=str(raw.get("task_id", "")), category=category)
        for attempt_raw in raw.get("attempts") or []:
            attempt = Attempt(
                task_id=str(attempt_raw.get("task_id", "")),
                passed=bool(attempt_raw.get("passed", False)),
                answer=str(attempt_raw.get("answer", "")),
                tools_used=list(attempt_raw.get("tools_used") or []),
                steps=int(attempt_raw.get("steps", 0)),
                latency_ms=int(attempt_raw.get("latency_ms", 0)),
                error=attempt_raw.get("error"),
            )
            attempt.checks = [
                CheckResult(
                    kind=str(c.get("kind", "")),
                    passed=bool(c.get("passed", False)),
                    detail=str(c.get("detail", "")),
                )
                for c in attempt_raw.get("checks") or []
            ]
            result.attempts.append(attempt)
        report.results.append(result)
    return report
