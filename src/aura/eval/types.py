"""What a task is, and what running one produces.

The shapes here are deliberately plain data. The runner, the scorers, the
report and the comparison all pass these around, and a report written today
has to stay readable by a comparison run in three months -- so they serialise
to JSON with nothing clever in them.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Category(StrEnum):
    """What a task is measuring.

    Kept as a closed set because the report breaks down by category, and a
    typo silently creating a new one would hide a regression in a group nobody
    is looking at.
    """

    TOOL_SELECTION = "tool_selection"
    SAFETY = "safety"
    MEMORY = "memory"
    # Named for the capability, not the failure. Every other category here
    # names the good thing, so "hallucination 100%" read as "hallucinates
    # always" when it meant the opposite. A score is only useful if its
    # direction is obvious at a glance.
    HONESTY = "honesty"
    ERROR_RECOVERY = "error_recovery"
    GENERAL = "general"


@dataclass(frozen=True, slots=True)
class Check:
    """One assertion about an answer.

    `kind` names a scorer in eval/scoring.py. Keeping checks as data rather
    than as callables means a suite is a YAML file rather than Python, which
    matters: the person adding a task should not have to touch the harness.
    """

    kind: str
    value: Any = None
    # Shown when the check fails, so a failing run explains itself.
    note: str = ""


@dataclass(frozen=True, slots=True)
class Task:
    """One thing AURA is asked to do, and how to tell if it worked."""

    id: str
    prompt: str
    checks: tuple[Check, ...]
    category: Category = Category.GENERAL
    # Tools a human has approved for this task. The model cannot add to it --
    # same rule as a real turn, because a benchmark that relaxed the permission
    # model would be measuring an AURA that does not exist.
    approve: tuple[str, ...] = ()
    # Facts to seed before the task runs, for memory tasks.
    given_facts: tuple[tuple[str, str], ...] = ()
    # Run through the agent loop rather than plain chat.
    use_tools: bool = False
    weight: float = 1.0


@dataclass(slots=True)
class CheckResult:
    kind: str
    passed: bool
    detail: str = ""


@dataclass(slots=True)
class Attempt:
    """One run of one task. Tasks run several times; see the phase doc §2."""

    task_id: str
    # Defaults to False: an attempt that has not been scored has not passed.
    # A default of True would make a crash before scoring look like a success.
    passed: bool = False
    checks: list[CheckResult] = field(default_factory=list)
    answer: str = ""
    # Every tool the model reached for. This is about *selection*.
    tools_used: list[str] = field(default_factory=list)
    # Only the ones that actually ran and returned ok. Kept separately
    # because a denied call and a successful one are the same name, and a
    # scorer that cannot tell them apart has to fall back to reading the
    # model's prose -- which tests its wording, not the guard.
    tools_succeeded: list[str] = field(default_factory=list)
    steps: int = 0
    latency_ms: int = 0
    error: str | None = None

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


@dataclass(slots=True)
class TaskResult:
    """Every attempt at one task."""

    task_id: str
    category: Category
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def runs(self) -> int:
        return len(self.attempts)

    @property
    def passes(self) -> int:
        return sum(1 for a in self.attempts if a.passed)

    @property
    def pass_rate(self) -> float:
        return self.passes / self.runs if self.runs else 0.0

    @property
    def is_flaky(self) -> bool:
        """Passed sometimes and failed others.

        Worth surfacing on its own: a task that passes two runs in three is not
        a pass, and averaging it into a headline hides the most interesting
        thing about it.
        """
        return 0 < self.passes < self.runs

    @property
    def median_latency_ms(self) -> float:
        values = [a.latency_ms for a in self.attempts if a.error is None]
        return statistics.median(values) if values else 0.0


@dataclass(slots=True)
class Report:
    """The outcome of a whole suite run."""

    results: list[TaskResult] = field(default_factory=list)
    label: str = ""
    model: str = ""
    provider: str = ""
    repeats: int = 1
    started_at: float = field(default_factory=time.time)
    duration_s: float = 0.0
    # Free-form description of the configuration under test, so a report is
    # self-describing when compared months later.
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def total_runs(self) -> int:
        return sum(r.runs for r in self.results)

    @property
    def score(self) -> float:
        """Weighted pass rate, 0..1.

        Phase 9 needs a scalar to climb. It is a hazard -- see the phase doc §5
        -- so nothing prints it without the category breakdown beside it.
        """
        if not self.results:
            return 0.0
        return statistics.fmean(r.pass_rate for r in self.results)

    @property
    def spread(self) -> float:
        """How much the per-task rates vary.

        Used as the noise floor for comparison: a difference smaller than this
        is not a difference.
        """
        rates = [r.pass_rate for r in self.results]
        if len(rates) < 2:
            return 0.0
        return statistics.pstdev(rates)

    def by_category(self) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for result in self.results:
            buckets.setdefault(str(result.category), []).append(result.pass_rate)
        return {name: statistics.fmean(rates) for name, rates in sorted(buckets.items())}

    def safety_failures(self) -> list[TaskResult]:
        """Reported separately and never averaged away (phase doc §5).

        A safety regression is not worth trading for two extra tool-selection
        passes, so it must not be able to hide inside a headline that improved.
        """
        return [
            r for r in self.results if r.category is Category.SAFETY and r.pass_rate < 1.0
        ]

    def flaky(self) -> list[TaskResult]:
        return [r for r in self.results if r.is_flaky]

    def median_latency_ms(self) -> float:
        values = [r.median_latency_ms for r in self.results if r.median_latency_ms]
        return statistics.median(values) if values else 0.0
