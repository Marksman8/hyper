"""Comparison, and its refusal to declare a winner it cannot see.

Phase 9 hill-climbs on this. Hill-climbing on noise produces a confident
sequence of changes that improve nothing, and because every step "measured
better", nobody notices for weeks. So the tests that matter most here are the
ones asserting that a small difference is reported as *no* difference.
"""

from __future__ import annotations

from pathlib import Path

from aura.eval.compare import compare, load_report, save_report
from aura.eval.types import Attempt, Category, Report, TaskResult


def _result(task_id: str, passes: int, runs: int, category: Category) -> TaskResult:
    result = TaskResult(task_id=task_id, category=category)
    for i in range(runs):
        result.attempts.append(Attempt(task_id=task_id, passed=i < passes, latency_ms=100))
    return result


def _report(spec: dict[str, tuple[int, int]], category: Category = Category.GENERAL) -> Report:
    """spec maps task id -> (passes, runs)."""
    return Report(
        results=[_result(tid, p, r, category) for tid, (p, r) in spec.items()],
        repeats=3,
    )


def _uniform(n_tasks: int, passes: int, runs: int = 3, **kw: object) -> Report:
    return _report({f"t{i}": (passes, runs) for i in range(n_tasks)}, **kw)  # type: ignore[arg-type]


# -- refusing to overclaim --------------------------------------------------


def test_an_identical_run_is_indistinguishable() -> None:
    before, after = _uniform(10, 2), _uniform(10, 2)
    assert compare(before, after).verdict == "indistinguishable"


def test_a_one_task_flip_is_inside_the_noise_floor() -> None:
    """The load-bearing test.

    With 10 tasks at 3 repeats, one attempt changing must not read as an
    improvement -- that is rounding, not a result.
    """
    before = _uniform(10, 2)
    after = _report({**{f"t{i}": (2, 3) for i in range(9)}, "t9": (3, 3)})

    verdict = compare(before, after)
    assert verdict.verdict == "indistinguishable"
    assert "noise floor" in verdict.reason
    assert not verdict.accepted


def test_the_reason_says_what_would_be_needed() -> None:
    """A user told 'inconclusive' should learn what to do about it."""
    reason = compare(_uniform(10, 2), _uniform(10, 2)).reason
    assert "repeats" in reason or "tasks" in reason


# -- real differences -------------------------------------------------------


def test_a_large_gain_is_reported_as_better() -> None:
    before, after = _uniform(10, 0), _uniform(10, 3)
    verdict = compare(before, after)
    assert verdict.verdict == "better"
    assert verdict.accepted
    assert verdict.change > 0


def test_a_large_loss_is_reported_as_worse() -> None:
    verdict = compare(_uniform(10, 3), _uniform(10, 0))
    assert verdict.verdict == "worse"
    assert not verdict.accepted


def test_an_improvement_still_names_what_it_broke() -> None:
    """A net gain can hide a task that went from always passing to never."""
    before = _report({f"t{i}": (0, 3) for i in range(9)} | {"kept": (3, 3)})
    after = _report({f"t{i}": (3, 3) for i in range(9)} | {"kept": (0, 3)})

    verdict = compare(before, after)
    assert verdict.verdict == "better"
    assert "kept" in verdict.reason
    assert [d.task_id for d in verdict.regressions] == ["kept"]


# -- safety is not tradeable ------------------------------------------------


def test_a_safety_regression_rejects_however_large_the_gains() -> None:
    """The other load-bearing test.

    A candidate that gains everywhere and loses one safety task has not
    improved, whatever the headline says.
    """
    before = Report(
        results=[_result(f"g{i}", 0, 3, Category.GENERAL) for i in range(9)]
        + [_result("guard", 3, 3, Category.SAFETY)],
        repeats=3,
    )
    after = Report(
        results=[_result(f"g{i}", 3, 3, Category.GENERAL) for i in range(9)]
        + [_result("guard", 0, 3, Category.SAFETY)],
        repeats=3,
    )

    verdict = compare(before, after)
    assert verdict.verdict == "rejected_on_safety"
    assert not verdict.accepted
    assert verdict.change > 0, "the headline really did improve; it must not matter"
    assert [d.task_id for d in verdict.safety_regressions] == ["guard"]
    assert "not traded" in verdict.reason


def test_safety_holding_does_not_block_an_improvement() -> None:
    before = Report(
        results=[_result(f"g{i}", 0, 3, Category.GENERAL) for i in range(9)]
        + [_result("guard", 3, 3, Category.SAFETY)],
        repeats=3,
    )
    after = Report(
        results=[_result(f"g{i}", 3, 3, Category.GENERAL) for i in range(9)]
        + [_result("guard", 3, 3, Category.SAFETY)],
        repeats=3,
    )
    assert compare(before, after).verdict == "better"


# -- guarding the comparison itself -----------------------------------------


def test_a_task_missing_from_the_baseline_is_not_counted() -> None:
    """Otherwise a candidate could 'win' by adding easy tasks to the suite."""
    before = _uniform(3, 3)
    after = _report({"t0": (3, 3), "t1": (3, 3), "t2": (3, 3), "brand_new": (3, 3)})

    verdict = compare(before, after)
    assert "brand_new" not in {d.task_id for d in verdict.improvements}


# -- the report itself ------------------------------------------------------


def test_the_headline_comes_with_a_category_breakdown() -> None:
    report = Report(
        results=[
            _result("a", 3, 3, Category.SAFETY),
            _result("b", 0, 3, Category.MEMORY),
        ],
        repeats=3,
    )
    assert report.score == 0.5
    assert report.by_category() == {"memory": 0.0, "safety": 1.0}


def test_safety_failures_are_listed_separately() -> None:
    report = Report(results=[_result("guard", 2, 3, Category.SAFETY)], repeats=3)
    assert [r.task_id for r in report.safety_failures()] == ["guard"]


def test_a_flaky_task_is_surfaced_not_averaged_away() -> None:
    """Two passes in three is not a pass, and it is the most interesting thing
    about that task."""
    report = Report(
        results=[
            _result("sometimes", 2, 3, Category.GENERAL),
            _result("always", 3, 3, Category.GENERAL),
        ],
        repeats=3,
    )
    assert [r.task_id for r in report.flaky()] == ["sometimes"]


# -- persistence ------------------------------------------------------------


def test_a_report_survives_a_round_trip(tmp_path: Path) -> None:
    report = _uniform(4, 2)
    report.label = "baseline"
    path = tmp_path / "baseline.json"
    save_report(report, path)

    loaded = load_report(path)
    assert loaded.label == "baseline"
    assert loaded.score == report.score
    assert {r.task_id for r in loaded.results} == {r.task_id for r in report.results}


def test_a_report_from_an_older_build_still_loads(tmp_path: Path) -> None:
    """A baseline that fails to load is a baseline nobody keeps."""
    path = tmp_path / "old.json"
    path.write_text('{"results": [{"task_id": "t0", "attempts": [{"passed": true}]}]}')

    loaded = load_report(path)
    assert loaded.results[0].task_id == "t0"
    assert loaded.score == 1.0


def test_an_unknown_category_loads_as_general(tmp_path: Path) -> None:
    path = tmp_path / "future.json"
    path.write_text(
        '{"results": [{"task_id": "t", "category": "invented_later", "attempts": []}]}'
    )
    assert load_report(path).results[0].category is Category.GENERAL
