"""Controlled evolution, and the two rules it must never break.

1. The optimiser cannot tune its own safety envelope. A hill-climber finds
   whatever makes the number go up; if loosening a guard scores better it will
   loosen the guard. The allow-list is what makes that inexpressible.
2. Nothing deploys without a named human. No flag, no config setting, no
   argument makes that automatic (spec §18).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aura.eval.types import Attempt, Category, Report, TaskResult
from aura.evolution.candidate import Candidate, CandidateStore, propose
from aura.evolution.deploy import (
    ApprovalRequiredError,
    NotDeployableError,
    apply_candidate,
)
from aura.evolution.search import HillClimber, best_of
from aura.evolution.space import (
    FORBIDDEN,
    KNOBS,
    KNOBS_BY_PATH,
    ForbiddenKnobError,
    as_nested,
    check_allowed,
    validate_overrides,
)

# -- rule 1: the safety envelope is not tunable -----------------------------


@pytest.mark.parametrize("path", sorted(FORBIDDEN))
def test_forbidden_paths_are_refused(path: str) -> None:
    with pytest.raises(ForbiddenKnobError):
        check_allowed(path)


@pytest.mark.parametrize("path", sorted(FORBIDDEN))
def test_a_forbidden_override_cannot_be_smuggled_through_validation(path: str) -> None:
    """The path a hand-edited candidate file would take."""
    with pytest.raises(ForbiddenKnobError):
        validate_overrides({path: 4})


def test_the_permission_ceiling_is_specifically_protected() -> None:
    """Named explicitly because it is the one that would matter most.

    An optimiser that can raise its own permission ceiling will, since more
    capability scores better on almost any benchmark.
    """
    assert "tools.max_autonomous_level" in FORBIDDEN
    with pytest.raises(ForbiddenKnobError, match="permission ceiling"):
        check_allowed("tools.max_autonomous_level")


def test_the_guards_are_all_still_on_the_forbidden_list() -> None:
    """A regression test against someone quietly shortening the list."""
    must_be_forbidden = {
        "tools.max_autonomous_level",
        "tools.deny_patterns",
        "tools.workspace_roots",
        "security.audit_enabled",
        "vision.mode",
        "app.emergency_stop",
    }
    assert must_be_forbidden <= set(FORBIDDEN)


def test_no_knob_is_also_forbidden() -> None:
    """A path in both lists would be a contradiction, and which one won would
    depend on the order of two ifs."""
    assert not (set(KNOBS_BY_PATH) & set(FORBIDDEN))


def test_an_unlisted_path_is_refused_even_though_it_is_harmless() -> None:
    """Allow-list, not deny-list: anything not enumerated stays fixed, so
    adding a tunable is a deliberate act."""
    with pytest.raises(ForbiddenKnobError, match="allow-list"):
        check_allowed("logging.level")


def test_every_forbidden_entry_explains_itself() -> None:
    """A refusal the user cannot understand is one they will route around."""
    assert all(len(reason) > 20 for reason in FORBIDDEN.values())


# -- bounds -----------------------------------------------------------------


def test_values_are_clamped_to_their_bounds() -> None:
    assert validate_overrides({"llm.temperature": 99.0})["llm.temperature"] == 1.2
    assert validate_overrides({"llm.temperature": -5.0})["llm.temperature"] == 0.0


def test_integer_knobs_stay_integers() -> None:
    assert validate_overrides({"agent.max_steps": 6.7})["agent.max_steps"] == 7.0


def test_neighbours_stay_inside_the_bounds() -> None:
    for knob in KNOBS:
        for value in (knob.low, knob.high, (knob.low + knob.high) / 2):
            for neighbour in knob.neighbours(value):
                assert knob.low <= neighbour <= knob.high, knob.path


def test_a_knob_at_its_limit_has_only_one_neighbour() -> None:
    temperature = KNOBS_BY_PATH["llm.temperature"]
    assert len(temperature.neighbours(0.0)) == 1
    assert len(temperature.neighbours(1.2)) == 1


def test_every_knob_explains_why_it_is_tunable() -> None:
    assert all(k.why for k in KNOBS)


def test_overrides_become_nested_config() -> None:
    assert as_nested({"llm.temperature": 0.4, "llm.top_p": 0.9, "agent.max_steps": 6}) == {
        "llm": {"temperature": 0.4, "top_p": 0.9},
        "agent": {"max_steps": 6},
    }


# -- rule 2: nothing deploys without a human --------------------------------


def _winner() -> Candidate:
    return Candidate(
        overrides={"llm.temperature": 0.45},
        baseline_values={"llm.temperature": 0.7},
        verdict="better",
        score=0.9,
        baseline_score=0.8,
    )


def test_a_deploy_without_an_approver_is_refused() -> None:
    with pytest.raises(ApprovalRequiredError):
        apply_candidate(_winner(), approved_by="")


def test_whitespace_is_not_an_approver() -> None:
    """The laziest possible bypass."""
    with pytest.raises(ApprovalRequiredError):
        apply_candidate(_winner(), approved_by="   ")


def test_there_is_no_force_or_auto_approve_argument() -> None:
    """A rule kept by making the unsafe path unwriteable, not by discipline.

    If someone later adds `force=` or `auto_approve=`, this fails and they have
    to argue with spec §18 on purpose.
    """
    import inspect

    params = set(inspect.signature(apply_candidate).parameters)
    assert params == {"candidate", "approved_by"}


@pytest.mark.parametrize(
    "verdict", ["indistinguishable", "worse", "rejected_on_safety", "unevaluated"]
)
def test_only_a_measured_winner_may_deploy(verdict: str) -> None:
    """Especially 'indistinguishable'. Deploying a change that could not be
    told apart from doing nothing would quietly undo Phase 8's noise floor."""
    candidate = _winner()
    candidate.verdict = verdict

    with pytest.raises(NotDeployableError):
        apply_candidate(candidate, approved_by="ajsal")


def test_a_candidate_that_never_ran_is_not_deployable() -> None:
    assert not Candidate(overrides={"llm.temperature": 0.4}).is_deployable


# -- the search -------------------------------------------------------------


def _report(rate: float, tasks: int = 10) -> Report:
    report = Report(repeats=3)
    for i in range(tasks):
        result = TaskResult(task_id=f"t{i}", category=Category.GENERAL)
        for j in range(3):
            result.attempts.append(Attempt(task_id=f"t{i}", passed=j / 3 < rate))
        report.results.append(result)
    return report


class FakeSettings:
    class llm:  # noqa: N801 - mirrors the config shape
        temperature = 0.7
        top_p = 0.95

    class agent:  # noqa: N801
        max_steps = 8
        tool_temperature = 0.2

    class memory:  # noqa: N801
        history_turns = 12
        injection_floor = 0.5
        search_results = 4


async def test_the_search_tries_both_directions() -> None:
    """Trying only the direction that helped last time is how a search talks
    itself into a local optimum and then sounds confident about it."""
    seen: list[float] = []

    async def evaluate(overrides: dict[str, float]) -> Report:
        seen.append(overrides["llm.temperature"])
        return _report(0.5)

    climber = HillClimber(FakeSettings(), evaluate)
    await climber.step(_report(0.5), [], KNOBS_BY_PATH["llm.temperature"])

    assert len(seen) == 2
    assert min(seen) < 0.7 < max(seen)


async def test_a_candidate_records_the_verdict_not_just_the_score() -> None:
    async def evaluate(overrides: dict[str, float]) -> Report:
        return _report(0.5)

    climber = HillClimber(FakeSettings(), evaluate)
    found = await climber.step(_report(0.5), [], KNOBS_BY_PATH["llm.temperature"])

    assert all(c.verdict == "indistinguishable" for c in found)
    assert all(c.reason for c in found)


def test_the_best_candidate_must_have_actually_won() -> None:
    """A higher raw score with an 'indistinguishable' verdict is not a winner.
    Picking on score here would quietly undo the noise floor."""
    high_but_noise = Candidate(overrides={}, score=0.99, verdict="indistinguishable")
    lower_but_real = Candidate(overrides={}, score=0.85, verdict="better")

    assert best_of([high_but_noise, lower_but_real]) is lower_but_real


def test_no_winner_returns_nothing_rather_than_the_least_bad() -> None:
    losers = [
        Candidate(overrides={}, score=0.9, verdict="indistinguishable"),
        Candidate(overrides={}, score=0.8, verdict="worse"),
    ]
    assert best_of(losers) is None


def test_proposing_changes_exactly_one_knob() -> None:
    """Coordinate ascent: when a candidate wins there must be no question
    about which change caused it."""
    candidate = propose(FakeSettings(), "llm.temperature", 0.55, 0.7)
    assert candidate.overrides == {"llm.temperature": 0.55}
    assert candidate.baseline_values == {"llm.temperature": 0.7}


def test_a_candidate_describes_itself_as_a_diff() -> None:
    """This is what a person approves, so it has to read as a change."""
    text = "\n".join(propose(FakeSettings(), "llm.temperature", 0.55, 0.7).describe())
    assert "0.7 -> 0.55" in text


# -- persistence ------------------------------------------------------------


def test_candidates_survive_to_be_approved_later(tmp_path: Path) -> None:
    """Proposal and approval are deliberately separate steps: a search that
    applied what it found would make the gate a formality."""
    store = CandidateStore(tmp_path / "candidates.json")
    candidate = _winner()
    store.save(candidate)

    loaded = CandidateStore(tmp_path / "candidates.json").get(candidate.id)
    assert loaded is not None
    assert loaded.overrides == candidate.overrides
    assert loaded.verdict == "better"


def test_a_corrupt_candidate_file_does_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text("{ not json")
    assert CandidateStore(path).list_all() == []
