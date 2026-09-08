"""Loading suites, and running them.

A malformed suite must fail at load, not twenty minutes into an evaluation --
discovering a broken task mid-run wastes the run. And the runner has to isolate
attempts from each other, or the suite stops measuring tasks and starts
measuring their order.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aura.core.types import Fact
from aura.eval.runner import SuiteRunner
from aura.eval.suite import DEFAULT_SUITE, SuiteError, load_suite
from aura.eval.types import Category, Check, Task


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(body, encoding="utf-8")
    return path


# -- the shipped suite ------------------------------------------------------


def test_the_shipped_suite_loads() -> None:
    tasks = load_suite()
    assert len(tasks) >= 10
    assert {str(t.category) for t in tasks} >= {"safety", "memory", "honesty"}


def test_every_shipped_task_has_a_check() -> None:
    """A task with no checks measures nothing and scores a free point."""
    assert all(t.checks for t in load_suite(DEFAULT_SUITE))


def test_every_shipped_scorer_exists() -> None:
    """A typo'd scorer name fails every run of that task forever, and the
    reason would only show up in a detail string nobody reads."""
    from aura.eval.scoring import SCORERS

    unknown = {c.kind for t in load_suite() for c in t.checks} - set(SCORERS)
    assert not unknown, f"suite uses scorers that do not exist: {sorted(unknown)}"


def test_tool_tasks_declare_the_tools_they_need() -> None:
    """A benchmark that relaxed the permission model would measure an AURA
    that does not exist."""
    for task in load_suite():
        if task.use_tools:
            assert task.approve, f"{task.id} uses tools but approves none"


# -- validation -------------------------------------------------------------


def test_a_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="No suite"):
        load_suite(tmp_path / "nope.yaml")


def test_broken_yaml_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="not valid YAML"):
        load_suite(_write(tmp_path, "tasks: [oh: no: yes"))


def test_a_task_with_no_checks_is_rejected(tmp_path: Path) -> None:
    body = "tasks:\n  - id: empty\n    prompt: hello\n    checks: []\n"
    with pytest.raises(SuiteError, match="at least one check"):
        load_suite(_write(tmp_path, body))


def test_a_task_with_no_prompt_is_rejected(tmp_path: Path) -> None:
    body = "tasks:\n  - id: silent\n    checks:\n      - contains: x\n"
    with pytest.raises(SuiteError, match="no prompt"):
        load_suite(_write(tmp_path, body))


def test_an_unknown_category_is_rejected(tmp_path: Path) -> None:
    """Not coerced to 'general': a typo would quietly move a safety task out
    of the category that gets special treatment."""
    body = (
        "tasks:\n  - id: t\n    prompt: hi\n    category: saftey\n"
        "    checks:\n      - contains: x\n"
    )
    with pytest.raises(SuiteError, match="unknown category"):
        load_suite(_write(tmp_path, body))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    """They would overwrite each other in a report, making a comparison
    quietly wrong."""
    body = (
        "tasks:\n  - id: same\n    prompt: a\n    checks:\n      - contains: x\n"
        "  - id: same\n    prompt: b\n    checks:\n      - contains: y\n"
    )
    with pytest.raises(SuiteError, match="duplicate task id"):
        load_suite(_write(tmp_path, body))


def test_a_check_naming_two_scorers_is_rejected(tmp_path: Path) -> None:
    body = (
        "tasks:\n  - id: t\n    prompt: hi\n    checks:\n"
        "      - contains: a\n        matches: b\n"
    )
    with pytest.raises(SuiteError, match="exactly one scorer"):
        load_suite(_write(tmp_path, body))


def test_notes_are_not_mistaken_for_scorers(tmp_path: Path) -> None:
    body = (
        "tasks:\n  - id: t\n    prompt: hi\n    checks:\n"
        "      - contains: a\n        note: why this matters\n"
    )
    task = load_suite(_write(tmp_path, body))[0]
    assert task.checks[0].kind == "contains"
    assert task.checks[0].note == "why this matters"


# -- the runner -------------------------------------------------------------


class FakeOrchestrator:
    """Answers from a script, and records what it was asked."""

    def __init__(self, answer: str = "Paris", fail: bool = False) -> None:
        self.answer = answer
        self.fail = fail
        self.prompts: list[str] = []
        self.seeds: list[int | None] = []
        self.remembered: list[str] = []
        self.forgotten: list[str] = []

    class _Provider:
        name = "fake"

        def info(self) -> object:
            return type("I", (), {"model": "fake-model"})()

    provider = _Provider()

    async def chat(self, prompt: str, *, session_id=None, params=None, **kw):  # type: ignore[no-untyped-def]
        self.prompts.append(prompt)
        self.seeds.append(getattr(params, "seed", None))
        if self.fail:
            raise RuntimeError("model server is down")
        return type("T", (), {"content": self.answer, "agent": None})()

    async def act(self, prompt: str, *, session_id=None, params=None, approve=()):  # type: ignore[no-untyped-def]
        self.prompts.append(prompt)
        return type("T", (), {"content": self.answer, "agent": None})()

    async def remember(self, key: str, value: str, permanence: str = "long_term") -> Fact:
        self.remembered.append(key)
        return Fact(scope="user", key=key, value=value)

    async def forget(self, key: str) -> bool:
        self.forgotten.append(key)
        return True


def _task(**kw: object) -> Task:
    base = {
        "id": "t",
        "prompt": "What is the capital of France?",
        "checks": (Check("contains", "Paris"),),
        "category": Category.GENERAL,
    }
    return Task(**{**base, **kw})  # type: ignore[arg-type]


async def test_each_task_runs_the_configured_number_of_times() -> None:
    orch = FakeOrchestrator()
    report = await SuiteRunner(orch, repeats=3).run([_task()])  # type: ignore[arg-type]

    assert report.results[0].runs == 3
    assert report.results[0].pass_rate == 1.0
    assert len(orch.prompts) == 3


async def test_repeats_use_different_seeds() -> None:
    """Otherwise all three repeats sample the same point and the spread is a
    fiction."""
    orch = FakeOrchestrator()
    await SuiteRunner(orch, repeats=3).run([_task()])  # type: ignore[arg-type]
    assert len(set(orch.seeds)) == 3


async def test_seeded_facts_are_cleaned_up_afterwards() -> None:
    """Without this, task 9 inherits task 8's memory and the suite measures
    task order."""
    orch = FakeOrchestrator(answer="neovim")
    task = _task(
        checks=(Check("contains", "neovim"),),
        given_facts=(("editor", "neovim"),),
        category=Category.MEMORY,
    )
    await SuiteRunner(orch, repeats=2).run([task])  # type: ignore[arg-type]

    assert orch.remembered == ["editor", "editor"]
    assert orch.forgotten == ["editor", "editor"]


async def test_a_crash_is_a_result_not_an_interruption() -> None:
    """One broken task must not destroy the rest of the report."""
    report = await SuiteRunner(FakeOrchestrator(fail=True), repeats=2).run(  # type: ignore[arg-type]
        [_task(), _task(id="second")]
    )

    assert len(report.results) == 2
    assert report.score == 0.0
    assert report.results[0].attempts[0].error is not None


async def test_facts_are_cleaned_up_even_when_the_task_crashes() -> None:
    """A leaked fact would silently change every later task's prompt."""
    orch = FakeOrchestrator(fail=True)
    task = _task(given_facts=(("editor", "neovim"),))
    await SuiteRunner(orch, repeats=1).run([task])  # type: ignore[arg-type]

    assert orch.forgotten == ["editor"]


async def test_a_wrong_answer_scores_zero() -> None:
    report = await SuiteRunner(FakeOrchestrator(answer="Berlin"), repeats=2).run(  # type: ignore[arg-type]
        [_task()]
    )
    assert report.results[0].pass_rate == 0.0
    assert report.results[0].attempts[0].failures
