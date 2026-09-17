"""Loading task suites.

Tasks live in YAML, not Python, so adding one does not mean touching the
harness. That matters for the phase's purpose: the suite should grow whenever
a real weakness is found, and a contribution barrier of "edit the runner"
guarantees it will not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from aura.errors import AuraError
from aura.eval.types import Category, Check, Task

_SUITES = Path(__file__).with_name("suites")

# Shipped with the project; `aura eval` uses all of these unless told otherwise.
#
# Two files rather than one, and the default is both. `core.yaml` measures
# capability; `injection.yaml` measures whether content AURA reads can make her
# act. Keeping them separate lets the injection corpus be run and published on
# its own -- it is the benchmark, and a benchmark nobody else can run against
# their own agent is not one -- while the default score still includes it.
#
# Including it in the default is the point. Phase 8 recorded that the suite was
# too easy (97.4%, 2.6% of headroom) and Phase 9 tried harder *capability*
# tasks, which moved it to 96.8%. Adversarial tasks are what actually restores
# headroom, because the model half of them genuinely fails.
DEFAULT_SUITES: tuple[Path, ...] = (_SUITES / "core.yaml", _SUITES / "injection.yaml")

# Kept for callers that want just the capability suite by name.
DEFAULT_SUITE = _SUITES / "core.yaml"
INJECTION_SUITE = _SUITES / "injection.yaml"


class SuiteError(AuraError):
    """The suite file is wrong. Raised at load, never mid-run: discovering a
    malformed task twenty minutes into an evaluation wastes the evaluation."""


def _check(raw: Any, task_id: str) -> Check:
    if not isinstance(raw, dict) or len(raw) < 1:
        raise SuiteError(f"{task_id}: each check must be a mapping, got {raw!r}")
    note = str(raw.get("note", ""))
    body = {k: v for k, v in raw.items() if k != "note"}
    if len(body) != 1:
        raise SuiteError(
            f"{task_id}: a check names exactly one scorer, got {sorted(body)}"
        )
    kind, value = next(iter(body.items()))
    return Check(kind=str(kind), value=value, note=note)


def _task(raw: Any, index: int) -> Task:
    if not isinstance(raw, dict):
        raise SuiteError(f"task #{index} is not a mapping")

    task_id = str(raw.get("id") or f"task_{index}")
    prompt = str(raw.get("prompt") or "").strip()
    if not prompt:
        raise SuiteError(f"{task_id}: no prompt")

    checks = raw.get("checks") or []
    if not isinstance(checks, list) or not checks:
        raise SuiteError(f"{task_id}: needs at least one check, or it measures nothing")

    category_raw = str(raw.get("category", "general"))
    try:
        category = Category(category_raw)
    except ValueError as exc:
        known = ", ".join(c.value for c in Category)
        raise SuiteError(f"{task_id}: unknown category {category_raw!r}. Known: {known}") from exc

    facts_raw = raw.get("given_facts") or {}
    if not isinstance(facts_raw, dict):
        raise SuiteError(f"{task_id}: given_facts must be a mapping of key to value")

    return Task(
        id=task_id,
        prompt=prompt,
        checks=tuple(_check(c, task_id) for c in checks),
        category=category,
        approve=tuple(str(t) for t in (raw.get("approve") or ())),
        given_facts=tuple((str(k), str(v)) for k, v in facts_raw.items()),
        use_tools=bool(raw.get("use_tools", False)),
        weight=float(raw.get("weight", 1.0)),
    )


def _load_one(source: Path, offset: int) -> list[Task]:
    if not source.exists():
        raise SuiteError(f"No suite at {source}")

    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SuiteError(f"{source} is not valid YAML: {exc}") from exc

    raw_tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise SuiteError(f"{source} has no `tasks:` list")

    return [_task(raw, offset + i) for i, raw in enumerate(raw_tasks)]


def load_suite(path: Path | None = None) -> list[Task]:
    """Read and validate a suite. Fails fast and completely.

    With no path, loads every file in `DEFAULT_SUITES`. Duplicate ids are an
    error *across* files as well as within one: two tasks sharing an id would
    silently overwrite each other in a report, which is the kind of thing that
    makes a comparison quietly wrong.
    """
    sources = [path] if path is not None else list(DEFAULT_SUITES)

    tasks: list[Task] = []
    for source in sources:
        tasks.extend(_load_one(source, len(tasks)))

    seen: set[str] = set()
    for task in tasks:
        if task.id in seen:
            raise SuiteError(f"duplicate task id {task.id!r}")
        seen.add(task.id)
    return tasks
