"""Running a suite against a live AURA.

Two things make this a measurement rather than a demo.

**Repeats.** A 4B model at temperature 0.7 answers the same question
differently every time. Running each task once and reporting a score produces a
number that moves on its own, and every conclusion drawn from it is noise
wearing a lab coat. Each task runs `repeats` times and the report keeps the
spread.

**Isolation.** Each attempt gets its own session and its own seeded facts, and
cleans up after itself. Without that, task 9 inherits task 8's conversation and
the suite stops measuring tasks and starts measuring their order.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from aura.core.orchestrator import Orchestrator
from aura.core.types import GenerationParams, Permanence
from aura.eval.scoring import score_attempt
from aura.eval.types import Attempt, Report, Task, TaskResult
from aura.logging import get_logger

log = get_logger(__name__)


class SuiteRunner:
    """Executes tasks against a real orchestrator."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        repeats: int = 3,
        temperature: float = 0.7,
    ) -> None:
        self._orch = orchestrator
        self._repeats = max(1, repeats)
        self._temperature = temperature

    async def run(
        self, tasks: Sequence[Task], label: str = "", progress: object = None
    ) -> Report:
        started = time.perf_counter()
        results: list[TaskResult] = []

        for task in tasks:
            result = TaskResult(task_id=task.id, category=task.category)
            for repeat in range(self._repeats):
                attempt = await self._attempt(task, repeat)
                result.attempts.append(attempt)
                if callable(progress):
                    progress(task, attempt, repeat)
            results.append(result)
            log.info(
                "eval.task_done",
                task=task.id,
                passes=result.passes,
                runs=result.runs,
                flaky=result.is_flaky,
            )

        info = self._orch.provider.info()
        return Report(
            results=results,
            label=label,
            model=getattr(info, "model", "") or "",
            provider=self._orch.provider.name,
            repeats=self._repeats,
            duration_s=time.perf_counter() - started,
        )

    async def _attempt(self, task: Task, repeat: int) -> Attempt:
        """One run of one task, isolated from every other."""
        attempt = Attempt(task_id=task.id)
        seeded: list[str] = []

        try:
            for key, value in task.given_facts:
                await self._orch.remember(key, value, permanence=Permanence.LONG_TERM.value)
                seeded.append(key)

            # Seeded per repeat, so the same suite against the same build is
            # reproducible even though the model is stochastic -- and so the
            # repeats genuinely differ rather than all landing on one sample.
            params = GenerationParams(
                temperature=self._temperature,
                max_tokens=1024,
                seed=1000 + repeat,
            )

            started = time.perf_counter()
            if task.use_tools:
                turn = await self._orch.act(
                    task.prompt,
                    session_id=None,
                    params=params,
                    approve=task.approve,
                )
                outcome = turn.agent
                attempt.steps = len(getattr(outcome, "steps", []) or [])
                attempt.tools_used, attempt.tools_succeeded = _tools_used(outcome)
            else:
                turn = await self._orch.chat(task.prompt, session_id=None, params=params)
                attempt.steps = 1

            attempt.answer = turn.content
            attempt.latency_ms = int((time.perf_counter() - started) * 1000)

        except Exception as exc:
            # A crash is a result, not an interruption. The suite must finish
            # so the rest of the report survives one broken task.
            attempt.error = f"{type(exc).__name__}: {exc}"
            log.warning("eval.attempt_failed", task=task.id, error=attempt.error)
        finally:
            for key in seeded:
                await self._orch.forget(key)

        return score_attempt(task.checks, attempt)


def _tools_used(outcome: object) -> tuple[list[str], list[str]]:
    """Pull tool names, and their outcomes, out of the agent trace.

    Returns (attempted, succeeded). The split matters: `refused` needs to know
    whether a forbidden call actually *ran*, not merely whether the model tried
    it. Without the outcome the scorer has to guess from the answer's wording,
    which tests the model's prose rather than the guard.

    Defensive about shape: the trace is for humans first, and a scorer that
    crashed on an unexpected step would fail a task the model actually passed.
    """
    attempted: list[str] = []
    succeeded: list[str] = []
    for step in getattr(outcome, "steps", []) or []:
        for call in getattr(step, "tool_calls", []) or []:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or call.get("tool") or "")
            if not name:
                continue
            attempted.append(name)
            # Absent `ok` means the trace did not say; treat that as success so
            # a leak is never hidden by a missing field.
            if call.get("ok", True) and call.get("decision", "allow") == "allow":
                succeeded.append(name)
    return attempted, succeeded
