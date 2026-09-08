"""Running the evaluation suite.

**In-process, and against a scratch database.** Unlike every other CLI here,
this is not an HTTP client. Two reasons, and both matter:

- It needs the agent's step trace to score tool selection, and the trace is
  richer in-process than over the wire.
- More importantly, it *seeds facts*. Running the memory tasks against the
  user's real database would write test data into their actual memory and then
  delete it again -- a side effect nobody asked a benchmark to have. So eval
  gets its own throwaway store and never touches `var/data/aura.db`.

It shares the model server with everything else, because that is the thing
under test.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aura.config import Settings, get_settings
from aura.core.orchestrator import Orchestrator
from aura.eval.compare import compare, load_report, save_report
from aura.eval.runner import SuiteRunner
from aura.eval.suite import SuiteError, load_suite
from aura.eval.types import Report, Task
from aura.memory.store import MemoryStore
from aura.models.registry import build_provider
from aura.security.audit import AuditLog

console = Console()


def _eval_settings(cfg: Settings, db_path: Path) -> Settings:
    """A copy of the real configuration pointed at a scratch database.

    Learning is switched off too: extraction would fire on the suite's own
    prompts and write inferences about a user who was never here.
    """
    return cfg.model_copy(
        update={
            "memory": cfg.memory.model_copy(
                update={"db_path": str(db_path), "learn_from_conversation": False}
            )
        }
    )


async def _build(cfg: Settings, db_path: Path) -> tuple[Orchestrator, MemoryStore]:
    settings = _eval_settings(cfg, db_path)
    store = MemoryStore(settings.memory.resolved_db_path())
    store.connect()

    provider = build_provider(settings)
    audit = AuditLog(store, enabled=False, required=False)

    runner = None
    if settings.tools.enabled:
        from aura.tools.factory import build_context, build_executor
        from aura.tools.runner import ExecutorToolRunner

        runner = ExecutorToolRunner(build_executor(settings, audit), build_context(settings))

    return Orchestrator(settings, provider, store, audit, tools=runner), store


def _render(report: Report) -> None:
    """Always the breakdown with the headline. See the phase doc §5: a lone
    number invites optimising the number."""
    table = Table(box=None, padding=(0, 2, 0, 0))
    table.add_column("task", style="bold")
    table.add_column("category")
    table.add_column("passed", justify="right")
    table.add_column("median ms", justify="right")
    table.add_column("")

    for result in report.results:
        if result.pass_rate == 1.0:
            mark, colour = "ok", "green"
        elif result.pass_rate == 0.0:
            mark, colour = "FAIL", "red"
        else:
            mark, colour = "flaky", "yellow"
        table.add_row(
            result.task_id,
            str(result.category),
            f"{result.passes}/{result.runs}",
            f"{result.median_latency_ms:.0f}",
            f"[{colour}]{mark}[/]",
        )
    console.print(table)

    console.print("\n[bold]By category[/bold]")
    for name, rate in report.by_category().items():
        console.print(f"  {name:<16} {rate:6.1%}")

    headline = (
        f"[bold]Score {report.score:.1%}[/bold]  "
        f"({report.total_runs} runs, {report.repeats}x each, "
        f"spread {report.spread:.1%}, {report.duration_s:.0f}s)"
    )
    console.print("\n" + headline)
    console.print(f"[dim]median latency {report.median_latency_ms():.0f} ms[/dim]")

    unsafe = report.safety_failures()
    if unsafe:
        names = ", ".join(r.task_id for r in unsafe)
        console.print(
            Panel(
                f"Safety tasks did not pass cleanly: {names}\n"
                "These are never averaged away or traded against gains elsewhere.",
                title="[red]Safety[/red]",
                border_style="red",
            )
        )

    flaky = report.flaky()
    if flaky:
        console.print(
            f"\n[yellow]Flaky:[/yellow] {', '.join(r.task_id for r in flaky)}"
            "\n[dim]Passed sometimes. Not a pass -- and the most interesting"
            " thing in the report.[/dim]"
        )


async def run_eval(
    suite_path: Path | None,
    repeats: int,
    save_to: Path | None,
    baseline: Path | None,
    label: str,
) -> None:
    cfg = get_settings()

    try:
        tasks: list[Task] = load_suite(suite_path)
    except SuiteError as exc:
        console.print(f"[red]Suite error:[/red] {exc}")
        return

    console.print(
        f"[dim]{len(tasks)} tasks x {repeats} repeats = {len(tasks) * repeats} runs. "
        "This talks to the model server; it will take a while.[/dim]\n"
    )

    with tempfile.TemporaryDirectory(prefix="aura-eval-") as scratch:
        orchestrator, store = await _build(cfg, Path(scratch) / "eval.db")

        health = await orchestrator.provider.health()
        if not health.available:
            console.print(
                Panel(
                    "The model server is not reachable, so every task would fail for "
                    "the same uninteresting reason.\n\n"
                    "Start it with: [bold]scripts/start_model_server.ps1[/bold]\n\n"
                    f"[dim]{health.detail}[/dim]",
                    title="[red]Nothing to measure[/red]",
                    border_style="red",
                )
            )
            store.close()
            return

        done = {"n": 0}
        total = len(tasks) * repeats
        started = time.perf_counter()

        def progress(task: Task, attempt: object, repeat: int) -> None:
            done["n"] += 1
            elapsed = time.perf_counter() - started
            rate = elapsed / done["n"]
            remaining = (total - done["n"]) * rate
            mark = "[green]ok[/green]" if getattr(attempt, "passed", False) else "[red]x[/red]"
            console.print(
                f"  {done['n']:>3}/{total}  {mark}  {task.id}"
                f"  [dim]~{remaining:.0f}s left[/dim]"
            )

        try:
            report = await SuiteRunner(orchestrator, repeats=repeats).run(
                tasks, label=label, progress=progress
            )
        finally:
            store.close()

    console.print()
    _render(report)

    if save_to:
        # Off the loop: these are small files, but a blocking write in an
        # async function is the habit that eventually stalls something real.
        await asyncio.to_thread(save_report, report, save_to)
        console.print(f"\n[dim]Saved to {save_to}[/dim]")

    if baseline:
        if not await asyncio.to_thread(baseline.exists):
            console.print(f"\n[yellow]No baseline at {baseline} to compare against.[/yellow]")
            return
        previous = await asyncio.to_thread(load_report, baseline)
        verdict = compare(previous, report)
        colour = {
            "better": "green",
            "worse": "red",
            "rejected_on_safety": "red",
            "indistinguishable": "yellow",
        }[verdict.verdict]
        console.print(
            Panel(
                verdict.summary(),
                title=f"[{colour}]{verdict.verdict.replace('_', ' ')}[/{colour}]",
                border_style=colour,
            )
        )
