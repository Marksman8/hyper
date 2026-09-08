"""The controlled-evolution chain, from the terminal.

    aura evolve search      propose candidates and measure them
    aura evolve list        what has been proposed, and how it did
    aura evolve show <id>   the full diff and verdict for one
    aura evolve apply <id>  deploy it -- asks for a name, every time
    aura evolve rollback    put the previous config back

`search` never applies anything. `apply` never runs without a name typed at
the prompt. Keeping those in separate commands is most of what stops the
approval gate becoming a formality nobody reads.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aura.config import PROJECT_ROOT, Settings, get_settings
from aura.eval.compare import load_report
from aura.eval.runner import SuiteRunner
from aura.eval.suite import load_suite
from aura.eval.types import Report
from aura.evolution.candidate import Candidate, CandidateStore
from aura.evolution.deploy import (
    ApprovalRequiredError,
    NotDeployableError,
    apply_candidate,
    list_snapshots,
)
from aura.evolution.deploy import rollback as do_rollback
from aura.evolution.search import HillClimber, best_of
from aura.evolution.space import KNOBS, KNOBS_BY_PATH, as_nested, current_value

console = Console()

CANDIDATES = PROJECT_ROOT / "var" / "data" / "candidates.json"
BASELINE = PROJECT_ROOT / "var" / "data" / "eval_baseline.json"


def store() -> CandidateStore:
    return CandidateStore(CANDIDATES)


def _verdict_colour(verdict: str) -> str:
    return {
        "better": "green",
        "worse": "red",
        "rejected_on_safety": "red",
        "indistinguishable": "yellow",
    }.get(verdict, "dim")


async def _score_with(cfg: Settings, overrides: dict[str, float], repeats: int) -> Report:
    """Run the suite against a configuration that differs by the overrides.

    The candidate config is built in memory and evaluated against a throwaway
    database. Nothing is written to config/local.yaml -- that only happens at
    `apply`, after a person has said yes.
    """
    from aura.clients.eval_cli import _build

    nested: dict[str, Any] = as_nested(overrides)
    update: dict[str, Any] = {}
    for section, values in nested.items():
        update[section] = getattr(cfg, section).model_copy(update=values)
    candidate_cfg = cfg.model_copy(update=update)

    tasks = load_suite()
    with tempfile.TemporaryDirectory(prefix="aura-evolve-") as scratch:
        orchestrator, memory = await _build(candidate_cfg, Path(scratch) / "eval.db")
        try:
            return await SuiteRunner(orchestrator, repeats=repeats).run(tasks)
        finally:
            memory.close()


async def run_search(knob_name: str | None, repeats: int) -> None:
    cfg = get_settings()

    if not BASELINE.exists():
        console.print(
            Panel(
                "No baseline to compare against.\n\n"
                "Record one first: [bold]aura eval --set-baseline[/bold]\n\n"
                "A search without a baseline cannot tell an improvement from noise.",
                title="[red]Nothing to compare[/red]",
                border_style="red",
            )
        )
        return

    baseline = load_report(BASELINE)

    knobs = list(KNOBS)
    if knob_name:
        knob = KNOBS_BY_PATH.get(knob_name)
        if knob is None:
            console.print(f"[red]{knob_name} is not tunable.[/red] Try:")
            for k in KNOBS:
                console.print(f"  {k.path}")
            return
        knobs = [knob]

    evaluations = sum(len(k.neighbours(current_value(cfg, k.path))) for k in knobs)
    console.print(
        f"[dim]Baseline {baseline.score:.1%}. {evaluations} candidates to evaluate, "
        f"{repeats} repeats each. Nothing will be applied.[/dim]\n"
    )

    saved = store()

    def note(candidate: Candidate) -> None:
        saved.save(candidate)
        colour = _verdict_colour(candidate.verdict)
        diff = ", ".join(candidate.describe()[:1])
        console.print(
            f"  [{colour}]{candidate.verdict:<20}[/{colour}] {diff}  "
            f"[dim]{candidate.score:.1%} vs {candidate.baseline_score:.1%} "
            f"({candidate.id})[/dim]"
        )

    async def evaluate(overrides: dict[str, float]) -> Report:
        return await _score_with(cfg, overrides, repeats)

    climber = HillClimber(cfg, evaluate)
    found = await climber.run(baseline, [], knobs=knobs, on_candidate=note)

    winner = best_of(found)
    console.print()
    if winner is None:
        console.print(
            Panel(
                "No candidate beat the incumbent by more than the noise floor.\n\n"
                "That is a result, not a failure: the current configuration is as "
                "good as anything one step away from it, as far as this suite can "
                "tell.",
                title="[yellow]Nothing to deploy[/yellow]",
                border_style="yellow",
            )
        )
        return

    console.print(
        Panel(
            "\n".join(winner.describe())
            + f"\n\n{winner.reason}\n\n"
            f"Deploy it with: [bold]aura evolve apply {winner.id}[/bold]\n"
            "[dim]That will ask for your name. There is no automatic path.[/dim]",
            title="[green]Candidate found[/green]",
            border_style="green",
        )
    )


async def _audit(action: str, detail: dict[str, Any], target: str = "") -> None:
    """Record a link in the chain to the real audit table.

    Structured logs are for operators reading a terminal; the audit table is
    what answers "why is AURA configured like this?" months later (spec §25).
    A config change that changed itself is exactly the case where that gets
    asked, so it belongs in both.
    """
    from aura.memory.store import MemoryStore
    from aura.security.audit import AuditLog
    from aura.security.permissions import PermissionLevel

    cfg = get_settings()
    store = MemoryStore(cfg.memory.resolved_db_path())
    store.connect()
    try:
        await AuditLog(store, enabled=True, required=False).record(
            actor="user",
            action=action,
            # CRITICAL: changing AURA's own configuration is the highest level
            # in the taxonomy, whatever the individual value happens to be.
            level=PermissionLevel.CRITICAL,
            target=target,
            detail=detail,
        )
    finally:
        store.close()


def run_list() -> None:
    candidates = store().list_all()
    if not candidates:
        console.print("[dim]Nothing proposed yet. Run [/dim][bold]aura evolve search[/bold].")
        return

    table = Table(box=None, padding=(0, 2, 0, 0))
    table.add_column("id", style="bold")
    table.add_column("change")
    table.add_column("score", justify="right")
    table.add_column("verdict")
    table.add_column("deployed")

    for candidate in candidates:
        change = ", ".join(
            f"{p}={v}" for p, v in sorted(candidate.overrides.items())
        )
        colour = _verdict_colour(candidate.verdict)
        table.add_row(
            candidate.id,
            change,
            f"{candidate.score:.1%}" if candidate.score is not None else "-",
            f"[{colour}]{candidate.verdict}[/{colour}]",
            "[green]yes[/green]" if candidate.deployed_at else "",
        )
    console.print(table)


def run_show(candidate_id: str) -> None:
    candidate = store().get(candidate_id)
    if candidate is None:
        console.print(f"[red]No candidate {candidate_id}.[/red]")
        return

    body = "\n".join(candidate.describe())
    if candidate.score is not None:
        body += (
            f"\n\nScored {candidate.score:.1%} against a baseline of "
            f"{candidate.baseline_score:.1%}."
        )
    body += f"\n\n{candidate.reason}"
    if candidate.deployed_at:
        body += f"\n\n[green]Deployed[/green], approved by {candidate.approved_by}."
    elif candidate.is_deployable:
        body += f"\n\nDeploy with: [bold]aura evolve apply {candidate.id}[/bold]"

    console.print(
        Panel(
            body,
            title=f"[{_verdict_colour(candidate.verdict)}]{candidate.verdict}[/]",
            border_style=_verdict_colour(candidate.verdict),
        )
    )


async def run_apply(candidate_id: str) -> None:
    """The approval gate. Deliberately interactive."""
    candidate = store().get(candidate_id)
    if candidate is None:
        console.print(f"[red]No candidate {candidate_id}.[/red]")
        return

    console.print(
        Panel(
            "\n".join(candidate.describe())
            + f"\n\n{candidate.reason}\n\n"
            "This writes config/local.yaml. The previous version is snapshotted "
            "first, and [bold]aura evolve rollback[/bold] restores it.\n"
            "AURA must be restarted for the change to take effect.",
            title="[bold]About to change AURA's configuration[/bold]",
            border_style="yellow",
        )
    )

    who = console.input("Type your name to approve, or press Enter to cancel: ").strip()
    if not who:
        console.print("[dim]Cancelled. Nothing was changed.[/dim]")
        return

    try:
        snapshot_path = apply_candidate(candidate, approved_by=who)
    except (ApprovalRequiredError, NotDeployableError) as exc:
        console.print(f"[red]Refused:[/red] {exc}")
        # A refused deploy is audited too. An attempt to change the safety
        # envelope is exactly the thing someone would want to find later.
        await _audit(
            "evolution.deploy_refused",
            {"candidate": candidate_id, "reason": str(exc), "attempted_by": who},
            target=candidate_id,
        )
        return

    store().save(candidate)
    await _audit(
        "evolution.deployed",
        {
            "candidate": candidate_id,
            "approved_by": who,
            "overrides": candidate.overrides,
            "was": candidate.baseline_values,
            "score": candidate.score,
            "baseline_score": candidate.baseline_score,
            "verdict": candidate.verdict,
            "snapshot": snapshot_path.name if snapshot_path else None,
        },
        target=candidate_id,
    )
    console.print(f"\n[green]Applied.[/green] Approved by {who}.")
    if snapshot_path:
        console.print(f"[dim]Previous config saved to {snapshot_path.name}[/dim]")
    console.print("[dim]Restart AURA Core for it to take effect.[/dim]")


async def run_rollback() -> None:
    snapshots = list_snapshots()
    if not snapshots:
        console.print("[yellow]No snapshots to roll back to.[/yellow]")
        return

    console.print(f"Most recent snapshot: [bold]{snapshots[0].name}[/bold]")
    if console.input("Restore it? Type yes to confirm: ").strip().lower() != "yes":
        console.print("[dim]Cancelled.[/dim]")
        return

    restored = do_rollback()
    if restored is None:
        console.print("[yellow]Nothing was restored.[/yellow]")
        return
    await _audit("evolution.rolled_back", {"restored": restored.name}, target=restored.name)
    console.print(f"[green]Restored[/green] {restored.name}.")
    console.print("[dim]Restart AURA Core for it to take effect.[/dim]")


def run_knobs() -> None:
    """What may be tuned, and what may never be."""
    cfg = get_settings()

    table = Table(box=None, padding=(0, 2, 0, 0))
    table.add_column("tunable", style="bold")
    table.add_column("now", justify="right")
    table.add_column("range")
    for knob in KNOBS:
        table.add_row(
            knob.path,
            str(current_value(cfg, knob.path)),
            f"{knob.low} .. {knob.high} (step {knob.step})",
        )
    console.print(table)

    from aura.evolution.space import FORBIDDEN

    console.print(
        f"\n[dim]{len(FORBIDDEN)} settings are permanently excluded -- permission "
        "levels, the workspace guard, audit, the privacy gate, the emergency\n"
        "stop. A search that could loosen a guard to score better would, so it "
        "cannot express that at all.[/dim]"
    )


def search_entry(knob: str | None, repeats: int) -> None:
    asyncio.run(run_search(knob, repeats))
