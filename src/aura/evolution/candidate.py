"""A proposed change, and its paperwork.

A candidate is inert. It records what would change, what it scored, and what
the comparison concluded -- and nothing about it applies itself. Deploy is a
separate, human-gated act (see deploy.py), and keeping the proposal and the
application in different modules is part of why that gate is hard to bypass by
accident.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aura.evolution.space import KNOBS_BY_PATH, validate_overrides
from aura.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class Candidate:
    """One proposed configuration change."""

    overrides: dict[str, float]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    # What the incumbent had, so a person reading this sees both sides and so
    # rollback does not depend on re-deriving it later.
    baseline_values: dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    # Filled in after evaluation.
    score: float | None = None
    baseline_score: float | None = None
    verdict: str = "unevaluated"
    reason: str = ""

    # Filled in at deploy.
    approved_by: str = ""
    approved_at: float = 0.0
    deployed_at: float = 0.0

    @property
    def is_deployable(self) -> bool:
        """Only a candidate that measurably beat the incumbent. Everything
        else -- indistinguishable, worse, rejected on safety, unevaluated --
        is not a thing to ship."""
        return self.verdict == "better"

    def describe(self) -> list[str]:
        """Human-readable diff. This is what a person approves, so it has to
        read as a change rather than as a data structure."""
        lines: list[str] = []
        for path, value in sorted(self.overrides.items()):
            was = self.baseline_values.get(path)
            knob = KNOBS_BY_PATH.get(path)
            arrow = f"{was} -> {value}" if was is not None else str(value)
            lines.append(f"{path}: {arrow}")
            if knob and knob.why:
                lines.append(f"    {knob.why}")
        return lines


def propose(settings: Any, path: str, value: float, baseline: float) -> Candidate:
    """Build one candidate that changes exactly one knob.

    One knob at a time is the whole method (coordinate ascent). It is slow, and
    it buys the thing that matters: when a candidate wins, there is no question
    about which change caused it.
    """
    overrides = validate_overrides({path: value})
    return Candidate(overrides=overrides, baseline_values={path: baseline})


class CandidateStore:
    """Candidates on disk, so approval can happen later than proposal.

    Deliberately a separate step: a search that proposed and applied in one
    breath would make the approval gate a formality nobody reads.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def _load_all(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("evolution.candidate_file_corrupt", path=str(self._path))
            return {}
        return payload if isinstance(payload, dict) else {}

    def save(self, candidate: Candidate) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        everything = self._load_all()
        everything[candidate.id] = asdict(candidate)
        self._path.write_text(
            json.dumps(everything, indent=2, default=str), encoding="utf-8"
        )
        log.info("evolution.candidate_saved", id=candidate.id, verdict=candidate.verdict)

    def get(self, candidate_id: str) -> Candidate | None:
        raw = self._load_all().get(candidate_id)
        if raw is None:
            return None
        return Candidate(**raw)

    def list_all(self) -> list[Candidate]:
        return sorted(
            (Candidate(**raw) for raw in self._load_all().values()),
            key=lambda c: c.created_at,
            reverse=True,
        )
