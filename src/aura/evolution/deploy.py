"""Applying an approved candidate, and getting back.

**The approval gate lives here and has no bypass.** `apply_candidate` takes an
`approved_by` string and refuses an empty one; there is no `force`, no
`--yes`, and no configuration setting that makes approval automatic. Spec §18
and standing constraint 3 both say the same thing, and the way to keep a rule
like that is to make the unsafe path unwriteable rather than merely
discouraged.

A deploy writes YAML into `config/local.yaml` -- the layer that already exists
for machine-specific settings, above the defaults and below the environment.
No code is written. No running process is patched. AURA is restarted, exactly
as it would be after any config edit.
"""

from __future__ import annotations

import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from aura.config import CONFIG_DIR, PROJECT_ROOT
from aura.evolution.candidate import Candidate
from aura.evolution.space import as_nested, validate_overrides
from aura.logging import get_logger

log = get_logger(__name__)

LOCAL_CONFIG = CONFIG_DIR / "local.yaml"
SNAPSHOT_DIR = PROJECT_ROOT / "var" / "config-snapshots"


class ApprovalRequiredError(Exception):
    """Raised when a deploy is attempted without a named human approver."""


class NotDeployableError(Exception):
    """Raised when a candidate that did not win is pushed anyway."""


def _read_local() -> dict[str, Any]:
    if not LOCAL_CONFIG.exists():
        return {}
    try:
        payload = yaml.safe_load(LOCAL_CONFIG.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        log.error("evolution.local_config_unparseable", path=str(LOCAL_CONFIG))
        raise
    return payload if isinstance(payload, dict) else {}


def snapshot() -> Path | None:
    """Copy the current local config aside before touching it.

    A file copy rather than a reconstruction, so rollback cannot get the
    previous state subtly wrong -- which is the failure mode that turns a
    rollback into a second incident.
    """
    if not LOCAL_CONFIG.exists():
        return None
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = SNAPSHOT_DIR / f"local-{stamp}.yaml"
    shutil.copy2(LOCAL_CONFIG, target)
    log.info("evolution.snapshot_taken", path=str(target))
    return target


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """One level deep, which is all the config shape needs.

    Not a full recursive merge: config is section -> key -> scalar, and a
    generic deep merge would silently do something surprising the first time
    someone nests a list.
    """
    merged = dict(base)
    for section, values in overrides.items():
        existing = merged.get(section)
        if isinstance(existing, dict) and isinstance(values, dict):
            merged[section] = {**existing, **values}
        else:
            merged[section] = values
    return merged


def apply_candidate(candidate: Candidate, approved_by: str) -> Path | None:
    """Write an approved candidate into config/local.yaml.

    Returns the snapshot path, for rollback. Raises rather than returning a
    failure value: deploying is the one place where a caller ignoring a return
    code would be dangerous.
    """
    if not approved_by.strip():
        raise ApprovalRequiredError(
            "A deploy needs a named human approver. There is no automatic path, "
            "by design (spec section 18)."
        )
    if not candidate.is_deployable:
        raise NotDeployableError(
            f"Candidate {candidate.id} is '{candidate.verdict}', not 'better'. "
            "Only a change that measurably beat the incumbent may be deployed."
        )

    # Validated a second time, deliberately. The candidate has been sitting in
    # a JSON file since it was proposed, and a file can be edited by hand.
    overrides = validate_overrides(candidate.overrides)

    previous = snapshot()
    merged = _merge(_read_local(), as_nested(overrides))

    LOCAL_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Machine-local configuration.\n"
        "#\n"
        f"# Last modified by `aura evolve apply` at "
        f"{datetime.now().isoformat(timespec='seconds')}\n"
        f"# Candidate {candidate.id}, approved by {approved_by}.\n"
        "# Roll back with: aura evolve rollback\n\n"
    )
    LOCAL_CONFIG.write_text(
        header + yaml.safe_dump(merged, sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )

    candidate.approved_by = approved_by
    candidate.approved_at = time.time()
    candidate.deployed_at = time.time()

    log.warning(
        "evolution.deployed",
        id=candidate.id,
        approved_by=approved_by,
        overrides=overrides,
        snapshot=str(previous) if previous else None,
    )
    return previous


def list_snapshots() -> list[Path]:
    if not SNAPSHOT_DIR.exists():
        return []
    return sorted(SNAPSHOT_DIR.glob("local-*.yaml"), reverse=True)


def rollback(snapshot_path: Path | None = None) -> Path | None:
    """Restore the most recent snapshot, or a named one.

    Returns the snapshot restored, or None when there is nothing to restore.
    """
    target = snapshot_path
    if target is None:
        available = list_snapshots()
        if not available:
            return None
        target = available[0]

    if not target.exists():
        return None

    # Snapshot the current state too, so a rollback is itself reversible. An
    # undo that cannot be undone is how a bad rollback becomes permanent.
    snapshot()
    shutil.copy2(target, LOCAL_CONFIG)
    log.warning("evolution.rolled_back", restored=str(target))
    return target
