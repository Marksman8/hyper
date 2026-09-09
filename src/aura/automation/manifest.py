"""The workflow manifest — AURA's authority over automation.

**This file implements ADR-0004 D2, and it is the security boundary of the
automation feature.** Everything else here is plumbing.

The rule: permission levels come from a local, version-controlled file, and
from nowhere else. n8n describes what a workflow *is*; the manifest determines
what AURA is *allowed to do with it*.

Why that matters, concretely. A workflow named `draft_email` is reasonably
declared LEVEL 2. Later, in a browser, it is renamed `send_email`, or its final
node changes from "save draft" to "send". Same id, same webhook. If AURA
derived permission from anything n8n reports, she would keep running it at
LEVEL 2 without confirmation -- now sending mail under a permission granted for
drafting.

So nothing is discovered. An undeclared workflow is not callable, and changing
a workflow's authority requires editing a file that shows up in a diff.

Validation is strict and happens at load. A manifest error must stop the
process at startup rather than surface halfway through an agent task, where the
tool would be missing and the model would improvise around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aura.errors import AuraError
from aura.logging import get_logger
from aura.security.permissions import PermissionLevel

log = get_logger(__name__)

# Levels a workflow may be declared at. LEVEL 0 is excluded deliberately:
# OBSERVE means "read AURA's own state with no side effects", and calling out
# to an external system is never that.
ALLOWED_LEVELS = frozenset({1, 2, 3, 4})

# Verbs that force LEVEL 4 regardless of what the manifest says. A belt-and-
# braces check against a declaration that is merely optimistic -- someone
# writing `send_slack` at level 2 has almost certainly made a mistake, and the
# cost of catching it is one loud failure at startup.
DESTRUCTIVE_HINTS = (
    "send",
    "post",
    "publish",
    "delete",
    "remove",
    "pay",
    "purchase",
    "transfer",
    "deploy",
    "email",
    "message",
    "tweet",
)

MAX_TIMEOUT_S = 300.0


class ManifestError(AuraError):
    """The manifest is wrong. Raised at load, never mid-run."""

    code = "automation_manifest_invalid"
    http_status = 500


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    """One declared workflow. The manifest's word is final on every field."""

    id: str
    description: str
    endpoint: str
    parameters: dict[str, Any]
    level: PermissionLevel
    confirm_always: bool = False
    timeout_s: float = 30.0
    idempotent: bool = True
    reversible: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def tool_name(self) -> str:
        return self.id


def _require(raw: dict[str, Any], key: str, where: str) -> Any:
    if key not in raw or raw[key] in (None, ""):
        raise ManifestError(f"{where}: missing required field {key!r}")
    return raw[key]


def _looks_destructive(spec_id: str, description: str) -> bool:
    haystack = f"{spec_id} {description}".lower()
    return any(hint in haystack for hint in DESTRUCTIVE_HINTS)


def parse_workflow(raw: Any, index: int) -> WorkflowSpec:
    """Validate one entry. Strict on purpose."""
    if not isinstance(raw, dict):
        raise ManifestError(f"workflow #{index} is not a mapping")

    where = f"workflow {raw.get('id', f'#{index}')!r}"

    spec_id = str(_require(raw, "id", where)).strip()
    if not spec_id.replace("_", "").isalnum():
        raise ManifestError(
            f"{where}: id must be alphanumeric with underscores -- it becomes a "
            "tool name the model calls by hand"
        )

    # Both of these are required, and their absence is the most likely mistake.
    # Without a description the agent cannot choose the tool; without a schema
    # ToolExecutor cannot validate arguments, and validation is the FIRST stage
    # of the chokepoint. A workflow missing either is registered and unusable.
    description = str(_require(raw, "description", where)).strip()
    if len(description) < 10:
        raise ManifestError(
            f"{where}: description is how the model decides to call this. "
            "Write a sentence, not a label."
        )

    parameters = _require(raw, "parameters", where)
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise ManifestError(
            f"{where}: parameters must be a JSON Schema object, so arguments "
            "can be validated before anything runs"
        )

    endpoint = str(_require(raw, "webhook", where)).strip()
    if not endpoint.startswith(("http://", "https://")):
        raise ManifestError(f"{where}: webhook must be an http(s) URL")

    raw_level = _require(raw, "level", where)
    if not isinstance(raw_level, int) or raw_level not in ALLOWED_LEVELS:
        raise ManifestError(
            f"{where}: level must be one of {sorted(ALLOWED_LEVELS)}. "
            "LEVEL 0 means no side effects, which calling an external system "
            "never is."
        )
    level = PermissionLevel(raw_level)

    confirm = str(raw.get("confirm", "")).lower() == "always"

    # ADR-0004 D5. Anything that sends, posts, pays or deletes is LEVEL 4 with
    # confirmation. Refuse at load rather than silently upgrading: a manifest
    # that says level 2 and behaves as level 4 is a manifest whose author has
    # misunderstood something, and they should find out now.
    understated = level < PermissionLevel.CRITICAL or not confirm
    if _looks_destructive(spec_id, description) and understated:
        raise ManifestError(
            f"{where}: looks like it sends, posts, deletes or pays. Those are "
            "LEVEL 4 with `confirm: always` (ADR-0004 D5). If this workflow "
            "really is harmless, rename it so that is obvious."
        )

    timeout_s = float(raw.get("timeout_s", 30.0))
    if not 0 < timeout_s <= MAX_TIMEOUT_S:
        raise ManifestError(f"{where}: timeout_s must be between 0 and {MAX_TIMEOUT_S}")

    return WorkflowSpec(
        id=spec_id,
        description=description,
        endpoint=endpoint,
        parameters=parameters,
        level=level,
        confirm_always=confirm,
        timeout_s=timeout_s,
        idempotent=bool(raw.get("idempotent", True)),
        reversible=bool(raw.get("reversible", True)),
        tags=tuple(str(t) for t in (raw.get("tags") or ())),
    )


def load_manifest(path: Path) -> list[WorkflowSpec]:
    """Read and validate the manifest.

    A missing file is not an error -- it means no workflows are declared, and
    therefore none are callable. That is the correct default: automation is
    opt-in by writing a file, not opt-out by deleting one.
    """
    if not path.exists():
        log.info("automation.no_manifest", path=str(path))
        return []

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path} is not valid YAML: {exc}") from exc

    raw = payload.get("workflows") if isinstance(payload, dict) else None
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ManifestError(f"{path}: `workflows` must be a list")

    specs = [parse_workflow(entry, i) for i, entry in enumerate(raw)]

    seen: set[str] = set()
    for spec in specs:
        if spec.id in seen:
            # Two entries with one id means one silently shadows the other,
            # and which one wins would depend on ordering. A permission level
            # decided by list order is not a permission level.
            raise ManifestError(f"duplicate workflow id {spec.id!r}")
        seen.add(spec.id)

    log.info(
        "automation.manifest_loaded",
        path=str(path),
        workflows=[s.id for s in specs],
    )
    return specs
