"""What may be tuned, and what may never be.

**This file is the security boundary of Phase 9.** A hill-climber's whole
nature is to find whatever makes the number go up. If loosening a guard makes
the number go up, it will loosen the guard -- and by the metric it will be
right. The metric is not the point.

So this is an **allow-list**, not a deny-list. A parameter that is not
enumerated here cannot be varied, which means adding a tunable is a deliberate
act by a person editing this file, reviewed like any other change. A deny-list
would invert that: every new config option would become tunable by default, and
the day someone adds `security.require_audit` nobody would notice.

Candidates are configuration, never generated code (see the phase doc §2).
Config is bounded, inspectable on one screen, trivially reversible, and cannot
introduce a capability that did not already exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aura.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Knob:
    """One tunable parameter, with hard bounds.

    `path` is dotted into Settings, e.g. "llm.temperature". Bounds are enforced
    at proposal time *and* again before deploy, because a candidate can be
    written to a file by hand between the two.
    """

    path: str
    low: float
    high: float
    # Step used when proposing a neighbour. Coordinate ascent moves one knob by
    # one step; a step that is too small wastes evaluations inside the noise
    # floor, and one that is too large steps over the optimum.
    step: float
    integer: bool = False
    why: str = ""

    def clamp(self, value: float) -> float:
        bounded = max(self.low, min(self.high, value))
        return float(round(bounded)) if self.integer else round(bounded, 4)

    def neighbours(self, current: float) -> list[float]:
        """The two adjacent values, minus any that are out of bounds or a
        no-op after clamping."""
        out: list[float] = []
        for candidate in (current - self.step, current + self.step):
            clamped = self.clamp(candidate)
            if clamped != self.clamp(current) and clamped not in out:
                out.append(clamped)
        return out


# Everything the optimiser is allowed to touch. Deliberately short.
KNOBS: tuple[Knob, ...] = (
    Knob(
        path="llm.temperature",
        low=0.0,
        high=1.2,
        step=0.15,
        why="Lower temperature should make flaky tasks consistent, which is "
        "where this suite's remaining headroom actually is.",
    ),
    Knob(
        path="llm.top_p",
        low=0.5,
        high=1.0,
        step=0.05,
        why="Narrows the sampling tail without flattening it entirely.",
    ),
    Knob(
        path="agent.tool_temperature",
        low=0.0,
        high=0.8,
        step=0.1,
        why="Choosing a tool is a decision; creativity there produces "
        "malformed calls.",
    ),
    Knob(
        path="agent.max_steps",
        low=3,
        high=12,
        step=1,
        integer=True,
        why="More steps can rescue a recovery task or let the loop wander. "
        "Bounded well below anything that could run away.",
    ),
    Knob(
        path="memory.history_turns",
        low=4,
        high=24,
        step=2,
        integer=True,
        why="More history is more context and more distraction.",
    ),
    Knob(
        path="memory.injection_floor",
        low=0.4,
        high=0.8,
        step=0.05,
        why="How sure AURA must be before acting on a preference. Tunable, but "
        "only within a range that keeps explicit statements believed.",
    ),
    Knob(
        path="memory.search_results",
        low=0,
        high=10,
        step=2,
        integer=True,
        why="How much past conversation is pulled back into context.",
    ),
)

KNOBS_BY_PATH: dict[str, Knob] = {k.path: k for k in KNOBS}


# Config paths that must never be varied, with the reason each one matters.
# Kept as data so the refusal can *explain itself* rather than just saying no,
# and so a test can assert the list has not quietly shrunk.
FORBIDDEN: dict[str, str] = {
    "tools.max_autonomous_level": (
        "An optimiser that can raise its own permission ceiling will, because "
        "more capability scores better on almost any benchmark."
    ),
    "tools.max_confirmable_level": (
        "Same reason: the confirmation ceiling is a human decision about risk, "
        "not a parameter with an optimal value."
    ),
    "tools.deny_patterns": (
        "The workspace guard is the thing being protected. It cannot also be a "
        "variable in the search."
    ),
    "tools.workspace_roots": (
        "Widening the workspace would let a candidate score better by reading "
        "more of the machine."
    ),
    "tools.python_enabled": "Whether AURA may execute code is not a tuning knob.",
    "security.api_token": "Authentication is not a performance parameter.",
    "security.audit_enabled": (
        "Auditing costs a little latency, so an optimiser would eventually "
        "switch it off to go faster. That is exactly the trade nobody wants."
    ),
    "security.require_audit": (
        "Whether an action must be recorded before it runs is a safety "
        "property, not a latency budget."
    ),
    "vision.mode": (
        "The privacy gate is the user's decision. It is never a search result."
    ),
    "vision.deny_apps": (
        "Which applications AURA may look at is something the user said out "
        "loud. A search must not be able to un-say it."
    ),
    "vision.allow_apps": (
        "Widening what may be captured would let a candidate score better by "
        "seeing more of the screen."
    ),
    "app.emergency_stop": (
        "The stop button cannot be a parameter that something else is "
        "allowed to press."
    ),
    "memory.learn_from_conversation": (
        "Switching learning off would score better on a suite that seeds its "
        "own facts -- a local optimum that makes AURA worse in real use."
    ),
    "personalization.enabled": (
        "Same shape of trap: the suite does not measure personalisation, so "
        "disabling it is free score and a real regression."
    ),
    "automation.manifest_path": (
        "The manifest is the sole authority for what AURA may do with an "
        "automation provider (ADR-0004 D2). Pointing the loader at a "
        "different file would replace that authority with a search result."
    ),
    "automation.enabled": (
        "Whether AURA can reach third-party services is a human decision, not "
        "a parameter with an optimal value."
    ),
    "automation.provider": (
        "Which external system executes on AURA's behalf is not a tuning knob."
    ),
    "automation.auth_token": "Authentication is not a performance parameter.",
    "automation.auth_token_file": "Same: where the secret lives is not tunable.",
    "automation.verify_tls": (
        "Turning off certificate verification would make some failures "
        "disappear, and a hill-climber would notice that before it noticed "
        "why the check was there."
    ),
    "automation.base_url": (
        "Redirecting automation at a different host is an egress decision a "
        "person makes, in a diff."
    ),
}


class ForbiddenKnobError(Exception):
    """Raised when something tries to tune a parameter that is off-limits."""


def check_allowed(path: str) -> None:
    """Gate every path before it is used. Raises rather than returning False:
    a caller that forgets to check a boolean would silently proceed."""
    if path in FORBIDDEN:
        raise ForbiddenKnobError(f"{path} may never be tuned. {FORBIDDEN[path]}")
    if path not in KNOBS_BY_PATH:
        raise ForbiddenKnobError(
            f"{path} is not in the tunable allow-list. Add it to "
            "evolution/space.py deliberately, or it stays fixed."
        )


def current_value(settings: Any, path: str) -> float:
    """Read a knob's value out of Settings by dotted path."""
    node: Any = settings
    for part in path.split("."):
        node = getattr(node, part)
    return float(node)


def validate_overrides(overrides: dict[str, float]) -> dict[str, float]:
    """Check and clamp a candidate's overrides.

    Run again immediately before deploy, not only at proposal: a candidate
    lives in a file between being proposed and being approved, and a file can
    be edited by hand.
    """
    clean: dict[str, float] = {}
    for path, value in overrides.items():
        check_allowed(path)
        knob = KNOBS_BY_PATH[path]
        clamped = knob.clamp(float(value))
        if clamped != float(value):
            log.warning(
                "evolution.value_clamped", path=path, requested=value, applied=clamped
            )
        clean[path] = clamped
    return clean


def as_nested(overrides: dict[str, float]) -> dict[str, Any]:
    """Turn {"llm.temperature": 0.4} into {"llm": {"temperature": 0.4}} for
    YAML and for Settings construction."""
    out: dict[str, Any] = {}
    for path, value in overrides.items():
        section, _, leaf = path.partition(".")
        out.setdefault(section, {})[leaf] = value
    return out
