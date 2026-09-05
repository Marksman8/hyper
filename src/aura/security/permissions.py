"""Permission levels (spec section 8).

Defined in v0.1 although nothing dangerous exists yet. The reason is ordering:
a permission system added after the tools exist has to be threaded through code
that was written assuming it was absent, and every missed call site is a hole.
Adding it first means every tool author in Phase 3 has to answer "what level?"
before their code will run.

    LEVEL 0  OBSERVE     read AURA's own state. No side effects.
    LEVEL 1  READ        read the user's data. Files, screen, clipboard.
    LEVEL 2  ACT         reversible side effects. Write a file, open an app.
    LEVEL 3  MUTATE      hard to reverse. Delete, overwrite, git push, install.
    LEVEL 4  CRITICAL    system-level or irreversible. Registry, credentials,
                         network exposure, anything touching AURA's own code.

Policy is a per-level decision of allow / confirm / deny, so the user can say
"never ask about reading files, always ask before deleting, never let it touch
level 4" without editing code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from aura.errors import PermissionDeniedError


class PermissionLevel(IntEnum):
    OBSERVE = 0
    READ = 1
    ACT = 2
    MUTATE = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return _LABELS[self]


_LABELS: dict[PermissionLevel, str] = {
    PermissionLevel.OBSERVE: "observe",
    PermissionLevel.READ: "read",
    PermissionLevel.ACT: "act",
    PermissionLevel.MUTATE: "mutate",
    PermissionLevel.CRITICAL: "critical",
}


class Decision(StrEnum):
    """Policy outcome for a requested action."""

    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"
    # Not a policy outcome: recorded when an audited action raised.
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PermissionPolicy:
    """What AURA may do without asking.

    Defaults are deliberately conservative: v0.1 has no tools, so anything
    above READ arriving here is a bug or an unreviewed new capability, and
    should stop rather than proceed.
    """

    max_autonomous: PermissionLevel = PermissionLevel.READ
    max_with_confirmation: PermissionLevel = PermissionLevel.MUTATE

    def evaluate(self, level: PermissionLevel) -> Decision:
        if level <= self.max_autonomous:
            return Decision.ALLOW
        if level <= self.max_with_confirmation:
            return Decision.CONFIRM
        return Decision.DENY

    def require(self, level: PermissionLevel, action: str) -> None:
        """Raise unless `action` at `level` may proceed without confirmation.

        Used by call sites that have no way to ask the user: background tasks,
        and the model's own tool calls before a confirmation UI exists.
        """
        decision = self.evaluate(level)
        if decision is Decision.ALLOW:
            return
        reason = (
            "above the autonomous ceiling and needs explicit confirmation"
            if decision is Decision.CONFIRM
            else "denied by policy"
        )
        raise PermissionDeniedError(
            action
            + " requires permission level "
            + level.label
            + " ("
            + str(int(level))
            + "), which is "
            + reason
            + "."
        )


DEFAULT_POLICY = PermissionPolicy()
