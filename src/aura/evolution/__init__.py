"""Phase 9: controlled evolution.

propose -> sandbox -> benchmark -> compare -> HUMAN APPROVAL -> deploy

Every arrow before the gate is automatic. The gate is not, and there is no
flag that makes it so (spec section 18).
"""

from aura.evolution.candidate import Candidate, CandidateStore, propose
from aura.evolution.deploy import (
    ApprovalRequiredError,
    NotDeployableError,
    apply_candidate,
    list_snapshots,
    rollback,
)
from aura.evolution.search import HillClimber, best_of
from aura.evolution.space import (
    FORBIDDEN,
    KNOBS,
    ForbiddenKnobError,
    Knob,
    check_allowed,
    validate_overrides,
)

__all__ = [
    "FORBIDDEN",
    "KNOBS",
    "ApprovalRequiredError",
    "Candidate",
    "CandidateStore",
    "ForbiddenKnobError",
    "HillClimber",
    "Knob",
    "NotDeployableError",
    "apply_candidate",
    "best_of",
    "check_allowed",
    "list_snapshots",
    "propose",
    "rollback",
    "validate_overrides",
]
