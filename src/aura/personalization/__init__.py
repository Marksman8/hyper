"""Phase 7: acting on what Phase 6 learned.

Knowing a preference and behaving differently because of it are separate
things, and only the second is worth anything to the user.
"""

from aura.personalization.profile import (
    UserProfile,
    apply_parameters,
    build_profile,
    directives,
)

__all__ = ["UserProfile", "apply_parameters", "build_profile", "directives"]
