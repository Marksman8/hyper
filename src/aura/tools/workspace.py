"""Path safety for filesystem tools.

This is where filesystem tools actually go wrong, so the rules are explicit:

**Allow-list, not deny-list.** `workspace_roots` says what AURA may touch and
everything else is refused. A deny-list is a promise that you thought of every
dangerous path in advance, and nobody has.

**Resolve before checking.** `Path.resolve()` collapses `..` and follows
symlinks, so containment is tested against the real target. Comparing the
string as written is the classic traversal hole: `project/../../.ssh/id_rsa`
looks contained and is not. A symlink inside the workspace pointing out of it
is the same hole wearing a different hat.

**Deny patterns run after containment**, for files that are inside the
workspace and still must not be read. AURA's own API tokens live under
`var/secrets/`; a `read_file` tool that can hand those to a model is a
credential leak, and the model does not have to be malicious for it to happen.

**Case-insensitivity is not optional on Windows.** `C:\\Users\\x` and
`c:\\users\\x` are the same file, and a containment check that compares them as
strings says otherwise.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path, PurePath

from aura.logging import get_logger

log = get_logger(__name__)

# Refused even inside an allowed root. Credentials, keys, and AURA's own
# secrets, which are in the workspace precisely because it is the project
# directory.
DEFAULT_DENY_PATTERNS: tuple[str, ...] = (
    "**/.env",
    "**/.env.*",
    "**/var/secrets/**",
    "**/secrets/**",
    "**/*.pem",
    "**/*.key",
    "**/id_rsa*",
    "**/id_ed25519*",
    "**/.ssh/**",
    "**/.aws/**",
    "**/.git/config",
    "**/*.sqlite-wal",
)


class PathNotAllowedError(Exception):
    """The path is outside the workspace, or matches a deny pattern."""


@dataclass(frozen=True, slots=True)
class Workspace:
    """The set of paths AURA is permitted to touch."""

    roots: tuple[Path, ...]
    deny_patterns: tuple[str, ...] = DEFAULT_DENY_PATTERNS

    @classmethod
    def from_config(
        cls, roots: tuple[str, ...] | list[str], deny: tuple[str, ...] | list[str] = ()
    ) -> Workspace:
        resolved = tuple(Path(r).expanduser().resolve() for r in roots if r)
        return cls(
            roots=resolved,
            deny_patterns=tuple(deny) if deny else DEFAULT_DENY_PATTERNS,
        )

    # -- containment ---------------------------------------------------------

    @staticmethod
    def _normalise(path: Path) -> str:
        """Comparable form of a resolved path.

        `os.path.normcase` is the piece that makes this correct on Windows: it
        lowercases and unifies separators, so two spellings of one file compare
        equal. On POSIX it is the identity.
        """
        return os.path.normcase(str(path))

    def _is_inside(self, resolved: Path) -> bool:
        target = self._normalise(resolved)
        for root in self.roots:
            root_str = self._normalise(root)
            if target == root_str:
                return True
            # The separator matters: without it, /home/user-secrets passes a
            # containment check against /home/user.
            if target.startswith(root_str.rstrip(os.sep) + os.sep):
                return True
        return False

    def _is_denied(self, resolved: Path) -> str | None:
        """Return the pattern that refuses this path, if any."""
        as_posix = PurePath(self._normalise(resolved)).as_posix()
        for pattern in self.deny_patterns:
            normalised = os.path.normcase(pattern).replace("\\", "/")
            if fnmatch.fnmatch(as_posix, normalised):
                return pattern
            # fnmatch does not treat "**" as spanning separators, so a pattern
            # anchored at the start misses nested hits. Check the tail too.
            if normalised.startswith("**/") and fnmatch.fnmatch(as_posix, "*/" + normalised[3:]):
                return pattern
        return None

    # -- the one entry point -------------------------------------------------

    def resolve(self, path: str | Path, *, must_exist: bool = False) -> Path:
        """Resolve `path` and confirm AURA may touch it.

        Raises PathNotAllowedError rather than returning a flag, because every
        caller must handle refusal and an ignored boolean is how these checks
        get bypassed.
        """
        if not self.roots:
            raise PathNotAllowedError(
                "No workspace roots are configured, so no path may be accessed. "
                "Set tools.workspace_roots."
            )

        candidate = Path(path).expanduser()
        # strict=False so a not-yet-existing file (a write target) still gets
        # its parents resolved and checked.
        resolved = candidate.resolve(strict=False)

        if not self._is_inside(resolved):
            raise PathNotAllowedError(
                f"{resolved} is outside the workspace. Allowed roots: "
                + ", ".join(str(r) for r in self.roots)
            )

        denied_by = self._is_denied(resolved)
        if denied_by is not None:
            log.warning("tools.path_denied", path=str(resolved), pattern=denied_by)
            raise PathNotAllowedError(
                f"{resolved} matches the deny pattern {denied_by!r}. "
                "Credentials and secrets are never readable by tools."
            )

        if must_exist and not resolved.exists():
            raise PathNotAllowedError(f"{resolved} does not exist.")

        return resolved

    def describe(self) -> dict[str, list[str]]:
        return {
            "roots": [str(r) for r in self.roots],
            "deny_patterns": list(self.deny_patterns),
        }
