"""The emergency stop.

**A kill switch that stops the model thinking but not the tools acting is not a
kill switch.** Before this module, that is what AURA had: `Orchestrator._guard`
refused model calls, and `POST /v1/tools/read_file` executed happily with the
stop engaged, because `ToolExecutor` had never heard of it. The audit
demonstrated it live.

Three properties, and each is here because the alternative failed:

**It is file-backed.** The previous stop was a boolean on the orchestrator,
initialised from config at construction, so engaging it and restarting AURA
released it. The situation where a kill switch matters most — something is
misbehaving — is exactly the situation where a process is most likely to
restart.

**It works without AURA.** The file is the switch. Create
`var/secrets/emergency_stop` in a file manager and AURA stops, whether or not
she is responding, whether or not her API is reachable, whether or not the
event loop is wedged. Delete it and she resumes. A control that requires the
malfunctioning system to cooperate is not much of a control.

**AURA cannot reach it.** It lives under `var/secrets/`, which
`tools/workspace.py` already denies to every file tool, and this module adds an
explicit pattern for it so the protection survives someone overriding the deny
list. Nothing in `aura.tools` imports this module; a test enforces that.

The state is read from disk on every check rather than cached. A stop that
takes effect on the next cache expiry is a stop that does not take effect.
"""

from __future__ import annotations

import time
from pathlib import Path

from aura.errors import EmergencyStopError
from aura.logging import get_logger

log = get_logger(__name__)

# The switch sits beside the API token, in whatever directory that is --
# already denied to every file tool, and the right neighbourhood for "things
# AURA must not touch". See `build_estop` for why it is not a fixed path.
FILENAME = "emergency_stop"

# Added to the workspace deny list so the switch is unreachable even if someone
# replaces the built-in patterns.
DENY_PATTERN = "**/var/secrets/emergency_stop"


class EmergencyStop:
    """Engaged or not, backed by the presence of a file."""

    def __init__(self, path: Path, *, engaged_by_config: bool = False) -> None:
        self._path = path
        # `app.emergency_stop: true` in config is a second way to engage, for a
        # machine that should boot stopped. It cannot be released at runtime,
        # by design -- someone who set it in a file meant it.
        self._by_config = engaged_by_config

    @property
    def path(self) -> Path:
        return self._path

    @property
    def engaged(self) -> bool:
        """Stat the file every time. Never cached.

        A tool call costs milliseconds at least; a stat costs microseconds, and
        the alternative is a window in which the stop is engaged and AURA is
        still acting.
        """
        if self._by_config:
            return True
        try:
            return self._path.exists()
        except OSError as exc:
            # Cannot tell. **Fail closed**: an unreadable kill switch is
            # treated as engaged, because the failure mode of guessing wrong in
            # the other direction is that a stopped AURA keeps acting.
            log.error("estop.unreadable", path=str(self._path), error=str(exc))
            return True

    @property
    def reason(self) -> str:
        if self._by_config:
            return "app.emergency_stop is true in configuration"
        return f"{self._path} exists"

    def engage(self, actor: str = "user") -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            f"Engaged by {actor} at {time.strftime('%Y-%m-%dT%H:%M:%S')}.\n"
            "Delete this file to release the emergency stop.\n",
            encoding="utf-8",
        )
        log.warning("estop.engaged", actor=actor, path=str(self._path))

    def release(self, actor: str = "user") -> None:
        if self._by_config:
            raise EmergencyStopError(
                "The stop is engaged by configuration (app.emergency_stop). "
                "Change it in config and restart; it cannot be released at "
                "runtime."
            )
        self._path.unlink(missing_ok=True)
        log.warning("estop.released", actor=actor, path=str(self._path))

    def require_clear(self, action: str = "This action") -> None:
        """Raise if engaged. For call sites that cannot return a value."""
        if self.engaged:
            raise EmergencyStopError(
                f"Emergency stop is engaged ({self.reason}). {action} is blocked. "
                'Release it with POST /v1/control/estop {"engaged": false}, or '
                f"delete {self._path}."
            )


def build_estop(settings: object) -> EmergencyStop:
    """Construct from settings.

    **The switch lives wherever the API token lives**, rather than at a fixed
    project path. Two reasons, and the second was found the hard way.

    That directory is already the one the workspace guard denies, so the switch
    inherits the protection instead of asking for its own.

    And it means a test that redirects `security.token_file` to a temporary
    directory redirects the switch with it. The first version resolved against
    `PROJECT_ROOT`, so a test that engaged the stop and did not release it left
    the file in the working tree -- and every later test, and the developer's
    next run, found AURA stopped. A kill switch with global side effects is a
    kill switch nobody will keep enabled in tests.
    """
    from aura.config import PROJECT_ROOT

    app = getattr(settings, "app", None)
    engaged = bool(getattr(app, "emergency_stop", False))

    security = getattr(settings, "security", None)
    resolver = getattr(security, "resolved_token_file", None)
    directory = resolver().parent if callable(resolver) else PROJECT_ROOT / "var" / "secrets"

    return EmergencyStop(directory / FILENAME, engaged_by_config=engaged)
