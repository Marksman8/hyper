"""Local API authentication.

"It only listens on loopback" is not authentication. Every process running as
this user -- including a browser tab via a malicious page, or any dependency in
any project on this machine -- can reach 127.0.0.1:8760. From Phase 3 that
endpoint executes code and reads files. So: a bearer token, from day one, with
no way to disable it.

The token is generated once into var/secrets/api_token and read from there by
both the server and the CLI client, so the common case needs no configuration.
"""

from __future__ import annotations

import os
import secrets
import stat
import subprocess
import sys
from pathlib import Path

from aura.errors import AuthError, ConfigError
from aura.logging import get_logger

log = get_logger(__name__)

_TOKEN_BYTES = 32


def generate_token() -> str:
    return secrets.token_urlsafe(_TOKEN_BYTES)


def load_or_create_token(token_file: Path, configured: str = "") -> str:
    """Resolve the API token.

    Precedence: explicit config/env value, then the token file, then a freshly
    generated token written to the file.
    """
    if configured:
        return configured

    if token_file.is_file():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            _warn_if_world_readable(token_file)
            return token

    token = generate_token()
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(token, encoding="utf-8")
    _restrict_permissions(token_file)
    log.info("auth.token_created", path=str(token_file))
    return token


def _restrict_permissions(path: Path) -> None:
    """Restrict the token file to the current user.

    POSIX mode bits do not exist on Windows -- os.chmod there only toggles the
    read-only flag, and st_mode always reports 0o666 regardless of the real
    ACL. So on Windows the actual restriction has to go through icacls:
    break inheritance, then grant full control to this user alone.
    """
    if sys.platform == "win32":
        user = os.environ.get("USERNAME")
        if not user:
            log.warning("auth.acl_skipped", reason="USERNAME not set")
            return
        # Absolute path from SystemRoot rather than PATH lookup: a token
        # file's ACL should not depend on what happens to be first in PATH.
        icacls = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "icacls.exe"
        if not icacls.is_file():
            log.warning("auth.acl_skipped", reason="icacls.exe not found")
            return
        try:
            subprocess.run(  # noqa: S603 - absolute argv[0], no shell
                [
                    str(icacls),
                    str(path),
                    "/inheritance:r",
                    "/grant:r",
                    user + ":F",
                ],
                capture_output=True,
                check=True,
                timeout=10,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            log.warning("auth.acl_failed", path=str(path), error=str(exc))
        return

    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as exc:  # pragma: no cover - platform dependent
        log.warning("auth.chmod_failed", path=str(path), error=str(exc))


def _warn_if_world_readable(path: Path) -> None:
    # Meaningless on Windows: st_mode is synthesized and always reports 0o666,
    # so checking it there produces a warning on every single read.
    if sys.platform == "win32":
        return
    try:
        mode = path.stat().st_mode
    except OSError:  # pragma: no cover
        return
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        log.warning("auth.token_file_permissive", path=str(path), mode=oct(mode & 0o777))


def verify_token(presented: str | None, expected: str) -> None:
    """Constant-time bearer check.

    Raises AuthError on any failure. Timing-safe comparison is not paranoia
    theatre here: the endpoint is reachable by local processes that can time it
    precisely.
    """
    if not expected:
        raise ConfigError("No API token configured; refusing to serve unauthenticated.")
    if not presented:
        raise AuthError("Missing Authorization header.")

    scheme, _, value = presented.partition(" ")
    if scheme.lower() != "bearer" or not value:
        raise AuthError("Authorization header must be 'Bearer <token>'.")
    if not secrets.compare_digest(value, expected):
        raise AuthError("Invalid API token.")
