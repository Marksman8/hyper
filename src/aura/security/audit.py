"""Audit trail (spec sections 23 and 25).

Thin policy layer over MemoryStore's append-only `audit` table. It exists as a
separate object so that the fail-closed rule lives in one place:

    If `require_audit` is set and the audit write fails, the operation that was
    about to be audited must not happen.

That rule is cheap now and load-bearing later. An assistant that can delete
files but silently loses its record of doing so is worse than one that refuses.
"""

from __future__ import annotations

from typing import Any

from aura.errors import AuraError
from aura.logging import get_logger
from aura.memory.store import MemoryStore
from aura.security.permissions import Decision, PermissionLevel

log = get_logger(__name__)


class AuditLogError(AuraError):
    """The audit trail could not be written and audit is mandatory."""

    http_status = 500
    code = "audit_unavailable"


class AuditLog:
    def __init__(self, store: MemoryStore, enabled: bool = True, required: bool = True) -> None:
        self._store = store
        self._enabled = enabled
        self._required = required

    async def record(
        self,
        actor: str,
        action: str,
        *,
        level: PermissionLevel = PermissionLevel.OBSERVE,
        decision: Decision = Decision.ALLOW,
        target: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        if not self._enabled:
            return
        try:
            await self._store.append_audit(
                actor=actor,
                action=action,
                permission_level=int(level),
                decision=decision.value,
                target=target,
                request_id=request_id,
                session_id=session_id,
                detail=detail,
            )
        except Exception as exc:
            log.error("audit.write_failed", action=action, error=str(exc))
            if self._required:
                raise AuditLogError(
                    "Audit write failed and security.require_audit is true; "
                    "refusing to proceed with " + action + "."
                ) from exc

    async def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        return await self._store.recent_audit(limit)
