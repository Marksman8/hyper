"""The approval authority path.

**Deliberately not part of the event bus, and deliberately not importable by
it.** The bus is one-way: it tells subscribers that a permission is required.
It has no way to grant one. Granting travels back over authenticated HTTP and
lands here, and the separation is the point -- if the bus were bidirectional,
every subscriber would become a potential source of authority.

The flow:

    ToolExecutor needs confirmation
        → ApprovalRegistry.issue()  → (approval_id, nonce)
        → PERMISSION_REQUIRED event carries both
        → user clicks Approve in the presence overlay
        → POST /v1/approvals/{approval_id}  {nonce, request_id}   [authenticated]
        → ApprovalRegistry.redeem()  → Granted | rejection reason

The nonce binds an approval to the one request that asked for it. Without it, a
subscriber could approve something it was never asked about, or replay an old
approval against a new request -- which is exactly the "confused deputy" this
whole design exists to prevent.

Four rejections, each for a different attack:

* wrong nonce        -- forgery
* already redeemed   -- replay
* wrong request      -- an approval moved to a different action
* expired            -- an approval left lying around and used later
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from enum import StrEnum

from aura.logging import get_logger

log = get_logger(__name__)

# How long an approval stays redeemable. Long enough for a person to read a
# dialog and decide; short enough that an unattended prompt does not stay
# loaded indefinitely.
DEFAULT_TTL_S = 300.0

# 256 bits. These are single-use and short-lived, but they are the only thing
# standing between "was asked" and "was granted".
NONCE_BYTES = 32


class ApprovalOutcome(StrEnum):
    GRANTED = "granted"
    UNKNOWN = "unknown"          # no such approval id
    BAD_NONCE = "bad_nonce"      # forgery
    ALREADY_USED = "already_used"  # replay
    WRONG_REQUEST = "wrong_request"  # bound to a different action
    EXPIRED = "expired"
    DENIED = "denied"            # the human said no


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """What the user is being asked to approve.

    The payload here is what appears in the confirmation dialog, so it must be
    enough for a person to make a real decision -- "click Send" is not, "send
    'I'll be 20 minutes' to Rahul on WhatsApp" is.
    """

    id: str
    nonce: str
    request_id: str
    tool: str
    summary: str
    level: int
    issued_at: float
    expires_at: float

    def is_expired(self, now: float | None = None) -> bool:
        return (now or time.time()) > self.expires_at

    def to_event_payload(self) -> dict[str, object]:
        """What goes into the PERMISSION_REQUIRED event.

        Carries the nonce, because the overlay needs it to redeem. That is safe:
        the event travels over the same authenticated channel as the approval,
        so anyone who can read it could already authenticate.
        """
        return {
            "approval_id": self.id,
            "nonce": self.nonce,
            "request_id": self.request_id,
            "tool": self.tool,
            "summary": self.summary,
            "level": self.level,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True, slots=True)
class ApprovalResult:
    outcome: ApprovalOutcome
    request: ApprovalRequest | None = None

    @property
    def granted(self) -> bool:
        return self.outcome is ApprovalOutcome.GRANTED

    @property
    def reason(self) -> str:
        return {
            ApprovalOutcome.GRANTED: "approved",
            ApprovalOutcome.UNKNOWN: "No such approval. It may have expired.",
            ApprovalOutcome.BAD_NONCE: "The approval token did not match.",
            ApprovalOutcome.ALREADY_USED: "That approval has already been used.",
            ApprovalOutcome.WRONG_REQUEST: (
                "That approval was issued for a different action."
            ),
            ApprovalOutcome.EXPIRED: "That approval expired.",
            ApprovalOutcome.DENIED: "Denied.",
        }[self.outcome]


class ApprovalRegistry:
    """Issues and redeems single-use approvals.

    Thread-safe: approvals are issued from the tool executor and redeemed from
    an HTTP handler, potentially on different threads.
    """

    def __init__(self, ttl_s: float = DEFAULT_TTL_S) -> None:
        self._ttl = ttl_s
        self._lock = threading.Lock()
        self._pending: dict[str, ApprovalRequest] = {}
        # Redeemed ids are remembered so a replay is reported as ALREADY_USED
        # rather than UNKNOWN. The difference matters: one is an attack and the
        # other is a timeout, and they should not look the same in the audit log.
        self._used: dict[str, float] = {}

    def issue(
        self, *, request_id: str, tool: str, summary: str, level: int
    ) -> ApprovalRequest:
        now = time.time()
        approval = ApprovalRequest(
            id=secrets.token_urlsafe(12),
            nonce=secrets.token_urlsafe(NONCE_BYTES),
            request_id=request_id,
            tool=tool,
            summary=summary,
            level=level,
            issued_at=now,
            expires_at=now + self._ttl,
        )
        with self._lock:
            self._prune(now)
            self._pending[approval.id] = approval
        log.info(
            "approvals.issued",
            approval_id=approval.id,
            tool=tool,
            level=level,
            request_id=request_id,
        )
        return approval

    def redeem(
        self, approval_id: str, nonce: str, request_id: str, *, granted: bool = True
    ) -> ApprovalResult:
        """Consume an approval. Single use, whatever the outcome.

        An approval is burned even when denied: a user who said no must not
        have that answer replayed into a yes by a second attempt.
        """
        now = time.time()
        with self._lock:
            self._prune(now)

            if approval_id in self._used:
                outcome = ApprovalOutcome.ALREADY_USED
                pending = None
            else:
                pending = self._pending.get(approval_id)
                if pending is None:
                    outcome = ApprovalOutcome.UNKNOWN
                elif pending.is_expired(now):
                    self._pending.pop(approval_id, None)
                    outcome = ApprovalOutcome.EXPIRED
                # Constant-time: the nonce is a secret, and a timing side
                # channel on it would let it be recovered a byte at a time.
                elif not secrets.compare_digest(pending.nonce, nonce):
                    outcome = ApprovalOutcome.BAD_NONCE
                elif not secrets.compare_digest(pending.request_id, request_id):
                    outcome = ApprovalOutcome.WRONG_REQUEST
                elif not granted:
                    outcome = ApprovalOutcome.DENIED
                else:
                    outcome = ApprovalOutcome.GRANTED

            # Burned on any outcome that identified a real, live approval. A
            # bad nonce does NOT burn it -- otherwise a wrong guess would be a
            # denial-of-service against a legitimate approval.
            if outcome in (
                ApprovalOutcome.GRANTED,
                ApprovalOutcome.DENIED,
                ApprovalOutcome.WRONG_REQUEST,
            ):
                self._pending.pop(approval_id, None)
                self._used[approval_id] = now

        log.info(
            "approvals.redeemed",
            approval_id=approval_id,
            outcome=str(outcome),
            request_id=request_id,
        )
        return ApprovalResult(outcome=outcome, request=pending)

    def _prune(self, now: float) -> None:
        """Drop expired pending approvals and old used-ids. Called under lock."""
        expired = [k for k, v in self._pending.items() if v.is_expired(now)]
        for key in expired:
            del self._pending[key]
        cutoff = now - (self._ttl * 2)
        stale = [k for k, ts in self._used.items() if ts < cutoff]
        for key in stale:
            del self._used[key]

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def get(self, approval_id: str) -> ApprovalRequest | None:
        with self._lock:
            return self._pending.get(approval_id)
