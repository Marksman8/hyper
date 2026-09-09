"""Where AURA sits, and — mostly — why she does not move.

The design is explicit that continuous movement is what makes a desktop
companion intolerable, so this module's job is more about **refusing** to move
than about moving.

**The four-trigger rule.** She moves only when:

1. the foreground application changed,
2. the user dragged her,
3. a state transition genuinely needs to be seen (`WARNING`, `ERROR`,
   `WAITING`), or
4. her current anchor became unusable — a display was unplugged, or the
   resolution changed.

Anything else — a new task, a reply, a mode change, the cursor moving — is not
a reason to move. Cursor tracking is **gaze only**: the eyes follow, the body
does not, and even that is off by default.

A rate limit sits on top: at most one move every 3 seconds, however many
triggers fire. Two applications fighting over focus must not turn the avatar
into a metronome.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from aura.core.presence import PresenceState
from aura.logging import get_logger

log = get_logger(__name__)

# However many triggers fire, at most one move per this interval.
MIN_MOVE_INTERVAL_S = 3.0

# How far from the screen edge she sits.
MARGIN = 24

# States urgent enough to justify moving somewhere visible.
NEEDS_ATTENTION = frozenset(
    {PresenceState.WARNING, PresenceState.ERROR, PresenceState.WAITING}
)


class Anchor(StrEnum):
    """The small set of places she is allowed to be.

    Slots rather than free positioning: an avatar that can be anywhere ends up
    somewhere unhelpful, and a bounded set is something a person can predict.
    """

    BOTTOM_RIGHT = "bottom_right"
    BOTTOM_LEFT = "bottom_left"
    TOP_RIGHT = "top_right"
    PINNED = "pinned"


class MoveReason(StrEnum):
    """Why a move was allowed. Logged, so "why did she move?" is answerable."""

    APP_CHANGED = "app_changed"
    USER_DRAG = "user_drag"
    NEEDS_ATTENTION = "needs_attention"
    ANCHOR_INVALID = "anchor_invalid"


@dataclass(frozen=True, slots=True)
class Screen:
    """The usable area of one display, in logical pixels."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class Placement:
    x: int
    y: int
    reason: MoveReason


class PlacementPolicy:
    """Decides whether AURA may move, and where to.

    Holds no Qt objects: it is given a `Screen` and returns coordinates, so it
    can be tested without a display.
    """

    def __init__(
        self,
        anchor: Anchor = Anchor.BOTTOM_RIGHT,
        *,
        min_interval_s: float = MIN_MOVE_INTERVAL_S,
    ) -> None:
        self._anchor = anchor
        self._min_interval = min_interval_s
        self._last_move_at = 0.0
        self._pinned: tuple[int, int] | None = None
        self._moves = 0
        self._suppressed = 0

    @property
    def anchor(self) -> Anchor:
        return self._anchor

    @property
    def move_count(self) -> int:
        return self._moves

    @property
    def suppressed_count(self) -> int:
        """Moves the rate limit refused. Useful for judging whether the limit
        is doing anything, or merely present."""
        return self._suppressed

    def pin(self, x: int, y: int, now: float) -> Placement:
        """The user dragged her. That always wins.

        Not rate-limited: refusing to follow a drag would feel broken, and a
        drag is by definition something the user is doing on purpose.
        """
        self._pinned = (x, y)
        self._anchor = Anchor.PINNED
        self._last_move_at = now
        self._moves += 1
        log.info("presence.pinned", x=x, y=y)
        return Placement(x=x, y=y, reason=MoveReason.USER_DRAG)

    def resolve(
        self, screen: Screen, size: tuple[int, int], *, anchor: Anchor | None = None
    ) -> tuple[int, int]:
        """Coordinates for an anchor on a given screen."""
        width, height = size
        target = anchor or self._anchor

        if target is Anchor.PINNED and self._pinned is not None:
            x, y = self._pinned
            # Clamped, so a pin from a monitor that is no longer attached does
            # not leave her off-screen and unreachable.
            x = max(screen.x, min(x, screen.x + screen.width - width))
            y = max(screen.y, min(y, screen.y + screen.height - height))
            return x, y

        right = screen.x + screen.width - width - MARGIN
        left = screen.x + MARGIN
        bottom = screen.y + screen.height - height - MARGIN
        top = screen.y + MARGIN

        return {
            Anchor.BOTTOM_RIGHT: (right, bottom),
            Anchor.BOTTOM_LEFT: (left, bottom),
            Anchor.TOP_RIGHT: (right, top),
            Anchor.PINNED: (right, bottom),
        }[target]

    def consider(
        self,
        reason: MoveReason,
        screen: Screen,
        size: tuple[int, int],
        now: float,
        *,
        state: PresenceState | None = None,
    ) -> Placement | None:
        """May she move, for this reason, right now?

        Returns None for "stay where you are", which is the answer most of the
        time and is the point of the module.
        """
        if reason is MoveReason.NEEDS_ATTENTION and (
            state is None or state not in NEEDS_ATTENTION
        ):
            # A state change is only a reason to move if the state is one that
            # actually needs looking at. SUCCESS is not.
            return None

        # An invalid anchor bypasses the rate limit: staying put is not an
        # option when "put" is a monitor that no longer exists.
        if reason is not MoveReason.ANCHOR_INVALID:
            since = now - self._last_move_at
            if since < self._min_interval:
                self._suppressed += 1
                return None

        x, y = self.resolve(screen, size)
        self._last_move_at = now
        self._moves += 1
        log.info("presence.moved", reason=str(reason), x=x, y=y)
        return Placement(x=x, y=y, reason=reason)

    def anchor_is_valid(self, screen: Screen, size: tuple[int, int], at: tuple[int, int]) -> bool:
        """Is the current position still on a usable part of a display?"""
        width, height = size
        return (
            screen.x <= at[0] <= screen.x + screen.width - width
            and screen.y <= at[1] <= screen.y + screen.height - height
        )
