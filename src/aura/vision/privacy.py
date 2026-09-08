"""The vision privacy gate.

Spec §10: when the user says "don't look", the pipeline stops. Not "the model
is told to ignore the screen" — stops. So this module is consulted by the code
that would *take* the screenshot, before it takes one, and a denial means no
pixels are read at all.

Everything else in `aura.vision` depends on that ordering. A design that
captured first and filtered afterwards would have already lost: the frame
existed, something held it, and the promise would only be as strong as every
consumer downstream.

Default is PRIVATE. Reading a file is bounded by the workspace guard; reading
the screen is bounded by nothing — whatever happens to be in front of the user,
which may be a password manager or someone else's messages. That is not a thing
to have on by default.
"""

from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass, field
from enum import StrEnum

from aura.logging import get_logger

log = get_logger(__name__)


class VisionMode(StrEnum):
    """How much AURA may see (spec §10)."""

    # No capture at all. The default.
    PRIVATE = "private"
    # Capture only when the user explicitly asks.
    ASSIST = "assist"
    # Capture on request and, once the Phase 5.1 loop exists, on events.
    ACTIVE = "active"
    # Everything paused, vision included.
    SLEEP = "sleep"


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether a capture may happen, and why not when it may not."""

    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


# Windows whose titles suggest credentials or private material. Denied
# regardless of mode: a user who turns vision on for their editor has not
# thereby consented to their password manager being read.
DEFAULT_DENY_TITLES: tuple[str, ...] = (
    "*bitwarden*",
    "*1password*",
    "*lastpass*",
    "*keepass*",
    "*credential*",
    "*sign in*",
    "*log in*",
    "*login*",
    "*password*",
    "*authenticator*",
    "*banking*",
    "*private browsing*",
    "*incognito*",
)


@dataclass
class PrivacyPolicy:
    """What AURA may look at, right now.

    Mutable and runtime-changeable, because "don't look at Chrome" has to take
    effect when it is said, not at the next restart.
    """

    mode: VisionMode = VisionMode.PRIVATE
    # Applications the user has explicitly excluded, matched against the
    # foreground window title and process name.
    denied_apps: set[str] = field(default_factory=set)
    # When non-empty, only these are allowed — an explicit "watch VS Code and
    # nothing else". Empty means "anything not denied".
    allowed_apps: set[str] = field(default_factory=set)
    deny_titles: tuple[str, ...] = DEFAULT_DENY_TITLES
    # Monitors the user has excluded entirely, by index.
    denied_monitors: set[int] = field(default_factory=set)
    changed_at: float = field(default_factory=time.time)

    # -- the one question that matters ---------------------------------------

    def may_capture(
        self,
        *,
        monitor: int | None = None,
        window_title: str | None = None,
        process_name: str | None = None,
        requested_by_user: bool = True,
    ) -> Decision:
        """May a screenshot be taken right now?

        Called before capture, never after. `requested_by_user` distinguishes
        "the user asked what is on screen" from a background poll, because
        ASSIST permits the first and not the second.
        """
        if self.mode is VisionMode.PRIVATE:
            return Decision(
                False,
                "Vision is off (private mode). Say 'resume vision' or set "
                "vision.mode to assist or active to allow it.",
            )
        if self.mode is VisionMode.SLEEP:
            return Decision(False, "AURA is asleep; vision is paused.")
        if self.mode is VisionMode.ASSIST and not requested_by_user:
            return Decision(
                False, "Assist mode captures only when you ask, not in the background."
            )

        if monitor is not None and monitor in self.denied_monitors:
            return Decision(False, f"Monitor {monitor} is excluded from vision.")

        # Title and process checks are last, because they are the ones that can
        # be answered wrongly: a window can change between the check and the
        # capture. That race is why the foreground window is re-read
        # immediately before the grab rather than trusted from a cache.
        haystacks = [h.lower() for h in (window_title, process_name) if h]

        for pattern in self.deny_titles:
            for haystack in haystacks:
                if fnmatch.fnmatch(haystack, pattern.lower()):
                    log.info("vision.denied_by_title", pattern=pattern)
                    return Decision(
                        False,
                        "That window looks like it contains credentials or "
                        "private information, so it is never captured.",
                    )

        for denied in self.denied_apps:
            for haystack in haystacks:
                if denied.lower() in haystack:
                    return Decision(False, f"You asked AURA not to look at {denied}.")

        if self.allowed_apps:
            if not any(
                allowed.lower() in haystack
                for allowed in self.allowed_apps
                for haystack in haystacks
            ):
                allowed_list = ", ".join(sorted(self.allowed_apps))
                return Decision(
                    False,
                    f"Vision is restricted to {allowed_list}, and that is not "
                    "what is in front of you.",
                )

        return Decision(True)

    # -- changing it ---------------------------------------------------------

    def set_mode(self, mode: VisionMode) -> None:
        previous, self.mode = self.mode, mode
        self.changed_at = time.time()
        log.info("vision.mode_changed", was=str(previous), now=str(mode))

    def deny_app(self, name: str) -> None:
        """"Don't look at Chrome."""
        cleaned = name.strip()
        if not cleaned:
            return
        self.denied_apps.add(cleaned)
        self.allowed_apps.discard(cleaned)
        self.changed_at = time.time()
        log.info("vision.app_denied", app=cleaned)

    def allow_only(self, name: str) -> None:
        """"Watch VS Code" — an allow-list of one, added to on repeat."""
        cleaned = name.strip()
        if not cleaned:
            return
        self.allowed_apps.add(cleaned)
        self.denied_apps.discard(cleaned)
        self.changed_at = time.time()
        log.info("vision.app_allowed", app=cleaned)

    def clear_restrictions(self) -> None:
        self.denied_apps.clear()
        self.allowed_apps.clear()
        self.denied_monitors.clear()
        self.changed_at = time.time()
        log.info("vision.restrictions_cleared")

    def describe(self) -> dict[str, object]:
        """What is in force, for the UI and for /v1/vision/status."""
        return {
            "mode": str(self.mode),
            "denied_apps": sorted(self.denied_apps),
            "allowed_apps": sorted(self.allowed_apps),
            "denied_monitors": sorted(self.denied_monitors),
            "deny_title_patterns": len(self.deny_titles),
        }
