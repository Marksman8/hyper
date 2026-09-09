"""The avatar seam.

Same discipline as `LLMProvider`, `OCRProvider` and `TTSProvider`: everything
above depends on this ABC, so the placeholder silhouette shipped in 10A can be
replaced by commissioned sprite art, a Live2D model, or a 3D character without
touching the overlay, the placement policy, or the state machine.

**A backend receives a `PresenceState` and a frame budget. It never receives an
event, a session, or a reference to AURA.** That is what keeps it a renderer
rather than a second place where behaviour accumulates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from aura.core.presence import PresenceState


@dataclass(frozen=True, slots=True)
class AvatarInfo:
    """Static description. Must not load anything."""

    name: str
    kind: str
    width: int
    height: int
    # False for a backend that draws itself (the 10A placeholder); True for one
    # that needs image files, a Cubism model, or a mesh.
    needs_assets: bool = False


class AvatarBackend(ABC):
    """Draws AURA.

    Implementations are told *what state she is in*, not what happened. The
    difference matters: a backend that knew about events would start making
    decisions, and those decisions belong in the state machine where they can
    be tested without a screen.
    """

    @abstractmethod
    def info(self) -> AvatarInfo:
        """Name, size and asset needs. Called before loading."""

    @abstractmethod
    def set_state(self, state: PresenceState, *, immediate: bool = False) -> None:
        """Change what she is doing.

        `immediate` skips any transition animation. Used for PRIVATE, where a
        graceful cross-fade would mean AURA stays visible for another 300 ms
        after being told to stop -- which is exactly the wrong behaviour.
        """

    @abstractmethod
    def advance(self, dt_ms: int) -> bool:
        """Step the animation. Returns True if a repaint is needed.

        Returning False when nothing moved is what makes the idle overlay cost
        no CPU: the overlay only repaints when a backend says something
        changed.
        """

    @abstractmethod
    def paint(self, painter: object, width: int, height: int) -> None:
        """Draw one frame into a QPainter.

        Typed loosely on purpose. This is the single point where a Qt object
        crosses the seam, and typing it as QPainter here would make the
        interface -- and everything that imports it -- depend on Qt.
        """

    def set_mouth(self, openness: float) -> None:
        """Lip sync, 0..1. Wired in 10B; a no-op until then."""

    def look_at(self, dx: float, dy: float) -> None:
        """Gaze offset, normalised -1..1. Eyes only, never the body."""

    def preload(self) -> None:
        """Load assets. Idempotent."""

    def unload(self) -> None:
        """Release them. Idempotent."""
