"""A drawn placeholder avatar.

Deliberately **not** artwork. It is a professional silhouette — head, shoulders,
a state-coloured accent ring — drawn with vector primitives so that 10A can be
judged on *behaviour*: does she change state correctly, move only when she
should, disappear on privacy, cost nothing when idle. Those are the questions
this milestone exists to answer, and none of them need a face.

Drawing rather than shipping PNGs is a deliberate choice for the placeholder:
there are no assets to version, license, or half-commit to, and swapping in the
real AURA character later is a new `AvatarBackend`, not an edit to this file.

**State is expressed by colour and one animated element**, never by a different
pose. A placeholder that implied a character design would quietly become the
design.
"""

from __future__ import annotations

import math
from typing import Any

from aura.core.presence import PresenceState
from aura.presence.avatar.base import AvatarBackend, AvatarInfo

# Accent colour per state, as (r, g, b). Chosen for meaning rather than variety:
# calm blues for ordinary activity, amber when a person is needed, red for
# failure, and near-invisible grey for the states that should not draw the eye.
PALETTE: dict[PresenceState, tuple[int, int, int]] = {
    PresenceState.IDLE: (110, 125, 150),
    PresenceState.LISTENING: (90, 170, 230),
    PresenceState.THINKING: (110, 140, 220),
    PresenceState.WORKING: (95, 175, 195),
    PresenceState.WAITING: (230, 175, 70),
    PresenceState.SUCCESS: (105, 190, 130),
    PresenceState.HAPPY: (120, 205, 155),
    PresenceState.CONFUSED: (200, 165, 105),
    PresenceState.WARNING: (230, 150, 60),
    PresenceState.ANNOYED: (190, 140, 120),
    PresenceState.ERROR: (215, 95, 95),
    PresenceState.SLEEP: (80, 88, 105),
    PresenceState.PRIVATE: (70, 76, 90),
}

# How fast the accent ring moves, per state. 0.0 means completely still --
# which is most of them, because a desktop companion that is always animating
# is a desktop companion that gets closed.
PULSE: dict[PresenceState, float] = {
    PresenceState.THINKING: 1.6,
    PresenceState.WORKING: 2.4,
    PresenceState.LISTENING: 1.1,
    PresenceState.WAITING: 0.9,
    PresenceState.ERROR: 2.0,
}

# How long a colour cross-fade takes.
FADE_MS = 260.0


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


class SilhouetteAvatar(AvatarBackend):
    """The 10A placeholder."""

    WIDTH = 132
    HEIGHT = 176

    def __init__(self) -> None:
        self._state = PresenceState.IDLE
        self._from = PALETTE[PresenceState.IDLE]
        self._to = PALETTE[PresenceState.IDLE]
        self._fade = 1.0
        self._phase = 0.0
        self._mouth = 0.0
        self._gaze = (0.0, 0.0)
        self._dirty = True

    def info(self) -> AvatarInfo:
        return AvatarInfo(
            name="silhouette",
            kind="vector-placeholder",
            width=self.WIDTH,
            height=self.HEIGHT,
            needs_assets=False,
        )

    # -- state ---------------------------------------------------------------

    def set_state(self, state: PresenceState, *, immediate: bool = False) -> None:
        if state is self._state:
            return
        self._from = self._current_colour()
        self._to = PALETTE.get(state, PALETTE[PresenceState.IDLE])
        # PRIVATE must not cross-fade: a graceful 260 ms transition means AURA
        # is still visible for a quarter-second after being told to stop.
        self._fade = 1.0 if immediate else 0.0
        self._state = state
        self._dirty = True

    def _current_colour(self) -> tuple[int, int, int]:
        return _mix(self._from, self._to, self._fade)

    # -- animation -----------------------------------------------------------

    def advance(self, dt_ms: int) -> bool:
        """Returns True only if something actually changed.

        This is what makes an idle overlay free: with no fade in progress and
        no pulse for the current state, nothing is dirty and the window is
        never repainted.
        """
        changed = self._dirty
        self._dirty = False

        if self._fade < 1.0:
            self._fade = min(1.0, self._fade + dt_ms / FADE_MS)
            changed = True

        speed = PULSE.get(self._state, 0.0)
        if speed > 0.0:
            self._phase = (self._phase + speed * dt_ms / 1000.0) % (2 * math.pi)
            changed = True

        return changed

    def set_mouth(self, openness: float) -> None:
        openness = max(0.0, min(1.0, openness))
        if abs(openness - self._mouth) > 0.02:
            self._mouth = openness
            self._dirty = True

    def look_at(self, dx: float, dy: float) -> None:
        """Eyes only. The body never turns -- continuous body-following is the
        fastest way to make a desktop companion intolerable."""
        target = (max(-1.0, min(1.0, dx)), max(-1.0, min(1.0, dy)))
        if target != self._gaze:
            self._gaze = target
            self._dirty = True

    # -- drawing -------------------------------------------------------------

    def paint(self, painter: Any, width: int, height: int) -> None:
        from PySide6.QtCore import QPointF, QRectF, Qt
        from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r, g, b = self._current_colour()
        accent = QColor(r, g, b)

        cx = width / 2.0
        head_r = width * 0.19
        head_cy = height * 0.26

        # Accent ring: the one animated element. Its radius breathes when the
        # state has a pulse, and sits still otherwise.
        breath = math.sin(self._phase) * 0.5 + 0.5
        ring_r = head_r * (1.55 + 0.12 * breath if self._state in PULSE else 1.55)
        ring = QColor(accent)
        ring.setAlpha(int(60 + 50 * breath) if self._state in PULSE else 70)
        pen = QPen(ring)
        pen.setWidthF(2.2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, head_cy), ring_r, ring_r)

        # Shoulders and head, as one filled silhouette.
        body = QPainterPath()
        body.addEllipse(QPointF(cx, head_cy), head_r, head_r * 1.06)
        shoulders = QRectF(
            cx - width * 0.31, height * 0.47, width * 0.62, height * 0.60
        )
        body.addRoundedRect(shoulders, width * 0.20, width * 0.20)

        fill = QColor(accent)
        fill.setAlpha(210)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawPath(body)

        # A collar line, so the silhouette reads as "professional" rather than
        # as a generic blob.
        collar = QColor(255, 255, 255, 40)
        painter.setPen(QPen(collar, 1.6))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(
            QPointF(cx - width * 0.10, height * 0.52),
            QPointF(cx, height * 0.60),
        )
        painter.drawLine(
            QPointF(cx + width * 0.10, height * 0.52),
            QPointF(cx, height * 0.60),
        )

        # Eyes: two marks that track the cursor when gaze is enabled. Kept
        # minimal deliberately -- anything more starts being a face, and a face
        # is a design decision this placeholder must not make.
        gx, gy = self._gaze
        eye_dx = head_r * 0.38
        eye_y = head_cy - head_r * 0.06 + gy * head_r * 0.12
        eye = QColor(250, 250, 252, 200)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(eye)
        for side in (-1, 1):
            painter.drawEllipse(
                QPointF(cx + side * eye_dx + gx * head_r * 0.16, eye_y),
                head_r * 0.085,
                head_r * 0.105,
            )

        # Mouth: a flat line at rest, opening with amplitude in 10B.
        if self._mouth > 0.02:
            mouth_h = head_r * 0.42 * self._mouth
            painter.setBrush(QColor(250, 250, 252, 170))
            painter.drawEllipse(
                QPointF(cx, head_cy + head_r * 0.42),
                head_r * 0.20,
                max(1.0, mouth_h),
            )
