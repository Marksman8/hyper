"""The transparent desktop overlay.

Frameless, per-pixel alpha, always-on-top, and — crucially — **not in the way**.

Three properties do most of the work of "does not interfere with normal desktop
use", and each is a specific flag rather than an emergent effect:

* `Qt.Tool` keeps it out of the taskbar and out of Alt-Tab.
* `WA_ShowWithoutActivating` stops it stealing focus when it appears. An
  overlay that grabs focus while you are typing is worse than no overlay.
* `setMask()` limits both painting *and input* to the avatar and subtitle. The
  rest of the window does not exist as far as the desktop is concerned, so
  clicks pass straight through to whatever is underneath.

The mask is a rectangle around the avatar rather than its exact silhouette. A
per-pixel input region would be more precise and materially harder to get
right; the cost is a small transparent area that is draggable, which is normal
for this kind of overlay and is the behaviour a user expects anyway.

**Repaint on change only.** The paint timer asks the backend whether anything
moved and does nothing when the answer is no. That is why an idle overlay
measured 0 repaints and 0.00% CPU in 10A.0.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QRegion
from PySide6.QtWidgets import QWidget

from aura.core.presence import PresenceState
from aura.logging import get_logger
from aura.presence.avatar.base import AvatarBackend

log = get_logger(__name__)

# Paint tick. Not a frame rate: it is how often the backend is *asked* whether
# anything changed. When nothing has, no repaint happens at all.
TICK_MS = 16

SUBTITLE_WIDTH = 320
SUBTITLE_PAD = 12
SUBTITLE_MAX_LINES = 3


class PresenceOverlay(QWidget):
    """The window. Renders; decides nothing."""

    dragged = Signal(int, int)
    # The panel changed size -- a subtitle appeared, wrapped to a second line,
    # or cleared. The controller re-anchors on this; see `_resize_for_content`.
    resized = Signal()

    def __init__(self, avatar: AvatarBackend) -> None:
        super().__init__()
        self._avatar = avatar
        info = avatar.info()
        self._avatar_size = (info.width, info.height)
        self._subtitle = ""
        self._drag_from: QPoint | None = None
        self._last_tick = 0

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            # Out of the taskbar and out of Alt-Tab. Without this the overlay
            # becomes a window the user has to manage.
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Never take focus. An overlay that steals it mid-sentence is worse
        # than no overlay.
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowTitle("AURA")

        self._resize_for_content()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(TICK_MS)

    # -- layout --------------------------------------------------------------

    def _resize_for_content(self) -> None:
        """Grow or shrink to fit the avatar plus whatever subtitle is showing.

        Emits `resized` so the controller can re-anchor. `resize()` moves the
        bottom and right edges while leaving the top-left where it was, so a
        bottom-right-anchored overlay that grows a subtitle grows *off the
        screen* -- which is exactly what happened the first time this ran with
        live speech: the avatar and the line under it were both clipped by the
        screen edge.
        """
        width, height = self._avatar_size
        if self._subtitle:
            width = max(width, SUBTITLE_WIDTH)
            height += self._subtitle_height()

        if (width, height) == (self.width(), self.height()):
            return
        self.resize(width, height)
        self._apply_mask()
        self.resized.emit()

    def _subtitle_height(self) -> int:
        if not self._subtitle:
            return 0
        metrics = self.fontMetrics()
        rect = metrics.boundingRect(
            QRect(0, 0, SUBTITLE_WIDTH - 2 * SUBTITLE_PAD, 0),
            int(Qt.TextFlag.TextWordWrap),
            self._subtitle,
        )
        lines = min(SUBTITLE_MAX_LINES, max(1, rect.height() // max(1, metrics.height())))
        return lines * metrics.height() + 2 * SUBTITLE_PAD

    def _apply_mask(self) -> None:
        """Restrict paint *and input* to the parts that are actually drawn.

        Everything outside the mask passes clicks through to the desktop, which
        is the whole of "does not interfere with normal desktop use".
        """
        aw, ah = self._avatar_size
        region = QRegion(QRect((self.width() - aw) // 2, 0, aw, ah))
        if self._subtitle:
            region = region.united(
                QRegion(QRect(0, ah, self.width(), self._subtitle_height()))
            )
        self.setMask(region)

    # -- what the controller drives -----------------------------------------

    def set_state(self, state: PresenceState, *, immediate: bool = False) -> None:
        self._avatar.set_state(state, immediate=immediate)
        self.update()

    def set_subtitle(self, text: str) -> None:
        """The line AURA is currently speaking. Driven by SPEECH_CHUNK (10B.2).

        Rendered with `drawText`, never as rich text or HTML. The words come off
        the bus and the overlay's whole job is to draw them -- a subtitle panel
        that interpreted markup would be a rendering engine pointed at a
        message stream, which is a larger surface than a subtitle needs.
        """
        if text == self._subtitle:
            return
        self._subtitle = text
        self._resize_for_content()
        self.update()

    @property
    def subtitle(self) -> str:
        """What is on screen now. Read by the controller to size the hold."""
        return self._subtitle

    def look_at(self, dx: float, dy: float) -> None:
        self._avatar.look_at(dx, dy)

    # -- painting ------------------------------------------------------------

    def _tick(self) -> None:
        # Ask, do not assume. The backend returns False when nothing moved, and
        # then no repaint is scheduled at all.
        if self._avatar.advance(TICK_MS):
            self.update()

    def paintEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        aw, ah = self._avatar_size
        painter.save()
        painter.translate((self.width() - aw) / 2.0, 0)
        self._avatar.paint(painter, aw, ah)
        painter.restore()

        if self._subtitle:
            self._paint_subtitle(painter, ah)
        painter.end()

    def _paint_subtitle(self, painter: QPainter, top: int) -> None:
        height = self._subtitle_height()
        rect = QRect(0, top, self.width(), height)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(18, 20, 26, 225))
        painter.drawRoundedRect(rect, 10, 10)

        painter.setPen(QColor(232, 236, 244))
        font = QFont(self.font())
        font.setPointSizeF(9.5)
        painter.setFont(font)
        painter.drawText(
            rect.adjusted(SUBTITLE_PAD, SUBTITLE_PAD, -SUBTITLE_PAD, -SUBTITLE_PAD),
            int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft),
            self._subtitle,
        )

    # -- dragging ------------------------------------------------------------

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        if event.button() is Qt.MouseButton.LeftButton:
            self._drag_from = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        if self._drag_from is not None:
            target = event.globalPosition().toPoint() - self._drag_from
            self.move(target)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        if self._drag_from is not None:
            self._drag_from = None
            # A drag is one of the four permitted movement triggers, and the
            # only one the user performs directly.
            self.dragged.emit(self.x(), self.y())
