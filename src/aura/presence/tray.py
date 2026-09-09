"""System tray icon for the presence process.

The overlay is click-through and has no title bar, so without this the only way
to close it is to kill the process. That is not a user interface, it is a
workaround.

**The tray is the one place the presence process talks back to Core.** Toggling
focus mode is a user action, and it travels the way every user action travels:
an authenticated POST, not an event. The bus stays one-way (event-bus contract
§C.5) — a subscriber that could publish would be a second source of authority.

Hiding is different, and deliberately so: it is *local presentation only*. It
does not touch privacy mode, because "get out of my way for a minute" and
"stop observing me" are different requests and conflating them would let a
cosmetic click silently change a security setting.
"""

from __future__ import annotations

from typing import Any

import httpx
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from aura.config import get_settings
from aura.logging import get_logger
from aura.security.auth import load_or_create_token

log = get_logger(__name__)

ICON_PX = 64


def _icon(colour: QColor) -> QIcon:
    """The avatar silhouette in miniature, so the tray matches the overlay."""
    pixmap = QPixmap(ICON_PX, ICON_PX)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    path = QPainterPath()
    path.addEllipse(22, 8, 20, 20)
    path.addRoundedRect(14, 32, 36, 28, 10, 10)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(colour)
    painter.drawPath(path)
    painter.end()
    return QIcon(pixmap)


class PresenceTray(QObject):
    """Tray icon and menu."""

    toggle_visible = Signal()
    quit_requested = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._settings = get_settings()
        self._focus = False

        self._tray = QSystemTrayIcon(_icon(QColor(110, 140, 220)), self)
        self._tray.setToolTip("AURA")

        menu = QMenu()
        self._visibility = QAction("Hide AURA", menu)
        self._visibility.triggered.connect(self.toggle_visible.emit)
        menu.addAction(self._visibility)

        self._focus_action = QAction("Focus mode", menu)
        self._focus_action.setCheckable(True)
        self._focus_action.toggled.connect(self._on_focus_toggled)
        menu.addAction(self._focus_action)

        menu.addSeparator()
        quit_action = QAction("Quit AURA presence", menu)
        quit_action.triggered.connect(self.quit_requested.emit)
        menu.addAction(quit_action)

        self._menu = menu
        self._tray.setContextMenu(menu)
        # Left click toggles visibility; a tray icon that only responds to
        # right click feels broken on Windows.
        self._tray.activated.connect(self._on_activated)

    @property
    def available(self) -> bool:
        return QSystemTrayIcon.isSystemTrayAvailable()

    def show(self) -> None:
        if self.available:
            self._tray.show()
        else:
            # Not fatal. The overlay still works; there is simply no menu.
            log.warning("presence.tray_unavailable")

    def hide(self) -> None:
        self._tray.hide()

    def set_state_colour(self, rgb: tuple[int, int, int]) -> None:
        """Match the avatar, so the tray says something at a glance."""
        self._tray.setIcon(_icon(QColor(*rgb)))

    def set_visible_label(self, *, overlay_visible: bool) -> None:
        self._visibility.setText("Hide AURA" if overlay_visible else "Show AURA")

    def set_focus_checked(self, enabled: bool) -> None:
        """Reflect Core's state without re-triggering the toggle."""
        if enabled == self._focus:
            return
        self._focus = enabled
        self._focus_action.blockSignals(True)
        self._focus_action.setChecked(enabled)
        self._focus_action.blockSignals(False)

    # -- the one call back to Core -------------------------------------------

    def _on_focus_toggled(self, enabled: bool) -> None:
        """Authenticated POST, never a bus publish.

        Focus mode is Core's state, not the overlay's: a second client, or a
        reconnect, must see the same answer. Setting it locally would make the
        overlay's opinion diverge from everyone else's.
        """
        self._focus = enabled
        cfg = self._settings
        url = f"http://{cfg.api.host}:{cfg.api.port}/v1/focus"
        try:
            token = load_or_create_token(
                cfg.security.resolved_token_file(), cfg.security.api_token
            )
            httpx.post(
                url,
                json={"enabled": enabled},
                headers={"Authorization": f"Bearer {token}"},
                timeout=5.0,
            ).raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("presence.focus_post_failed", error=str(exc))
            # Put the tick back: the menu must not claim a change Core refused.
            self.set_focus_checked(not enabled)

    def _on_activated(self, reason: Any) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.toggle_visible.emit()
