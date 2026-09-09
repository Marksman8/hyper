"""The presence process's connection to AURA Core.

A `QThread` reading `/v1/events`, emitting a Qt signal per event. Qt signals
across threads are queued and delivered on the GUI thread, which is what makes
this safe -- and is a genuine advantage over the tkinter shell, where a
cross-thread widget call corrupts Tcl state and crashes somewhere unrelated
later (see `desktop/app.py`).

Three behaviours matter more than the plumbing:

**Reconnect with `Last-Event-ID`.** The overlay resumes where it left off, so a
brief drop does not cost it the current state.

**Lag means resync, not carry on.** If Core sends the terminal
`subscriber_lagged` frame, the client reconnects *without* a Last-Event-ID and
takes a fresh snapshot. Reconnecting with the stale id would silently continue
with a hole -- exactly what the bus contract's rule 2 for consumers forbids.

**Observations are never acted on.** The overlay renders; it does not decide.
The trust label is passed through so a consumer can honour it, and this client
never treats a payload as an instruction.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from PySide6.QtCore import QThread, Signal

from aura.config import get_settings
from aura.logging import get_logger
from aura.security.auth import load_or_create_token

log = get_logger(__name__)

# How long to wait before retrying a dropped connection. Short enough that a
# Core restart is invisible; long enough not to spin when Core is simply down.
RETRY_S = 2.0

# Read timeout. Must exceed Core's 15 s heartbeat, or a quiet connection would
# be torn down on every keepalive gap.
READ_TIMEOUT_S = 45.0


class CoreEventClient(QThread):
    """Streams `/v1/events` and re-emits each one as a Qt signal."""

    # The decoded event dict, exactly as it came off the wire.
    #
    # Named `received`, not `event`: QObject already has an `event()`
    # method, and shadowing it would break Qt's own dispatch in ways that
    # surface far from the cause.
    received = Signal(dict)
    # True when connected, False when the connection is down.
    connected = Signal(bool)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._stop = False
        self._last_seq: int | None = None
        self._settings = get_settings()

    def stop(self) -> None:
        self._stop = True

    def _url(self) -> str:
        cfg = self._settings
        return f"http://{cfg.api.host}:{cfg.api.port}/v1/events"

    def _headers(self) -> dict[str, str]:
        cfg = self._settings
        token = load_or_create_token(
            cfg.security.resolved_token_file(), cfg.security.api_token
        )
        headers = {"Authorization": f"Bearer {token}"}
        if self._last_seq is not None:
            headers["Last-Event-ID"] = str(self._last_seq)
        return headers

    def run(self) -> None:
        while not self._stop:
            try:
                self._stream()
            except httpx.HTTPError as exc:
                log.info("presence.core_unreachable", error=str(exc))
                self.connected.emit(False)
            except Exception as exc:
                log.warning("presence.client_failed", error=str(exc))
                self.connected.emit(False)
            if not self._stop:
                self.msleep(int(RETRY_S * 1000))

    def _stream(self) -> None:
        timeout = httpx.Timeout(READ_TIMEOUT_S, connect=3.0)
        with httpx.Client(timeout=timeout) as client, client.stream(
            "GET", self._url(), headers=self._headers()
        ) as response:
            response.raise_for_status()
            self.connected.emit(True)
            log.info("presence.connected", resumed_from=self._last_seq)

            buffer = ""
            for chunk in response.iter_text():
                if self._stop:
                    return
                buffer += chunk
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    if block.startswith(":"):
                        continue  # heartbeat
                    if self._handle(block):
                        # Lagged: drop the resume position so the reconnect
                        # takes a snapshot rather than continuing with a hole.
                        self._last_seq = None
                        return

    def _handle(self, block: str) -> bool:
        """Parse one frame. Returns True if the stream must be resynced."""
        payload: dict[str, Any] | None = None
        for line in block.splitlines():
            if line.startswith("data: "):
                try:
                    payload = json.loads(line[6:])
                except json.JSONDecodeError:
                    log.debug("presence.bad_frame")
                    return False

        if payload is None:
            return False

        seq = payload.get("seq")
        if isinstance(seq, int):
            self._last_seq = seq

        if (
            payload.get("type") == "error"
            and (payload.get("payload") or {}).get("code") == "subscriber_lagged"
        ):
            log.warning("presence.lagged_resyncing")
            return True

        self.received.emit(payload)
        return False
