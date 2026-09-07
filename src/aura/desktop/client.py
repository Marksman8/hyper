"""HTTP client for the desktop shell.

Synchronous on purpose. The shell runs tkinter's `mainloop()` on the main
thread and everything here on a worker; mixing an asyncio loop into that adds a
third scheduling model for no benefit, since the shell makes one request at a
time and waits for it.

Every call returns a result object rather than raising, because the caller is a
UI: a network error is something to display in the status line, not something
to crash the window over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from aura.config import get_settings
from aura.logging import get_logger
from aura.security.auth import load_or_create_token

log = get_logger(__name__)


@dataclass(slots=True)
class CoreStatus:
    """What /v1/status reported, or why it could not be reached."""

    online: bool
    model: str = ""
    model_available: bool = False
    emergency_stop: bool = False
    ram_percent: float = 0.0
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None
    detail: str = ""


@dataclass(slots=True)
class TaskResult:
    """The outcome of one agent task."""

    ok: bool
    content: str = ""
    error: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    pending_confirmations: list[str] = field(default_factory=list)
    latency_ms: int = 0
    session_id: str | None = None
    stopped_early: bool = False


class CoreClient:
    """Talks to AURA Core."""

    def __init__(self, timeout_s: float = 600.0) -> None:
        cfg = get_settings()
        self._base = f"http://{cfg.api.host}:{cfg.api.port}"
        token = load_or_create_token(cfg.security.resolved_token_file(), cfg.security.api_token)
        self._client = httpx.Client(
            base_url=self._base,
            headers={"Authorization": f"Bearer {token}"},
            # Short connect timeout so "Core is not running" shows up
            # immediately, long read timeout because a task can take minutes.
            timeout=httpx.Timeout(timeout_s, connect=2.0),
        )

    def close(self) -> None:
        self._client.close()

    @property
    def base_url(self) -> str:
        return self._base

    def status(self) -> CoreStatus:
        try:
            resp = self._client.get("/v1/status", timeout=httpx.Timeout(5.0, connect=1.5))
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return CoreStatus(online=False, detail=str(exc)[:120])

        data = resp.json()
        provider, res = data["provider"], data["resources"]
        return CoreStatus(
            online=True,
            model=str(provider.get("model") or "unknown"),
            model_available=bool(provider.get("available")),
            emergency_stop=bool(data.get("emergency_stop")),
            ram_percent=float(res.get("ram_percent") or 0.0),
            vram_used_gb=res.get("vram_used_gb"),
            vram_total_gb=res.get("vram_total_gb"),
            detail=str(provider.get("detail") or ""),
        )

    def run_task(
        self, message: str, approve: list[str] | None = None, session_id: str | None = None
    ) -> TaskResult:
        payload: dict[str, Any] = {"message": message, "approve": approve or []}
        if session_id:
            payload["session_id"] = session_id

        try:
            resp = self._client.post("/v1/agent", json=payload)
        except httpx.HTTPError as exc:
            return TaskResult(ok=False, error=f"Cannot reach AURA Core: {exc}")

        if resp.status_code >= 400:
            return TaskResult(ok=False, error=_error_text(resp))

        data = resp.json()
        calls = [call for step in data.get("steps", []) for call in step.get("tools", [])]
        return TaskResult(
            ok=True,
            content=data.get("content") or "",
            tool_calls=calls,
            pending_confirmations=data.get("pending_confirmations") or [],
            latency_ms=int(data.get("latency_ms") or 0),
            session_id=data.get("session_id"),
            stopped_early=bool(data.get("stopped_early")),
        )

    def set_emergency_stop(self, engaged: bool) -> bool:
        """Returns the state actually in force, so the UI never lies."""
        try:
            resp = self._client.post("/v1/control/estop", json={"engaged": engaged})
            resp.raise_for_status()
            return bool(resp.json().get("emergency_stop"))
        except httpx.HTTPError as exc:
            log.warning("desktop.estop_failed", error=str(exc))
            return not engaged

    def tools(self) -> list[dict[str, Any]]:
        try:
            resp = self._client.get("/v1/tools", timeout=httpx.Timeout(5.0, connect=1.5))
            resp.raise_for_status()
            result: list[dict[str, Any]] = resp.json()
            return result
        except httpx.HTTPError:
            return []


def _error_text(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error", {})
        return str(err.get("message") or resp.text)[:400]
    except ValueError:
        return resp.text[:400]
