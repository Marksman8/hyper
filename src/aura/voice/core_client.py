"""HTTP client for AURA Core, used by the voice process.

Synchronous on purpose. The voice process is a conversation: one utterance,
one reply, one thing happening at a time. An async client here would buy
nothing and would fight the audio callbacks, which are threads.

This is the same authenticated `/v1/chat` surface the CLI and any third-party
client uses -- voice gets no privileged back door into the orchestrator.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from aura.config import Settings
from aura.logging import get_logger
from aura.security.auth import load_or_create_token

log = get_logger(__name__)


class CoreUnavailableError(RuntimeError):
    """AURA Core is not reachable."""


class CoreClient:
    """Streams replies out of AURA Core."""

    def __init__(self, settings: Settings, timeout_s: float = 300.0) -> None:
        self._base = f"http://{settings.api.host}:{settings.api.port}"
        token = load_or_create_token(
            settings.security.resolved_token_file(), settings.security.api_token
        )
        self._headers = {"Authorization": f"Bearer {token}"}
        self._client = httpx.Client(
            base_url=self._base,
            headers=self._headers,
            timeout=httpx.Timeout(timeout_s, connect=3.0),
        )
        self.session_id: str | None = None

    @property
    def base_url(self) -> str:
        return self._base

    def health(self) -> bool:
        try:
            return self._client.get("/healthz", timeout=3.0).status_code == 200
        except httpx.HTTPError:
            return False

    def stream_reply(self, text: str) -> Iterator[str]:
        """Yield reply text deltas for `text`, continuing the same session.

        Raises CoreUnavailableError if Core cannot be reached, so the caller
        can say something useful out loud instead of failing silently.
        """
        # Ask Core to shape the reply for the ear, not the eye.
        payload: dict[str, object] = {
            "message": text,
            "stream": True,
            "spoken": True,
        }
        if self.session_id:
            payload["session_id"] = self.session_id

        try:
            with self._client.stream("POST", "/v1/chat", json=payload) as resp:
                if resp.status_code >= 400:
                    body = resp.read().decode("utf-8", errors="replace")
                    raise CoreUnavailableError(
                        f"AURA Core returned {resp.status_code}: {body[:300]}"
                    )
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        frame = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    kind = frame.get("type")
                    if kind == "error":
                        err = frame.get("error", {})
                        raise CoreUnavailableError(str(err.get("message", err)))
                    if kind == "delta":
                        if frame.get("session_id"):
                            self.session_id = str(frame["session_id"])
                        yield str(frame.get("content", ""))
        except httpx.HTTPError as exc:
            raise CoreUnavailableError(
                f"Cannot reach AURA Core at {self._base}. Start it with: aura serve"
            ) from exc

    def new_session(self) -> None:
        self.session_id = None

    def close(self) -> None:
        self._client.close()
