"""The n8n provider — webhooks, and nothing above them.

**This is the only module in AURA that knows n8n exists** (ADR-0004 D10).
Everything above it speaks `AutomationProvider`. Swapping in Node-RED,
Windmill or Home Assistant is a new file next to this one.

Three things it deliberately does not do:

*It does not decide anything.* It receives an endpoint and arguments that the
manifest authorised and `ToolExecutor` validated, and it performs one POST.
It never consults n8n about what a workflow is, because that would be n8n
assigning AURA permissions (D2).

*It does not raise for ordinary failure* (D9). n8n being switched off is the
normal state on this machine, not an exception. An unreachable backend returns
`AutomationResult(ok=False)`, the agent loop reads it as an observation, and
AURA carries on. A provider that raised would make n8n a Core dependency by
accident.

*It does not hold credentials* (D7). n8n owns the OAuth tokens for Gmail,
Slack and the rest; AURA holds a URL. The one secret that may exist here is a
shared header authenticating AURA *to* n8n, and it is never logged, never
returned, and never put in a result.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx

from aura.automation.base import AutomationProvider, AutomationResult
from aura.logging import get_logger

log = get_logger(__name__)

# How much of a response we are willing to take. A workflow that reads a
# mailbox can return megabytes; that text becomes model context, and an
# unbounded observation is both a cost problem and a prompt-injection surface
# with more room to work in.
MAX_RESPONSE_BYTES = 64_000

# Trimmed further before it reaches the model. The full body stays in `data`
# for a caller that genuinely wants it.
MAX_SUMMARY_CHARS = 4_000


class N8NProvider(AutomationProvider):
    """Executes a declared workflow by POSTing to its webhook."""

    def __init__(
        self,
        *,
        base_url: str = "",
        auth_header: str = "",
        auth_token: str = "",
        auth_token_file: str = "",
        verify_tls: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth_header = auth_header
        self._auth_token = auth_token or _read_secret(auth_token_file)
        self._verify = verify_tls

    def name(self) -> str:
        return "n8n"

    # -- the seam --------------------------------------------------------

    def run(
        self, endpoint: str, arguments: dict[str, Any], timeout_s: float
    ) -> AutomationResult:
        started = time.perf_counter()
        headers = {"Content-Type": "application/json"}
        if self._auth_header and self._auth_token:
            headers[self._auth_header] = self._auth_token

        try:
            response = httpx.post(
                endpoint,
                json=arguments,
                headers=headers,
                timeout=timeout_s,
                verify=self._verify,
                # A workflow that answers with a redirect is a misconfiguration,
                # and following one would send the payload -- and the auth
                # header -- somewhere the manifest never authorised.
                follow_redirects=False,
            )
        except httpx.TimeoutException:
            return self._failed(
                started,
                f"The workflow did not respond within {timeout_s:.0f}s. "
                "It may still be running in n8n.",
            )
        except httpx.HTTPError as exc:
            # Includes connection refused, which is what "n8n is not running"
            # looks like. Deliberately not a raise: see the module docstring.
            log.info("automation.unreachable", error=type(exc).__name__)
            return self._failed(
                started, f"The automation service could not be reached ({type(exc).__name__})."
            )

        return self._from_response(response, started)

    def health(self) -> bool:
        """Best-effort liveness. Never used to decide whether a tool exists.

        Registration is independent of liveness (D9): a tool surface that
        flapped with n8n's uptime would make AURA's capabilities unpredictable
        from one turn to the next.
        """
        if not self._base_url:
            return False
        try:
            reply = httpx.get(f"{self._base_url}/healthz", timeout=3.0, verify=self._verify)
        except httpx.HTTPError:
            return False
        return reply.status_code < 500

    # -- response handling -----------------------------------------------

    def _from_response(self, response: httpx.Response, started: float) -> AutomationResult:
        elapsed = _ms(started)
        body = response.content[:MAX_RESPONSE_BYTES]
        truncated = len(response.content) > MAX_RESPONSE_BYTES

        if response.status_code >= 400:
            # n8n returns the workflow's error text here. It is external
            # content, so it is reported as a failure message and never
            # interpreted -- "404" most often means the workflow is not
            # active, which is worth saying plainly because it is the single
            # most common n8n mistake.
            hint = (
                " The workflow may be inactive in n8n, or its webhook path may "
                "have changed."
                if response.status_code == 404
                else ""
            )
            return AutomationResult(
                ok=False,
                summary="",
                error=f"The workflow returned HTTP {response.status_code}.{hint}",
                duration_ms=elapsed,
            )

        data: dict[str, Any] = {}
        text = body.decode("utf-8", errors="replace").strip()
        try:
            parsed = response.json()
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            data = parsed
        elif parsed is not None:
            # A workflow answering with a list or a bare string is legal. Wrap
            # it rather than discarding it, so `data` is always an object.
            data = {"result": parsed}

        summary = text[:MAX_SUMMARY_CHARS] if text else "The workflow completed."
        if truncated or len(text) > MAX_SUMMARY_CHARS:
            summary += "\n[response truncated]"

        # The body is not logged. It may contain mail, messages or documents,
        # and this line exists to say a workflow ran, not to archive its output.
        log.info(
            "automation.completed",
            status=response.status_code,
            bytes=len(response.content),
            duration_ms=elapsed,
        )
        return AutomationResult(ok=True, summary=summary, data=data, duration_ms=elapsed)

    def _failed(self, started: float, error: str) -> AutomationResult:
        return AutomationResult(ok=False, summary="", error=error, duration_ms=_ms(started))


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _read_secret(path_str: str) -> str:
    """Read a shared secret from a file, if one is configured.

    utf-8-sig for the same reason as `llm.api_key_file`: a file written by
    Windows PowerShell carries a BOM, and a BOM inside a token is a 401 that
    looks like the wrong secret.
    """
    if not path_str:
        return ""
    path = Path(path_str)
    if not path.is_absolute():
        from aura.config import PROJECT_ROOT

        path = PROJECT_ROOT / path
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8-sig").strip()
