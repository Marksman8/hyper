"""VLM over an OpenAI-compatible server (10C.1).

`llama-server --mmproj <projector>` serves a vision model on the *same*
`chat/completions` endpoint AURA already speaks, with images passed as
`image_url` parts carrying a `data:` URI. So this file is deliberately close to
`models/providers/openai_compat.py` — it is the same protocol with one extra
content type, and the similarity is the point: no second inference stack, and
torch stays out of the project.

**This module is the only place in AURA allowed to know how a VLM request is
shaped.** Everything above it holds a `VLMProvider`.

Note what it does *not* do. It never retries a failed describe. A VLM call on
this hardware costs seconds and evicts the language model to run at all
(10C.3); a silent retry would double the worst case and hide the fact that
perception is struggling, at exactly the moment a caller most needs to know.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import httpx

from aura.logging import get_logger
from aura.vision.vlm.base import (
    ObservationSource,
    VLMHealth,
    VLMObservation,
    VLMProvider,
)

log = get_logger(__name__)

# What a data: URI needs to say about the bytes. Sniffed rather than configured
# because the caller encodes the image and should not also have to declare it.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"


def _mime_type(image: bytes) -> str:
    if image.startswith(_PNG_MAGIC):
        return "image/png"
    if image.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    # Neither. Say PNG and let the server reject it with something specific --
    # guessing here and being wrong produces a clearer error than refusing.
    return "image/png"


def to_data_uri(image: bytes) -> str:
    """Encode image bytes for an `image_url` content part."""
    return f"data:{_mime_type(image)};base64,{base64.b64encode(image).decode('ascii')}"


class OpenAICompatVLM(VLMProvider):
    """Talks OpenAI chat-completions with an image part."""

    def __init__(
        self,
        base_url: str,
        model: str = "default",
        api_key: str = "",
        timeout_s: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
        )

    @property
    def name(self) -> str:
        return "openai_compat_vlm"

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "endpoint": self._base_url, "model": self._model}

    async def describe(
        self,
        image: bytes,
        prompt: str,
        *,
        max_tokens: int = 512,
        timeout_s: float | None = None,
    ) -> VLMObservation:
        if not image:
            return VLMObservation(
                text="",
                source=ObservationSource.UNAVAILABLE,
                caveats=("no image was supplied",),
            )

        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            # Low, and not a knob. A VLM asked to describe a screen at high
            # temperature invents plausible UI that is not there, and an
            # invented button is worse than no answer -- 10C.7 onward may act
            # on this.
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": to_data_uri(image)},
                        },
                    ],
                }
            ],
        }

        started = time.perf_counter()
        try:
            response = await self._client.post(
                "/chat/completions",
                json=payload,
                timeout=timeout_s or self._timeout_s,
            )
        except httpx.TimeoutException:
            return self._unavailable(
                started, "the vision model did not answer in time", timed_out=True
            )
        except httpx.HTTPError as exc:
            # Connection refused is the ordinary state: the VLM is evicted far
            # more often than it is resident.
            return self._unavailable(started, f"no vision model reachable ({type(exc).__name__})")

        if response.status_code >= 400:
            body = response.text[:200]
            log.info("vlm.rejected", status=response.status_code)
            return self._unavailable(
                started, f"the vision model returned {response.status_code}: {body}"
            )

        return self._from_response(response, started)

    def _from_response(self, response: httpx.Response, started: float) -> VLMObservation:
        elapsed = int((time.perf_counter() - started) * 1000)
        try:
            data = response.json()
            text = str(data["choices"][0]["message"]["content"] or "").strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            return self._unavailable(started, f"unreadable response ({type(exc).__name__})")

        caveats: list[str] = []
        finish = (data.get("choices") or [{}])[0].get("finish_reason")
        if finish == "length":
            # Worth carrying: a description cut off mid-sentence can lose the
            # very element the caller asked about.
            caveats.append("the description was cut off by the token limit")

        # The text is NOT logged. It describes the user's screen, and this line
        # exists to say a call happened, not to archive what was on it.
        log.info(
            "vlm.observed",
            model=self._model,
            chars=len(text),
            duration_ms=elapsed,
            finish=finish,
        )
        return VLMObservation(
            text=text,
            source=ObservationSource.VLM if text else ObservationSource.UNAVAILABLE,
            model=self._model,
            latency_ms=elapsed,
            caveats=tuple(caveats),
        )

    def _unavailable(
        self, started: float, detail: str, *, timed_out: bool = False
    ) -> VLMObservation:
        return VLMObservation(
            text="",
            source=ObservationSource.UNAVAILABLE,
            model=self._model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            caveats=(detail,) if not timed_out else (detail, "it may still be running"),
        )

    async def health(self) -> VLMHealth:
        started = time.perf_counter()
        try:
            response = await self._client.get("/models", timeout=3.0)
        except httpx.HTTPError as exc:
            return VLMHealth(
                available=False,
                detail=f"No vision model at {self._base_url} ({type(exc).__name__}).",
                model=self._model,
            )
        elapsed = int((time.perf_counter() - started) * 1000)
        if response.status_code >= 400:
            return VLMHealth(
                available=False,
                detail=f"HTTP {response.status_code} from {self._base_url}.",
                model=self._model,
                latency_ms=elapsed,
            )
        return VLMHealth(
            available=True, detail="", model=self._model, latency_ms=elapsed
        )

    async def close(self) -> None:
        await self._client.aclose()
