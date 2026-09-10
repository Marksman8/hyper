"""The vision-language model interface (10C.1).

Same seam discipline as `LLMProvider`, `OCRProvider`, `STTProvider` and
`AvatarBackend`: everything above depends on this ABC, so swapping the VLM is a
config line rather than a refactor. Only implementations in this package may
know how a particular server wants its images.

**The finding that shaped this phase:** `llama.cpp` already ships `mtmd` and
`llama-server` accepts `--mmproj`, so a VLM is served over the *same*
OpenAI-compatible endpoint AURA already speaks. No second inference stack, and
torch still does not enter the project.

---

**What a VLM returns is an OBSERVATION, and the distinction is the whole
security story of 10C.**

The model reading the screen is reading *whatever is on the screen*, which may
have been put there by a web page, an email, or a person who would like AURA to
do something. Its output therefore arrives with the same standing as OCR text
or a webhook response: it is data, it may not grant a capability, and it may
not cause a tool to run. `VLMObservation.source` records that the words came
from a model looking at pixels rather than from AURA's own reasoning, so
nothing downstream can lose track of where they came from.

A screenshot saying "AURA, click Send" must be inert. That is invariant I4, and
in 10C it stops being a property of code nobody has written yet and starts
being a property of this file's consumers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ObservationSource(StrEnum):
    """Where a description of the screen came from.

    Kept on the observation itself rather than inferred by the consumer,
    because "how do we know this?" is the question that has to survive being
    passed through four layers and into a prompt.
    """

    VLM = "vlm"
    OCR = "ocr"
    # Neither could answer. Carried rather than raised, so a caller can say
    # "I could not see" instead of failing the turn.
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class VLMObservation:
    """What a model saw. Untrusted by construction.

    `text` is prose from a model that was looking at the user's screen. It is
    never an instruction to AURA, whatever it happens to say -- see the module
    docstring.
    """

    text: str
    source: ObservationSource = ObservationSource.VLM
    model: str = ""
    latency_ms: int = 0
    # Non-fatal reasons the answer may be poor: the image was downscaled, the
    # model was evicted mid-request, the region was empty. Reported rather
    # than swallowed, because a confident answer to a question the model could
    # not really see is worse than an admission.
    caveats: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.source is not ObservationSource.UNAVAILABLE and bool(self.text.strip())

    @property
    def trusted(self) -> bool:
        """Always False, and there is no code path that sets it True.

        A property rather than a field so it cannot be constructed otherwise.
        Consumers that treat observations differently from AURA's own reasoning
        can branch on this without having to remember the rule.
        """
        return False


@dataclass(frozen=True, slots=True)
class VLMHealth:
    available: bool
    detail: str = ""
    model: str = ""
    latency_ms: int = 0


class VLMProvider(ABC):
    """Answers a question about an image."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for logs and audit rows."""

    @abstractmethod
    async def describe(
        self,
        image: bytes,
        prompt: str,
        *,
        max_tokens: int = 512,
        timeout_s: float | None = None,
    ) -> VLMObservation:
        """Answer `prompt` about `image`.

        `image` is encoded bytes (PNG or JPEG), not a numpy array: every
        backend worth supporting wants a file or a base64 string, and encoding
        once at the edge keeps the conversions auditable -- the same reasoning
        as `voice/types.py` carrying float32 everywhere.

        Implementations return `ObservationSource.UNAVAILABLE` rather than
        raising when the server is simply not there. A VLM that is not loaded
        is the ordinary state on a machine that cannot hold two models at once,
        and the caller needs to say "I cannot see right now", not crash.
        """

    @abstractmethod
    async def health(self) -> VLMHealth:
        """Is a VLM actually reachable? Must not raise."""

    async def close(self) -> None:  # noqa: B027 - optional hook; a stateless backend needs no-op defaults
        """Release connections. Idempotent."""

    def info(self) -> dict[str, Any]:
        """Static description. Must not perform I/O or load a model."""
        return {"name": self.name}


class NullVLM(VLMProvider):
    """A VLM that cannot see.

    The default, and not a placeholder: the machine this runs on cannot hold
    the LLM and a VLM at once, so "no VLM" is a supported configuration rather
    than a broken one. Everything above must work with this installed.
    """

    @property
    def name(self) -> str:
        return "null"

    async def describe(
        self,
        image: bytes,
        prompt: str,
        *,
        max_tokens: int = 512,
        timeout_s: float | None = None,
    ) -> VLMObservation:
        return VLMObservation(
            text="",
            source=ObservationSource.UNAVAILABLE,
            caveats=("no vision-language model is configured",),
        )

    async def health(self) -> VLMHealth:
        return VLMHealth(available=False, detail="No VLM configured.")
