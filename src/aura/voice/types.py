"""Voice domain types.

Audio is carried as `numpy.ndarray` of float32 in [-1, 1], mono, at
`SAMPLE_RATE`. One representation everywhere: every device, model and library
in this pipeline wants something slightly different (int16, 16 kHz, 22.05 kHz,
bytes), and converting at the edges rather than in the middle is the only way
to keep the conversions auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

# numpy is imported at runtime, not just for typing. That is safe because
# nothing outside `aura.voice` imports this module, and every entry point into
# `aura.voice` is a lazy import inside a CLI command -- so `aura --help` still
# works on a machine with no voice extra installed.
AudioArray = NDArray[np.float32]

# 16 kHz mono is the common denominator: openWakeWord, Silero VAD and Whisper
# all require exactly this. Resampling happens once, at capture.
SAMPLE_RATE = 16_000

# 80 ms. openWakeWord wants 1280-sample (80 ms) frames; matching that here
# means the wake detector never has to re-buffer.
FRAME_SAMPLES = 1280
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE


class VoiceState(StrEnum):
    """Where the session loop is.

    Maps onto spec section 10's modes: SLEEP is the resting state that only a
    wake word leaves, and PRIVATE (mic off) is represented by the capture
    stream being stopped rather than by a state here -- a state the loop could
    ignore would not be a kill switch.
    """

    SLEEP = "sleep"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass(slots=True, frozen=True)
class WakeEvent:
    """A wake word fired."""

    model: str
    score: float
    at: float


@dataclass(slots=True, frozen=True)
class Transcript:
    """The result of transcribing one utterance."""

    text: str
    language: str = "en"
    duration_s: float = 0.0
    latency_ms: int = 0
    # Mean log-probability from the decoder, when the backend reports one.
    # Used to drop confident-sounding garbage from a half-second of noise.
    confidence: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(slots=True, frozen=True)
class SpeechChunk:
    """A synthesised audio fragment, ready to play."""

    audio: AudioArray
    sample_rate: int
    text: str = ""

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate


@dataclass(slots=True, frozen=True)
class VoiceInfo:
    """Static description of a speech backend, for status output."""

    name: str
    kind: str
    model: str | None = None
    detail: str = ""


@dataclass(slots=True)
class TurnTimings:
    """Where the time went in one voice turn (spec section 25).

    Recorded per turn because the 1.5 s end-to-end budget is only defensible if
    each stage is measured separately; a regression in any one of them looks
    identical from the outside.
    """

    endpoint_ms: int = 0
    stt_ms: int = 0
    llm_first_token_ms: int = 0
    tts_first_chunk_ms: int = 0
    total_to_first_audio_ms: int = 0
    extra: dict[str, int] = field(default_factory=dict)
