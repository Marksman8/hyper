"""Speech-to-text via faster-whisper (CTranslate2).

CTranslate2, not torch. `base.en` at int8 is ~150 MB resident and transcribes a
3-second utterance in a few hundred milliseconds on 16 threads, which fits both
the RAM budget and the 1.5 s end-to-end target without a GPU.

Running on CPU is a deliberate trade. CTranslate2's CUDA build needs cuDNN 9
DLLs present on Windows, which is real setup friction for ~200 ms on short
utterances. The `STTProvider` seam means moving to GPU later is
`voice.stt.device: cuda`, not a rewrite.

Only this package may import an STT library.
"""

from __future__ import annotations

import math
import time
from typing import Any

from aura.logging import get_logger
from aura.voice.stt.base import STTProvider
from aura.voice.types import SAMPLE_RATE, AudioArray, Transcript, VoiceInfo

log = get_logger(__name__)

# Whisper hallucinates confident boilerplate on silence or noise -- these are
# the artefacts of its training data, not transcriptions. Dropping them is far
# cheaper than making the user listen to AURA answer a YouTube outro.
_HALLUCINATIONS = frozenset(
    {
        "thank you.",
        "thanks for watching!",
        "thank you for watching.",
        "thank you for watching!",
        "you",
        "bye.",
        "please subscribe to my channel.",
        ".",
        "...",
        "[blank_audio]",
        "(silence)",
    }
)

# Below this, an "utterance" is a click or a breath, not speech.
_MIN_AUDIO_S = 0.25


class FasterWhisperSTT(STTProvider):
    def __init__(
        self,
        model: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        beam_size: int = 1,
        min_confidence: float = -1.0,
        cpu_threads: int = 0,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size
        self._min_confidence = min_confidence
        self._cpu_threads = cpu_threads
        self._model: Any | None = None

    def info(self) -> VoiceInfo:
        return VoiceInfo(
            name="faster_whisper",
            kind=f"ctranslate2_{self._device}",
            model=self._model_name,
            detail=self._compute_type,
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        started = time.perf_counter()
        self._model = WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
            cpu_threads=self._cpu_threads,
        )
        log.info(
            "stt.loaded",
            model=self._model_name,
            device=self._device,
            compute_type=self._compute_type,
            load_ms=int((time.perf_counter() - started) * 1000),
        )

    def unload(self) -> None:
        if self._model is not None:
            self._model = None
            log.debug("stt.unloaded", model=self._model_name)

    def transcribe(self, audio: AudioArray) -> Transcript:
        duration_s = len(audio) / SAMPLE_RATE
        if duration_s < _MIN_AUDIO_S:
            return Transcript(text="", duration_s=duration_s, confidence=-10.0)

        self.load()
        assert self._model is not None
        started = time.perf_counter()

        segments, _info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            # Whisper's own VAD, second-guessing our endpointing. It costs
            # little and reliably removes trailing silence that would
            # otherwise become a hallucinated sentence.
            vad_filter=True,
            condition_on_previous_text=False,
        )

        parts: list[str] = []
        logprobs: list[float] = []
        for seg in segments:  # generator: this is where the work happens
            parts.append(seg.text)
            logprobs.append(float(seg.avg_logprob))

        latency_ms = int((time.perf_counter() - started) * 1000)
        text = " ".join(p.strip() for p in parts).strip()
        confidence = sum(logprobs) / len(logprobs) if logprobs else -10.0

        if text.lower().strip() in _HALLUCINATIONS:
            log.info("stt.hallucination_dropped", text=text)
            text = ""
        elif confidence < self._min_confidence:
            # INFO, not DEBUG: discarding what the user actually said is a
            # decision they deserve to be able to see and tune.
            log.info(
                "stt.low_confidence_dropped",
                text=text,
                confidence=round(confidence, 2),
                threshold=self._min_confidence,
            )
            text = ""

        log.info(
            "stt.transcribed",
            chars=len(text),
            audio_s=round(duration_s, 2),
            latency_ms=latency_ms,
            rtf=round(latency_ms / 1000 / max(duration_s, 0.01), 2),
            confidence=round(confidence, 2) if not math.isnan(confidence) else None,
        )
        return Transcript(
            text=text,
            language=self._language,
            duration_s=duration_s,
            latency_ms=latency_ms,
            confidence=confidence,
        )
