"""Wake word detection with openWakeWord.

Runs on every frame of live microphone audio, forever, so it is the one
component where cost is a hard constraint rather than a preference: an ONNX
model of a few hundred KB, ~1% of one core, no network, nothing retained.

Nothing the detector hears leaves this process. Audio only becomes a
transcript, and only reaches AURA Core, after a detection.

Only this package may import a wake-word library.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from aura.config import voice_models_dir
from aura.logging import get_logger
from aura.voice.types import AudioArray, VoiceInfo, WakeEvent
from aura.voice.wake.base import WakeWordDetector

log = get_logger(__name__)

_INT16_MAX = 32767.0

# Phrases openWakeWord ships models for. "aura" is not among them; training a
# custom model is an offline job (synthetic data plus a notebook), so the
# default is the closest thing in spirit to the spec's JARVIS framing.
PRETRAINED = ("alexa", "hey_mycroft", "hey_jarvis", "hey_rhasspy", "timer", "weather")


class OpenWakeWordDetector(WakeWordDetector):
    """Fires when the configured phrase is spoken."""

    def __init__(
        self,
        model: str = "hey_jarvis",
        threshold: float = 0.5,
        cooldown_s: float = 2.0,
    ) -> None:
        self._model_name = model
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._model: Any | None = None
        self._last_fired = 0.0

    def info(self) -> VoiceInfo:
        return VoiceInfo(
            name="openwakeword",
            kind="onnx_cpu",
            model=self._model_name,
            detail=f"threshold {self._threshold}",
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _resolve(self) -> str | None:
        """A custom .onnx in var/models/voice wins over a pretrained name."""
        custom = voice_models_dir() / f"{self._model_name}.onnx"
        if custom.is_file():
            return str(custom)
        if self._model_name in PRETRAINED:
            return None  # openWakeWord resolves pretrained names itself
        raise FileNotFoundError(
            f"Wake word {self._model_name!r} is neither a pretrained model "
            f"({', '.join(PRETRAINED)}) nor a file at {custom}. "
            "Run: uv run python scripts/download_wakeword.py"
        )

    def load(self) -> None:
        if self._model is not None:
            return
        from openwakeword.model import Model

        custom_path = self._resolve()
        started = time.perf_counter()
        # inference_framework="onnx" everywhere: the tflite path is Linux-only
        # and pulls a dependency with no Python 3.12 wheels.
        self._model = Model(
            wakeword_models=[custom_path] if custom_path else [self._model_name],
            inference_framework="onnx",
        )
        log.info(
            "wake.loaded",
            model=self._model_name,
            custom=bool(custom_path),
            load_ms=int((time.perf_counter() - started) * 1000),
        )

    def process(self, frame: AudioArray) -> WakeEvent | None:
        if self._model is None:
            self.load()
        assert self._model is not None

        now = time.time()
        if now - self._last_fired < self._cooldown_s:
            # Still feed the model so its buffers stay continuous, but ignore
            # the scores: one spoken phrase spans several frames and would
            # otherwise fire repeatedly.
            self._model.predict(self._to_int16(frame))
            return None

        scores: dict[str, float] = self._model.predict(self._to_int16(frame))
        for name, score in scores.items():
            if score >= self._threshold:
                self._last_fired = now
                self.reset()
                log.info("wake.detected", model=name, score=round(float(score), 3))
                return WakeEvent(model=name, score=float(score), at=now)
        return None

    def reset(self) -> None:
        if self._model is not None:
            # Without this the scores that just crossed the threshold stay in
            # the model's buffer and re-trigger on the following frame.
            self._model.reset()

    @staticmethod
    def _to_int16(frame: AudioArray) -> Any:
        """openWakeWord expects 16-bit PCM, not float."""
        return (np.clip(frame, -1.0, 1.0) * _INT16_MAX).astype(np.int16)


def ensure_pretrained_models(target: Path | None = None) -> None:
    """Download openWakeWord's pretrained models if absent.

    They are small (a few MB total) and fetched once. Called by
    scripts/download_wakeword.py rather than on first use, so a voice session
    never stalls on a network request.
    """
    import openwakeword.utils

    dest = target or voice_models_dir()
    dest.mkdir(parents=True, exist_ok=True)
    openwakeword.utils.download_models()
    log.info("wake.models_downloaded")
