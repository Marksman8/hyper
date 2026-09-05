"""Streaming voice activity detection (Silero v6, ONNX).

Two jobs, both timing-critical:

- **Endpointing.** Deciding when the user stopped talking. Too eager and it
  cuts them off mid-sentence; too patient and every reply feels sluggish. This
  is the single largest term in the latency budget that is under our control.
- **Barge-in.** Noticing that the user started talking while AURA is speaking,
  so playback can be cut immediately.

The model costs no new dependency and, notably, no torch -- the `silero-vad`
package on PyPI requires it, while the raw ONNX graph needs only onnxruntime.
scripts/download_vad.py places it in var/models/voice/.

This drives the ONNX graph with *persistent* recurrent state and a 512-sample
window, which is what Silero expects frame to frame. Batch wrappers (including
faster-whisper's) zero the LSTM state on every call -- correct for transcribing
a finished recording, wrong for a live stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aura.config import voice_models_dir
from aura.logging import get_logger
from aura.voice.types import SAMPLE_RATE, AudioArray

log = get_logger(__name__)

# Silero v6 at 16 kHz: 512-sample windows with 64 samples of carried context.
_WINDOW = 512
_CONTEXT = 64


VAD_MODEL_FILENAME = "silero_vad.onnx"


def vad_model_path() -> Path:
    """Where AURA keeps the Silero VAD model.

    Deliberately AURA's own directory rather than reaching into
    faster-whisper's package assets. Importing another library purely to read a
    file out of its install tree is a broken seam -- this module is not a
    speech-provider package -- and fragile, since the asset filename changes
    between their releases. scripts/download_vad.py puts the file here.
    """
    return voice_models_dir() / VAD_MODEL_FILENAME


@dataclass(slots=True)
class SpeechSegment:
    """What the endpointer decided about the audio it collected."""

    audio: AudioArray
    speech_s: float
    ended_by: str  # "silence" | "max_duration" | "no_speech"

    @property
    def has_speech(self) -> bool:
        return self.speech_s > 0.0


class StreamingVAD:
    """Per-frame speech probability with persistent recurrent state."""

    def __init__(self, threshold: float = 0.5, model_path: Path | None = None) -> None:
        self._threshold = threshold
        self._path = model_path or vad_model_path()
        self._session: Any | None = None
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0

    def load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime

        if not self._path.is_file():
            raise FileNotFoundError(
                f"Silero VAD model not found at {self._path}. "
                "Run: uv run python scripts/download_vad.py"
            )

        opts = onnxruntime.SessionOptions()
        # One thread each: this runs on every 80 ms frame forever, so the goal
        # is to stay out of the way, not to finish 2 ms sooner.
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            str(self._path), providers=["CPUExecutionProvider"], sess_options=opts
        )
        log.debug("vad.loaded", path=str(self._path))

    def reset(self) -> None:
        """Clear recurrent state between utterances."""
        self._h[:] = 0.0
        self._c[:] = 0.0
        self._context[:] = 0.0
        self._pending = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0

    @property
    def last_probability(self) -> float:
        return self._last_prob

    def process(self, frame: AudioArray) -> float:
        """Feed audio; return the most recent speech probability.

        Frames need not be a multiple of the window: leftover samples are
        carried to the next call, so callers can use whatever size suits the
        wake-word model.
        """
        if self._session is None:
            self.load()
        assert self._session is not None

        self._pending = np.concatenate([self._pending, np.asarray(frame, dtype=np.float32)])

        while len(self._pending) >= _WINDOW:
            window = self._pending[:_WINDOW]
            self._pending = self._pending[_WINDOW:]

            batch = np.concatenate([self._context, window])[None, :].astype(np.float32)
            probs, self._h, self._c = self._session.run(
                None, {"input": batch, "h": self._h, "c": self._c}
            )
            self._context = window[-_CONTEXT:].copy()
            self._last_prob = float(np.asarray(probs).ravel()[-1])

        return self._last_prob

    def is_speech(self, frame: AudioArray) -> bool:
        return self.process(frame) >= self._threshold


class Endpointer:
    """Collects one utterance, deciding when it ended.

    Deliberately requires speech to *start* before silence can end anything:
    without that, the trailing silence of a false wake would immediately
    "end" an utterance that never began, and Whisper would be handed noise.
    """

    def __init__(
        self,
        silence_s: float = 0.6,
        max_duration_s: float = 20.0,
        start_timeout_s: float = 6.0,
        threshold: float = 0.5,
        vad: StreamingVAD | None = None,
    ) -> None:
        self._vad = vad or StreamingVAD(threshold=threshold)
        self._silence_s = silence_s
        self._max_s = max_duration_s
        self._start_timeout_s = start_timeout_s

    @property
    def vad(self) -> StreamingVAD:
        return self._vad

    def begin(
        self,
        preroll: AudioArray | None = None,
        start_timeout_s: float | None = None,
    ) -> _Collector:
        """Start collecting.

        `preroll` is audio from before the wake word. `start_timeout_s`
        overrides how long to wait for speech to begin, which a follow-up
        window sets differently from a wake-word utterance.
        """
        self._vad.reset()
        return _Collector(
            vad=self._vad,
            silence_s=self._silence_s,
            max_s=self._max_s,
            start_timeout_s=(self._start_timeout_s if start_timeout_s is None else start_timeout_s),
            preroll=preroll,
        )


class _Collector:
    """Accumulates frames for one utterance."""

    def __init__(
        self,
        vad: StreamingVAD,
        silence_s: float,
        max_s: float,
        start_timeout_s: float,
        preroll: AudioArray | None,
    ) -> None:
        self._vad = vad
        self._silence_s = silence_s
        self._max_s = max_s
        self._start_timeout_s = start_timeout_s
        self._chunks: list[AudioArray] = []
        if preroll is not None and len(preroll):
            self._chunks.append(np.asarray(preroll, dtype=np.float32))
        self._elapsed_s = 0.0
        self._speech_s = 0.0
        self._silence_run_s = 0.0
        self._started = False

    def push(self, frame: AudioArray) -> SpeechSegment | None:
        """Add a frame. Returns a segment once the utterance is complete."""
        self._chunks.append(np.asarray(frame, dtype=np.float32))
        frame_s = len(frame) / SAMPLE_RATE
        self._elapsed_s += frame_s

        speaking = self._vad.is_speech(frame)
        if speaking:
            self._started = True
            self._speech_s += frame_s
            self._silence_run_s = 0.0
        elif self._started:
            self._silence_run_s += frame_s
            if self._silence_run_s >= self._silence_s:
                return self._finish("silence")

        if not self._started and self._elapsed_s >= self._start_timeout_s:
            # A false wake: nothing was ever said.
            return self._finish("no_speech")

        if self._elapsed_s >= self._max_s:
            return self._finish("max_duration")

        return None

    def _finish(self, reason: str) -> SpeechSegment:
        audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)
        log.debug(
            "vad.segment",
            ended_by=reason,
            audio_s=round(len(audio) / SAMPLE_RATE, 2),
            speech_s=round(self._speech_s, 2),
        )
        return SpeechSegment(audio=audio, speech_s=self._speech_s, ended_by=reason)


class BargeInDetector:
    """Decides whether the user is talking over AURA.

    Requires sustained speech rather than a single frame: a cough, a keyboard
    clack, or AURA's own voice bleeding through the microphone should not stop
    playback, but a person starting a sentence should.
    """

    def __init__(self, sustain_s: float = 0.35, threshold: float = 0.6) -> None:
        self._sustain_s = sustain_s
        self._vad = StreamingVAD(threshold=threshold)
        self._run_s = 0.0

    def reset(self) -> None:
        self._vad.reset()
        self._run_s = 0.0

    def process(self, frame: AudioArray) -> bool:
        """True once speech has been sustained long enough to interrupt."""
        if self._vad.is_speech(frame):
            self._run_s += len(frame) / SAMPLE_RATE
        else:
            self._run_s = 0.0
        return self._run_s >= self._sustain_s
