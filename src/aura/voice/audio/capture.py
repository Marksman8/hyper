"""Microphone capture.

Design points that matter:

*The kill switch is real.* `stop()` closes the PortAudio stream. It does not
set a flag that a consumer checks. Spec section 23 asks for a microphone kill
switch, and a switch that leaves the OS recording while the application
promises not to look is not one -- the Windows mic-in-use indicator going dark
is the observable difference.

*Resampling happens here, once.* Devices default to 44.1 or 48 kHz; everything
downstream needs exactly 16 kHz mono float32. Converting at the edge means no
other module has to know a sample rate.

*The audio callback never blocks.* It drops frames into a bounded queue and
returns. Anything slow -- ONNX inference, HTTP, disk -- happens on the consumer
side, because overrunning the callback produces audible dropouts.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
import sounddevice as sd

from aura.logging import get_logger
from aura.voice.types import FRAME_SAMPLES, SAMPLE_RATE, AudioArray

log = get_logger(__name__)

# ~4 seconds of 80 ms frames. Deep enough to ride out a GC pause or a slow
# ONNX call, shallow enough that a wedged consumer is noticed rather than
# silently accumulating stale audio.
_QUEUE_FRAMES = 50


class MicrophoneClosedError(RuntimeError):
    """Audio was requested while the microphone is stopped."""


class Microphone:
    """A 16 kHz mono float32 frame source.

    Not reusable across threads for reading; one consumer at a time.
    """

    def __init__(
        self,
        device: int | str | None = None,
        frame_samples: int = FRAME_SAMPLES,
    ) -> None:
        self._device = device
        self._frame_samples = frame_samples
        self._queue: queue.Queue[AudioArray] = queue.Queue(maxsize=_QUEUE_FRAMES)
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._dropped = 0

    # -- lifecycle -----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        """Open the device. Idempotent."""
        with self._lock:
            if self._stream is not None:
                return
            # blocksize in device frames. PortAudio resamples to `samplerate`
            # for us, so the callback always receives exactly what we asked for.
            self._stream = sd.InputStream(
                device=self._device,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=self._frame_samples,
                dtype="float32",
                callback=self._on_audio,
            )
            self._stream.start()
        info = sd.query_devices(self._device if self._device is not None else sd.default.device[0])
        log.info("mic.started", device=str(info["name"]), sample_rate=SAMPLE_RATE)

    def stop(self) -> None:
        """Close the device. Idempotent. This is the kill switch."""
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
            log.info("mic.stopped", dropped_frames=self._dropped)
        self._drain()

    def __enter__(self) -> Microphone:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- capture -------------------------------------------------------------

    def _on_audio(self, indata: Any, _frames: int, _time: Any, status: sd.CallbackFlags) -> None:
        if status:
            # Overflow here means the OS buffer filled: usually another
            # process hogging the device, or the machine swapping.
            log.warning("mic.stream_status", status=str(status))
        try:
            self._queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            # Drop the newest frame rather than block the callback. Dropping
            # is audible as a gap; blocking is audible as a crackle across the
            # whole stream, and can wedge PortAudio.
            self._dropped += 1
            if self._dropped % 50 == 1:
                log.warning("mic.frames_dropped", total=self._dropped)

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def read(self, timeout: float = 1.0) -> AudioArray | None:
        """Return the next frame, or None if none arrived within `timeout`."""
        if self._stream is None:
            raise MicrophoneClosedError("Microphone is stopped.")
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def frames(self, timeout: float = 1.0) -> Iterator[AudioArray]:
        """Yield frames until the microphone is stopped."""
        while self.is_open:
            frame = self.read(timeout=timeout)
            if frame is not None:
                yield frame

    @property
    def dropped_frames(self) -> int:
        return self._dropped


class RingBuffer:
    """Fixed-length rolling audio history.

    Exists so that the audio *preceding* a wake word is available. Detection
    fires a beat after the phrase completes, and users habitually run the
    request into the trigger ("Hey Jarvis what's the weather"), so the first
    word is already in the past by the time listening starts.
    """

    def __init__(self, seconds: float, sample_rate: int = SAMPLE_RATE) -> None:
        self._capacity = int(seconds * sample_rate)
        self._buf: AudioArray = np.zeros(self._capacity, dtype=np.float32)
        self._filled = 0
        self._pos = 0

    def write(self, frame: AudioArray) -> None:
        n = len(frame)
        if n >= self._capacity:
            self._buf[:] = frame[-self._capacity :]
            self._pos = 0
            self._filled = self._capacity
            return
        end = self._pos + n
        if end <= self._capacity:
            self._buf[self._pos : end] = frame
        else:
            split = self._capacity - self._pos
            self._buf[self._pos :] = frame[:split]
            self._buf[: end - self._capacity] = frame[split:]
        self._pos = end % self._capacity
        self._filled = min(self._capacity, self._filled + n)

    def read(self) -> AudioArray:
        """Return the buffered audio in chronological order."""
        if self._filled < self._capacity:
            return self._buf[: self._filled].copy()
        return np.concatenate([self._buf[self._pos :], self._buf[: self._pos]])

    def clear(self) -> None:
        self._filled = 0
        self._pos = 0


def list_devices() -> list[dict[str, Any]]:
    """Input devices, for `aura voice devices`."""
    out: list[dict[str, Any]] = []
    default_in = sd.default.device[0]
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            out.append(
                {
                    "index": idx,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "sample_rate": int(dev["default_samplerate"]),
                    "default": idx == default_in,
                }
            )
    return out
