"""Speaker playback.

Built around one requirement: **stopping must be immediate.** Barge-in is the
difference between a voice assistant and a voice-activated announcement, and an
assistant that finishes its sentence after being interrupted is worse than one
that never started.

So playback is a queue feeding a PortAudio output callback, and `stop()` clears
the queue and drops the current buffer mid-frame. The audible result is a cut,
which is exactly right: the user is already talking over it.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

import numpy as np
import sounddevice as sd

from aura.logging import get_logger
from aura.voice.types import AudioArray, SpeechChunk

log = get_logger(__name__)


class Speaker:
    """Streaming, interruptible audio output."""

    def __init__(self, device: int | str | None = None, sample_rate: int = 22_050) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._queue: queue.Queue[AudioArray | None] = queue.Queue()
        self._stream: sd.OutputStream | None = None
        self._current: AudioArray | None = None
        self._offset = 0
        self._lock = threading.Lock()
        # Set whenever the queue is empty AND nothing is mid-buffer, so callers
        # can await the natural end of speech without polling.
        self._idle = threading.Event()
        self._idle.set()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_playing(self) -> bool:
        return not self._idle.is_set()

    def start(self, sample_rate: int | None = None) -> None:
        """Open the output device. Idempotent.

        `sample_rate` may differ per TTS voice (Piper models are 16/22.05 kHz),
        so reopening at a new rate is supported and cheap.
        """
        with self._lock:
            if sample_rate is not None and sample_rate != self._sample_rate:
                self._close_locked()
                self._sample_rate = sample_rate
            if self._stream is not None:
                return
            self._stream = sd.OutputStream(
                device=self._device,
                channels=1,
                samplerate=self._sample_rate,
                dtype="float32",
                callback=self._on_output,
            )
            self._stream.start()
        log.debug("speaker.started", sample_rate=self._sample_rate)

    def _close_locked(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    def close(self) -> None:
        self.stop()
        with self._lock:
            self._close_locked()

    def _on_output(self, outdata: Any, frames: int, _time: Any, status: Any) -> None:
        if status:
            log.debug("speaker.stream_status", status=str(status))
        written = 0
        while written < frames:
            if self._current is None or self._offset >= len(self._current):
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    # Underrun: fill the rest with silence rather than repeat
                    # the last buffer, which would sound like a stutter.
                    outdata[written:, 0] = 0.0
                    if self._current is None:
                        self._idle.set()
                    self._current = None
                    return
                if nxt is None:
                    outdata[written:, 0] = 0.0
                    self._current = None
                    self._idle.set()
                    return
                self._current = nxt
                self._offset = 0

            take = min(frames - written, len(self._current) - self._offset)
            outdata[written : written + take, 0] = self._current[self._offset : self._offset + take]
            self._offset += take
            written += take

    def play(self, chunk: SpeechChunk) -> None:
        """Queue a chunk. Returns immediately."""
        if chunk.sample_rate != self._sample_rate:
            self.start(sample_rate=chunk.sample_rate)
        elif self._stream is None:
            self.start()
        self._idle.clear()
        self._queue.put(np.asarray(chunk.audio, dtype=np.float32))

    def stop(self) -> None:
        """Cut playback now: drop the queue and the buffer being played."""
        with self._lock:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._current = None
            self._offset = 0
        self._idle.set()

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Block until playback drains naturally. True if it did."""
        return self._idle.wait(timeout=timeout)


def list_output_devices() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    default_out = sd.default.device[1]
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0:
            out.append(
                {
                    "index": idx,
                    "name": dev["name"],
                    "channels": dev["max_output_channels"],
                    "sample_rate": int(dev["default_samplerate"]),
                    "default": idx == default_out,
                }
            )
    return out
