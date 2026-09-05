"""The voice session loop.

Threading model, which is the part worth understanding:

- **Main thread** owns the microphone and the state machine. It must never
  block for long, because it is also the barge-in detector: if it stalls while
  AURA is speaking, the user cannot interrupt.
- **Reply worker** (one per turn) does everything slow -- STT, the HTTP stream
  to AURA Core, TTS synthesis -- and pushes audio into the speaker queue.
- **PortAudio callbacks** own the actual device I/O and are never blocked by
  either of the above.

That split is what makes barge-in work. A single-threaded loop that transcribes
and then streams a reply cannot listen while it does so, and an assistant that
cannot be interrupted is a worse experience than one that is merely slow.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from aura.config import Settings
from aura.logging import get_logger, new_request_id
from aura.voice.audio.capture import Microphone, RingBuffer
from aura.voice.audio.playback import Speaker
from aura.voice.core_client import CoreClient, CoreUnavailableError
from aura.voice.stt.base import STTProvider
from aura.voice.tts.base import TTSProvider
from aura.voice.tts.piper import SentenceStreamer
from aura.voice.types import SAMPLE_RATE, AudioArray, TurnTimings, VoiceState
from aura.voice.vad import BargeInDetector, Endpointer
from aura.voice.wake.base import WakeWordDetector

log = get_logger(__name__)

StateCallback = Callable[[VoiceState], None]
TextCallback = Callable[[str, str], None]  # (speaker, text)


class VoiceSession:
    """Wake word -> listen -> think -> speak, forever."""

    def __init__(
        self,
        settings: Settings,
        wake: WakeWordDetector,
        stt: STTProvider,
        tts: TTSProvider,
        core: CoreClient,
        on_state: StateCallback | None = None,
        on_text: TextCallback | None = None,
        mic: Microphone | None = None,
        speaker: Speaker | None = None,
        endpointer: Endpointer | None = None,
        barge_in: BargeInDetector | None = None,
    ) -> None:
        """Audio devices and the VAD are injectable.

        That lets the state machine be driven from scripted frames with a
        deterministic voice detector, so a test of "does a false wake
        return to sleep" asserts on the loop rather than on whether Silero
        happens to classify synthetic audio as speech.
        """
        self._cfg = settings.voice
        self._wake = wake
        self._stt = stt
        self._tts = tts
        self._core = core
        self._on_state = on_state or (lambda _s: None)
        self._on_text = on_text or (lambda _who, _t: None)

        self._mic = mic or Microphone(device=self._cfg.input_device or None)
        self._speaker = speaker or Speaker(device=self._cfg.output_device or None)
        self._ring = RingBuffer(seconds=self._cfg.preroll_s)
        self._endpointer = endpointer or Endpointer(
            silence_s=self._cfg.endpoint_silence_s,
            max_duration_s=self._cfg.max_utterance_s,
            start_timeout_s=self._cfg.listen_timeout_s,
        )
        self._barge = barge_in or BargeInDetector(sustain_s=self._cfg.barge_in_s)

        self._state = VoiceState.SLEEP
        self._stop = threading.Event()
        self._interrupt = threading.Event()
        self._worker: threading.Thread | None = None
        self._last_activity = time.monotonic()
        self.timings = TurnTimings()

    # -- state ---------------------------------------------------------------

    @property
    def state(self) -> VoiceState:
        return self._state

    def _set_state(self, state: VoiceState) -> None:
        if state is self._state:
            return
        self._state = state
        log.info("voice.state", state=state.value)
        self._on_state(state)

    # -- microphone kill switch (spec section 23) ----------------------------

    @property
    def microphone_open(self) -> bool:
        return self._mic.is_open

    def mute(self) -> None:
        """Stop capturing. Closes the device; does not merely ignore audio."""
        self._mic.stop()
        self._set_state(VoiceState.SLEEP)
        log.warning("voice.microphone_off")

    def unmute(self) -> None:
        self._mic.start()
        log.info("voice.microphone_on")

    def stop(self) -> None:
        self._stop.set()
        self._interrupt.set()

    # -- main loop -----------------------------------------------------------

    def run(self) -> None:
        """Block until stop() is called."""
        self._wake.load()
        self._endpointer.vad.load()
        self._mic.start()
        self._set_state(VoiceState.SLEEP)

        collector = None
        try:
            while not self._stop.is_set():
                # Reap a finished turn before touching the microphone. Doing
                # this only when a frame arrives couples a state transition to
                # audio flowing, so a stalled or unplugged device would strand
                # the session in THINKING forever.
                self._reap_finished_turn()

                frame = self._mic.read(timeout=0.25)
                if frame is None:
                    self._maybe_unload_idle()
                    continue
                self._ring.write(frame)

                if self._state is VoiceState.SLEEP:
                    if self._wake.process(frame) is not None:
                        self._on_text("system", "listening")
                        collector = self._endpointer.begin(preroll=self._ring.read())
                        self._set_state(VoiceState.LISTENING)

                elif self._state is VoiceState.LISTENING:
                    assert collector is not None
                    segment = collector.push(frame)
                    if segment is not None:
                        collector = None
                        if not segment.has_speech:
                            log.info("voice.false_wake", ended_by=segment.ended_by)
                            self._ring.clear()
                            self._set_state(VoiceState.SLEEP)
                        else:
                            self._start_turn(segment.audio, segment.speech_s)

                elif self._state in (VoiceState.THINKING, VoiceState.SPEAKING):
                    self._check_barge_in(frame)
        finally:
            self._shutdown()

    def _reap_finished_turn(self) -> None:
        """Return to SLEEP once the reply worker has exited."""
        if self._worker is None or self._worker.is_alive():
            return
        self._worker = None
        self._ring.clear()
        self._barge.reset()
        if self._state in (VoiceState.THINKING, VoiceState.SPEAKING):
            self._set_state(VoiceState.SLEEP)

    def _check_barge_in(self, frame: AudioArray) -> None:
        if not self._cfg.barge_in_enabled or self._state is not VoiceState.SPEAKING:
            return
        if self._interrupt.is_set():
            # Already interrupted; the worker is winding down. Re-firing here
            # just fills the log with duplicates.
            return
        if self._barge.process(frame):
            log.info("voice.barge_in")
            self._on_text("system", "interrupted")
            self._interrupt.set()
            self._speaker.stop()

    def _maybe_unload_idle(self) -> None:
        """Free STT after a quiet spell (spec section 26).

        On a machine with under a gigabyte free, holding 150 MB of Whisper
        weights for a conversation that ended twenty minutes ago is the
        difference between AURA being welcome and being closed.
        """
        if self._cfg.idle_unload_s <= 0 or not self._stt.is_loaded:
            return
        if time.monotonic() - self._last_activity > self._cfg.idle_unload_s:
            self._stt.unload()
            self._last_activity = time.monotonic()

    # -- one turn ------------------------------------------------------------

    def _start_turn(self, audio: AudioArray, speech_s: float) -> None:
        self._interrupt.clear()
        self._barge.reset()
        self._set_state(VoiceState.THINKING)
        self._worker = threading.Thread(
            target=self._run_turn, args=(audio, speech_s), name="aura-voice-turn", daemon=True
        )
        self._worker.start()

    def _run_turn(self, audio: AudioArray, speech_s: float) -> None:
        """STT -> AURA Core -> TTS -> speaker. Runs off the main loop."""
        new_request_id()
        turn_started = time.perf_counter()
        timings = TurnTimings()

        try:
            transcript = self._stt.transcribe(audio)
            timings.stt_ms = transcript.latency_ms
            self._last_activity = time.monotonic()

            if transcript.is_empty:
                log.info("voice.nothing_heard", speech_s=round(speech_s, 2))
                return
            self._on_text("user", transcript.text)

            streamer = SentenceStreamer()
            spoke_anything = False
            first_token_at = 0.0

            for delta in self._core.stream_reply(transcript.text):
                if self._interrupt.is_set():
                    log.info("voice.turn_interrupted")
                    return
                if not first_token_at:
                    first_token_at = time.perf_counter()
                    timings.llm_first_token_ms = int(
                        (first_token_at - turn_started) * 1000 - timings.stt_ms
                    )
                for sentence in streamer.push(delta):
                    if self._speak(sentence, timings, turn_started, first=not spoke_anything):
                        spoke_anything = True
                    if self._interrupt.is_set():
                        return

            for sentence in streamer.flush():
                if self._interrupt.is_set():
                    return
                if self._speak(sentence, timings, turn_started, first=not spoke_anything):
                    spoke_anything = True

            if spoke_anything:
                # Wait for playback to drain, but keep checking for an
                # interrupt so a barge-in during the tail still cuts through.
                while self._speaker.is_playing and not self._interrupt.is_set():
                    self._speaker.wait_until_idle(timeout=0.1)

        except CoreUnavailableError as exc:
            log.error("voice.core_unavailable", error=str(exc))
            self._on_text("system", str(exc))
            self._speak_now("I can't reach AURA Core right now.")
        except Exception as exc:
            log.error("voice.turn_failed", error=str(exc), error_type=type(exc).__name__)
            self._on_text("system", f"error: {exc}")
        finally:
            self.timings = timings
            self._last_activity = time.monotonic()
            log.info(
                "voice.turn_complete",
                stt_ms=timings.stt_ms,
                llm_first_token_ms=timings.llm_first_token_ms,
                tts_first_chunk_ms=timings.tts_first_chunk_ms,
                to_first_audio_ms=timings.total_to_first_audio_ms,
            )

    def _speak(self, text: str, timings: TurnTimings, turn_started: float, first: bool) -> bool:
        """Synthesise and queue one sentence. True if audio was produced."""
        produced = False
        started = time.perf_counter()
        for chunk in self._tts.synthesize(text):
            if self._interrupt.is_set():
                return produced
            if not produced:
                produced = True
                if first:
                    timings.tts_first_chunk_ms = int((time.perf_counter() - started) * 1000)
                    timings.total_to_first_audio_ms = int(
                        (time.perf_counter() - turn_started) * 1000
                    )
                    self._set_state(VoiceState.SPEAKING)
                    self._on_text("aura", text)
                else:
                    self._on_text("aura", text)
            self._speaker.play(chunk)
        return produced

    def _speak_now(self, text: str) -> None:
        """Say something immediately, outside the normal turn flow."""
        try:
            self._set_state(VoiceState.SPEAKING)
            for chunk in self._tts.synthesize(text):
                self._speaker.play(chunk)
            self._speaker.wait_until_idle(timeout=10)
        except Exception as exc:
            log.error("voice.speak_failed", error=str(exc))

    def _shutdown(self) -> None:
        self._interrupt.set()
        self._speaker.stop()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=3.0)
        self._speaker.close()
        self._mic.stop()
        self._core.close()
        self._set_state(VoiceState.SLEEP)
        log.info("voice.stopped", dropped_frames=self._mic.dropped_frames)


def utterance_seconds(samples: int) -> float:
    return samples / SAMPLE_RATE
