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

import re
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from aura.config import Settings
from aura.logging import get_logger, new_request_id
from aura.voice.audio.capture import Microphone, RingBuffer
from aura.voice.audio.cues import dismiss_cue, follow_up_cue, wake_cue
from aura.voice.audio.playback import Speaker
from aura.voice.core_client import CoreClient, CoreUnavailableError
from aura.voice.stt.base import STTProvider
from aura.voice.tts.base import TTSProvider
from aura.voice.tts.normalize import for_speech
from aura.voice.tts.piper import SentenceStreamer
from aura.voice.types import SAMPLE_RATE, AudioArray, TurnTimings, VoiceState
from aura.voice.vad import BargeInDetector, Endpointer
from aura.voice.wake.base import WakeWordDetector

log = get_logger(__name__)

# Speech shorter than this after a wake word is treated as a false trigger
# rather than a failed utterance, so AURA stays silent instead of
# apologising to an empty room.
_FEEDBACK_SPEECH_S = 0.4

# The pre-roll buffer intentionally keeps audio from before the wake word
# fired, so the transcript usually starts with the trigger phrase itself.
# Sending "Hey Jarvis, what is X?" to the model wastes context and
# occasionally derails the answer, so strip it.
_WAKE_PREFIX = re.compile(
    r"^\s*(?:hey|hi|ok|okay)?[\s,]*"
    r"(?:jarvis|aura|mycroft|rhasspy|alexa)\b[\s,.!?-]*",
    re.IGNORECASE,
)


def strip_wake_phrase(text: str) -> str:
    """Remove a leading wake phrase from a transcript.

    Only strips a prefix, and only if something is left: "Hey Jarvis" on
    its own stays intact so it reads as a greeting rather than an empty
    prompt.
    """
    stripped = _WAKE_PREFIX.sub("", text, count=1).strip()
    return stripped or text.strip()


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
        self._speaking_since = 0.0
        self._collector: Any = None
        # Whether the turn that just ended earns a follow-up window. A turn
        # that errored or heard nothing does not: reopening the microphone
        # after a failure invites a second failure.
        self._follow_up_ok = False
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
                        # Cue first, before anything that could take time. This
                        # is the user's only confirmation that AURA woke up.
                        self._play_cue(wake_cue)
                        self._on_text("system", "listening")
                        self._collector = self._endpointer.begin(preroll=self._ring.read())
                        self._set_state(VoiceState.LISTENING)

                elif self._state in (VoiceState.LISTENING, VoiceState.FOLLOW_UP):
                    assert self._collector is not None
                    segment = self._collector.push(frame)
                    if segment is not None:
                        was_follow_up = self._state is VoiceState.FOLLOW_UP
                        self._collector = None
                        if not segment.has_speech:
                            # In a follow-up window this is the normal ending:
                            # the user had nothing more to ask.
                            log.info(
                                "voice.follow_up_expired" if was_follow_up else "voice.false_wake",
                                ended_by=segment.ended_by,
                            )
                            self._play_cue(dismiss_cue)
                            self._ring.clear()
                            self._set_state(VoiceState.SLEEP)
                        else:
                            self._start_turn(segment.audio, segment.speech_s)

                elif self._state in (VoiceState.THINKING, VoiceState.SPEAKING):
                    self._check_barge_in(frame)
        finally:
            self._shutdown()

    def _play_cue(self, make_cue: Callable[..., Any]) -> None:
        """Play a short cue. Never lets an audio failure break the loop."""
        if not self._cfg.wake_cue:
            return
        try:
            self._speaker.play(make_cue(volume=self._cfg.cue_volume))
        except Exception as exc:
            log.warning("voice.cue_failed", error=str(exc))

    def _reap_finished_turn(self) -> None:
        """Once the reply worker exits, either listen again or go to sleep."""
        if self._worker is None or self._worker.is_alive():
            return
        self._worker = None
        # The ring buffer holds AURA's own voice from the frames captured while
        # it was speaking. Clearing it is what keeps that audio out of the
        # follow-up utterance.
        self._ring.clear()
        self._barge.reset()
        self._speaking_since = 0.0
        if self._state not in (VoiceState.THINKING, VoiceState.SPEAKING):
            return

        if self._cfg.follow_up_enabled and self._follow_up_ok:
            self._begin_follow_up()
        else:
            self._set_state(VoiceState.SLEEP)

    def _begin_follow_up(self) -> None:
        """Listen again without requiring the wake word.

        Entered with no pre-roll: the buffered audio is AURA's own reply, and
        feeding that to Whisper would transcribe the assistant talking to
        itself.
        """
        self._play_cue(follow_up_cue)
        self._collector = self._endpointer.begin(start_timeout_s=self._cfg.follow_up_timeout_s)
        log.info("voice.follow_up_open", timeout_s=self._cfg.follow_up_timeout_s)
        self._on_text("system", "still listening")
        self._set_state(VoiceState.FOLLOW_UP)

    def _check_barge_in(self, frame: AudioArray) -> None:
        """Interrupt playback when the *user* talks over AURA.

        The hard part is that AURA hears itself. Without acoustic echo
        cancellation, speaker output bleeds into the microphone and the VAD
        classifies it as speech -- perfectly correctly, since it is speech.
        Observed live: AURA cut itself off one second into its own answer.

        Two cheap defences instead of an AEC, which is a project of its own:

        1. A grace period after playback starts. Echo begins the instant the
           speaker does, so ignoring the first moments removes the common case.
        2. A loudness floor. Speaker bleed reaching a laptop's mic array is
           much quieter than a person addressing it, so requiring real level
           separates the two without any signal processing.

        Headphones remove the problem entirely, which is worth saying out loud
        rather than pretending the heuristics are a solution.
        """
        if not self._cfg.barge_in_enabled or self._state is not VoiceState.SPEAKING:
            return
        if self._interrupt.is_set():
            # Already interrupted; the worker is winding down. Re-firing here
            # just fills the log with duplicates.
            return

        if self._speaking_since and (
            time.monotonic() - self._speaking_since < self._cfg.barge_in_grace_s
        ):
            return

        rms = float(np.sqrt(np.mean(np.square(frame))))
        if rms < self._cfg.barge_in_min_rms:
            # Present but too quiet to be someone talking to us. Feed the
            # detector nothing so echo cannot accumulate toward the sustain
            # threshold.
            self._barge.reset()
            return

        if self._barge.process(frame):
            log.info("voice.barge_in", rms=round(rms, 4))
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
        self._follow_up_ok = False
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
                # If there was real speech but nothing came back, say so.
                # Silence here is indistinguishable from AURA being broken:
                # the user talked, something was heard, and nothing happened.
                # A false wake with no speech stays quiet, because announcing
                # every stray trigger would be worse than the problem.
                if speech_s >= _FEEDBACK_SPEECH_S:
                    self._on_text("system", "did not catch that")
                    self._speak_now("Sorry, I didn't catch that.")
                return
            prompt = strip_wake_phrase(transcript.text)
            self._on_text("user", prompt)

            streamer = SentenceStreamer()
            spoke_anything = False
            first_token_at = 0.0

            for delta in self._core.stream_reply(prompt):
                if self._interrupt.is_set():
                    log.info("voice.turn_interrupted")
                    # The user talked over the answer, so they clearly have
                    # something to say. Reopen the microphone rather than
                    # making them start again with the wake word.
                    self._follow_up_ok = True
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
                # An answer was actually delivered, so a follow-up question is
                # a natural next move.
                self._follow_up_ok = True
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
        """Synthesise and queue one sentence. True if audio was produced.

        The text is normalised for speech first. The prompt asks the model for
        prose, but prompting is a request rather than a guarantee, and a
        synthesiser handed `**bold**` says "star star bold star star".
        """
        spoken_text = for_speech(text)
        if not spoken_text:
            # Markup with nothing behind it - a lone bullet or a rule.
            return False
        produced = False
        started = time.perf_counter()
        for chunk in self._tts.synthesize(spoken_text):
            if self._interrupt.is_set():
                return produced
            if not produced:
                produced = True
                if first:
                    timings.tts_first_chunk_ms = int((time.perf_counter() - started) * 1000)
                    timings.total_to_first_audio_ms = int(
                        (time.perf_counter() - turn_started) * 1000
                    )
                    self._speaking_since = time.monotonic()
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
