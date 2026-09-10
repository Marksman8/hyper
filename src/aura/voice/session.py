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
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from aura.config import Settings
from aura.logging import get_logger, new_request_id
from aura.voice.audio.capture import Microphone, RingBuffer
from aura.voice.audio.cues import dismiss_cue, follow_up_cue, wake_cue
from aura.voice.audio.playback import Speaker
from aura.voice.clips import ClipLibrary
from aura.voice.core_client import CoreClient, CoreUnavailableError
from aura.voice.envelope import envelope
from aura.voice.personality import CATALOGUE, Moment, build_engine
from aura.voice.reporter import SpeechReporter
from aura.voice.stt.base import STTProvider
from aura.voice.tts.base import TTSProvider
from aura.voice.tts.normalize import for_speech
from aura.voice.tts.piper import SentenceStreamer
from aura.voice.types import SAMPLE_RATE, AudioArray, SpeechChunk, TurnTimings, VoiceState
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


def is_stop_command(text: str, stop_words: Sequence[str]) -> bool:
    """Is this utterance just "stop", rather than a new question?

    Matched on the whole utterance after stripping punctuation and any wake
    phrase, never as a substring: "stop the build" and "how do I stop a
    container" are questions, not cancellations.
    """
    cleaned = re.sub(r"[^a-z0-9\s]", "", strip_wake_phrase(text).lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned in {w.lower() for w in stop_words}


def _looks_like_a_stop_attempt(text: str, stop_words: Sequence[str]) -> bool:
    """A short utterance that shares a word with the stop list but did not match.

    Purely diagnostic. It changes no behaviour -- guessing at a cancellation
    would be worse than missing one, since acting on a half-heard word means
    cutting off an answer the user wanted.
    """
    cleaned = re.sub(r"[^a-z0-9\s]", "", strip_wake_phrase(text).lower()).strip()
    if not cleaned or is_stop_command(text, stop_words):
        return False
    words = set(cleaned.split())
    if len(words) > 3:
        return False
    vocabulary = {w for phrase in stop_words for w in phrase.lower().split()}
    return bool(words & vocabulary)


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
        # 10B.1. Speech becomes a bus event stream so it can be *seen* as well
        # as heard. Off the speech path entirely -- see reporter.py.
        self._reporter = SpeechReporter(core)
        # True between SPEECH_STARTED and SPEECH_ENDED, so the end is published
        # exactly once however the turn unwinds: finished, interrupted, or
        # thrown out of.
        self._speech_open = False
        # 10B.5/10B.6. Built here, restricted to what has audio in run().
        # Either being None simply means no micro-expressions, which is a
        # supported state and the one an unrendered checkout is in.
        self._personality = build_engine(settings)
        self._clips: ClipLibrary | None = None

        self._state = VoiceState.SLEEP
        self._stop = threading.Event()
        self._interrupt = threading.Event()
        self._worker: threading.Thread | None = None
        self._last_activity = time.monotonic()
        self._speaking_since = 0.0
        self._collector: Any = None
        # Whether the turn that just ended was cut off by the user speaking.
        self._interrupted = False
        # Consecutive barge-ins with no wake word in between. A microphone that
        # can hear the speakers turns barge-in into a feedback loop: AURA
        # interrupts itself, transcribes its own voice, answers the result, and
        # repeats. Counting is what makes that terminate.
        self._barge_streak = 0
        self._barge_disabled = False
        # Microphone levels sampled during the grace period, which is AURA's
        # own voice and nothing else. The median of these is the echo floor.
        self._echo_samples: list[float] = []
        self._barge_threshold = 0.0
        # Loudest thing heard during this reply that was still too quiet to
        # count as an interruption. Logged when the turn ends, because a
        # barge-in that never fires otherwise leaves no evidence at all.
        self._loudest_ignored = 0.0
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
        self._reporter.start()
        # The catalogue lists what AURA could say; the library is what she has
        # audio for. Restricting to the intersection means a phrase added but
        # never rendered is inert rather than a silent pause mid-reply.
        self._clips = ClipLibrary()
        self._clips.load(CATALOGUE)
        self._personality.restrict_to(self._clips.available)
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
                        # A person said the wake word, so whatever barge-ins
                        # came before were not a runaway loop.
                        self._barge_streak = 0
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
                    # Two independent ways to cut her off: by loudness, which
                    # needs a threshold this machine cannot supply, and by
                    # pattern, which needs none. See _wake_word_interrupt.
                    self._wake_word_interrupt(frame)
                    self._check_barge_in(frame)
        finally:
            self._shutdown()

    def _current_barge_threshold(self) -> float:
        """How loud an interruption has to be, for this reply.

        The median of the grace-period samples rather than the maximum: if the
        user did start talking early, that contaminates a minority of frames
        and the median ignores them.
        """
        if self._barge_threshold:
            return self._barge_threshold
        floor = 0.0
        if self._echo_samples:
            ordered = sorted(self._echo_samples)
            floor = ordered[len(ordered) // 2]
        self._barge_threshold = max(
            floor * self._cfg.barge_in_echo_factor, self._cfg.barge_in_min_rms
        )
        log.info(
            "voice.barge_threshold",
            echo_floor=round(floor, 4),
            threshold=round(self._barge_threshold, 4),
            samples=len(self._echo_samples),
        )
        return self._barge_threshold

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
        if self._loudest_ignored:
            # Without this, "I said stop and nothing happened" produces no log
            # line at all, and the only way to find the right threshold is to
            # guess again.
            log.info(
                "voice.barge_in_not_triggered",
                loudest_heard=round(self._loudest_ignored, 4),
                threshold=round(self._barge_threshold, 4),
                hint="raise voice.barge_in_echo_factor if this was you talking",
            )
            self._loudest_ignored = 0.0
        # The ring buffer holds AURA's own voice from the frames captured while
        # it was speaking, so it is cleared before listening again. The
        # exception is a barge-in: the tail of the buffer is then the word the
        # user interrupted with, and throwing it away is what used to make
        # saying "stop" cut the reply off and then wait for a new question.
        interrupt_preroll = None
        if self._interrupted:
            interrupt_preroll = self._ring.read(seconds=self._cfg.barge_in_s + 0.4)
        self._ring.clear()
        self._barge.reset()
        self._speaking_since = 0.0
        if self._state not in (VoiceState.THINKING, VoiceState.SPEAKING):
            return

        if self._cfg.follow_up_enabled and self._follow_up_ok:
            self._begin_follow_up(preroll=interrupt_preroll)
        else:
            self._set_state(VoiceState.SLEEP)

    def _begin_follow_up(self, preroll: AudioArray | None = None) -> None:
        """Listen again without requiring the wake word.

        Normally entered with no pre-roll: the buffered audio is AURA's own
        reply, and feeding that to Whisper would transcribe the assistant
        talking to itself. After a barge-in the caller passes the tail of the
        buffer instead, which is what the user just said.
        """
        self._play_cue(follow_up_cue)
        self._collector = self._endpointer.begin(
            preroll=preroll, start_timeout_s=self._cfg.follow_up_timeout_s
        )
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
        if (
            not self._cfg.barge_in_enabled
            or self._barge_disabled
            or self._state is not VoiceState.SPEAKING
        ):
            return
        if self._interrupt.is_set():
            # Already interrupted; the worker is winding down. Re-firing here
            # just fills the log with duplicates.
            return

        rms = float(np.sqrt(np.mean(np.square(frame))))

        if self._speaking_since and (
            time.monotonic() - self._speaking_since < self._cfg.barge_in_grace_s
        ):
            # Grace period: whatever the microphone hears now is the echo,
            # because the user has not had time to react yet. Sample it.
            self._echo_samples.append(rms)
            return

        threshold = self._current_barge_threshold()
        if rms < threshold:
            self._loudest_ignored = max(self._loudest_ignored, rms)
            # Too quiet to be someone talking to us. Feed the detector nothing,
            # so echo cannot accumulate toward the sustain threshold.
            self._barge.reset()
            return

        if self._barge.process(frame):
            self._barge_streak += 1
            log.info("voice.barge_in", rms=round(rms, 4), streak=self._barge_streak)

            if self._barge_streak >= self._cfg.max_consecutive_barge_ins:
                # Nobody interrupts three answers in a row without ever saying
                # the wake word. This is the microphone hearing the speakers.
                self._barge_disabled = True
                log.warning(
                    "voice.barge_in_feedback_loop",
                    streak=self._barge_streak,
                    rms=round(rms, 4),
                    threshold=self._cfg.barge_in_min_rms,
                    hint=(
                        "AURA is hearing itself. Barge-in is off for this "
                        "session. Run `aura calibrate`, raise "
                        "voice.barge_in_min_rms, or use headphones."
                    ),
                )
                self._on_text("system", "barge-in disabled: hearing myself")
                return

            self._on_text("system", "interrupted")
            self._interrupt_speech("barge_in")

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
        self._interrupted = False
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
            if _looks_like_a_stop_attempt(transcript.text, self._cfg.stop_words):
                # Short, and close to something in the stop list, but not an
                # exact match. Worth a log line: "I said stop and it answered
                # anyway" is otherwise impossible to diagnose from the outside.
                log.info(
                    "voice.stop_word_near_miss",
                    text=transcript.text,
                    hint="add this phrasing to voice.stop_words if it was a cancellation",
                )

            if is_stop_command(transcript.text, self._cfg.stop_words):
                # A cancellation, not a question. Sending "stop" to the model
                # would have it explain what stopping means.
                log.info("voice.stop_command", text=transcript.text)
                self._on_text("user", transcript.text)
                self._on_text("system", "stopped")
                # Through the same route as every other interruption, so the
                # overlay learns about this one too. It did not, before.
                self._interrupt_speech("stop_command")
                self._follow_up_ok = True
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
            # In `finally` rather than at the end of the happy path, because
            # every other way out of this method -- barge-in, Core dropping
            # mid-reply, an unexpected exception -- also stops the audio, and a
            # subtitle whose closing event depends on the turn ending tidily
            # would sit on screen after the room went quiet.
            self._end_speech(completed=not self._interrupt.is_set())
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
                    self._echo_samples = []
                    self._barge_threshold = 0.0
                    self._set_state(VoiceState.SPEAKING)
                    self._on_text("aura", text)
                    self._begin_speech(len(text))
                    # 10B.6. Before the first sentence of a reply and never
                    # mid-reply: an interjection between two sentences reads
                    # as an interruption rather than as an acknowledgement.
                    if first:
                        self._maybe_micro_expression()
                else:
                    self._on_text("aura", text)
                # Published when the audio for this sentence starts, not when
                # the sentence was synthesised: the subtitle should track what
                # is being heard, and synthesis runs ahead of playback.
                #
                # The envelope rides along with it (10B.3). Piper yields a
                # sentence's audio in one piece, so the samples that drive the
                # mouth are already in hand here -- computing it costs no
                # latency and needs no second event.
                self._reporter.chunk(text, envelope(chunk.audio, chunk.sample_rate))
            self._speaker.play(chunk)
        return produced

    # -- interruption (10B.7) --------------------------------------------------

    def _interrupt_speech(self, method: str) -> bool:
        """Stop talking, and say so. The one route out of speaking.

        Every way of cutting AURA off ends here -- barge-in by loudness, the
        wake word spoken over her, a stop command -- so they cannot drift into
        behaving differently. Before this existed each path stopped the speaker
        and did its own subset of the rest, and none of them told anyone it had
        happened.

        The order matters. The speaker stops first, because that is the part
        the user is waiting for; everything after it is bookkeeping and none of
        it should delay silence by a network round trip.

        Returns False if speech was already interrupted, so a detector firing
        twice does not publish twice.
        """
        if self._interrupt.is_set():
            return False

        self._interrupted = True
        self._interrupt.set()
        self._speaker.stop()

        # SPEECH_ENDED(completed=False) rather than True: the overlay clears a
        # cut-off line in 400 ms instead of holding it up to be read, because a
        # fragment left on screen presents itself as the final word.
        self._end_speech(completed=False)
        self._reporter.interrupted(method)
        log.info("voice.interrupted", method=method)
        return True

    def _wake_word_interrupt(self, frame: AudioArray) -> None:
        """Saying the wake word over AURA stops her.

        **This is the interrupt that does not depend on a threshold.**

        Barge-in by loudness has failed three times on this machine, in both
        directions, and `KNOWN-ISSUES.md` records why: AURA's own voice reaches
        the microphone at RMS 0.024-0.040, and if the user's voice arrives at a
        similar level then no threshold can separate them. That is a property
        of the room and the hardware, not a number to tune.

        The wake word sidesteps it entirely by matching a *pattern* rather than
        a level. AURA's own speech does not sound like "hey jarvis", however
        loudly it comes back through the microphone, so echo cannot trigger
        this however bad the coupling is.

        Cheap, because the detector is already loaded and already running --
        it was simply never fed during playback.
        """
        if not self._cfg.wake_interrupt or self._state is not VoiceState.SPEAKING:
            return
        if self._interrupt.is_set():
            return
        if self._wake.process(frame) is None:
            return

        # Reset, or the scores that fired stay in the buffer and immediately
        # re-trigger against the next reply.
        self._wake.reset()
        if self._interrupt_speech("wake_word"):
            self._on_text("system", "interrupted")
            # The user said the wake word, so they have something to ask.
            # Reopening the microphone is the whole point of interrupting.
            self._follow_up_ok = True

    # -- micro-expressions (10B.6) --------------------------------------------

    def _maybe_micro_expression(self) -> None:
        """Prepend a short fixed phrase, if the personality layer offers one.

        Queued ahead of the reply's own audio, so it is heard first. The
        subtitle gets the phrase and its English gloss as a chunk of its own,
        which is what makes it comprehensible with the sound off -- the case
        this whole milestone exists for.

        Everything here is best-effort. A missing clip, an unreadable file or a
        personality layer that is switched off all produce the same outcome:
        the reply is spoken exactly as it would have been anyway.
        """
        if self._personality is None or self._clips is None:
            return
        try:
            choice = self._personality.select(Moment.ACKNOWLEDGING)
            if choice is None:
                return
            loaded = self._clips.get(choice.id)
            if loaded is None:
                return

            audio, rate = loaded
            self._speaker.play(SpeechChunk(audio=audio, sample_rate=rate, text=choice.text))
            # Its own subtitle line and its own envelope, so the mouth moves
            # for the interjection too rather than sitting shut through it.
            self._reporter.chunk(choice.subtitle(), envelope(audio, rate))
            log.info("voice.micro_expression", expression=choice.id, gloss=choice.gloss)
        except Exception as exc:
            # A garnish must never cost the answer.
            log.debug("voice.micro_expression_failed", error=str(exc))

    # -- speech lifecycle (10B.1) --------------------------------------------

    def _begin_speech(self, length: int) -> None:
        """SPEECH_STARTED, once per reply.

        `length` is the FIRST SENTENCE, not the whole reply, and cannot be
        anything else: the model is still generating when the mouth opens.
        That is the price of streaming, and streaming is what meets the latency
        budget. A consumer wanting the full text accumulates SPEECH_CHUNK.
        """
        if self._speech_open:
            return
        self._speech_open = True
        self._reporter.started(length)

    def _end_speech(self, *, completed: bool) -> None:
        """SPEECH_ENDED, once, however the turn unwound.

        Idempotent because the paths out of a turn are many -- finished,
        barge-in, stop word, an exception, Core going away mid-reply -- and a
        subtitle that never got its closing event stays on screen forever.
        """
        if not self._speech_open:
            return
        self._speech_open = False
        self._reporter.ended(completed=completed)

    def _speak_now(self, text: str) -> None:
        """Say something immediately, outside the normal turn flow.

        Reported like any other speech. These are AURA's own words -- "Sorry, I
        didn't catch that" is said out loud in the room -- and a subtitle that
        showed only the replies routed through Core would silently omit exactly
        the utterances a user is most likely to have missed.
        """
        try:
            self._set_state(VoiceState.SPEAKING)
            self._begin_speech(len(text))
            self._reporter.chunk(text)
            for chunk in self._tts.synthesize(text):
                self._speaker.play(chunk)
            self._speaker.wait_until_idle(timeout=10)
        except Exception as exc:
            log.error("voice.speak_failed", error=str(exc))
        finally:
            self._end_speech(completed=True)

    def _shutdown(self) -> None:
        self._interrupt.set()
        self._speaker.stop()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=3.0)
        self._speaker.close()
        self._mic.stop()
        # Before the client it posts through, and after the worker that feeds
        # it: closing in the other order would strand a queued subtitle against
        # a shut client and log a failure on every clean exit.
        self._end_speech(completed=False)
        self._reporter.close()
        self._core.close()
        self._set_state(VoiceState.SLEEP)
        log.info("voice.stopped", dropped_frames=self._mic.dropped_frames)


def utterance_seconds(samples: int) -> float:
    return samples / SAMPLE_RATE
