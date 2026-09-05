"""The voice state machine, driven by scripted audio.

No microphone, no speaker, no models. A fake mic yields frames, the null wake
detector fires on command, and the null STT returns whatever we queued -- so
these tests assert on the *loop*: does a wake word start listening, does
silence end an utterance, does a false wake return to sleep, does barge-in cut
playback.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

import numpy as np
import pytest

from aura.config import (
    Settings,
    STTConfig,
    TTSConfig,
    VoiceConfig,
    WakeConfig,
)
from aura.voice.session import VoiceSession
from aura.voice.stt.null import NullSTT
from aura.voice.tts.null import NullTTS
from aura.voice.types import VoiceState
from aura.voice.vad import BargeInDetector, Endpointer
from aura.voice.wake.null import NullWakeWord

FRAME = 1280  # 80 ms


def speech_frame() -> Any:
    """A frame the StubVAD treats as speech (marked by a non-zero amplitude)."""
    return np.full(FRAME, 0.5, dtype=np.float32)


def silent_frame() -> Any:
    return np.zeros(FRAME, dtype=np.float32)


class StubVAD:
    """Deterministic stand-in for Silero.

    Speech is "the frame is not all zeros". Real VAD behaviour is covered
    separately against real audio; here the point is to test the loop, not the
    model, and a state machine test that depends on an ONNX model's judgement
    of synthetic audio is a flaky test.
    """

    def load(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def is_speech(self, frame: Any) -> bool:
        return bool(np.any(np.abs(frame) > 0.01))

    def process(self, frame: Any) -> float:
        return 1.0 if self.is_speech(frame) else 0.0


class FakeMicrophone:
    """Yields queued frames, then silence forever."""

    def __init__(self) -> None:
        self._q: queue.Queue[Any] = queue.Queue()
        self.is_open = True
        self.dropped_frames = 0
        self.stop_calls = 0

    def feed(self, frame: Any, times: int = 1) -> None:
        for _ in range(times):
            self._q.put(frame)

    def start(self) -> None:
        self.is_open = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.is_open = False

    def read(self, timeout: float = 1.0) -> Any:
        """Blocks when empty rather than inventing silence.

        Returning silence on an empty queue made the loop race ahead and blow
        through the listen timeout before the test had fed anything.
        """
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


class FakeSpeaker:
    """Records what it was asked to play."""

    def __init__(self) -> None:
        self.chunks: list[Any] = []
        self.stop_calls = 0
        self.closed = False
        self.is_playing = False

    def play(self, chunk: Any) -> None:
        self.chunks.append(chunk)

    def stop(self) -> None:
        self.stop_calls += 1
        self.is_playing = False

    def close(self) -> None:
        self.closed = True

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        return True


class FakeCore:
    """Stands in for AURA Core's SSE stream."""

    def __init__(self, reply: str = "Hello there. How can I help?") -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.session_id: str | None = "test-session"
        self.delay_s = 0.0

    def stream_reply(self, text: str) -> Any:
        self.prompts.append(text)
        for word in self.reply.split(" "):
            if self.delay_s:
                time.sleep(self.delay_s)
            yield word + " "

    def close(self) -> None:
        pass


def make_settings(**voice_overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "endpoint_silence_s": 0.2,
        "max_utterance_s": 3.0,
        "listen_timeout_s": 0.5,
        "preroll_s": 0.2,
        "barge_in_s": 0.15,
        "barge_in_grace_s": 0.15,
        "follow_up_enabled": True,
        "follow_up_timeout_s": 0.5,
        "barge_in_min_rms": 0.02,
        "idle_unload_s": 0.0,
        "wake": WakeConfig(model="null"),
        "stt": STTConfig(provider="null"),
        "tts": TTSConfig(provider="null"),
    }
    defaults.update(voice_overrides)
    return Settings(voice=VoiceConfig(**defaults))


def run_session(session: VoiceSession, seconds: float = 3.0) -> threading.Thread:
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    return thread


def wait_for(predicate: Any, timeout: float = 4.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _stub_barge_in(sustain_s: float) -> BargeInDetector:
    """A barge-in detector wired to the deterministic VAD."""
    detector = BargeInDetector(sustain_s=sustain_s)
    detector._vad = StubVAD()  # type: ignore[assignment]
    return detector


@pytest.fixture
def rig() -> Any:
    mic, speaker, core = FakeMicrophone(), FakeSpeaker(), FakeCore()
    wake, stt, tts = NullWakeWord(), NullSTT(), NullTTS()
    states: list[VoiceState] = []
    texts: list[tuple[str, str]] = []
    cfg = make_settings()
    stub_vad = StubVAD()
    session = VoiceSession(
        cfg,
        wake,
        stt,
        tts,
        core,  # type: ignore[arg-type]
        on_state=states.append,
        on_text=lambda who, t: texts.append((who, t)),
        mic=mic,  # type: ignore[arg-type]
        speaker=speaker,  # type: ignore[arg-type]
        endpointer=Endpointer(
            silence_s=cfg.voice.endpoint_silence_s,
            max_duration_s=cfg.voice.max_utterance_s,
            start_timeout_s=cfg.voice.listen_timeout_s,
            vad=stub_vad,  # type: ignore[arg-type]
        ),
        barge_in=_stub_barge_in(cfg.voice.barge_in_s),
    )
    return {
        "session": session,
        "mic": mic,
        "speaker": speaker,
        "core": core,
        "wake": wake,
        "stt": stt,
        "tts": tts,
        "states": states,
        "texts": texts,
    }


def test_starts_asleep(rig: Any) -> None:
    assert rig["session"].state is VoiceState.SLEEP


def test_wake_word_starts_listening(rig: Any) -> None:
    session, mic, wake = rig["session"], rig["mic"], rig["wake"]
    thread = run_session(session)
    try:
        wake.trigger()
        mic.feed(silent_frame())
        assert wait_for(lambda: session.state is VoiceState.LISTENING)
    finally:
        session.stop()
        thread.join(timeout=3)


def test_a_full_turn_transcribes_asks_core_and_speaks(rig: Any) -> None:
    session, mic, wake, stt, core = (
        rig["session"],
        rig["mic"],
        rig["wake"],
        rig["stt"],
        rig["core"],
    )
    stt.push("what is the time")
    thread = run_session(session)
    try:
        wake.trigger()
        mic.feed(silent_frame())
        assert wait_for(lambda: session.state is VoiceState.LISTENING)

        mic.feed(speech_frame(), times=12)  # ~1 s of speech
        mic.feed(silent_frame(), times=10)  # silence ends the utterance

        assert wait_for(lambda: core.prompts, timeout=6.0), "Core was never asked"
        assert core.prompts == ["what is the time"]
        assert wait_for(lambda: rig["tts"].spoken, timeout=6.0), "nothing was spoken"
        assert wait_for(lambda: rig["speaker"].chunks, timeout=6.0)
    finally:
        session.stop()
        thread.join(timeout=3)


def test_false_wake_returns_to_sleep_without_calling_core(rig: Any) -> None:
    """A wake word with no speech after it must not reach the model."""
    session, mic, wake, core = rig["session"], rig["mic"], rig["wake"], rig["core"]
    thread = run_session(session)
    try:
        wake.trigger()
        mic.feed(silent_frame())
        assert wait_for(lambda: session.state is VoiceState.LISTENING)

        mic.feed(silent_frame(), times=20)  # past listen_timeout_s
        assert wait_for(lambda: session.state is VoiceState.SLEEP, timeout=5.0)
        assert core.prompts == [], "a false wake reached AURA Core"
    finally:
        session.stop()
        thread.join(timeout=3)


def test_empty_transcript_does_not_reach_core(rig: Any) -> None:
    """Speech that transcribes to nothing must not become a prompt."""
    session, mic, wake, core = rig["session"], rig["mic"], rig["wake"], rig["core"]
    # NullSTT with an empty queue returns an empty transcript.
    thread = run_session(session)
    try:
        wake.trigger()
        mic.feed(silent_frame())
        assert wait_for(lambda: session.state is VoiceState.LISTENING)
        mic.feed(speech_frame(), times=12)
        mic.feed(silent_frame(), times=10)
        assert wait_for(lambda: session.state is VoiceState.SLEEP, timeout=6.0)
        assert core.prompts == []
    finally:
        session.stop()
        thread.join(timeout=3)


def test_mute_closes_the_microphone(rig: Any) -> None:
    """Spec section 23: the kill switch stops capture, it does not gate it."""
    session, mic = rig["session"], rig["mic"]
    session.mute()
    assert mic.stop_calls == 1
    assert not mic.is_open
    assert session.state is VoiceState.SLEEP


def test_shutdown_closes_devices(rig: Any) -> None:
    session, mic, speaker = rig["session"], rig["mic"], rig["speaker"]
    thread = run_session(session)
    time.sleep(0.1)
    session.stop()
    thread.join(timeout=3)
    assert speaker.closed
    assert mic.stop_calls >= 1


def test_barge_in_stops_playback_mid_reply(rig: Any) -> None:
    """Speaking over AURA must cut it off immediately.

    An assistant that finishes its sentence after being interrupted is worse
    than one that never started, so this is the behaviour the whole threading
    model exists to support.
    """
    session, mic, wake, stt, core, speaker = (
        rig["session"],
        rig["mic"],
        rig["wake"],
        rig["stt"],
        rig["core"],
        rig["speaker"],
    )
    stt.push("tell me a long story")
    core.reply = " ".join(f"Sentence number {i}." for i in range(1, 40))
    core.delay_s = 0.03  # outlast the barge-in grace period

    thread = run_session(session)
    try:
        wake.trigger()
        mic.feed(silent_frame())
        assert wait_for(lambda: session.state is VoiceState.LISTENING)

        mic.feed(speech_frame(), times=8)
        mic.feed(silent_frame(), times=8)

        assert wait_for(lambda: session.state is VoiceState.SPEAKING, timeout=6.0), (
            "never reached SPEAKING"
        )
        spoken_before = len(rig["tts"].spoken)

        # Wait out the grace period, then talk over the reply.
        time.sleep(0.2)
        mic.feed(speech_frame(), times=20)

        assert wait_for(lambda: speaker.stop_calls > 0, timeout=5.0), (
            "playback was not stopped by barge-in"
        )
        # Interrupting means the user has something to say, so AURA reopens the
        # microphone rather than making them start again with the wake word.
        # (What happens next depends on whether they keep talking, which
        # test_follow_up_expires_into_sleep_when_nothing_is_said covers.)
        assert wait_for(lambda: session.state is VoiceState.FOLLOW_UP, timeout=6.0)

        # And it stopped early rather than running to the end of the reply.
        assert len(rig["tts"].spoken) < 39, (
            f"spoke {len(rig['tts'].spoken)} sentences; barge-in did not cut the reply short"
        )
        assert len(rig["tts"].spoken) >= spoken_before
    finally:
        session.stop()
        thread.join(timeout=3)


def test_barge_in_can_be_disabled(rig: Any) -> None:
    """With barge_in_enabled off, speech during playback must not interrupt."""
    session = rig["session"]
    session._cfg = session._cfg.model_copy(update={"barge_in_enabled": False})
    session._set_state(VoiceState.SPEAKING)
    session._check_barge_in(speech_frame())
    assert rig["speaker"].stop_calls == 0


def test_quiet_audio_does_not_trigger_barge_in(rig: Any) -> None:
    """AURA's own voice bleeding into the mic must not interrupt it.

    Without acoustic echo cancellation the VAD correctly classifies speaker
    output as speech. The loudness floor is what separates "AURA hearing
    itself" from "the user talking", and it is why AURA cut itself off one
    second into its first real answer before this guard existed.
    """
    session = rig["session"]
    session._speaking_since = time.monotonic() - 10.0
    session._set_state(VoiceState.SPEAKING)

    quiet = np.full(FRAME, 0.005, dtype=np.float32)  # below barge_in_min_rms
    for _ in range(30):
        session._check_barge_in(quiet)
    assert rig["speaker"].stop_calls == 0, "quiet echo interrupted playback"


def test_grace_period_suppresses_immediate_echo(rig: Any) -> None:
    """Echo starts the instant the speaker does, so the first moments are ignored."""
    session = rig["session"]
    session._speaking_since = time.monotonic()
    session._set_state(VoiceState.SPEAKING)

    for _ in range(30):
        session._check_barge_in(speech_frame())
    assert rig["speaker"].stop_calls == 0, "barge-in fired inside the grace period"


def test_loud_speech_after_grace_does_trigger_barge_in(rig: Any) -> None:
    """The guards must not make barge-in unusable."""
    session = rig["session"]
    session._speaking_since = time.monotonic() - 10.0
    session._set_state(VoiceState.SPEAKING)

    fired = False
    for _ in range(30):
        session._check_barge_in(speech_frame())
        if rig["speaker"].stop_calls:
            fired = True
            break
    assert fired, "loud sustained speech after the grace period should interrupt"


# -- follow-up window --------------------------------------------------------
# Without this, every question starts with the wake word, which makes AURA feel
# like a vending machine instead of something you are talking to.


def _drive_one_turn(rig: Any, said: str) -> None:
    """Wake, speak, and let one full turn complete."""
    rig["stt"].push(said)
    rig["wake"].trigger()
    rig["mic"].feed(silent_frame())
    assert wait_for(lambda: rig["session"].state is VoiceState.LISTENING)
    rig["mic"].feed(speech_frame(), times=10)
    rig["mic"].feed(silent_frame(), times=8)


def test_after_answering_it_keeps_listening_without_a_wake_word(rig: Any) -> None:
    session, mic, stt, core = rig["session"], rig["mic"], rig["stt"], rig["core"]
    thread = run_session(session)
    try:
        _drive_one_turn(rig, "first question")
        assert wait_for(lambda: session.state is VoiceState.FOLLOW_UP, timeout=8.0), (
            "did not open a follow-up window after answering"
        )

        # A second question, with no wake word this time.
        stt.push("second question")
        mic.feed(speech_frame(), times=10)
        mic.feed(silent_frame(), times=8)

        assert wait_for(lambda: len(core.prompts) == 2, timeout=8.0), (
            f"follow-up never reached Core; prompts={core.prompts}"
        )
        assert core.prompts == ["first question", "second question"]
        # And it stays one conversation, so context carries over.
        assert core.session_id == "test-session"
    finally:
        session.stop()
        thread.join(timeout=3)


def test_follow_up_expires_into_sleep_when_nothing_is_said(rig: Any) -> None:
    session, mic = rig["session"], rig["mic"]
    thread = run_session(session)
    try:
        _drive_one_turn(rig, "only question")
        assert wait_for(lambda: session.state is VoiceState.FOLLOW_UP, timeout=8.0)

        mic.feed(silent_frame(), times=25)  # past follow_up_timeout_s
        assert wait_for(lambda: session.state is VoiceState.SLEEP, timeout=8.0), (
            "follow-up window never closed"
        )
    finally:
        session.stop()
        thread.join(timeout=3)


def test_no_follow_up_when_nothing_was_heard(rig: Any) -> None:
    """A failed turn must not reopen the microphone.

    Whatever caused the failure -- noise, a false wake, a bad transcript -- is
    likely still happening, so listening again just invites a second failure.
    """
    session, mic, wake = rig["session"], rig["mic"], rig["wake"]
    thread = run_session(session)
    try:
        # NullSTT with an empty queue returns an empty transcript.
        wake.trigger()
        mic.feed(silent_frame())
        assert wait_for(lambda: session.state is VoiceState.LISTENING)
        mic.feed(speech_frame(), times=10)
        mic.feed(silent_frame(), times=8)

        assert wait_for(lambda: session.state is VoiceState.SLEEP, timeout=8.0)
        assert session.state is not VoiceState.FOLLOW_UP
    finally:
        session.stop()
        thread.join(timeout=3)


def test_follow_up_can_be_disabled(rig: Any) -> None:
    session = rig["session"]
    session._cfg = session._cfg.model_copy(update={"follow_up_enabled": False})
    thread = run_session(session)
    try:
        _drive_one_turn(rig, "one and done")
        assert wait_for(lambda: session.state is VoiceState.SLEEP, timeout=8.0)
    finally:
        session.stop()
        thread.join(timeout=3)
