"""Interrupting AURA mid-sentence (10B.7).

Two halves, and only one of them was ever blocked.

**The response** — what happens once an interrupt is detected — needed no
measurement and had never been built: nothing published `USER_INTERRUPTED`, so
the overlay was never told, and each of the three ways to stop her did its own
partial subset of the work.

**The detection by loudness** is still blocked, and `KNOWN-ISSUES.md` explains
why it is not a tuning problem. So the interrupt that matters here is the one
that needs no threshold at all: saying the wake word over her. It matches a
pattern rather than a level, so echo cannot trigger it however bad the acoustic
coupling is.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from aura.voice.types import SAMPLE_RATE, VoiceState

FRAME = np.zeros(int(SAMPLE_RATE * 0.02), dtype=np.float32)


class FakeWake:
    """Fires when told to, and records resets."""

    def __init__(self) -> None:
        self.armed = False
        self.resets = 0
        self.frames = 0

    def process(self, frame: Any) -> object | None:
        self.frames += 1
        if self.armed:
            self.armed = False
            return object()
        return None

    def reset(self) -> None:
        self.resets += 1

    def load(self) -> None:
        return None


class FakeSpeaker:
    def __init__(self) -> None:
        self.stops = 0
        self.is_playing = False

    def stop(self) -> None:
        self.stops += 1

    def play(self, chunk: Any) -> None:
        return None

    def wait_until_idle(self, timeout: float = 0.0) -> None:
        return None

    def close(self) -> None:
        return None


class FakeReporter:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def started(self, length: int) -> None:
        self.events.append(("started", length))

    def chunk(self, text: str, envelope: str = "") -> None:
        self.events.append(("chunk", text))

    def ended(self, *, completed: bool) -> None:
        self.events.append(("ended", completed))

    def interrupted(self, method: str) -> None:
        self.events.append(("interrupted", method))

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A VoiceSession with every device faked. No audio, no network."""
    from aura.config import Settings
    from aura.voice.session import VoiceSession

    settings = Settings()

    class Null:
        def __getattr__(self, _name: str) -> Any:
            return lambda *a, **k: None

    sess = VoiceSession.__new__(VoiceSession)
    sess._cfg = settings.voice
    sess._wake = FakeWake()
    sess._speaker = FakeSpeaker()
    sess._reporter = FakeReporter()
    sess._state = VoiceState.SPEAKING
    sess._speech_open = True
    sess._interrupted = False
    sess._follow_up_ok = False
    sess._on_text = lambda who, text: None
    sess._on_state = lambda state: None

    import threading

    sess._interrupt = threading.Event()
    return sess


# ===========================================================================
# One route out
# ===========================================================================


def test_an_interrupt_stops_the_speaker_first(session: Any) -> None:
    """The part the user is waiting for. Everything after it is bookkeeping
    and none of it should delay silence."""
    session._interrupt_speech("barge_in")

    assert session._speaker.stops == 1
    assert session._interrupt.is_set()
    assert session._interrupted is True


def test_an_interrupt_is_announced(session: Any) -> None:
    """Nothing published USER_INTERRUPTED before this, so the overlay was
    never told AURA had been cut off."""
    session._interrupt_speech("wake_word")

    assert ("interrupted", "wake_word") in session._reporter.events


def test_an_interrupt_ends_speech_as_incomplete(session: Any) -> None:
    """completed=False, so the overlay clears the cut-off line in 400 ms
    rather than holding it up to be read as the final word."""
    session._interrupt_speech("barge_in")

    assert ("ended", False) in session._reporter.events


def test_interrupting_twice_publishes_once(session: Any) -> None:
    """A detector firing twice -- or two detectors firing together -- must not
    produce two interruptions."""
    assert session._interrupt_speech("barge_in") is True
    assert session._interrupt_speech("wake_word") is False

    methods = [e for e in session._reporter.events if e[0] == "interrupted"]
    assert methods == [("interrupted", "barge_in")]
    assert session._speaker.stops == 1


@pytest.mark.parametrize("method", ["barge_in", "wake_word", "stop_command"])
def test_every_route_behaves_identically(method: str, session: Any) -> None:
    """Before this, each path stopped the speaker and did its own subset of
    the rest. They cannot drift apart now."""
    session._interrupt_speech(method)

    kinds = dict(session._reporter.events)
    assert session._speaker.stops == 1
    assert kinds["interrupted"] == method
    assert kinds["ended"] is False


# ===========================================================================
# The wake word — the interrupt that needs no threshold
# ===========================================================================


def test_the_wake_word_stops_her(session: Any) -> None:
    session._wake.armed = True

    session._wake_word_interrupt(FRAME)

    assert session._interrupt.is_set()
    assert ("interrupted", "wake_word") in session._reporter.events


def test_silence_does_not_stop_her(session: Any) -> None:
    session._wake_word_interrupt(FRAME)

    assert not session._interrupt.is_set()
    assert session._speaker.stops == 0


def test_the_detector_is_reset_after_firing(session: Any) -> None:
    """Otherwise the scores that fired stay in the buffer and immediately
    re-trigger against the next reply."""
    session._wake.armed = True

    session._wake_word_interrupt(FRAME)

    assert session._wake.resets == 1


def test_it_reopens_the_microphone(session: Any) -> None:
    """Someone who said the wake word has something to ask. Reopening is the
    entire point of interrupting."""
    session._wake.armed = True

    session._wake_word_interrupt(FRAME)

    assert session._follow_up_ok is True


def test_it_only_listens_while_she_is_speaking(session: Any) -> None:
    """During LISTENING the wake word is a normal trigger, not an interrupt."""
    session._state = VoiceState.LISTENING
    session._wake.armed = True

    session._wake_word_interrupt(FRAME)

    assert not session._interrupt.is_set()
    assert session._wake.frames == 0, "the detector was fed outside SPEAKING"


def test_it_can_be_switched_off(session: Any) -> None:
    from aura.config import Settings, VoiceConfig

    session._cfg = Settings(voice=VoiceConfig(wake_interrupt=False)).voice
    session._wake.armed = True

    session._wake_word_interrupt(FRAME)

    assert not session._interrupt.is_set()


def test_it_is_on_by_default() -> None:
    """It is the interrupt that actually works on this machine, so it should
    not need switching on."""
    from aura.config import Settings

    assert Settings().voice.wake_interrupt is True


def test_it_does_not_re_fire_once_interrupted(session: Any) -> None:
    session._interrupt.set()
    session._wake.armed = True

    session._wake_word_interrupt(FRAME)

    assert session._wake.frames == 0


# ===========================================================================
# Why this path exists at all
# ===========================================================================


def test_the_wake_interrupt_reads_no_threshold() -> None:
    """The whole point.

    Loudness-based barge-in has failed three times on this machine in both
    directions, because AURA's own voice and the user's arrive at the
    microphone at similar levels. A pattern match is immune to that, and this
    asserts the implementation did not quietly grow a level check anyway.
    """
    import inspect

    from aura.voice.session import VoiceSession

    source = inspect.getsource(VoiceSession._wake_word_interrupt)

    for level_check in ("rms", "barge_in_min_rms", "_current_barge_threshold", "np.sqrt"):
        assert level_check not in source, f"the wake interrupt consults {level_check}"


def test_echo_of_auras_own_speech_cannot_trigger_it(session: Any) -> None:
    """Modelled directly: loud audio that is not the wake word.

    This is the case that breaks the loudness path -- AURA's own voice at
    RMS 0.03 coming back through the mic array. Here it is simply not a match.
    """
    loud_echo = (0.35 * np.sin(np.linspace(0, 40, len(FRAME)))).astype(np.float32)
    session._wake.armed = False

    for _ in range(50):
        session._wake_word_interrupt(loud_echo)

    assert not session._interrupt.is_set()
    assert session._speaker.stops == 0
