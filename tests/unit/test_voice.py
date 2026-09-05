"""Voice components that can be tested without a microphone or a model."""

from __future__ import annotations

import numpy as np
import pytest

from aura.voice.audio.capture import RingBuffer
from aura.voice.stt.null import NullSTT
from aura.voice.tts.null import NullTTS
from aura.voice.tts.piper import SentenceStreamer, split_sentences
from aura.voice.types import SAMPLE_RATE, VoiceState
from aura.voice.wake.null import NullWakeWord

# -- sentence streaming ------------------------------------------------------
# The mechanism that keeps time-to-first-audio near 1 s: speak sentence one
# while the model is still writing sentence two.


def test_streamer_releases_a_sentence_as_soon_as_it_is_complete() -> None:
    s = SentenceStreamer()
    assert s.push("The capital of France ") == []
    assert s.push("is Paris. ") == ["The capital of France is Paris."]


def test_streamer_holds_incomplete_text() -> None:
    s = SentenceStreamer()
    assert s.push("This has no ending yet") == []
    assert s.flush() == ["This has no ending yet"]


def test_streamer_handles_several_sentences_in_one_delta() -> None:
    s = SentenceStreamer()
    out = s.push("One. Two! Three? ")
    assert out == ["One.", "Two!", "Three?"]


def test_streamer_does_not_emit_punctuation_only_fragments() -> None:
    """A fragment with no letters would make the vocoder click."""
    s = SentenceStreamer()
    assert s.push("... ") == []
    assert s.flush() == []


def test_streamer_releases_long_unpunctuated_text() -> None:
    """Otherwise a comma-spliced paragraph is silent until the stream ends."""
    s = SentenceStreamer(max_chars=60)
    out = s.push("word " * 40)
    assert out, "long text should be released at a word boundary"
    assert all(len(chunk) <= 60 for chunk in out)


def test_streamer_flush_is_idempotent() -> None:
    s = SentenceStreamer()
    s.push("Trailing text")
    assert s.flush() == ["Trailing text"]
    assert s.flush() == []


def test_split_sentences_drops_empty_parts() -> None:
    assert split_sentences("Hi. There!  ") == ["Hi.", "There!"]
    assert split_sentences("   ") == []


# -- ring buffer -------------------------------------------------------------
# Holds the audio from before the wake word fired, so "Hey Jarvis what's the
# time" does not lose its first word.


def test_ring_buffer_returns_audio_in_order() -> None:
    rb = RingBuffer(seconds=1.0)
    rb.write(np.arange(4000, dtype=np.float32))
    rb.write(np.arange(4000, 8000, dtype=np.float32))
    out = rb.read()
    assert len(out) == 8000
    assert out[0] == 0.0
    assert out[-1] == 7999.0


def test_ring_buffer_keeps_only_the_most_recent_audio() -> None:
    rb = RingBuffer(seconds=0.5)  # 8000 samples
    rb.write(np.arange(20000, dtype=np.float32))
    out = rb.read()
    assert len(out) == 8000
    # The newest sample must be last, and the oldest discarded.
    assert out[-1] == 19999.0
    assert out[0] == 12000.0


def test_ring_buffer_wraps_correctly_across_writes() -> None:
    rb = RingBuffer(seconds=0.5)  # 8000
    for start in range(0, 12000, 1000):
        rb.write(np.arange(start, start + 1000, dtype=np.float32))
    out = rb.read()
    assert len(out) == 8000
    assert np.array_equal(out, np.arange(4000, 12000, dtype=np.float32))


def test_ring_buffer_clear() -> None:
    rb = RingBuffer(seconds=1.0)
    rb.write(np.ones(1000, dtype=np.float32))
    rb.clear()
    assert len(rb.read()) == 0


# -- stubs -------------------------------------------------------------------
# These exist so the whole session loop can be exercised with no models. If a
# voice test ever needs real weights, the seam has leaked.


def test_null_tts_produces_correctly_shaped_audio() -> None:
    tts = NullTTS()
    chunks = list(tts.synthesize("hello there"))
    assert len(chunks) == 1
    assert chunks[0].sample_rate == 22_050
    assert chunks[0].duration_s > 0
    assert tts.spoken == ["hello there"]


def test_null_tts_ignores_empty_text() -> None:
    assert list(NullTTS().synthesize("   ")) == []


def test_null_stt_returns_queued_transcripts_then_empties() -> None:
    stt = NullSTT(["first", "second"])
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    assert stt.transcribe(audio).text == "first"
    assert stt.transcribe(audio).text == "second"
    assert stt.transcribe(audio).is_empty


def test_null_wake_word_fires_only_when_triggered() -> None:
    wake = NullWakeWord()
    frame = np.zeros(1280, dtype=np.float32)
    assert wake.process(frame) is None
    wake.trigger()
    assert wake.process(frame) is not None
    # And does not fire twice from one trigger.
    assert wake.process(frame) is None


# -- state -------------------------------------------------------------------


def test_voice_states_are_distinct() -> None:
    assert VoiceState.SLEEP != VoiceState.LISTENING
    assert VoiceState.SLEEP.value == "sleep"


@pytest.mark.parametrize(
    "state", [VoiceState.SLEEP, VoiceState.LISTENING, VoiceState.THINKING, VoiceState.SPEAKING]
)
def test_voice_state_round_trips_through_its_value(state: VoiceState) -> None:
    assert VoiceState(state.value) is state
