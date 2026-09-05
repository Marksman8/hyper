"""Stop commands.

Saying "stop" while AURA is talking should cut it off and return to listening.
Barge-in already stopped the audio, but the word itself was discarded, so the
cancellation was then treated as the start of a new question -- or, worse, sent
to the model, which would explain what stopping means.
"""

from __future__ import annotations

import pytest

from aura.config import VoiceConfig
from aura.voice.session import is_stop_command

STOP_WORDS = VoiceConfig().stop_words


@pytest.mark.parametrize(
    "said",
    [
        "stop",
        "Stop.",
        "STOP!",
        "stop it",
        "stop aura",
        "shut up",
        "be quiet",
        "cancel",
        "never mind",
        "nevermind",
        "forget it",
        "that's enough",
        "okay stop",
        "  stop  ",
    ],
)
def test_recognised_as_a_cancellation(said: str) -> None:
    assert is_stop_command(said, STOP_WORDS)


@pytest.mark.parametrize(
    "said",
    [
        # Each of these contains a stop word but is plainly a question.
        "stop the build",
        "how do I stop a docker container",
        "why did the training stop",
        "what does cancel culture mean",
        "tell me about the enough is enough campaign",
        "can you be quiet about the details and just summarise",
        "what is a stop word",
        "explain quiet quitting",
    ],
)
def test_not_mistaken_for_a_cancellation(said: str) -> None:
    """Matched on the whole utterance, never as a substring."""
    assert not is_stop_command(said, STOP_WORDS)


def test_wake_phrase_is_ignored_when_matching() -> None:
    """ "Hey Jarvis, stop" is a cancellation, not a question about stopping."""
    assert is_stop_command("Hey Jarvis, stop", STOP_WORDS)
    assert is_stop_command("Aura stop it", STOP_WORDS)


def test_empty_is_not_a_stop_command() -> None:
    assert not is_stop_command("", STOP_WORDS)
    assert not is_stop_command("   ", STOP_WORDS)


def test_stop_words_are_configurable() -> None:
    assert is_stop_command("basta", ["basta"])
    assert not is_stop_command("stop", ["basta"])
