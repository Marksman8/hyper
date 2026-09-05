"""Short audible cues.

A voice assistant that gives no sign it woke up is indistinguishable from one
that is broken. The user cannot see a log line, and the gap between the wake
word and the first spoken word is seconds long -- during which the only honest
question is "is this thing on?".

So: a short rising blip when AURA starts listening, and a lower one when it
gives up without answering. Generated rather than shipped as audio files, both
to avoid another asset to download and because tones want to be tunable.

Deliberately quiet and brief. A cue that is louder than the assistant's voice
becomes the thing you notice, and this is one people hear dozens of times a day.
"""

from __future__ import annotations

import numpy as np

from aura.voice.types import SpeechChunk

CUE_SAMPLE_RATE = 22_050


def _tone(
    frequencies: tuple[float, ...],
    duration_s: float,
    sample_rate: int,
    volume: float,
) -> np.ndarray:
    """A sequence of pure tones with click-free edges."""
    per_tone = duration_s / len(frequencies)
    samples = max(1, int(per_tone * sample_rate))
    t = np.arange(samples) / sample_rate

    # A raised-cosine envelope on each segment. Without it the discontinuity
    # at the boundaries is audible as a click, which sounds like a fault.
    envelope = np.sin(np.pi * np.linspace(0.0, 1.0, samples)) ** 2

    parts = [np.sin(2 * np.pi * f * t) * envelope for f in frequencies]
    audio = np.concatenate(parts) * volume
    return audio.astype(np.float32)


def wake_cue(volume: float = 0.18, sample_rate: int = CUE_SAMPLE_RATE) -> SpeechChunk:
    """Rising two-tone blip: AURA is listening."""
    return SpeechChunk(
        audio=_tone((660.0, 880.0), 0.14, sample_rate, volume),
        sample_rate=sample_rate,
        text="[wake]",
    )


def dismiss_cue(volume: float = 0.14, sample_rate: int = CUE_SAMPLE_RATE) -> SpeechChunk:
    """Falling two-tone blip: AURA stopped listening without answering.

    Distinct from silence, which is what a false wake used to sound like.
    """
    return SpeechChunk(
        audio=_tone((660.0, 440.0), 0.14, sample_rate, volume),
        sample_rate=sample_rate,
        text="[dismiss]",
    )


def follow_up_cue(volume: float = 0.12, sample_rate: int = CUE_SAMPLE_RATE) -> SpeechChunk:
    """A single soft tone: AURA finished answering and is still listening.

    Quieter and shorter than the wake cue. It fires after every reply, so it
    has to be easy to ignore -- it means "your turn", not "pay attention".
    """
    return SpeechChunk(
        audio=_tone((760.0,), 0.09, sample_rate, volume),
        sample_rate=sample_rate,
        text="[follow_up]",
    )
