"""Measuring the amplitude envelope for lip sync (10B.3).

**Amplitude, not phonemes.** Piper already hands back PCM; RMS per 20 ms frame
is roughly 90% of the perceived quality of visemes for about 5% of the work,
and it needs no phoneme aligner, no second model, and no per-language data. A
real viseme backend can arrive later behind the same `set_mouth()` interface.

The envelope is computed where the audio is -- here, in the voice process --
and travels with the sentence it belongs to. That is what makes it cheap:
`SPEECH_CHUNK` is already published at the moment a sentence's audio starts,
and Piper yields a sentence's audio in one piece, so the envelope is in hand at
exactly the moment the event goes out. No extra events, and nothing streaming
at 50 Hz across a bus built to carry metadata.

The *format* -- frame length, alphabet, how to read a frame -- lives in
`aura.core.envelope`, so the presence process can render a mouth without
importing the voice package.
"""

from __future__ import annotations

import numpy as np

from aura.core.envelope import FRAME_MS, LEVELS, MAX_FRAMES
from aura.voice.types import AudioArray

# Below this peak RMS the sentence is treated as silence and the mouth stays
# shut. Without it, per-sentence normalisation would take the noise floor of a
# near-silent clip and stretch it to a full open-and-close -- a mouth flapping
# at nothing.
SILENCE_FLOOR = 0.01


def envelope(audio: AudioArray, sample_rate: int, *, frame_ms: int = FRAME_MS) -> str:
    """RMS per frame, quantised to digits.

    Returns a string like `"0135789631"` -- one character per `frame_ms`.

    Normalised against this sentence's own peak, deliberately. Absolute
    normalisation would make a quietly-spoken sentence barely move the mouth,
    which reads as a rendering fault rather than as quiet speech; a talking
    head is expected to open its mouth when it talks.
    """
    if audio is None or len(audio) == 0 or sample_rate <= 0:
        return ""

    samples = np.asarray(audio, dtype=np.float32)
    frame = max(1, int(sample_rate * frame_ms / 1000))

    # Whole frames only. A trailing partial frame would have a lower RMS purely
    # because it is short, putting a spurious dip at the end of every sentence.
    usable = (len(samples) // frame) * frame
    if usable == 0:
        return ""

    frames = samples[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))

    peak = float(rms.max())
    if peak < SILENCE_FLOOR:
        return "0" * min(len(rms), MAX_FRAMES)

    levels = np.clip(np.rint(rms / peak * LEVELS).astype(int), 0, LEVELS)[:MAX_FRAMES]
    return "".join(str(int(v)) for v in levels)
