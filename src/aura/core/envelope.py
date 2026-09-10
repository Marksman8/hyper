"""The amplitude-envelope format carried by `SPEECH_CHUNK` (10B.3).

This is the *format*, not the measurement: the frame duration, the alphabet,
and how a consumer reads one frame. It lives here, beside the event catalog,
because it describes a field in a catalogued event and therefore belongs to
whoever reads that event as much as to whoever writes it.

The split matters. Computing an envelope needs audio and numpy and lives in
`aura.voice`; *reading* one needs neither, and the presence process should not
grow a dependency on the voice package to move a mouth. `voice/types.py` says
outright that nothing outside `aura.voice` imports it, and that invariant is
worth more than saving a file.

Pure Python, no imports. That is deliberate -- it is the one module both a
producer with an audio stack and a consumer with a GUI toolkit can share.
"""

from __future__ import annotations

# One value per 20 ms. Fast enough that a mouth looks driven rather than
# stepped, slow enough that a whole sentence is a short string.
FRAME_MS = 20

# Levels per frame, 0-9. More would be invisible: the silhouette's mouth is a
# few pixels tall, and no viewer can tell level 6 from level 7.
LEVELS = 9

# 8 seconds of mouth. A cap rather than a limit anyone should hit: sentences
# are short, and an envelope is not a place to put a whole reply.
MAX_FRAMES = 400


def openness(envelope: str, index: int) -> float:
    """One frame of an envelope as a 0..1 mouth position.

    Out of range reads as a closed mouth, so a consumer that runs past the end
    of the envelope closes rather than holding the last shape open. A
    non-digit reads as closed too -- the field is validated at the API
    boundary, and a renderer is the wrong place to raise about it.
    """
    if index < 0 or index >= len(envelope):
        return 0.0
    char = envelope[index]
    return (int(char) / LEVELS) if char.isdigit() else 0.0


def duration_ms(envelope: str) -> int:
    """How long this envelope will take to play."""
    return len(envelope) * FRAME_MS
