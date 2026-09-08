"""Change detection.

Nothing calls this on a timer yet (the observation loop is deferred), but it is
the part of the loop with a correctness question, so it is tested now: the
trigger has to ignore a blinking cursor and notice a new window, or the loop it
eventually feeds will either burn CPU on nothing or miss what happened.
"""

from __future__ import annotations

import numpy as np

from aura.vision.change import ChangeDetector, compare, fingerprint


def _screen(value: int = 30, width: int = 1920, height: int = 1080) -> np.ndarray:
    """A flat frame of a given brightness."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_first_frame_is_always_a_change() -> None:
    """With nothing to compare against, the first look is new information."""
    result = compare(None, fingerprint(_screen()))
    assert result.changed
    assert result.score == 1.0


def test_identical_frames_do_not_trigger() -> None:
    frame = _screen()
    detector = ChangeDetector()
    detector.check(frame)
    assert not detector.check(frame)


def test_a_new_window_triggers() -> None:
    """A dialog covering a quarter of the screen is a change."""
    before = _screen(30)
    after = before.copy()
    after[300:800, 600:1400] = 230

    detector = ChangeDetector()
    detector.check(before)
    result = detector.check(after)

    assert result.changed
    assert result.score > result.threshold


def test_a_blinking_cursor_does_not_trigger() -> None:
    """The noise floor: two characters' worth of pixels must not wake the loop.

    This is the case that decides whether a background observer is usable. If a
    text cursor counts as a change, OCR runs every half-second forever.
    """
    before = _screen(30)
    after = before.copy()
    after[500:520, 800:803] = 255  # a caret

    detector = ChangeDetector()
    detector.check(before)
    assert not detector.check(after)


def test_colour_change_at_equal_brightness_triggers() -> None:
    """Luminance weighting, not a flat mean.

    A red error banner and a blue title bar have the same channel average. If
    the fingerprint used a flat mean, swapping one for the other would be
    invisible.
    """
    before = np.zeros((1080, 1920, 3), dtype=np.uint8)
    before[:, :, 2] = 200  # blue
    after = np.zeros((1080, 1920, 3), dtype=np.uint8)
    after[:, :, 0] = 200  # red

    detector = ChangeDetector()
    detector.check(before)
    assert detector.check(after)


def test_resolution_change_counts_as_change() -> None:
    detector = ChangeDetector()
    detector.check(_screen(width=1920, height=1080))
    # Small enough to skip block-reduction, so the shapes genuinely differ.
    assert detector.check(_screen(width=16, height=16))


def test_monitors_are_tracked_separately() -> None:
    """Monitor 2 changing must not be reported as monitor 1 changing."""
    detector = ChangeDetector()
    frame_a, frame_b = _screen(30), _screen(200)

    detector.check(frame_a, monitor=1)
    detector.check(frame_a, monitor=2)

    assert detector.check(frame_b, monitor=2)
    assert not detector.check(frame_a, monitor=1)


def test_reset_forgets_the_last_frame() -> None:
    """Turning vision off must not leave a reduced copy of the screen behind."""
    detector = ChangeDetector()
    frame = _screen()
    detector.check(frame)
    detector.reset()
    assert detector.check(frame), "compared against a frame from before the reset"


def test_grayscale_input_is_accepted() -> None:
    """Two dimensions, no channel axis - must not raise."""
    gray = np.full((1080, 1920), 40, dtype=np.uint8)
    assert fingerprint(gray).size > 0


def test_threshold_is_configurable() -> None:
    before = _screen(30)
    after = before.copy()
    after[0:100, 0:100] = 60

    strict = ChangeDetector(threshold=0.0001)
    strict.check(before)
    assert strict.check(after)

    lax = ChangeDetector(threshold=0.9)
    lax.check(before)
    assert not lax.check(after)
