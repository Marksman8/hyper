"""Has the screen changed enough to be worth looking at again?

Built now, called by nothing on a timer. Spec §9 wants a continuous
observation loop; the loop is deferred (see the phase doc §3), but its trigger
is the part with a correctness question, so it is written and tested here and
the loop can be added later without inventing this at the same time.

Why it matters when the loop does arrive: OCR on a 2560x1440 frame costs real
CPU. Running it on every tick when nothing moved is the difference between a
background observer and a space heater. A cheap numeric comparison decides
whether the expensive step happens.

numpy only — deliberately no opencv. The comparison is a downscale and a mean
absolute difference, and pulling an image-processing stack in for that would be
weight without benefit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aura.logging import get_logger

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    Image = NDArray[np.uint8]
else:
    Image = Any

log = get_logger(__name__)

# The comparison grid. Small enough that the diff is microseconds, coarse
# enough that a blinking cursor or an antialiasing wobble does not register as
# a change, fine enough that a new dialog box does.
GRID = 32

# Mean absolute difference, on a 0-1 scale, above which the screen is
# "different". Tuned to ignore a clock ticking over and notice a window opening.
DEFAULT_THRESHOLD = 0.02


@dataclass(frozen=True, slots=True)
class ChangeResult:
    """How different two frames are."""

    changed: bool
    score: float
    threshold: float

    def __bool__(self) -> bool:
        return self.changed


def fingerprint(image: Image) -> Any:
    """Reduce a frame to a small grayscale grid.

    Cheap, allocation-light, and stable under the noise that does not matter:
    subpixel rendering, cursor blink, a progress bar advancing one pixel.
    """
    import numpy as np

    array = np.asarray(image)
    if array.ndim == 3:
        # Luminance weights, not a flat mean: a flat mean makes a red error
        # banner and a blue title bar look identical.
        array = array[:, :, :3].astype(np.float32)
        gray = array[:, :, 0] * 0.299 + array[:, :, 1] * 0.587 + array[:, :, 2] * 0.114
    else:
        gray = array.astype(np.float32)

    height, width = gray.shape[:2]
    if height < GRID or width < GRID:
        # Too small to block-reduce meaningfully; use it as-is, normalised.
        return gray / 255.0

    # Trim to a multiple of GRID, then average each block. Integer slicing
    # rather than an interpolating resize: no dependency, and averaging is the
    # right operation for "did this region change" anyway.
    rows, cols = (height // GRID) * GRID, (width // GRID) * GRID
    block = gray[:rows, :cols].reshape(GRID, rows // GRID, GRID, cols // GRID)
    return block.mean(axis=(1, 3)) / 255.0


def compare(
    previous: Any | None, current: Any, threshold: float = DEFAULT_THRESHOLD
) -> ChangeResult:
    """Compare two fingerprints.

    With no previous frame the answer is "changed": the first observation is
    always new information.
    """
    import numpy as np

    if previous is None:
        return ChangeResult(changed=True, score=1.0, threshold=threshold)
    if previous.shape != current.shape:
        # A resolution change is itself a change worth noticing.
        return ChangeResult(changed=True, score=1.0, threshold=threshold)

    score = float(np.abs(current - previous).mean())
    return ChangeResult(changed=score >= threshold, score=score, threshold=threshold)


class ChangeDetector:
    """Remembers the last frame it was shown, per monitor."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self._threshold = threshold
        self._last: dict[int, Any] = {}

    def check(self, image: Image, monitor: int = 1) -> ChangeResult:
        """Compare against this monitor's last frame and remember this one."""
        current = fingerprint(image)
        result = compare(self._last.get(monitor), current, self._threshold)
        self._last[monitor] = current
        return result

    def reset(self, monitor: int | None = None) -> None:
        """Forget. Called when vision is turned off, so that turning it back on
        does not compare against a frame from before the user said stop."""
        if monitor is None:
            self._last.clear()
        else:
            self._last.pop(monitor, None)
