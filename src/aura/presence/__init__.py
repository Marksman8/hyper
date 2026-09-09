"""Phase 10A: AURA's desktop presence.

A SEPARATE PROCESS from AURA Core, holding no reasoning and no state of its
own. It subscribes to `/v1/events`, renders what the presence state machine
decided, and nothing else -- so killing it costs AURA a face and nothing more.

Qt is imported only inside this package. `tests/unit/test_presence.py` asserts
that no module in `aura/core` depends on a UI toolkit.
"""

from aura.presence.placement import Anchor, MoveReason, PlacementPolicy, Screen

__all__ = ["Anchor", "MoveReason", "PlacementPolicy", "Screen"]
