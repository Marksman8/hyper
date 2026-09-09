"""The presence process (10A.5 to 10A.11).

Split by what needs a display and what does not. The placement policy, the
avatar backend and the event-to-presentation mapping are all testable headless,
which is the point of keeping them out of the widget.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aura.core.presence import PresenceState
from aura.presence.avatar.base import AvatarBackend
from aura.presence.avatar.silhouette import PALETTE, PULSE, SilhouetteAvatar
from aura.presence.placement import (
    MIN_MOVE_INTERVAL_S,
    Anchor,
    MoveReason,
    PlacementPolicy,
    Screen,
)

SRC = Path(__file__).resolve().parents[2] / "src" / "aura"
SCREEN = Screen(x=0, y=0, width=1920, height=1040)
SIZE = (132, 176)


# ===========================================================================
# PLACEMENT — mostly about refusing to move
# ===========================================================================


def test_the_four_triggers_are_the_only_ones() -> None:
    """The rule, asserted against the enum rather than against behaviour, so
    adding a fifth is a deliberate act."""
    assert {str(r) for r in MoveReason} == {
        "app_changed",
        "user_drag",
        "needs_attention",
        "anchor_invalid",
    }


def test_a_move_is_allowed_for_a_permitted_trigger() -> None:
    policy = PlacementPolicy()
    placement = policy.consider(MoveReason.APP_CHANGED, SCREEN, SIZE, now=100.0)
    assert placement is not None
    assert placement.reason is MoveReason.APP_CHANGED


def test_moves_are_rate_limited_to_one_every_three_seconds() -> None:
    """Two applications fighting over focus must not turn the avatar into a
    metronome."""
    policy = PlacementPolicy()
    assert policy.consider(MoveReason.APP_CHANGED, SCREEN, SIZE, now=100.0) is not None

    for offset in (0.1, 0.5, 1.0, 2.9):
        assert (
            policy.consider(MoveReason.APP_CHANGED, SCREEN, SIZE, now=100.0 + offset)
            is None
        ), f"moved again after {offset}s"

    assert (
        policy.consider(
            MoveReason.APP_CHANGED, SCREEN, SIZE, now=100.0 + MIN_MOVE_INTERVAL_S + 0.1
        )
        is not None
    )
    assert policy.suppressed_count == 4


def test_a_rapid_burst_of_app_changes_produces_one_move() -> None:
    policy = PlacementPolicy()
    moved = [
        policy.consider(MoveReason.APP_CHANGED, SCREEN, SIZE, now=100.0 + i * 0.05)
        for i in range(60)
    ]
    assert sum(1 for m in moved if m is not None) == 1


def test_only_states_that_need_attention_justify_moving() -> None:
    """A SUCCESS is not a reason to move across the screen."""
    policy = PlacementPolicy()
    for state in (PresenceState.SUCCESS, PresenceState.THINKING, PresenceState.IDLE):
        assert (
            policy.consider(
                MoveReason.NEEDS_ATTENTION, SCREEN, SIZE, now=100.0, state=state
            )
            is None
        ), state


@pytest.mark.parametrize(
    "state", [PresenceState.ERROR, PresenceState.WARNING, PresenceState.WAITING]
)
def test_states_that_do_need_attention_may_move(state: PresenceState) -> None:
    policy = PlacementPolicy()
    assert (
        policy.consider(MoveReason.NEEDS_ATTENTION, SCREEN, SIZE, now=100.0, state=state)
        is not None
    )


def test_a_user_drag_always_wins() -> None:
    """Refusing to follow a drag would feel broken, and a drag is by definition
    deliberate."""
    policy = PlacementPolicy()
    policy.consider(MoveReason.APP_CHANGED, SCREEN, SIZE, now=100.0)

    pinned = policy.pin(400, 300, now=100.1)  # inside the rate limit
    assert (pinned.x, pinned.y) == (400, 300)
    assert policy.anchor is Anchor.PINNED


def test_an_invalid_anchor_bypasses_the_rate_limit() -> None:
    """Staying put is not an option when 'put' is a monitor that no longer
    exists."""
    policy = PlacementPolicy()
    policy.consider(MoveReason.APP_CHANGED, SCREEN, SIZE, now=100.0)
    assert (
        policy.consider(MoveReason.ANCHOR_INVALID, SCREEN, SIZE, now=100.1) is not None
    )


def test_anchors_resolve_inside_the_screen() -> None:
    policy = PlacementPolicy()
    for anchor in (Anchor.BOTTOM_RIGHT, Anchor.BOTTOM_LEFT, Anchor.TOP_RIGHT):
        x, y = policy.resolve(SCREEN, SIZE, anchor=anchor)
        assert 0 <= x <= SCREEN.width - SIZE[0]
        assert 0 <= y <= SCREEN.height - SIZE[1]


def test_a_pin_from_a_detached_monitor_is_clamped_back_on_screen() -> None:
    """Otherwise she is stranded off-screen and unreachable."""
    policy = PlacementPolicy()
    policy.pin(5000, 4000, now=100.0)
    x, y = policy.resolve(SCREEN, SIZE)
    assert x <= SCREEN.width - SIZE[0]
    assert y <= SCREEN.height - SIZE[1]


def test_anchor_validity_detects_an_off_screen_position() -> None:
    policy = PlacementPolicy()
    assert policy.anchor_is_valid(SCREEN, SIZE, (100, 100))
    assert not policy.anchor_is_valid(SCREEN, SIZE, (3000, 100))


# ===========================================================================
# THE AVATAR BACKEND
# ===========================================================================


def test_every_state_has_a_colour() -> None:
    """A missing entry would silently fall back to IDLE, so the avatar would
    lie about what AURA is doing."""
    assert set(PALETTE) == set(PresenceState)


def test_most_states_do_not_animate() -> None:
    """A desktop companion that is always moving is one that gets closed."""
    assert len(PULSE) < len(PresenceState) / 2
    for still in (PresenceState.IDLE, PresenceState.SUCCESS, PresenceState.SLEEP):
        assert still not in PULSE


def test_an_idle_avatar_becomes_clean_and_stays_clean() -> None:
    """The property behind 0 repaints and 0.00% CPU at idle."""
    avatar = SilhouetteAvatar()
    avatar.set_state(PresenceState.IDLE, immediate=True)

    # Bounded: IDLE has no pulse, so this settles -- but an unbounded loop here
    # would hang forever on any state that does.
    for _ in range(100):
        if not avatar.advance(16):
            break

    assert not any(avatar.advance(16) for _ in range(100)), (
        "an idle avatar kept asking for repaints"
    )


def test_an_animated_state_keeps_asking_for_repaints() -> None:
    avatar = SilhouetteAvatar()
    avatar.set_state(PresenceState.THINKING)
    assert all(avatar.advance(16) for _ in range(20))


def test_private_does_not_cross_fade() -> None:
    """A graceful 260 ms transition means AURA is still visible a quarter of a
    second after being told to stop."""
    avatar = SilhouetteAvatar()
    avatar.set_state(PresenceState.WORKING, immediate=True)
    avatar.set_state(PresenceState.PRIVATE, immediate=True)

    avatar.advance(16)
    assert not avatar.advance(16), "PRIVATE was still fading"


def test_setting_the_same_state_changes_nothing() -> None:
    """Re-asserting a state must not restart the fade or reset the pulse."""
    avatar = SilhouetteAvatar()
    avatar.set_state(PresenceState.THINKING, immediate=True)
    for _ in range(10):
        avatar.advance(16)

    phase, fade = avatar._phase, avatar._fade
    avatar.set_state(PresenceState.THINKING)

    assert avatar._phase == phase
    assert avatar._fade == fade


def test_gaze_is_clamped() -> None:
    avatar = SilhouetteAvatar()
    avatar.look_at(50.0, -50.0)
    assert avatar._gaze == (1.0, -1.0)


def test_the_placeholder_needs_no_assets() -> None:
    """So 10A can be judged on behaviour without committing to artwork."""
    info = SilhouetteAvatar().info()
    assert info.needs_assets is False
    assert info.kind == "vector-placeholder"


def test_the_backend_is_a_real_implementation_of_the_seam() -> None:
    assert isinstance(SilhouetteAvatar(), AvatarBackend)


def test_the_seam_hides_qt_from_its_callers() -> None:
    """`paint` takes an untyped painter on purpose: typing it as QPainter would
    make the interface, and everything importing it, depend on Qt."""
    tree = ast.parse((SRC / "presence" / "avatar" / "base.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(m.startswith("PySide6") for m in imported)


# ===========================================================================
# ARCHITECTURE
# ===========================================================================


def test_qt_is_confined_to_the_presence_package() -> None:
    """The architectural rule from the 10A brief: the presentation layer must
    be replaceable without touching presence logic."""
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if path.parts[-2] == "presence" or "presence" in path.parts:
            continue
        if "PySide6" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, f"Qt leaked outside aura/presence: {offenders}"


def test_the_presence_process_holds_no_reasoning() -> None:
    """It renders decisions; it must not make them. If it starts importing the
    orchestrator, the agent or the tool layer, that has stopped being true.
    """
    forbidden = ("aura.core.orchestrator", "aura.core.agent", "aura.tools", "aura.memory")
    offenders: list[str] = []
    for path in (SRC / "presence").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                module = node.names[0].name
            if module and any(module.startswith(f) for f in forbidden):
                offenders.append(f"{path.name} imports {module}")
    assert not offenders, offenders


def test_the_presence_process_cannot_run_a_tool() -> None:
    """The bus contract's rule for consumers, checked at the package level:
    rendering an event must not be able to cause anything to happen."""
    for path in (SRC / "presence").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "ToolExecutor" not in text, path.name
        assert "/v1/tools" not in text, path.name
        assert "/v1/approvals" not in text, path.name
