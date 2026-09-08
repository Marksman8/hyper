"""The privacy gate (spec section 10).

The property under test is not "the model was told not to look". It is that
`grab()` refuses and **no pixels are read**. Every capture test therefore uses a
fake mss backend that records whether it was touched, and asserts on that
recording rather than on the return value: a refusal that still grabbed the
frame and threw it away would pass a weaker test and fail the promise.
"""

from __future__ import annotations

import pytest

from aura.vision.capture import CaptureRefusedError, ScreenCapture
from aura.vision.privacy import PrivacyPolicy, VisionMode
from aura.vision.windows import ForegroundWindow


class SpyBackend:
    """Stands in for mss. Counts grabs so a refusal can be proven."""

    def __init__(self, monitors: int = 2) -> None:
        self.grabs = 0
        self.monitors = [{"left": 0, "top": 0, "width": 3840, "height": 1080}] + [
            {"left": 1920 * i, "top": 0, "width": 1920, "height": 1080}
            for i in range(monitors)
        ]

    def grab(self, region: dict[str, int]) -> object:
        self.grabs += 1
        return _Frame(region["width"], region["height"])

    def close(self) -> None:
        pass


class _Frame:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def __array__(self, dtype: object = None) -> object:
        import numpy as np

        return np.zeros((self.height, self.width, 4), dtype=np.uint8)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> SpyBackend:
    """A capture backend that cannot reach a real screen."""
    backend = SpyBackend()
    monkeypatch.setattr(ScreenCapture, "_backend", lambda self: backend)
    return backend


def _looking_at(monkeypatch: pytest.MonkeyPatch, title: str, process: str = "app.exe") -> None:
    """Pretend a particular window has focus."""
    window = ForegroundWindow(title=title, process_name=process, pid=42, known=True)
    monkeypatch.setattr("aura.vision.capture.foreground_window", lambda: window)


# -- modes ------------------------------------------------------------------


def test_default_is_private() -> None:
    """Vision starts off. Not on-with-restrictions - off."""
    assert PrivacyPolicy().mode is VisionMode.PRIVATE
    assert not PrivacyPolicy().may_capture()


def test_private_mode_reads_no_pixels(
    spy: SpyBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exit criterion, in its simplest form."""
    _looking_at(monkeypatch, "Untitled - Notepad", "notepad.exe")
    capture = ScreenCapture(PrivacyPolicy(mode=VisionMode.PRIVATE))

    with pytest.raises(CaptureRefusedError):
        capture.grab()

    assert spy.grabs == 0, "refused but still captured the frame"


def test_sleep_mode_reads_no_pixels(spy: SpyBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    _looking_at(monkeypatch, "Untitled - Notepad", "notepad.exe")
    capture = ScreenCapture(PrivacyPolicy(mode=VisionMode.SLEEP))

    with pytest.raises(CaptureRefusedError):
        capture.grab()
    assert spy.grabs == 0


def test_assist_mode_allows_a_request_and_refuses_a_poll(
    spy: SpyBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assist is 'when you ask', which is a distinction the gate must make."""
    _looking_at(monkeypatch, "Untitled - Notepad", "notepad.exe")
    capture = ScreenCapture(PrivacyPolicy(mode=VisionMode.ASSIST))

    capture.grab(requested_by_user=True)
    assert spy.grabs == 1

    with pytest.raises(CaptureRefusedError):
        capture.grab(requested_by_user=False)
    assert spy.grabs == 1, "background poll captured in assist mode"


def test_active_mode_allows_background_capture(
    spy: SpyBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    _looking_at(monkeypatch, "Untitled - Notepad", "notepad.exe")
    capture = ScreenCapture(PrivacyPolicy(mode=VisionMode.ACTIVE))
    capture.grab(requested_by_user=False)
    assert spy.grabs == 1


# -- the exit criterion -----------------------------------------------------


def test_dont_look_at_chrome_stops_frames_from_chrome(
    spy: SpyBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Phase 5 exit criterion, verified by instrumentation.

    Vision stays on and other windows are still captured; Chrome specifically
    stops producing frames. Asserting on `spy.grabs` is the point - asking the
    model whether it looked would prove nothing.
    """
    policy = PrivacyPolicy(mode=VisionMode.ACTIVE)
    capture = ScreenCapture(policy)

    _looking_at(monkeypatch, "AURA - Google Chrome", "chrome.exe")
    capture.grab()
    assert spy.grabs == 1, "baseline: Chrome is captured before the restriction"

    policy.deny_app("Chrome")

    with pytest.raises(CaptureRefusedError):
        capture.grab()
    assert spy.grabs == 1, "captured Chrome after being told not to"

    # Vision itself is still on: a different window still works.
    _looking_at(monkeypatch, "notes.md - VS Code", "Code.exe")
    capture.grab()
    assert spy.grabs == 2


def test_denial_matches_process_name_not_only_title(
    spy: SpyBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A window titled 'Inbox' owned by chrome.exe is still Chrome."""
    policy = PrivacyPolicy(mode=VisionMode.ACTIVE)
    policy.deny_app("chrome")
    capture = ScreenCapture(policy)

    _looking_at(monkeypatch, "Inbox (12)", "chrome.exe")
    with pytest.raises(CaptureRefusedError):
        capture.grab()
    assert spy.grabs == 0


# -- credential windows -----------------------------------------------------


@pytest.mark.parametrize(
    "title",
    [
        "Bitwarden",
        "1Password 8",
        "Sign in - Google Accounts",
        "Change your password",
        "New Incognito Tab",
        "HSBC Online Banking",
        "Microsoft Authenticator",
    ],
)
def test_credential_windows_are_never_captured(
    title: str, spy: SpyBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Denied in ACTIVE mode with no restrictions set.

    Turning vision on for your editor is not consent to have your password
    manager read.
    """
    _looking_at(monkeypatch, title, "app.exe")
    capture = ScreenCapture(PrivacyPolicy(mode=VisionMode.ACTIVE))

    with pytest.raises(CaptureRefusedError):
        capture.grab()
    assert spy.grabs == 0


def test_credential_denial_beats_an_allow_list(
    spy: SpyBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicitly allowing Chrome does not allow Chrome's password page."""
    policy = PrivacyPolicy(mode=VisionMode.ACTIVE)
    policy.allow_only("chrome")
    capture = ScreenCapture(policy)

    _looking_at(monkeypatch, "Sign in - Google Accounts", "chrome.exe")
    with pytest.raises(CaptureRefusedError):
        capture.grab()
    assert spy.grabs == 0


# -- allow-lists and monitors -----------------------------------------------


def test_allow_only_excludes_everything_else(
    spy: SpyBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = PrivacyPolicy(mode=VisionMode.ACTIVE)
    policy.allow_only("Code")
    capture = ScreenCapture(policy)

    _looking_at(monkeypatch, "notes.md - VS Code", "Code.exe")
    capture.grab()
    assert spy.grabs == 1

    _looking_at(monkeypatch, "Spotify Premium", "Spotify.exe")
    with pytest.raises(CaptureRefusedError):
        capture.grab()
    assert spy.grabs == 1


def test_denied_monitor_is_never_captured(
    spy: SpyBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    _looking_at(monkeypatch, "Untitled - Notepad", "notepad.exe")
    policy = PrivacyPolicy(mode=VisionMode.ACTIVE, denied_monitors={2})
    capture = ScreenCapture(policy)

    capture.grab(monitor=1)
    assert spy.grabs == 1

    with pytest.raises(CaptureRefusedError):
        capture.grab(monitor=2)
    assert spy.grabs == 1


def test_allow_only_supersedes_a_previous_denial() -> None:
    policy = PrivacyPolicy(mode=VisionMode.ACTIVE)
    policy.deny_app("Chrome")
    policy.allow_only("Chrome")
    assert "Chrome" not in policy.denied_apps
    assert "Chrome" in policy.allowed_apps


def test_refusal_says_why() -> None:
    """A refusal the user cannot act on is a bug report waiting to happen."""
    decision = PrivacyPolicy().may_capture()
    assert not decision
    assert "private" in decision.reason.lower()
    assert decision.reason.endswith(".")


# -- the window-switch race -------------------------------------------------


def test_foreground_window_is_reread_for_every_grab(
    spy: SpyBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached window identity is a race, and the case it loses matters.

    Focus moves between "may I?" and "click". If the answer were cached, a
    password manager would be captured because Notepad was in front a moment
    ago.
    """
    windows = [
        ForegroundWindow(title="Untitled - Notepad", process_name="notepad.exe", known=True),
        ForegroundWindow(title="Bitwarden", process_name="Bitwarden.exe", known=True),
    ]
    calls = {"n": 0}

    def next_window() -> ForegroundWindow:
        window = windows[min(calls["n"], len(windows) - 1)]
        calls["n"] += 1
        return window

    monkeypatch.setattr("aura.vision.capture.foreground_window", next_window)
    capture = ScreenCapture(PrivacyPolicy(mode=VisionMode.ACTIVE))

    capture.grab()
    assert spy.grabs == 1

    # Focus has moved to the password manager. Nothing told the policy; it has
    # to notice by itself.
    with pytest.raises(CaptureRefusedError):
        capture.grab()
    assert spy.grabs == 1


# -- monitor enumeration ----------------------------------------------------


def test_monitors_are_listed_without_capturing(spy: SpyBackend) -> None:
    """Layout is not content: enumeration is not gated, and reads no pixels."""
    capture = ScreenCapture(PrivacyPolicy(mode=VisionMode.PRIVATE))
    monitors = capture.monitors()

    assert [m.index for m in monitors] == [1, 2]
    assert monitors[0].is_primary
    assert spy.grabs == 0


def test_unknown_monitor_is_refused(spy: SpyBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    _looking_at(monkeypatch, "Untitled - Notepad", "notepad.exe")
    capture = ScreenCapture(PrivacyPolicy(mode=VisionMode.ACTIVE))

    with pytest.raises(CaptureRefusedError):
        capture.grab(monitor=9)
    assert spy.grabs == 0
