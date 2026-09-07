"""Opening applications.

The safety property under test: the model supplies a *name*, and what launches
is an AppID taken from the discovered list. A name that matches nothing must
fail rather than becoming a command, because a tool that accepted a path would
be `run_anything` with a friendlier label.
"""

from __future__ import annotations

import sys

import pytest

from aura.security.permissions import PermissionLevel
from aura.tools.applications import (
    InstalledApp,
    ListApplicationsTool,
    OpenApplicationTool,
    application_tools,
    find_app,
)
from aura.tools.types import ToolCall, ToolContext

# Modelled on what Get-StartApps actually returned on this machine, including
# the noise entries.
INSTALLED = [
    InstalledApp("WhatsApp", "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App"),
    InstalledApp("Google Chrome", "Chrome"),
    InstalledApp("Microsoft Edge", "MSEdge"),
    InstalledApp("Notepad", "Microsoft.WindowsNotepad_8wekyb3d8bbwe!App"),
    InstalledApp("Visual Studio Code", "Microsoft.VisualStudioCode"),
    InstalledApp("Visual Studio Installer", "VSInstaller"),
    InstalledApp("Spotify", "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"),
    InstalledApp("Calculator", "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"),
]


# -- matching ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("WhatsApp", "WhatsApp"),
        ("whatsapp", "WhatsApp"),
        ("  WhatsApp  ", "WhatsApp"),
        ("chrome", "Google Chrome"),
        ("google chrome", "Google Chrome"),
        ("notepad", "Notepad"),
        ("spotify", "Spotify"),
        ("calc", "Calculator"),
        # Token matching: every word present, any order.
        ("code visual", "Visual Studio Code"),
        # Initials, which substring matching cannot reach.
        ("vs code", "Visual Studio Code"),
    ],
)
def test_resolves_how_people_actually_say_it(spoken: str, expected: str) -> None:
    match = find_app(spoken, INSTALLED)
    assert match is not None, f"{spoken!r} matched nothing"
    assert match.name == expected


def test_prefers_the_shortest_containing_match() -> None:
    """'visual studio' must not land on the Installer."""
    match = find_app("visual studio", INSTALLED)
    assert match is not None
    assert match.name == "Visual Studio Code"


@pytest.mark.parametrize("spoken", ["", "   ", "photoshop", "a nonexistent program"])
def test_an_unknown_name_matches_nothing(spoken: str) -> None:
    """Failing to match is the correct outcome; there is no fallback to exec."""
    assert find_app(spoken, INSTALLED) is None


def test_matching_against_an_empty_machine_is_safe() -> None:
    assert find_app("chrome", []) is None


# -- the tools --------------------------------------------------------------


def test_opening_requires_confirmation() -> None:
    """Launching something on the user's desktop is not a read."""
    assert OpenApplicationTool().spec.level is PermissionLevel.ACT


def test_listing_does_not() -> None:
    assert ListApplicationsTool().spec.level is PermissionLevel.READ


def test_the_schema_accepts_a_name_and_nothing_else() -> None:
    """No path, no command, no arguments — the safety property, as a schema."""
    params = OpenApplicationTool().spec.parameters
    assert set(params["properties"]) == {"name"}
    assert params["additionalProperties"] is False
    assert params["required"] == ["name"]


def test_an_unknown_application_fails_without_launching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aura.tools.applications.discover_apps", lambda *a, **k: INSTALLED)
    monkeypatch.setattr(sys, "platform", "win32")

    launched: list[object] = []
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda *a, **k: launched.append(a),  # type: ignore[misc]
    )

    result = OpenApplicationTool().run(
        ToolCall(name="open_application", arguments={"name": "photoshop"}), ToolContext()
    )
    assert not result.ok
    assert launched == [], "an unmatched name reached the launcher"


def test_launching_uses_the_discovered_app_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """What runs comes from the discovered list, never from the model."""
    monkeypatch.setattr("aura.tools.applications.discover_apps", lambda *a, **k: INSTALLED)
    monkeypatch.setattr(sys, "platform", "win32")

    calls: list[list[str]] = []

    def fake_popen(argv: list[str], **_kwargs: object) -> object:
        calls.append(argv)
        return object()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    result = OpenApplicationTool().run(
        ToolCall(name="open_application", arguments={"name": "whatsapp"}), ToolContext()
    )

    assert result.ok, result.error
    assert len(calls) == 1
    argv = calls[0]
    assert argv[0].lower().endswith("explorer.exe")
    assert argv[1] == "shell:AppsFolder\\5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App"
    assert result.data["name"] == "WhatsApp"


def test_a_near_miss_suggests_alternatives(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong guess should teach the model the right name, not just say no."""
    monkeypatch.setattr("aura.tools.applications.discover_apps", lambda *a, **k: INSTALLED)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: object())

    result = OpenApplicationTool().run(
        ToolCall(name="open_application", arguments={"name": "notepadd plus"}), ToolContext()
    )
    assert not result.ok
    assert "Notepad" in (result.error or "")


def test_web_links_are_not_offered_as_applications() -> None:
    """Get-StartApps returns release notes and help pages alongside programs."""
    assert not InstalledApp("Release Notes", "file://C:/x/NEWS.txt").is_launchable
    assert not InstalledApp("Docs", "https://example.com").is_launchable
    assert InstalledApp("WhatsApp", "5319275A.WhatsAppDesktop!App").is_launchable


@pytest.mark.skipif(sys.platform == "win32", reason="checks the non-Windows path")
def test_nothing_is_registered_off_windows() -> None:
    """An unusable tool must not be advertised to the model."""
    assert application_tools() == []
