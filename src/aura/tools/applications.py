"""Open installed applications.

**The safety property, and it is the whole design:** the model never supplies a
path or a command. It supplies a *name*, which is matched against the list of
applications Windows already knows about, and what gets launched is that
entry's AppID. A tool that took an executable path would be `run_anything`
wearing a friendlier label.

Discovery uses `Get-StartApps`, which is what the Start menu itself is built
on, so it covers desktop programs and Store apps alike — WhatsApp, Chrome,
Spotify and VS Code all resolve the same way. No hardcoded install paths to rot
when something updates.

Launching goes through `explorer.exe shell:AppsFolder\\<AppID>`, the documented
way to start an app by identity rather than by executable.

Windows only. `application_tools()` returns nothing elsewhere, so the model is
never offered a tool that cannot work.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

from aura.logging import get_logger
from aura.security.permissions import PermissionLevel
from aura.tools.base import Tool
from aura.tools.types import ToolCall, ToolContext, ToolResult, ToolSpec

log = get_logger(__name__)

# Enumerating the Start menu takes a moment and the answer rarely changes.
_CACHE_TTL_S = 300.0
_cache: tuple[float, list[InstalledApp]] | None = None

_POWERSHELL = (
    os.path.join(
        os.environ.get("SystemRoot", r"C:\Windows"),
        "System32",
        "WindowsPowerShell",
        "v1.0",
        "powershell.exe",
    )
    if sys.platform == "win32"
    else "powershell"
)
_EXPLORER = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "explorer.exe")

# Entries that are documents or web links rather than programs. Get-StartApps
# returns release notes and help pages alongside real applications; offering
# those as "apps" produces confusing behaviour.
_NOT_AN_APP = ("file://", "http://", "https://")

# How long to wait for evidence that a launch worked. Long enough for a Store
# app to spawn its host process, short enough that an approved action still
# feels immediate.
_LAUNCH_CONFIRM_S = 3.0


@dataclass(frozen=True, slots=True)
class InstalledApp:
    name: str
    app_id: str

    @property
    def is_launchable(self) -> bool:
        return not self.app_id.lower().startswith(_NOT_AN_APP)


def discover_apps(refresh: bool = False) -> list[InstalledApp]:
    """Applications Windows knows how to start, newest answer cached briefly."""
    global _cache
    now = time.monotonic()
    if not refresh and _cache is not None and now - _cache[0] < _CACHE_TTL_S:
        return _cache[1]

    if sys.platform != "win32":
        return []

    try:
        raw = subprocess.run(  # noqa: S603 - absolute argv[0], fixed command, no shell
            [
                _POWERSHELL,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-StartApps | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
        parsed = json.loads(raw or "[]")
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as exc:
        log.warning("tools.app_discovery_failed", error=str(exc))
        return []

    if isinstance(parsed, dict):  # a single result is not wrapped in a list
        parsed = [parsed]

    apps = [
        InstalledApp(name=str(item["Name"]), app_id=str(item["AppID"]))
        for item in parsed
        if isinstance(item, dict) and item.get("Name") and item.get("AppID")
    ]
    apps = [a for a in apps if a.is_launchable]
    _cache = (now, apps)
    log.info("tools.apps_discovered", count=len(apps))
    return apps


def find_app(query: str, apps: list[InstalledApp]) -> InstalledApp | None:
    """Match a spoken or typed name to one installed application.

    Progressively looser: exact, then prefix, then substring. Ordered that way
    so "notepad" reaches Notepad rather than "Notepad++ Installer Release
    Notes", and so a precise name is never beaten by a longer accidental match.
    """
    wanted = query.strip().lower()
    if not wanted:
        return None

    for app in apps:
        if app.name.lower() == wanted:
            return app
    for app in apps:
        if app.name.lower().startswith(wanted):
            return app
    # Shortest containing match: "chrome" should find "Google Chrome", not
    # "Google Chrome Beta Uninstall Helper".
    contains = [a for a in apps if wanted in a.name.lower()]
    if contains:
        return min(contains, key=lambda a: len(a.name))

    # Every word present, in any order. Handles the way people actually name
    # things — "vs code" for Visual Studio Code, "chrome google" for Google
    # Chrome — which plain substring matching cannot.
    words = [w for w in wanted.split() if w]
    if words:
        token_hits = [a for a in apps if all(w in a.name.lower() for w in words)]
        if token_hits:
            return min(token_hits, key=lambda a: len(a.name))

        # Abbreviations, so "vs code" reaches "Visual Studio Code" even though
        # "vs" appears nowhere in it.
        abbreviated = [a for a in apps if _matches_abbreviated(words, a.name)]
        if abbreviated:
            return min(abbreviated, key=lambda a: len(a.name))

    return None


def _running_processes() -> set[int]:
    """PIDs currently running, for detecting what a launch created."""
    try:
        import psutil

        return set(psutil.pids())
    except Exception:
        return set()


def _wait_for_new_process(before: set[int]) -> bool:
    """Did anything new start? Best effort, and honest when it cannot tell.

    Matching on process *name* does not work: WhatsApp runs as
    `WhatsApp.Root`, Chrome forks helpers, and Store apps are hosted under
    names that have nothing to do with the app. A new PID appearing is the
    signal that generalises.
    """
    if not before:
        return False

    import time as _time

    deadline = _time.monotonic() + _LAUNCH_CONFIRM_S
    while _time.monotonic() < deadline:
        _time.sleep(0.25)
        if _running_processes() - before:
            return True
    return False


def _matches_abbreviated(query_words: list[str], app_name: str) -> bool:
    """Does the query name this app, allowing initials for any run of words?

    People compress the boring part and keep the distinctive one: "vs code" for
    Visual Studio Code, "ms word" for Microsoft Word. So each query word must
    either appear in the next app word, or be the initials of a run of app
    words starting there — consumed left to right, so word order still matters.
    """
    app_words = app_name.lower().split()
    position = 0

    for token in query_words:
        matched = False

        # Initials of consecutive words: "vs" over ["visual", "studio"].
        acronym = ""
        for index in range(position, len(app_words)):
            acronym += app_words[index][0]
            if acronym == token:
                position = index + 1
                matched = True
                break
            if len(acronym) > len(token):
                break
        if matched:
            continue

        # Otherwise the token has to appear in one of the remaining words.
        for index in range(position, len(app_words)):
            if token in app_words[index]:
                position = index + 1
                matched = True
                break
        if not matched:
            return False

    return True


class ListApplicationsTool(Tool):
    """What can be opened on this machine."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list_applications",
            description=(
                "List the applications installed on this computer that can be "
                "opened. Use this when unsure whether something is installed, "
                "or what it is called."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Optional substring to narrow the list, e.g. 'chrome'.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
            level=PermissionLevel.READ,
            timeout_s=25.0,
        )

    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        apps = discover_apps()
        if not apps:
            return ToolResult.failure(
                call,
                "Could not list applications. This tool works on Windows only.",
            )

        needle = str(call.arguments.get("filter") or "").strip().lower()
        if needle:
            apps = [a for a in apps if needle in a.name.lower()]

        if not apps:
            return ToolResult.success(
                call, f"No installed application matches {needle!r}.", data={"count": 0}
            )

        names = sorted(a.name for a in apps)
        shown = names[:120]
        body = "\n".join(shown)
        if len(names) > len(shown):
            body += f"\n[{len(names)} total, {len(shown)} shown]"
        return ToolResult.success(call, body, data={"count": len(names)})


class OpenApplicationTool(Tool):
    """Launch an installed application by name."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="open_application",
            description=(
                "Open an installed application by name, for example 'Chrome', "
                "'WhatsApp' or 'Notepad'. Only applications already installed "
                "on this computer can be opened. Requires confirmation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Application name, e.g. 'WhatsApp'.",
                        "minLength": 1,
                    }
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            level=PermissionLevel.ACT,
            timeout_s=25.0,
            # Opening a window is undone by closing it.
            reversible=True,
            # Running it twice may open a second window.
            idempotent=False,
        )

    def run(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        if sys.platform != "win32":
            return ToolResult.failure(call, "Opening applications is supported on Windows only.")

        wanted = str(call.arguments["name"])
        apps = discover_apps()
        if not apps:
            return ToolResult.failure(call, "Could not enumerate installed applications.")

        app = find_app(wanted, apps)
        if app is None:
            close = [a.name for a in apps if wanted.lower()[:4] in a.name.lower()][:8]
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            return ToolResult.failure(call, f"No installed application named {wanted!r}.{hint}")

        # The AppID comes from the discovered list, never from the model. That
        # is what makes this "open one of the things already installed" rather
        # than "run whatever string you were given".
        target = f"shell:AppsFolder\\{app.app_id}"
        before = _running_processes()

        try:
            subprocess.Popen(  # noqa: S603 - absolute argv[0], AppID from discovery
                [_EXPLORER, target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )
        except OSError as exc:
            return ToolResult.failure(call, f"Could not open {app.name}: {exc}")

        # explorer.exe returns immediately and successfully whether or not the
        # application actually starts, so its exit code proves nothing. Without
        # a check, a stale AppID produces a confident "Opened WhatsApp." for an
        # app that never appeared — which is the kind of unverified claim that
        # makes an assistant untrustworthy.
        started = _wait_for_new_process(before)

        log.info("tools.app_opened", app=app.name, app_id=app.app_id, confirmed=bool(started))
        data = {
            "name": app.name,
            "app_id": app.app_id,
            "requested": wanted,
            "confirmed": bool(started),
        }
        if started:
            return ToolResult.success(call, f"Opened {app.name}.", data=data)

        # Not confirmed is not the same as failed: the app may already have
        # been running, or may simply be slow. Say what is actually known.
        return ToolResult.success(
            call,
            f"Asked Windows to open {app.name}. No new process appeared within "
            f"{_LAUNCH_CONFIRM_S:.0f}s — it may already have been running, or "
            "may still be starting.",
            data=data,
        )


def application_tools() -> list[Tool]:
    """Nothing outside Windows: an unusable tool should not be advertised."""
    if sys.platform != "win32":
        return []
    return [ListApplicationsTool(), OpenApplicationTool()]
