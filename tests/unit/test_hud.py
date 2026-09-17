"""The operator console.

The property worth testing here is not how it looks. It is that **every panel
is backed by something real** -- the rule the console is written under, and the
one the reference images it borrows from break on every panel.

This is the trick 10A.12 used for snapshot completeness, which caught a real
contract violation: assert the declaration against the thing it declares, so
drift is a failing test rather than a discovery.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from aura.api.app import PUBLIC_PATHS, create_app
from aura.config import Settings
from aura.core.events import EventType

HUD = Path("src/aura/api/static/hud.html")

_PANEL = re.compile(r'data-source="([^"]+)"')


def _sources() -> list[str]:
    return _PANEL.findall(HUD.read_text(encoding="utf-8"))


def test_the_console_exists_where_the_route_looks_for_it() -> None:
    assert HUD.is_file()


def test_every_panel_declares_a_source() -> None:
    """A panel with no declaration cannot be checked, so it is not allowed."""
    html = HUD.read_text(encoding="utf-8")
    panels = html.count('<section class="panel')
    assert panels >= 5, "the console lost panels"
    assert len(_sources()) == panels, "a panel is missing data-source"


def test_every_api_panel_names_a_route_that_exists(settings: Settings) -> None:
    """The check that stops a panel outliving its endpoint."""
    app = create_app(settings)
    routes = {getattr(r, "path", "") for r in app.routes}

    named = [s.split(":", 1)[1] for s in _sources() if s.startswith("api:")]
    assert named, "no panel reads the API at all"
    for path in named:
        assert path in routes, f"panel declares {path}, which is not a route"


def test_every_event_panel_names_a_real_event_type() -> None:
    """Same, for the ones driven by the bus rather than by a fetch."""
    known = {str(t) for t in EventType}
    named = [s.split(":", 1)[1] for s in _sources() if s.startswith("event:")]
    assert named, "no panel is driven by the event stream"
    for name in named:
        assert name in known, f"panel declares event {name!r}, which is not in the catalog"


def test_no_panel_declares_something_that_is_neither() -> None:
    for source in _sources():
        assert source.startswith(("api:", "event:")), f"unrecognised source {source!r}"


def test_the_console_handles_every_event_type_or_deliberately_ignores_it() -> None:
    """New event types should reach the console or be consciously skipped.

    Not a demand that each one get a panel -- most are stream lines. It fails
    when the catalog grows and nobody looked at the console, which is exactly
    how a dashboard starts lying.
    """
    script = HUD.read_text(encoding="utf-8")
    missing = [str(t) for t in EventType if f'"{t}"' not in script]
    assert not missing, f"the console never mentions: {missing}"


def test_the_console_shows_no_number_it_did_not_receive() -> None:
    """A guard against the reference images' favourite habit.

    Catches the obvious regression: a hard-coded percentage or a fake meter
    width left in the markup. Real values are written by script at runtime.
    """
    html = HUD.read_text(encoding="utf-8")
    markup = html.split("<script>")[0]
    assert "%<" not in markup, "a literal percentage is sitting in the markup"
    # Inline styles only. `width:100%` in the stylesheet is layout; a meter
    # pre-filled in the markup would be an inline style on an element.
    assert not re.search(r'style="[^"]*width:\s*\d', markup), "a meter is pre-filled"


async def test_the_shell_is_served_without_a_token(settings: Settings) -> None:
    """It is markup. The data behind it is not."""
    assert "/hud" in PUBLIC_PATHS
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://aura.test"
        ) as anon:
            page = await anon.get("/hud")
            assert page.status_code == 200
            assert "text/html" in page.headers["content-type"]

            # ...and the routes it reads are still closed.
            for path in ("/v1/status", "/v1/audit", "/v1/tools", "/v1/events"):
                assert (await anon.get(path)).status_code in (401, 403), path


def test_anything_toggled_with_hidden_survives_the_cascade() -> None:
    """The bug that shipped: a dialog that would not close.

    `[hidden]` gets its `display:none` from the browser's default stylesheet.
    A class rule that sets `display` is an author rule, so it wins, and
    `el.hidden = true` silently does nothing -- the console loaded perfectly
    behind an overlay that stayed put.

    No other test here could see it. The markup was right, the script was
    right, and the two were correct separately. So: every class that both sets
    `display` and is toggled via `.hidden` in the script must also carry a
    `[hidden]` rule turning it off.
    """
    html = HUD.read_text(encoding="utf-8")
    css = html.split("</style>")[0]
    script = html.split("<script>")[1] if "<script>" in html else ""

    toggled = set(re.findall(r'\$\("([\w-]+)"\)\.hidden\s*=', script))
    assert toggled, "nothing is toggled with .hidden; update this test if that changed"

    for element_id in sorted(toggled):
        block = re.search(
            rf'id="{re.escape(element_id)}"[^>]*class="([^"]+)"', html
        ) or re.search(rf'class="([^"]+)"[^>]*id="{re.escape(element_id)}"', html)
        if block is None:
            continue  # no class, so no author rule can be fighting [hidden]
        for cls in block.group(1).split():
            sets_display = re.search(rf"\.{re.escape(cls)}\s*\{{[^}}]*display\s*:", css)
            if not sets_display:
                continue
            guard = re.search(rf"\.{re.escape(cls)}\[hidden\]\s*\{{[^}}]*display\s*:\s*none", css)
            assert guard, (
                f'.{cls} sets display and #{element_id} is toggled with .hidden, '
                f"so it needs a `.{cls}[hidden]{{display:none}}` rule or it will never hide"
            )


def test_the_console_javascript_parses() -> None:
    """The one failure no other test here can see.

    Every assertion above reads the HTML as text, so a syntax error in the
    script would pass all of them and leave a console that renders its panels
    and then does nothing at all -- no stream, no status, no approvals. Node is
    already on this machine; where it is not, this skips rather than pretending
    to have checked.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; cannot parse the console script")

    html = HUD.read_text(encoding="utf-8")
    body = re.search(r"<script>(.*?)</script>", html, re.S)
    assert body, "the console has no script block"

    with tempfile.TemporaryDirectory() as tmp:
        js = Path(tmp) / "hud.js"
        js.write_text(body.group(1), encoding="utf-8")
        done = subprocess.run(
            [node, "--check", str(js)], capture_output=True, text=True, timeout=30
        )
    assert done.returncode == 0, "the console script does not parse: " + done.stderr


def test_the_console_never_hardcodes_a_token() -> None:
    html = HUD.read_text(encoding="utf-8")
    assert "var/secrets/api_token" in html, "it should tell the user where to find theirs"
    assert not re.search(r'Bearer\s+[A-Za-z0-9_\-]{16,}', html), "a literal token is embedded"
