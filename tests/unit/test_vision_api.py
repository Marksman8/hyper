"""The vision HTTP surface.

Two properties. First, the controls actually reach the process that takes
screenshots — a CLI that flipped a local copy of the policy would report
success while the server kept looking. Second, turning vision on is
authenticated, audited, and reachable only here: there is no tool that does it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from aura.api.app import create_app
from aura.config import (
    ApiConfig,
    LLMConfig,
    LoggingConfig,
    MemoryConfig,
    SecurityConfig,
    Settings,
    ToolsConfig,
    VisionConfig,
)
from aura.vision.service import get_vision, reset_vision

TEST_TOKEN = "test-token-not-a-real-secret"


@pytest.fixture
def vision_settings(tmp_path: Path) -> Settings:
    return Settings(
        api=ApiConfig(host="127.0.0.1", port=8760),
        llm=LLMConfig(provider="echo"),
        memory=MemoryConfig(db_path=str(tmp_path / "aura.db")),
        security=SecurityConfig(api_token=TEST_TOKEN, token_file=str(tmp_path / "token")),
        logging=LoggingConfig(level="WARNING", file=str(tmp_path / "aura.log")),
        tools=ToolsConfig(workspace_roots=[str(tmp_path)]),
        # `null` so no OCR model is loaded and no real screen is touched.
        vision=VisionConfig(mode="private", ocr_backend="null"),
    )


@pytest_asyncio.fixture
async def client(vision_settings: Settings) -> AsyncIterator[AsyncClient]:
    # The service is a process-wide singleton by design (one policy, one
    # answer), so tests reset it explicitly rather than leaking state forward.
    reset_vision()
    app = create_app(vision_settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://aura.test",
            headers={"Authorization": "Bearer " + TEST_TOKEN},
        ) as c:
            yield c
    reset_vision()


async def test_status_reports_private_by_default(client: AsyncClient) -> None:
    body = (await client.get("/v1/vision")).json()
    assert body["mode"] == "private"
    assert body["deny_title_patterns"] > 0


async def test_mode_change_reaches_the_capture_process(
    client: AsyncClient, vision_settings: Settings
) -> None:
    """The point of routing this over HTTP.

    After the request, the policy object the *capture path* consults must be
    the one that changed. Asserting on the response body alone would pass even
    if the endpoint mutated a throwaway copy.
    """
    resp = await client.post("/v1/vision/mode", json={"mode": "assist"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "assist"

    service = get_vision(vision_settings)
    assert service.capture.policy.may_capture(
        window_title="notes.md", process_name="Code.exe"
    )


async def test_turning_vision_off_takes_effect_immediately(
    client: AsyncClient, vision_settings: Settings
) -> None:
    await client.post("/v1/vision/mode", json={"mode": "active"})
    service = get_vision(vision_settings)
    assert service.policy.may_capture(window_title="x", process_name="x.exe")

    await client.post("/v1/vision/mode", json={"mode": "private"})
    assert not service.policy.may_capture(window_title="x", process_name="x.exe")


async def test_deny_app_reaches_the_capture_process(
    client: AsyncClient, vision_settings: Settings
) -> None:
    await client.post("/v1/vision/mode", json={"mode": "active"})
    await client.post("/v1/vision/deny", json={"app": "Chrome"})

    policy = get_vision(vision_settings).capture.policy
    assert not policy.may_capture(window_title="AURA - Google Chrome", process_name="chrome.exe")
    assert policy.may_capture(window_title="notes.md", process_name="Code.exe")


async def test_allow_app_restricts_to_it(client: AsyncClient, vision_settings: Settings) -> None:
    await client.post("/v1/vision/mode", json={"mode": "active"})
    await client.post("/v1/vision/allow", json={"app": "Code"})

    policy = get_vision(vision_settings).capture.policy
    assert policy.may_capture(window_title="notes.md", process_name="Code.exe")
    assert not policy.may_capture(window_title="Spotify", process_name="Spotify.exe")


async def test_mode_change_is_audited(client: AsyncClient) -> None:
    """Changing what AURA can see is exactly the kind of thing section 25 means."""
    await client.post("/v1/vision/mode", json={"mode": "assist"})
    entries = (await client.get("/v1/audit")).json()
    assert any(e["action"] == "vision.set_mode" for e in entries)


async def test_an_invalid_mode_is_rejected(client: AsyncClient) -> None:
    assert (await client.post("/v1/vision/mode", json={"mode": "everything"})).status_code == 422


async def test_vision_controls_require_authentication(vision_settings: Settings) -> None:
    """An unauthenticated request must not be able to switch the screen on."""
    reset_vision()
    app = create_app(vision_settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://aura.test"
        ) as anon:
            assert (await anon.get("/v1/vision")).status_code == 401
            assert (
                await anon.post("/v1/vision/mode", json={"mode": "active"})
            ).status_code == 401

    assert get_vision(vision_settings).policy.mode.value == "private"
    reset_vision()


async def test_vision_tools_are_registered(client: AsyncClient) -> None:
    """And no registered tool can widen what AURA sees."""
    names = {t["name"] for t in (await client.get("/v1/tools")).json()}
    assert {"read_screen", "stop_looking", "vision_status", "list_monitors"} <= names

    by_name = {t["name"]: t for t in (await client.get("/v1/tools")).json()}
    assert by_name["read_screen"]["level"] == 1
    assert by_name["stop_looking"]["level"] == 0
