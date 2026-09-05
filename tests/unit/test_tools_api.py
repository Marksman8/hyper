"""The tool endpoints.

The property under test is that the HTTP surface cannot be used to get around
the permission gate. A client that forgets `confirmed`, or sends a path outside
the workspace, must be refused with a structured answer rather than a 500 — and
the refusal must be recorded.
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
)

TEST_TOKEN = "test-token-not-a-real-secret"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    (root / "var" / "secrets").mkdir(parents=True)
    (root / "notes.txt").write_text("hello from the workspace")
    (root / "var" / "secrets" / "api_token").write_text("SEKRIT")
    (tmp_path / "outside.txt").write_text("not yours")
    return root


@pytest.fixture
def tool_settings(tmp_path: Path, workspace: Path) -> Settings:
    return Settings(
        api=ApiConfig(host="127.0.0.1", port=8760),
        llm=LLMConfig(provider="echo"),
        memory=MemoryConfig(db_path=str(tmp_path / "aura.db")),
        security=SecurityConfig(api_token=TEST_TOKEN, token_file=str(tmp_path / "token")),
        logging=LoggingConfig(level="WARNING", file=str(tmp_path / "aura.log")),
        tools=ToolsConfig(workspace_roots=[str(workspace)]),
    )


@pytest_asyncio.fixture
async def tool_client(tool_settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(tool_settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://aura.test",
            headers={"Authorization": "Bearer " + TEST_TOKEN},
        ) as c:
            yield c


# -- discovery --------------------------------------------------------------


async def test_lists_tools_with_their_permission_levels(tool_client: AsyncClient) -> None:
    tools = (await tool_client.get("/v1/tools")).json()
    by_name = {t["name"]: t for t in tools}

    assert "read_file" in by_name
    assert by_name["read_file"]["level"] == 1
    assert by_name["read_file"]["needs_confirmation"] is False

    assert by_name["write_file"]["level"] == 2
    assert by_name["write_file"]["needs_confirmation"] is True
    assert by_name["run_python"]["needs_confirmation"] is True


async def test_tool_endpoints_require_authentication(tool_settings: Settings) -> None:
    app = create_app(tool_settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://aura.test"
        ) as anon:
            assert (await anon.get("/v1/tools")).status_code == 401
            assert (await anon.post("/v1/tools/read_file", json={})).status_code == 401


# -- execution --------------------------------------------------------------


async def test_reading_a_workspace_file_works(tool_client: AsyncClient, workspace: Path) -> None:
    resp = await tool_client.post(
        "/v1/tools/read_file", json={"arguments": {"path": str(workspace / "notes.txt")}}
    )
    body = resp.json()
    assert body["ok"] is True
    assert body["content"] == "hello from the workspace"


async def test_unknown_tool_is_a_structured_refusal_not_a_crash(
    tool_client: AsyncClient,
) -> None:
    resp = await tool_client.post("/v1/tools/rm_rf", json={"arguments": {}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["decision"] == "deny"


# -- the gate ---------------------------------------------------------------


async def test_writing_without_confirmation_is_refused_and_writes_nothing(
    tool_client: AsyncClient, workspace: Path
) -> None:
    target = workspace / "created.txt"
    resp = await tool_client.post(
        "/v1/tools/write_file",
        json={"arguments": {"path": str(target), "content": "x"}},
    )
    body = resp.json()
    assert body["ok"] is False
    assert body["decision"] == "confirm"
    assert not target.exists(), "the HTTP path bypassed the confirmation gate"


async def test_writing_works_once_confirmed(tool_client: AsyncClient, workspace: Path) -> None:
    target = workspace / "created.txt"
    resp = await tool_client.post(
        "/v1/tools/write_file",
        json={"arguments": {"path": str(target), "content": "written"}, "confirmed": True},
    )
    assert resp.json()["ok"] is True, resp.json()["error"]
    assert target.read_text() == "written"


async def test_a_refused_write_is_audited(tool_client: AsyncClient, workspace: Path) -> None:
    """The confirm decision must reach the audit table.

    It could not until Phase 3: the table's CHECK constraint listed only
    allow/deny/error, so recording a confirmation refusal raised instead —
    and with require_audit on, that turned a clean refusal into a 500.
    """
    await tool_client.post(
        "/v1/tools/write_file",
        json={"arguments": {"path": str(workspace / "x.txt"), "content": "x"}},
    )
    rows = (await tool_client.get("/v1/audit")).json()
    assert any(r["action"] == "tool.write_file" and r["decision"] == "confirm" for r in rows)


# -- containment ------------------------------------------------------------


async def test_reading_outside_the_workspace_is_denied(
    tool_client: AsyncClient, tmp_path: Path
) -> None:
    resp = await tool_client.post(
        "/v1/tools/read_file",
        json={"arguments": {"path": str(tmp_path / "outside.txt")}},
    )
    body = resp.json()
    assert body["ok"] is False
    assert "outside the workspace" in body["error"]


async def test_reading_a_secret_is_denied_even_inside_the_workspace(
    tool_client: AsyncClient, workspace: Path
) -> None:
    """AURA's own token lives in the workspace and must stay unreadable."""
    resp = await tool_client.post(
        "/v1/tools/read_file",
        json={"arguments": {"path": str(workspace / "var" / "secrets" / "api_token")}},
    )
    body = resp.json()
    assert body["ok"] is False
    assert "SEKRIT" not in (body["content"] or "")
    assert "deny pattern" in body["error"]


async def test_confirmation_does_not_unlock_a_denied_path(
    tool_client: AsyncClient, tmp_path: Path
) -> None:
    """Approval raises the permission ceiling, not the workspace boundary."""
    resp = await tool_client.post(
        "/v1/tools/write_file",
        json={
            "arguments": {"path": str(tmp_path / "escaped.txt"), "content": "x"},
            "confirmed": True,
        },
    )
    assert resp.json()["ok"] is False
    assert not (tmp_path / "escaped.txt").exists()


# -- python -----------------------------------------------------------------


async def test_python_runs_when_confirmed(tool_client: AsyncClient) -> None:
    resp = await tool_client.post(
        "/v1/tools/run_python",
        json={"arguments": {"code": "print(6 * 7)"}, "confirmed": True},
    )
    body = resp.json()
    assert body["ok"] is True, body["error"]
    assert "42" in body["content"]


async def test_python_reports_a_traceback_rather_than_swallowing_it(
    tool_client: AsyncClient,
) -> None:
    """A failing snippet is an observation the model must be able to act on."""
    resp = await tool_client.post(
        "/v1/tools/run_python",
        json={"arguments": {"code": "raise ValueError('boom')"}, "confirmed": True},
    )
    body = resp.json()
    assert body["ok"] is False
    assert "ValueError" in body["error"]
    assert body["data"]["exit_code"] != 0
