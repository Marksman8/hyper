"""Shared fixtures.

Every fixture here builds AURA against a temporary database and the `echo`
provider. If a test ever needs real weights to pass, something above the
provider seam has reached below it -- see tests/unit/test_seam.py.
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
)
from aura.core.orchestrator import Orchestrator
from aura.memory.store import MemoryStore
from aura.models.providers.echo import EchoProvider
from aura.security.audit import AuditLog

TEST_TOKEN = "test-token-not-a-real-secret"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings: temp DB, temp logs, echo provider, fixed token."""
    return Settings(
        api=ApiConfig(host="127.0.0.1", port=8760),
        llm=LLMConfig(provider="echo"),
        memory=MemoryConfig(
            db_path=str(tmp_path / "aura.db"), history_turns=6, history_char_budget=2000
        ),
        security=SecurityConfig(
            api_token=TEST_TOKEN,
            token_file=str(tmp_path / "token"),
            audit_enabled=True,
            require_audit=True,
        ),
        logging=LoggingConfig(level="WARNING", file=str(tmp_path / "aura.log")),
    )


@pytest_asyncio.fixture
async def store(settings: Settings) -> AsyncIterator[MemoryStore]:
    s = MemoryStore(settings.memory.resolved_db_path())
    s.connect()
    try:
        yield s
    finally:
        s.close()


@pytest_asyncio.fixture
async def orchestrator(settings: Settings, store: MemoryStore) -> Orchestrator:
    return Orchestrator(settings, EchoProvider(), store, AuditLog(store))


@pytest_asyncio.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncClient]:
    """An authenticated client against the real ASGI app, lifespan included."""
    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=transport,
            base_url="http://aura.test",
            headers={"Authorization": "Bearer " + TEST_TOKEN},
        ) as c:
            yield c


@pytest_asyncio.fixture
async def anon_client(settings: Settings) -> AsyncIterator[AsyncClient]:
    """Same app, no credentials."""
    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://aura.test") as c:
            yield c
