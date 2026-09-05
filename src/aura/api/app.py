"""FastAPI application.

The local API (spec section 24). Every route except /healthz requires a bearer
token, including from localhost.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from aura import __version__
from aura.api.schemas import (
    ChatRequest,
    ChatResponse,
    EStopRequest,
    FactIn,
    FactOut,
    OpenAIChatRequest,
    ProviderStatus,
    ResourceStatus,
    SessionOut,
    StatusResponse,
    TurnOut,
    UsageOut,
)
from aura.config import Settings, get_settings
from aura.core.orchestrator import Orchestrator
from aura.core.types import GenerationParams
from aura.errors import AuraError, AuthError, NotFoundError
from aura.logging import (
    bind_session_id,
    configure_logging,
    current_session_id,
    get_logger,
    new_request_id,
)
from aura.memory.store import MemoryStore
from aura.models.registry import build_provider
from aura.runtime.resources import snapshot
from aura.security.audit import AuditLog
from aura.security.auth import load_or_create_token, verify_token

log = get_logger(__name__)

PUBLIC_PATHS = frozenset({"/healthz", "/docs", "/openapi.json", "/redoc"})


@dataclass
class AppState:
    """Wiring, assembled once at startup and shared by every request."""

    settings: Settings
    store: MemoryStore
    audit: AuditLog
    orchestrator: Orchestrator
    token: str
    started_at: float


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.aura
    return state


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()

    configure_logging(
        level=cfg.logging.level,
        fmt=cfg.logging.format,
        log_file=cfg.logging.resolved_file(),
        redact_keys=tuple(cfg.logging.redact_keys),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = MemoryStore(cfg.memory.resolved_db_path())
        store.connect()
        # Session-scoped facts never outlive the process that created them.
        await store.forget_expired()

        provider = build_provider(cfg)
        audit = AuditLog(
            store, enabled=cfg.security.audit_enabled, required=cfg.security.require_audit
        )
        token = load_or_create_token(cfg.security.resolved_token_file(), cfg.security.api_token)

        app.state.aura = AppState(
            settings=cfg,
            store=store,
            audit=audit,
            orchestrator=Orchestrator(cfg, provider, store, audit),
            token=token,
            started_at=time.time(),
        )

        health = await provider.health()
        log.info(
            "aura.started",
            version=__version__,
            host=cfg.api.host,
            port=cfg.api.port,
            provider=provider.name,
            model_server=("up" if health.available else "down"),
        )
        if not health.available:
            log.warning("aura.model_unavailable", detail=health.detail)

        try:
            yield
        finally:
            await provider.close()
            store.close()
            log.info("aura.stopped")

    app = FastAPI(
        title="AURA",
        version=__version__,
        description="Local-first personal AI runtime.",
        lifespan=lifespan,
    )

    if cfg.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.api.cors_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    _install_middleware(app)
    _install_error_handlers(app)
    _install_routes(app)
    return app


# -- middleware --------------------------------------------------------------


def _install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def correlate_and_authenticate(request: Request, call_next: Any) -> Any:
        rid = new_request_id()
        bind_session_id(None)
        started = time.perf_counter()

        if request.url.path not in PUBLIC_PATHS:
            try:
                state: AppState = request.app.state.aura
                verify_token(request.headers.get("authorization"), state.token)
            except AuraError as exc:
                return _error_response(exc, rid)

        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        log.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        return response


def _error_response(exc: AuraError, request_id: str | None = None) -> JSONResponse:
    payload: dict[str, Any] = {
        "error": {
            "code": exc.code,
            "message": str(exc),
            "type": type(exc).__name__,
        }
    }
    if request_id:
        payload["error"]["request_id"] = request_id
    headers = {"WWW-Authenticate": "Bearer"} if isinstance(exc, AuthError) else None
    return JSONResponse(status_code=exc.http_status, content=payload, headers=headers)


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AuraError)
    async def handle_aura_error(_request: Request, exc: AuraError) -> JSONResponse:
        # Expected failures (model server down, bad session id) are logged at
        # warning; only genuine 5xx bugs deserve an error-level line.
        level = log.warning if exc.http_status < 500 else log.error
        level("api.error", code=exc.code, message=str(exc))
        return _error_response(exc)


# -- routes ------------------------------------------------------------------


def _install_routes(app: FastAPI) -> None:
    @app.get("/healthz", tags=["status"])
    async def healthz() -> dict[str, str]:
        """Unauthenticated liveness probe. Reveals nothing beyond 'running'."""
        return {"status": "ok", "version": __version__}

    @app.get("/v1/status", response_model=StatusResponse, tags=["status"])
    async def status(state: AppState = Depends(get_state)) -> StatusResponse:
        provider = state.orchestrator.provider
        info = provider.info()
        health = await provider.health()
        res = snapshot()
        return StatusResponse(
            name=state.settings.app.name,
            version=__version__,
            uptime_s=round(time.time() - state.started_at, 1),
            emergency_stop=state.orchestrator.emergency_stop,
            provider=ProviderStatus(
                name=info.name,
                kind=info.kind,
                endpoint=info.endpoint,
                model=health.model or info.model,
                available=health.available,
                detail=health.detail,
                latency_ms=health.latency_ms,
            ),
            memory=await state.store.stats(),
            resources=ResourceStatus(
                cpu_percent=res.cpu_percent,
                ram_used_gb=res.ram_used_gb,
                ram_total_gb=res.ram_total_gb,
                ram_percent=res.ram_percent,
                process_ram_mb=res.process_ram_mb,
                vram_used_gb=res.vram_used_gb,
                vram_total_gb=res.vram_total_gb,
            ),
        )

    @app.get("/v1/config", tags=["status"])
    async def config(state: AppState = Depends(get_state)) -> dict[str, Any]:
        """Effective configuration, secrets removed."""
        return state.settings.redacted()

    # -- chat ----------------------------------------------------------------

    @app.post("/v1/chat", tags=["chat"])
    async def chat(body: ChatRequest, state: AppState = Depends(get_state)) -> Any:
        params = GenerationParams(
            temperature=state.settings.llm.temperature,
            top_p=state.settings.llm.top_p,
            max_tokens=state.settings.llm.max_tokens,
        ).merged(
            temperature=body.temperature,
            top_p=body.top_p,
            max_tokens=body.max_tokens,
        )

        if body.stream:
            return StreamingResponse(
                _sse(state.orchestrator, body.message, body.session_id, params),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result = await state.orchestrator.chat(
            body.message, session_id=body.session_id, params=params
        )
        return ChatResponse(
            session_id=result.session_id,
            content=result.content,
            model=result.model,
            provider=result.provider,
            usage=UsageOut(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.total_tokens,
            ),
            finish_reason=result.finish_reason,
            latency_ms=result.latency_ms,
        )

    @app.post("/v1/chat/completions", tags=["chat"])
    async def openai_chat(body: OpenAIChatRequest, state: AppState = Depends(get_state)) -> Any:
        """OpenAI-compatible surface (spec section 24).

        Only the final user message is taken; AURA supplies system prompt and
        history from its own memory. Clients that want AURA-as-a-dumb-model
        should talk to the model server directly.
        """
        user_text = next((m.content for m in reversed(body.messages) if m.role == "user"), "")
        if not user_text:
            raise NotFoundError("Request contained no user message.")

        params = GenerationParams(
            temperature=state.settings.llm.temperature,
            top_p=state.settings.llm.top_p,
            max_tokens=state.settings.llm.max_tokens,
        ).merged(
            temperature=body.temperature,
            top_p=body.top_p,
            max_tokens=body.max_tokens,
        )

        if body.stream:
            return StreamingResponse(
                _sse(
                    state.orchestrator,
                    user_text,
                    body.session_id,
                    params,
                    openai_shape=True,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result = await state.orchestrator.chat(user_text, session_id=body.session_id, params=params)
        return {
            "id": "chatcmpl-" + result.session_id[:16],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.content},
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            },
            "aura": {"session_id": result.session_id, "provider": result.provider},
        }

    # -- sessions ------------------------------------------------------------

    @app.get("/v1/sessions", response_model=list[SessionOut], tags=["memory"])
    async def list_sessions(
        limit: int = 50, state: AppState = Depends(get_state)
    ) -> list[SessionOut]:
        sessions = await state.store.list_sessions(min(limit, 500))
        return [
            SessionOut(
                id=s.id,
                title=s.title,
                created_at=s.created_at,
                updated_at=s.updated_at,
            )
            for s in sessions
        ]

    @app.get("/v1/sessions/{session_id}", response_model=list[TurnOut], tags=["memory"])
    async def session_turns(
        session_id: str, limit: int = 100, state: AppState = Depends(get_state)
    ) -> list[TurnOut]:
        await state.store.get_session(session_id)  # 404s if absent
        turns = await state.store.recent_turns(session_id, min(limit, 1000))
        return [
            TurnOut(
                id=t.id,
                role=t.role.value,
                content=t.content,
                created_at=t.created_at,
                model=t.model,
                latency_ms=t.latency_ms,
                error=t.error,
            )
            for t in turns
        ]

    @app.delete("/v1/sessions/{session_id}", tags=["memory"])
    async def delete_session(
        session_id: str, state: AppState = Depends(get_state)
    ) -> dict[str, bool]:
        deleted = await state.store.delete_session(session_id)
        if not deleted:
            raise NotFoundError("No session with id " + repr(session_id))
        await state.audit.record(actor="user", action="memory.delete_session", target=session_id)
        return {"deleted": True}

    # -- facts (spec sections 15, 16) ----------------------------------------

    @app.get("/v1/memory", response_model=list[FactOut], tags=["memory"])
    async def list_memory(state: AppState = Depends(get_state)) -> list[FactOut]:
        """Everything AURA durably knows about the user. Always inspectable."""
        facts = await state.orchestrator.known_facts()
        return [
            FactOut(
                key=f.key,
                value=f.value,
                confidence=f.confidence,
                evidence_count=f.evidence_count,
                source=f.source,
                permanence=f.permanence.value,
                last_confirmed_at=f.last_confirmed_at,
            )
            for f in facts
        ]

    @app.post("/v1/memory", response_model=FactOut, tags=["memory"])
    async def remember(body: FactIn, state: AppState = Depends(get_state)) -> FactOut:
        fact = await state.orchestrator.remember(body.key, body.value, body.permanence)
        return FactOut(
            key=fact.key,
            value=fact.value,
            confidence=fact.confidence,
            evidence_count=fact.evidence_count,
            source=fact.source,
            permanence=fact.permanence.value,
            last_confirmed_at=fact.last_confirmed_at,
        )

    @app.delete("/v1/memory/{key}", tags=["memory"])
    async def forget(key: str, state: AppState = Depends(get_state)) -> dict[str, bool]:
        deleted = await state.orchestrator.forget(key)
        if not deleted:
            raise NotFoundError("Nothing remembered under key " + repr(key))
        return {"deleted": True}

    # -- control & observability ---------------------------------------------

    @app.post("/v1/control/estop", tags=["control"])
    async def estop(body: EStopRequest, state: AppState = Depends(get_state)) -> dict[str, bool]:
        state.orchestrator.set_emergency_stop(body.engaged)
        await state.audit.record(
            actor="user",
            action="control.emergency_stop",
            detail={"engaged": body.engaged},
        )
        return {"emergency_stop": body.engaged}

    @app.get("/v1/audit", tags=["control"])
    async def audit_log(
        limit: int = 100, state: AppState = Depends(get_state)
    ) -> list[dict[str, Any]]:
        """Why did AURA do this? (spec section 25)"""
        return await state.audit.recent(min(limit, 1000))


# -- streaming ---------------------------------------------------------------


async def _sse(
    orchestrator: Orchestrator,
    user_text: str,
    session_id: str | None,
    params: GenerationParams,
    openai_shape: bool = False,
) -> AsyncIterator[str]:
    """Server-sent events for a streaming turn.

    Errors are delivered as a terminal `data:` frame rather than an HTTP status,
    because by the time generation fails the 200 response headers are already
    on the wire. Clients must inspect frames, not just the status code.
    """
    created = int(time.time())
    sent_session = False
    try:
        async for chunk in orchestrator.stream(user_text, session_id=session_id, params=params):
            # Do NOT break here. The orchestrator's generator persists the
            # assistant turn after its final yield; abandoning it mid-flight
            # means that epilogue never runs.
            if chunk.done or not chunk.delta:
                continue
            if openai_shape:
                frame: dict[str, Any] = {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "choices": [
                        {"index": 0, "delta": {"content": chunk.delta}, "finish_reason": None}
                    ],
                }
            else:
                frame = {"type": "delta", "content": chunk.delta}
                if not sent_session:
                    # The client needs the session id to continue the
                    # conversation, and only learns it once the turn opens.
                    frame["session_id"] = current_session_id()
                    sent_session = True
            yield "data: " + json.dumps(frame) + "\n\n"
    except AuraError as exc:
        log.warning("stream.failed", code=exc.code, message=str(exc))
        yield (
            "data: "
            + json.dumps({"type": "error", "error": {"code": exc.code, "message": str(exc)}})
            + "\n\n"
        )
    else:
        if not openai_shape:
            yield "data: " + json.dumps({"type": "done"}) + "\n\n"
    yield "data: [DONE]\n\n"


app_factory = create_app
