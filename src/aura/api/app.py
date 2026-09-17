"""FastAPI application.

The local API (spec section 24). Every route except /healthz requires a bearer
token, including from localhost.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from aura import __version__
from aura.api.schemas import (
    AgentRequest,
    AgentResponse,
    AgentStepOut,
    ApprovalRedeem,
    ChatRequest,
    ChatResponse,
    ClientEvent,
    EStopRequest,
    EvidenceOut,
    FactExplanation,
    FactIn,
    FactOut,
    FocusModeRequest,
    OpenAIChatRequest,
    ProfileDimension,
    ProfileOut,
    ProviderStatus,
    ResourceStatus,
    SessionOut,
    SpeechRequest,
    StatusResponse,
    ToolCallRequest,
    ToolCallResponse,
    ToolSpecOut,
    TurnMatchOut,
    TurnOut,
    UsageOut,
    VisionAppRequest,
    VisionModeRequest,
)
from aura.config import Settings, get_settings
from aura.core.emitter import EventEmitter
from aura.core.events import EventBus
from aura.core.orchestrator import Orchestrator
from aura.core.presence import PresenceDriver
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
from aura.security.approvals import ApprovalRegistry
from aura.security.audit import AuditLog
from aura.security.auth import load_or_create_token, verify_token

log = get_logger(__name__)

# `/hud` is public because it is markup and contains no data: every byte it
# displays comes from a route below, each of which still requires the bearer
# token. Serving the shell without auth is what lets a browser open it at all
# -- the page then asks for the token once and keeps it locally.
PUBLIC_PATHS = frozenset({"/healthz", "/docs", "/openapi.json", "/redoc", "/hud"})


@dataclass
class AppState:
    """Wiring, assembled once at startup and shared by every request."""

    settings: Settings
    store: MemoryStore
    audit: AuditLog
    orchestrator: Orchestrator
    token: str
    started_at: float
    tools: Any = None
    # Phase 10A. One-way notification out; nonce-bound authority back in.
    # Deliberately two separate objects: see docs/architecture/10-event-bus-contract.md
    events: EventBus = field(default_factory=EventBus)
    approvals: ApprovalRegistry = field(default_factory=ApprovalRegistry)
    presence: Any = None
    watcher: Any = None
    estop: Any = None
    # Phase 10B extension. Spawns `python -m aura.voice.synth` per utterance,
    # so it holds no model and Core stays torch-free and 69 MB.
    speech: Any = None


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

        # One bus per process, one emitter over it. Built before the
        # subsystems so they can be handed it at construction rather than
        # having it injected later, which would leave a window where events
        # are silently dropped.
        bus = EventBus()
        emitter = EventEmitter(bus)
        # ONE kill switch, shared by the orchestrator and the executor. They
        # held separate booleans until the Stage 1 audit showed the cost: the
        # stop refused model calls and let every tool run.
        from aura.security.estop import build_estop

        estop = build_estop(cfg)
        # Derives the visible state from the event contract and publishes
        # PRESENCE_STATE. Knows nothing about rendering; the overlay is just
        # another subscriber.
        presence = PresenceDriver(bus)

        tools = None
        runner = None
        if cfg.tools.enabled:
            from aura.tools.factory import build_context, build_executor
            from aura.tools.runner import ExecutorToolRunner

            tools = build_executor(cfg, audit, events=emitter, estop=estop)
            runner = ExecutorToolRunner(tools, build_context(cfg))

        app.state.aura = AppState(
            settings=cfg,
            store=store,
            audit=audit,
            orchestrator=Orchestrator(
                cfg, provider, store, audit, tools=runner, events=emitter
            ),
            events=bus,
            presence=presence,
            estop=estop,
            token=token,
            started_at=time.time(),
            tools=tools,
        )

        # The vision service is a process-wide singleton (one policy, one
        # answer). Attach the emitter here so a mode change announces itself
        # however it was triggered -- CLI, API, or the stop_looking tool.
        if cfg.vision.mode is not None:
            try:
                from aura.vision.service import get_vision

                get_vision(cfg).events = emitter
            except Exception as exc:
                log.debug("events.vision_not_wired", error=str(exc))

        await presence.start()

        # The APP_CHANGED producer. Optional: it needs pywin32, and a machine
        # without the vision extra should still get everything else.
        watcher = None
        try:
            from aura.vision.service import get_vision
            from aura.vision.watcher import WindowWatcher

            watcher = WindowWatcher(emitter, get_vision(cfg).policy)
            await watcher.start()
            app.state.aura.watcher = watcher
            log.info("aura.window_watcher_started")
        except Exception as exc:
            log.info("aura.window_watcher_unavailable", error=str(exc))

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
            # Presence first: it holds a subscription, and the bus's drain
            # should not be waiting on a consumer that is about to stop.
            if watcher is not None:
                await watcher.stop()
            await presence.stop()
            await bus.aclose(drain_timeout=1.0)
            if tools is not None:
                tools.close()
            live: AppState = app.state.aura
            if live.speech is not None:
                live.speech.close()
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


def _snapshot_provider(state: AppState) -> Any:
    """Build a snapshot provider for one subscriber.

    **Runs under the bus's publish lock**, so it must not block, do I/O, or
    publish. Anything that could be slow is resolved out here, when the closure
    is built, rather than inside it.

    In 10A.2 the snapshot is deliberately thin: presence state does not exist
    until 10A.4, and inventing fields now would mean inventing their semantics
    too. It grows as consumers arrive.
    """
    vision_mode = "unknown"
    try:
        from aura.vision.service import get_vision

        vision_mode = str(get_vision(state.settings).policy.mode)
    except Exception as exc:  # vision is optional; a snapshot must still work
        log.debug("events.snapshot_vision_unavailable", error=str(exc))

    emergency = state.orchestrator.emergency_stop
    seq = state.events.seq
    # Resolved out here: the provider runs under the publish lock.
    presence = (
        state.presence.machine.snapshot() if state.presence is not None else {}
    )
    # APP_CHANGED is catalogued as STATE owning `foreground_application`. A
    # state event that never reaches the snapshot is not recoverable, and a
    # client that reconnects has no way to learn it -- which is the whole
    # difference between STATE and OCCURRENCE.
    foreground = getattr(state.watcher, "current", "") if state.watcher else ""
    

    def provider() -> dict[str, Any]:
        return {
            "version": __version__,
            "seq_at_snapshot": seq,
            "emergency_stop": emergency,
            "vision_mode": vision_mode,
            "started_at": state.started_at,
            "foreground_application": foreground,
            # Requirement 10: a client recovers by being told what is TRUE NOW,
            # never by replaying the occurrences that produced it.
            **presence,
        }

    return provider


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

    @app.get("/hud", tags=["status"], include_in_schema=False)
    async def hud() -> Any:
        """The operator console.

        A single static file, served rather than built. See the comment at the
        top of `static/hud.html` for the rule it is written under: every lit
        element resolves to a value one of these routes returned.
        """
        from fastapi.responses import FileResponse

        page = Path(__file__).with_name("static") / "hud.html"
        if not page.is_file():  # pragma: no cover - packaging failure
            raise AuraError("The console is missing from this installation.")
        return FileResponse(page, media_type="text/html")

    @app.post("/v1/speech", tags=["voice"])
    async def speak(body: SpeechRequest, state: AppState = Depends(get_state)) -> Any:
        """Turn text into audio, for clients that have no voice of their own.

        The console uses this so a typed conversation can still be heard.

        Core does not synthesise: it spawns `python -m aura.voice.synth`, which
        loads a voice, writes a WAV to stdout and exits. An in-process version
        was built first and measured -- 69 MB became 115 MB permanently -- and
        `test_voice_seam.py` forbids the import that made it possible. See the
        module docstring.
        """
        import anyio
        from fastapi.responses import Response

        from aura.api.speech_route import SpeechService

        if state.speech is None:
            state.speech = SpeechService(state.settings)

        # Synthesis is blocking and CPU-bound; off the event loop it goes, or
        # the SSE stream and every other request stall behind it.
        wav, seconds = await anyio.to_thread.run_sync(
            state.speech.synthesize_wav, body.text
        )
        await state.audit.record(
            actor="user",
            action="speech.synthesize",
            target=f"{len(body.text)} chars",
            detail={"audio_s": round(seconds, 2)},
        )
        return Response(
            content=wav,
            media_type="audio/wav",
            headers={"X-Audio-Seconds": f"{seconds:.2f}", "Cache-Control": "no-store"},
        )

    @app.get("/v1/speech", tags=["voice"])
    async def speech_state(state: AppState = Depends(get_state)) -> dict[str, Any]:
        """Whether speech is available. It is never resident; see the module."""
        from aura.api.speech_route import speech_status

        return speech_status(state.speech)

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
                _sse(
                    state.orchestrator,
                    body.message,
                    body.session_id,
                    params,
                    spoken=body.spoken,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result = await state.orchestrator.chat(
            body.message,
            session_id=body.session_id,
            params=params,
            spoken=body.spoken,
            personalize=body.personalize,
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

    @app.get("/v1/memory/search", response_model=list[TurnMatchOut], tags=["memory"])
    async def search_memory(
        q: str, limit: int = 8, state: AppState = Depends(get_state)
    ) -> list[TurnMatchOut]:
        """Search past conversation (FTS5/BM25, not embeddings -- see
        docs/architecture/05-phase6-memory.md section 2)."""
        hits = await state.store.search_turns(q, limit=min(limit, 50))
        return [
            TurnMatchOut(
                session_id=h.session_id,
                turn_id=h.turn.id,
                role=str(h.turn.role),
                content=h.turn.content,
                created_at=h.turn.created_at,
                score=h.score,
            )
            for h in hits
        ]

    @app.get("/v1/memory/{key}/why", response_model=FactExplanation, tags=["memory"])
    async def why_memory(key: str, state: AppState = Depends(get_state)) -> FactExplanation:
        """Why does AURA believe this? The Phase 6 exit criterion."""
        from aura.memory.beliefs import assess

        fact = await state.store.get_fact("user", key)
        if fact is None:
            raise NotFoundError(f"AURA knows nothing under {key!r}.")

        evidence = await state.store.evidence_for(fact.id, limit=50)
        belief = assess(evidence)
        return FactExplanation(
            fact=FactOut(
                key=fact.key,
                value=fact.value,
                confidence=fact.confidence,
                evidence_count=fact.evidence_count,
                source=fact.source,
                permanence=fact.permanence.value,
                last_confirmed_at=fact.last_confirmed_at,
            ),
            reason=belief.explain(),
            believed=fact.confidence >= state.settings.memory.injection_floor,
            evidence=[
                EvidenceOut(
                    quote=e.quote,
                    source=e.source,
                    session_id=e.session_id,
                    turn_id=e.turn_id,
                    created_at=e.created_at,
                )
                for e in evidence
            ],
        )

    @app.delete("/v1/memory/{key}", tags=["memory"])
    async def forget(key: str, state: AppState = Depends(get_state)) -> dict[str, bool]:
        deleted = await state.orchestrator.forget(key)
        if not deleted:
            raise NotFoundError("Nothing remembered under key " + repr(key))
        return {"deleted": True}

    # -- agent (spec section 7) ------------------------------------------------

    @app.post("/v1/agent", response_model=AgentResponse, tags=["agent"])
    async def agent(body: AgentRequest, state: AppState = Depends(get_state)) -> AgentResponse:
        """Run a turn that may use tools.

        Falls back to a plain conversational turn when tools are disabled, so a
        client does not have to check first.
        """
        params = GenerationParams(
            temperature=state.settings.llm.temperature,
            top_p=state.settings.llm.top_p,
            max_tokens=state.settings.llm.max_tokens,
        ).merged(temperature=body.temperature)

        result = await state.orchestrator.act(
            body.message,
            session_id=body.session_id,
            params=params,
            approve=body.approve,
        )
        outcome = result.agent
        return AgentResponse(
            session_id=result.session_id,
            content=result.content,
            model=result.model,
            provider=result.provider,
            usage=UsageOut(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.total_tokens,
            ),
            latency_ms=result.latency_ms,
            tool_calls_made=getattr(outcome, "tool_calls_made", 0),
            steps=[
                AgentStepOut(
                    index=st.index,
                    kind=st.kind,
                    content=st.content,
                    tools=st.tool_calls,
                    latency_ms=st.latency_ms,
                )
                for st in getattr(outcome, "steps", [])
            ],
            stopped_early=getattr(outcome, "stopped_early", False),
            stop_reason=getattr(outcome, "stop_reason", result.finish_reason),
            pending_confirmations=getattr(outcome, "pending_confirmations", []),
        )

    # -- tools (spec section 8) ----------------------------------------------

    @app.get("/v1/tools", response_model=list[ToolSpecOut], tags=["tools"])
    async def list_tools(state: AppState = Depends(get_state)) -> list[ToolSpecOut]:
        """What AURA can do, and what needs approval first."""
        if state.tools is None:
            return []
        from aura.security.permissions import Decision

        policy = state.tools._policy
        return [
            ToolSpecOut(
                name=spec.name,
                description=spec.description,
                parameters=spec.parameters,
                level=int(spec.level),
                level_label=spec.level.label,
                timeout_s=spec.timeout_s,
                reversible=spec.reversible,
                needs_confirmation=policy.evaluate(spec.level) is Decision.CONFIRM,
            )
            for spec in state.tools._registry.specs()
        ]

    @app.post("/v1/tools/{name}", response_model=ToolCallResponse, tags=["tools"])
    async def call_tool(
        name: str, body: ToolCallRequest, state: AppState = Depends(get_state)
    ) -> ToolCallResponse:
        """Run one tool.

        Phase 3 only: the caller chooses the tool. Wiring this into the model's
        own reasoning is Phase 4, and the permission gate exists so that
        arriving does not change what is allowed.
        """
        if state.tools is None:
            raise NotFoundError("Tools are disabled (tools.enabled is false).")

        from aura.tools.factory import build_context
        from aura.tools.types import ToolCall

        result = await state.tools.execute(
            ToolCall(name=name, arguments=body.arguments, confirmed=body.confirmed),
            build_context(state.settings, body.session_id),
        )
        return ToolCallResponse(
            call_id=result.call_id,
            name=result.name,
            ok=result.ok,
            content=result.content,
            error=result.error,
            data=result.data,
            duration_ms=result.duration_ms,
            decision=result.decision,
        )

    # -- personalization (Phase 7) -------------------------------------------

    @app.get("/v1/profile", response_model=ProfileOut, tags=["memory"])
    async def profile(state: AppState = Depends(get_state)) -> ProfileOut:
        """What AURA adapts for this user, and which fact drives each change."""
        from aura.personalization import directives as build_directives

        user = await state.orchestrator.user_profile()
        return ProfileOut(
            enabled=state.settings.personalization.enabled,
            dimensions=[ProfileDimension(**d) for d in user.describe()],  # type: ignore[arg-type]
            directives=build_directives(user),
            other_facts=[f"{f.key}: {f.value}" for f in user.other_facts],
        )

    # -- events (Phase 10A.2) -------------------------------------------------
    #
    # One-way out, nonce-bound authority back in. The stream cannot grant
    # anything; the approval endpoint cannot publish anything. See
    # docs/architecture/10-event-bus-contract.md sections 6 and 7.

    @app.get("/v1/events", tags=["events"])
    async def events(request: Request, state: AppState = Depends(get_state)) -> Any:
        """Subscribe to AURA's event stream.

        Resumable: send `Last-Event-ID` with the last sequence number seen. If
        the gap is still in the replay buffer it is replayed; otherwise a
        SNAPSHOT is sent first and the stream continues from that consistent
        cut.
        """
        from aura.api.events_route import event_stream, open_stream, parse_last_event_id

        last = parse_last_event_id(request.headers.get("last-event-id"))
        sub, first = open_stream(state.events, last, _snapshot_provider(state))
        log.info(
            "events.stream_opened",
            subscription=sub.id,
            resumed_from=last,
            snapshot=first is not None,
        )
        return StreamingResponse(
            event_stream(state.events, sub, first, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Without this, a reverse proxy will buffer the stream and the
                # overlay goes blank until enough bytes accumulate.
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/v1/focus", tags=["control"])
    async def set_focus_mode(
        body: FocusModeRequest, state: AppState = Depends(get_state)
    ) -> dict[str, Any]:
        """Quieter, not less capable.

        Audited because it changes how much AURA interrupts, which is the kind
        of setting a person will later wonder about having changed.
        """
        EventEmitter(state.events).focus_mode(enabled=body.enabled)
        await state.audit.record(
            actor="user", action="control.focus_mode", detail={"enabled": body.enabled}
        )
        return {"focus_mode": body.enabled}

    @app.post("/v1/events/client", tags=["events"])
    async def publish_client_event(
        body: ClientEvent, state: AppState = Depends(get_state)
    ) -> dict[str, Any]:
        """Bridge an occurrence from a client process onto the bus.

        Exists because the voice loop is a separate process: wake-word
        detection, barge-in and speech playback happen there and cannot reach
        Core's in-process bus. Deliberately not a general publish endpoint --
        see ClientEvent.

        **Origin is never taken from the request.** Each type is dispatched to
        its typed emitter method, and the emitter reads origin and delivery out
        of the catalog. A client chooses *which catalogued thing happened*, and
        nothing else about how it is labelled.
        """
        emitter = EventEmitter(state.events)
        if body.type == "user_spoke":
            emitter.user_spoke("x" * body.length, spoken=body.spoken)
        elif body.type == "user_interrupted":
            emitter.user_interrupted(body.method or "unknown")
        elif body.type == "speech_started":
            emitter.speech_started(length=body.length)
        elif body.type == "speech_chunk":
            emitter.speech_chunk(body.text, envelope=body.envelope)
        else:
            emitter.speech_ended(completed=body.completed)
        return {"accepted": True, "seq": state.events.seq}

    @app.post("/v1/approvals/{approval_id}", tags=["events"])
    async def redeem_approval(
        approval_id: str, body: ApprovalRedeem, state: AppState = Depends(get_state)
    ) -> dict[str, Any]:
        """Grant or deny a pending permission request.

        **The only path by which authority flows back into AURA.** Requires the
        nonce issued with the request and the request id it was bound to, so an
        approval cannot be forged, replayed, or moved onto a different action.
        """
        from aura.security.permissions import Decision, PermissionLevel

        result = state.approvals.redeem(
            approval_id,
            body.nonce,
            body.request_id,
            granted=body.granted,
        )
        # Audited at CRITICAL whatever the outcome: a refused or replayed
        # approval is exactly the thing someone would want to find later.
        await state.audit.record(
            actor="user",
            action="approval.redeem",
            level=PermissionLevel.CRITICAL,
            target=approval_id,
            decision=Decision.ALLOW if result.granted else Decision.DENY,
            detail={
                "outcome": str(result.outcome),
                "request_id": body.request_id,
                "granted_flag": body.granted,
            },
        )
        return {
            "approval_id": approval_id,
            "outcome": str(result.outcome),
            "granted": result.granted,
            "reason": result.reason,
        }

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

    # -- automation -----------------------------------------------------------

    @app.get("/v1/automation", tags=["automation"])
    async def automation_status(state: AppState = Depends(get_state)) -> dict[str, Any]:
        """What AURA is permitted to automate, and at what level.

        **Read-only, and it stays that way** (ADR-0004 D12 rule 13). Declaring a
        workflow means editing a version-controlled file, so that the answer to
        "what can AURA do?" is always a diff. An endpoint that could write the
        manifest would move that authority onto the network.
        """
        from aura.automation.factory import load_workflows
        from aura.tools.automation import describe_workflows

        cfg = state.settings.automation
        return {
            "enabled": cfg.enabled,
            "provider": cfg.provider if cfg.enabled else None,
            "manifest": cfg.manifest_path,
            "workflows": describe_workflows(load_workflows(state.settings)),
        }

    # -- vision ---------------------------------------------------------------
    #
    # The human half of the vision controls. The model gets tools that can only
    # narrow what AURA sees; widening happens here, where a person is asking.

    @app.get("/v1/vision", tags=["vision"])
    async def vision_status(state: AppState = Depends(get_state)) -> dict[str, Any]:
        from aura.vision.service import get_vision

        return dict(get_vision(state.settings).status())

    @app.post("/v1/vision/mode", tags=["vision"])
    async def set_vision_mode(
        body: VisionModeRequest, state: AppState = Depends(get_state)
    ) -> dict[str, Any]:
        """Turn vision on or off. Audited, because it changes what AURA can see."""
        from aura.vision.privacy import VisionMode
        from aura.vision.service import get_vision

        service = get_vision(state.settings)
        service.set_mode(VisionMode(body.mode))
        await state.audit.record(
            actor="user", action="vision.set_mode", detail={"mode": body.mode}
        )
        return dict(service.status())

    @app.post("/v1/vision/deny", tags=["vision"])
    async def deny_vision_app(
        body: VisionAppRequest, state: AppState = Depends(get_state)
    ) -> dict[str, Any]:
        from aura.vision.service import get_vision

        service = get_vision(state.settings)
        service.policy.deny_app(body.app)
        await state.audit.record(
            actor="user", action="vision.deny_app", detail={"app": body.app}
        )
        return dict(service.status())

    @app.post("/v1/vision/allow", tags=["vision"])
    async def allow_vision_app(
        body: VisionAppRequest, state: AppState = Depends(get_state)
    ) -> dict[str, Any]:
        """Restrict vision to named applications - an allow-list of one, grown
        by repeating the call."""
        from aura.vision.service import get_vision

        service = get_vision(state.settings)
        service.policy.allow_only(body.app)
        await state.audit.record(
            actor="user", action="vision.allow_app", detail={"app": body.app}
        )
        return dict(service.status())

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
    spoken: bool = False,
) -> AsyncIterator[str]:
    """Server-sent events for a streaming turn.

    Errors are delivered as a terminal `data:` frame rather than an HTTP status,
    because by the time generation fails the 200 response headers are already
    on the wire. Clients must inspect frames, not just the status code.
    """
    created = int(time.time())
    sent_session = False
    try:
        async for chunk in orchestrator.stream(
            user_text, session_id=session_id, params=params, spoken=spoken
        ):
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
