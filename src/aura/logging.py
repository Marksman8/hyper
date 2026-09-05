"""Structured logging.

Spec section 25 asks a specific question of this system: "Why did AURA do this?"
That is only answerable if every log line is a queryable record rather than a
sentence. So: structlog, key-value events, correlation ids bound to a context
var so they propagate through async call chains without being threaded through
every signature.

Two sinks, always:
  - stdout, human-readable in dev (`console`) or JSON in production
  - var/logs/aura.log, always JSON, for later analysis by the debugging
    dashboard (section 25) and the learning pipeline (section 17)
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import structlog

_request_id: ContextVar[str | None] = ContextVar("aura_request_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("aura_session_id", default=None)

_configured = False


def new_request_id() -> str:
    """Mint a short correlation id and bind it to the current context."""
    rid = uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def bind_request_id(rid: str) -> None:
    _request_id.set(rid)


def bind_session_id(sid: str | None) -> None:
    _session_id.set(sid)


def current_session_id() -> str | None:
    """Session bound to the current async context, if any."""
    return _session_id.get()


def _inject_context(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Attach correlation ids to every event without touching call sites."""
    rid = _request_id.get()
    if rid is not None:
        event_dict.setdefault("request_id", rid)
    sid = _session_id.get()
    if sid is not None:
        event_dict.setdefault("session_id", sid)
    return event_dict


def _redactor(keys: tuple[str, ...]) -> Any:
    """Drop secret-shaped values before they can reach a sink.

    Defence in depth: config.redacted() is the intended path, but a stray
    `log.info("x", api_key=...)` must not defeat it.
    """
    lowered = tuple(k.lower() for k in keys)

    def _is_secret(key: str) -> bool:
        """Redact on an exact name or a `*_<secret>` suffix, not a substring.

        Secrets are named for what they are and the name ends there:
        `api_token`, `auth_token`, `client_secret`, `api_key`. Telemetry puts
        the noun in the middle: `llm_first_token_ms`, `prompt_tokens`.

        A plain `"token" in key` check hit both, which silently destroyed the
        latency and usage numbers that section 25 exists to provide -- observed
        live as `llm_first_token_ms=***redacted***` in a real voice turn.
        Whole-word matching was not enough either, since `llm_first_token_ms`
        does contain the word "token".
        """
        k = key.lower()
        return any(k == secret or k.endswith("_" + secret) for secret in lowered)

    def processor(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        for k in list(event_dict):
            if _is_secret(k) and event_dict[k]:
                event_dict[k] = "***redacted***"
        return event_dict

    return processor


def configure_logging(
    level: str = "INFO",
    fmt: str = "console",
    log_file: Path | None = None,
    redact_keys: tuple[str, ...] = ("api_key", "token", "password", "secret"),
) -> None:
    """Idempotent global logging setup. Safe to call more than once."""
    global _configured
    if _configured:
        return

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _inject_context,
        _redactor(redact_keys),
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())

    console_renderer: Any = (
        structlog.dev.ConsoleRenderer(colors=True)
        if fmt == "console"
        else structlog.processors.JSONRenderer()
    )
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(
        structlog.stdlib.ProcessorFormatter(processor=console_renderer, foreign_pre_chain=shared)
    )
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # 10 MB x 5 keeps a couple of days of chatty debug logging bounded.
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=shared,
            )
        )
        root.addHandler(file_handler)

    # uvicorn installs its own handlers; route them through ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True

    # Third-party libraries that log every HTTP request at INFO. A model
    # download would otherwise bury AURA's own events under a hundred lines of
    # CDN redirects, which defeats the point of structured logging.
    for noisy in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "filelock",
        "faster_whisper",
        "openwakeword",
        "urllib3",
    ):
        logging.getLogger(noisy).setLevel(
            logging.DEBUG if level.upper() == "DEBUG" else logging.WARNING
        )

    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
