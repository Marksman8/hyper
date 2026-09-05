"""Log redaction.

Regression cover for a bug found in a live voice turn: the redactor matched
`token` as a substring, so `llm_first_token_ms=***redacted***` appeared in the
logs. It silently destroyed exactly the latency telemetry that spec section 25
exists to provide, while looking like the security control was working.
"""

from __future__ import annotations

import pytest

from aura.logging import _redactor

SECRETS = ("api_token", "api_key", "password", "secret", "token")


def redact(event: dict[str, object]) -> dict[str, object]:
    processor = _redactor(SECRETS)
    result: dict[str, object] = processor(None, "info", dict(event))
    return result


@pytest.mark.parametrize(
    "field",
    [
        "llm_first_token_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "tts_first_chunk_ms",
        "stt_ms",
        "token_count",
    ],
)
def test_telemetry_fields_survive(field: str) -> None:
    """Measurements that merely mention tokens are not secrets."""
    assert redact({field: 123})[field] == 123


@pytest.mark.parametrize(
    "field",
    ["api_token", "token", "auth_token", "access_token", "api_key", "password", "client_secret"],
)
def test_secret_fields_are_redacted(field: str) -> None:
    assert redact({field: "hunter2"})[field] == "***redacted***"


def test_empty_secrets_are_left_alone() -> None:
    """An unset secret should read as unset, not as a redacted value."""
    assert redact({"api_token": ""})["api_token"] == ""


def test_redaction_is_case_insensitive() -> None:
    assert redact({"API_TOKEN": "hunter2"})["API_TOKEN"] == "***redacted***"
