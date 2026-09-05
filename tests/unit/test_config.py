"""Configuration behaviour, including the two security-relevant rules."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aura.config import ApiConfig, LoggingConfig, SecurityConfig, Settings


def test_defaults_come_from_yaml() -> None:
    """default.yaml is actually read, not just documented."""
    s = Settings()
    assert s.app.name == "aura"
    assert s.api.port == 8760
    assert "AURA" in s.persona.system_prompt


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AURA__API__PORT", "9999")
    monkeypatch.setenv("AURA__LLM__PROVIDER", "echo")
    s = Settings()
    assert s.api.port == 9999
    assert s.llm.provider == "echo"


def test_init_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AURA__API__PORT", "9999")
    s = Settings(api=ApiConfig(port=7777))
    assert s.api.port == 7777


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::"])
def test_non_loopback_bind_is_rejected(host: str) -> None:
    """AURA must not be exposed to the network by a config typo."""
    with pytest.raises(ValidationError, match="loopback"):
        ApiConfig(host=host)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_bind_is_allowed(host: str) -> None:
    assert ApiConfig(host=host).host == host


def test_base_url_trailing_slash_is_normalised() -> None:
    s = Settings()
    assert not s.llm.base_url.endswith("/")


def test_redacted_hides_secrets() -> None:
    s = Settings(
        security=SecurityConfig(api_token="super-secret-value"),
        logging=LoggingConfig(redact_keys=["api_token", "api_key"]),
    )
    dumped = s.redacted()
    assert dumped["security"]["api_token"] == "***redacted***"
    assert "super-secret-value" not in str(dumped)


def test_redacted_leaves_empty_secrets_alone() -> None:
    """An unset secret should read as unset, not as a redacted value."""
    s = Settings(security=SecurityConfig(api_token=""))
    assert s.redacted()["security"]["api_token"] == ""


def test_settings_are_immutable() -> None:
    s = Settings()
    with pytest.raises(ValidationError):
        s.api.port = 1  # type: ignore[misc]
