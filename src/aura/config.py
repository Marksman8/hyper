"""Configuration system.

Layered, with strictly increasing precedence:

    1. config/default.yaml      committed defaults
    2. config/local.yaml        gitignored per-machine overrides
    3. .env / process env       AURA__SECTION__FIELD
    4. explicit constructor kwargs

Rationale for this shape rather than plain env vars: the config is large and
structured (spec sections 5, 8, 10 and 26 all add sections later), and a
commented YAML file is the only form of it that stays readable at that size.
Env vars remain the override path because that is how secrets and CI arrive.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from aura.errors import ConfigError

# Project root: src/aura/config.py -> src/aura -> src -> <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

_REDACTED = "***redacted***"


class _Section(BaseModel):
    """Base for config sections.

    Frozen, so a section cannot be mutated through a loaded Settings object.
    Without this, `settings.api.port = 0` succeeds even though Settings
    itself is frozen, and a subsystem can quietly change global config at
    runtime.
    """

    model_config = ConfigDict(frozen=True)


class AppConfig(_Section):
    name: str = "aura"
    version: str = "0.1.0"
    emergency_stop: bool = False


class ApiConfig(_Section):
    host: str = "127.0.0.1"
    port: int = Field(default=8760, ge=1024, le=65535)
    cors_origins: list[str] = Field(default_factory=list)
    request_timeout_s: int = Field(default=300, gt=0)

    @field_validator("host")
    @classmethod
    def _loopback_only(cls, v: str) -> str:
        # AURA's threat model assumes it is reachable only by the local user.
        # Binding to a wildcard address quietly exposes memory, audit logs and
        # (from Phase 3) tool execution to anything on the network. Refuse.
        if v not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError(
                "api.host must be a loopback address, got "
                + repr(v)
                + ". AURA must not be exposed beyond this machine."
            )
        return v


class LLMConfig(_Section):
    provider: str = "openai_compat"
    base_url: str = "http://127.0.0.1:8081/v1"
    api_key: str = ""
    model: str = "default"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    max_tokens: int = Field(default=1024, gt=0)
    timeout_s: int = Field(default=180, gt=0)
    connect_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


class MemoryConfig(_Section):
    db_path: str = "var/data/aura.db"
    history_turns: int = Field(default=12, ge=0, le=200)
    history_char_budget: int = Field(default=12000, gt=0)

    def resolved_db_path(self) -> Path:
        p = Path(self.db_path)
        return p if p.is_absolute() else PROJECT_ROOT / p


class SecurityConfig(_Section):
    api_token: str = ""
    token_file: str = "var/secrets/api_token"  # noqa: S105 - a path, not a secret
    audit_enabled: bool = True
    require_audit: bool = True

    def resolved_token_file(self) -> Path:
        p = Path(self.token_file)
        return p if p.is_absolute() else PROJECT_ROOT / p


class RuntimeConfig(_Section):
    manage_model_server: bool = False
    resource_poll_s: int = Field(default=10, gt=0)


class LoggingConfig(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["console", "json"] = "console"
    file: str = "var/logs/aura.log"
    redact_keys: list[str] = Field(
        default_factory=lambda: ["api_token", "api_key", "password", "secret", "token"]
    )

    def resolved_file(self) -> Path:
        p = Path(self.file)
        return p if p.is_absolute() else PROJECT_ROOT / p


class PersonaConfig(_Section):
    name: str = "AURA"
    system_prompt: str = "You are AURA, a local-first personal AI assistant."


class _YamlSource(PydanticBaseSettingsSource):
    """Settings source that reads a YAML file, tolerating absence."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._path = path

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # Unused: __call__ is overridden to return the whole mapping at once.
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        if not self._path.is_file():
            return {}
        try:
            raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError("Invalid YAML in " + str(self._path) + ": " + str(exc)) from exc
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ConfigError(str(self._path) + " must contain a mapping at the top level.")
        return raw


class Settings(BaseSettings):
    """Root configuration object. Immutable once loaded."""

    model_config = SettingsConfigDict(
        env_prefix="AURA__",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    app: AppConfig = Field(default_factory=AppConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # First source wins in pydantic-settings, so this tuple is in
        # DESCENDING precedence order.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls, CONFIG_DIR / "local.yaml"),
            _YamlSource(settings_cls, CONFIG_DIR / "default.yaml"),
        )

    def redacted(self) -> dict[str, Any]:
        """Dump config with secrets removed.

        Every path that shows config to a human, a log, or a model goes through
        this. There is no other public serializer.
        """
        keys = {k.lower() for k in self.logging.redact_keys}

        def scrub(node: Any) -> Any:
            if isinstance(node, dict):
                return {
                    k: (_REDACTED if (k.lower() in keys and v) else scrub(v))
                    for k, v in node.items()
                }
            if isinstance(node, list):
                return [scrub(v) for v in node]
            return node

        result: dict[str, Any] = scrub(self.model_dump())
        return result


class ModelEntry(BaseModel):
    """One entry in the model catalogue."""

    display_name: str
    repo_id: str
    filename: str
    quantization: str = "Q4_K_M"
    context_length: int = 8192
    vram_gb: float = 0.0
    gpu_layers: int = -1
    supports_tools: bool = False
    notes: str = ""


class ModelCatalogue(BaseModel):
    active: str
    catalogue: dict[str, ModelEntry]

    def active_entry(self) -> ModelEntry:
        try:
            return self.catalogue[self.active]
        except KeyError as exc:
            raise ConfigError(
                "models.yaml: active model "
                + repr(self.active)
                + " is not in the catalogue. Known: "
                + str(sorted(self.catalogue))
            ) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings. Call `get_settings.cache_clear()` in tests."""
    return Settings()


@lru_cache(maxsize=1)
def get_model_catalogue() -> ModelCatalogue:
    path = CONFIG_DIR / "models.yaml"
    if not path.is_file():
        raise ConfigError("Missing model catalogue at " + str(path))
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelCatalogue.model_validate(raw)


def models_dir() -> Path:
    """Where GGUF weights live. Overridable for users with a second drive."""
    override = os.environ.get("AURA__MODELS_DIR")
    return Path(override) if override else PROJECT_ROOT / "var" / "models"
