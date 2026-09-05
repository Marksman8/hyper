"""Provider registry.

One place decides which concrete provider exists, driven by config. Adding a
backend later (a Transformers in-process provider, a fine-tuned AURA endpoint)
means registering a factory here -- and changing nothing else in AURA.
"""

from __future__ import annotations

from collections.abc import Callable

from aura.config import Settings
from aura.core.types import GenerationParams
from aura.errors import ConfigError
from aura.logging import get_logger
from aura.models.base import LLMProvider
from aura.models.providers.echo import EchoProvider
from aura.models.providers.openai_compat import OpenAICompatProvider

log = get_logger(__name__)

ProviderFactory = Callable[[Settings], LLMProvider]


def _build_openai_compat(settings: Settings) -> LLMProvider:
    cfg = settings.llm
    return OpenAICompatProvider(
        base_url=cfg.base_url,
        model=cfg.model,
        api_key=cfg.api_key,
        timeout_s=float(cfg.timeout_s),
        connect_retries=cfg.connect_retries,
        default_params=GenerationParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
        ),
    )


def _build_echo(_settings: Settings) -> LLMProvider:
    return EchoProvider()


_REGISTRY: dict[str, ProviderFactory] = {
    "openai_compat": _build_openai_compat,
    "echo": _build_echo,
}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register an additional provider factory (used by tests and plugins)."""
    _REGISTRY[name] = factory


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def build_provider(settings: Settings) -> LLMProvider:
    """Instantiate the provider named by config."""
    name = settings.llm.provider
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise ConfigError(
            "Unknown llm.provider " + repr(name) + ". Available: " + str(available_providers())
        ) from exc
    provider = factory(settings)
    log.info(
        "provider.built",
        provider=provider.name,
        endpoint=provider.info().endpoint,
        model=provider.info().model,
    )
    return provider
