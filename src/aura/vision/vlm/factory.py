"""Build the VLM provider from config (10C.1).

The one place a provider name becomes a class, so nothing above learns which
server is answering -- the same arrangement as the model and automation
providers.
"""

from __future__ import annotations

from aura.config import Settings
from aura.errors import ConfigError
from aura.logging import get_logger
from aura.vision.vlm.base import NullVLM, VLMProvider

log = get_logger(__name__)


def build_vlm(settings: Settings) -> VLMProvider:
    """The configured VLM, or one that cannot see.

    `NullVLM` is the default and is not a failure mode. This machine cannot
    hold the language model and a vision model at once, so running without a
    VLM is a supported configuration that everything above has to work with.
    """
    cfg = settings.vision.vlm
    if not cfg.enabled or cfg.provider == "null":
        return NullVLM()

    if cfg.provider == "openai_compat":
        from aura.vision.vlm.openai_compat import OpenAICompatVLM

        return OpenAICompatVLM(
            base_url=cfg.base_url,
            model=cfg.model,
            api_key=cfg.resolved_api_key(),
            timeout_s=cfg.timeout_s,
        )

    raise ConfigError(
        f"Unknown vision.vlm.provider {cfg.provider!r}. Known: openai_compat, null"
    )
