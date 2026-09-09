"""Build the automation provider from config.

The one place a provider name becomes a class, so `tools/automation.py` never
learns that n8n exists (ADR-0004 D10) — the same arrangement as the model
providers behind `LLMProvider`.
"""

from __future__ import annotations

from pathlib import Path

from aura.automation.base import AutomationProvider
from aura.automation.manifest import WorkflowSpec, load_manifest
from aura.config import PROJECT_ROOT, Settings
from aura.errors import ConfigError
from aura.logging import get_logger

log = get_logger(__name__)


def build_provider(settings: Settings) -> AutomationProvider | None:
    """The configured provider, or None when automation is switched off."""
    cfg = settings.automation
    if not cfg.enabled:
        return None

    if cfg.provider == "n8n":
        from aura.automation.n8n import N8NProvider

        return N8NProvider(
            base_url=cfg.base_url,
            auth_header=cfg.auth_header,
            auth_token=cfg.auth_token,
            auth_token_file=cfg.auth_token_file,
            verify_tls=cfg.verify_tls,
        )

    raise ConfigError(
        f"Unknown automation provider {cfg.provider!r}. "
        "Add it to aura/automation/factory.py deliberately."
    )


def load_workflows(settings: Settings) -> list[WorkflowSpec]:
    """The declared workflows.

    Read even when `enabled` is false, so a manifest mistake is caught by
    tests and by startup validation rather than lying in wait until the day
    automation is switched on.
    """
    raw = settings.automation.manifest_path
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return load_manifest(path)
