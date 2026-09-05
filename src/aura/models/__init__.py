"""Model abstraction layer. The replaceability seam (spec section 32)."""

from aura.models.base import LLMProvider
from aura.models.registry import available_providers, build_provider, register_provider

__all__ = ["LLMProvider", "available_providers", "build_provider", "register_provider"]
