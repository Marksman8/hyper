"""AURA error hierarchy.

Every error AURA raises deliberately derives from `AuraError` so the API layer
can distinguish "something we modelled" from "something we did not", and map
the former to a useful HTTP status instead of a 500.
"""

from __future__ import annotations


class AuraError(Exception):
    """Base class for all AURA errors."""

    http_status: int = 500
    code: str = "aura_error"


class ConfigError(AuraError):
    """Configuration is missing, malformed, or unsafe."""

    http_status = 500
    code = "config_error"


class ProviderError(AuraError):
    """A model provider failed."""

    http_status = 502
    code = "provider_error"


class ProviderUnavailableError(ProviderError):
    """The model server could not be reached at all.

    Distinct from ProviderError because this is the expected state when the
    user simply has not started llama-server yet, and deserves an actionable
    message rather than a stack trace.
    """

    http_status = 503
    code = "provider_unavailable"


class ProviderTimeoutError(ProviderError):
    """The model server accepted the request but did not finish in time."""

    http_status = 504
    code = "provider_timeout"


class StorageError(AuraError):
    """Persistence layer failure."""

    http_status = 500
    code = "storage_error"


class AuthError(AuraError):
    """Missing or invalid credentials."""

    http_status = 401
    code = "auth_error"


class PermissionDeniedError(AuraError):
    """The operation exceeds the caller's permission level."""

    http_status = 403
    code = "permission_denied"


class EmergencyStopError(AuraError):
    """Emergency stop is engaged; no model or tool calls are permitted."""

    http_status = 503
    code = "emergency_stop"


class NotFoundError(AuraError):
    """A requested resource does not exist."""

    http_status = 404
    code = "not_found"
