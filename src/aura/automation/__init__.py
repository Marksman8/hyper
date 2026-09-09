"""Automation providers — third-party integrations behind one seam.

The authority model is in `manifest.py` and in ADR-0004. In one line: a
provider executes; AURA's local, version-controlled manifest decides what it
is permitted to execute.
"""

from __future__ import annotations

from aura.automation.base import AutomationProvider, AutomationResult
from aura.automation.manifest import ManifestError, WorkflowSpec, load_manifest

__all__ = [
    "AutomationProvider",
    "AutomationResult",
    "ManifestError",
    "WorkflowSpec",
    "load_manifest",
]
