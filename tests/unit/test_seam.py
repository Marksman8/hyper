"""The architecture invariant (spec section 32).

AURA's whole premise is that its brain is replaceable. That is not enforced by
good intentions; it is enforced here. If this test fails, some module has
started depending on a specific inference backend and the model is no longer a
configuration value.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "aura"
PROVIDERS_PKG = SRC / "models" / "providers"

# Libraries that bind AURA to one way of running a model. Importing any of them
# outside models/providers/ means the seam has leaked.
FORBIDDEN_ROOTS = frozenset(
    {
        "anthropic",
        "openai",
        "transformers",
        "torch",
        "llama_cpp",
        "vllm",
        "ollama",
        "ctransformers",
        "sentence_transformers",
        "google",
        "cohere",
        "mistralai",
    }
)


def _python_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if PROVIDERS_PKG not in p.parents)


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.name))
def test_no_inference_library_outside_providers(path: Path) -> None:
    leaked = _imported_roots(path) & FORBIDDEN_ROOTS
    assert not leaked, (
        f"{path.relative_to(SRC)} imports {sorted(leaked)}. "
        "Backend-specific imports belong in aura/models/providers/ only."
    )


def test_orchestrator_depends_only_on_the_abstraction() -> None:
    """The orchestrator must never name a concrete provider class."""
    source = (SRC / "core" / "orchestrator.py").read_text(encoding="utf-8")
    for concrete in ("OpenAICompatProvider", "EchoProvider", "llama", "gguf"):
        assert concrete not in source, (
            f"orchestrator.py references {concrete!r}; it may only use LLMProvider."
        )


def test_no_model_filenames_outside_config() -> None:
    """Weight filenames live in config/models.yaml, never in code."""
    offenders = [
        p.relative_to(SRC) for p in SRC.rglob("*.py") if ".gguf" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"Model filenames hard-coded in: {offenders}"


# The same discipline, one layer down. Vision backends are replaceable too: mss
# is a screen-capture shim and RapidOCR is one recogniser among several, and a
# VLM should slot in behind OCRProvider without touching anything above it.
VISION_LIBS = frozenset({"mss", "rapidocr_onnxruntime", "win32gui", "win32process"})
VISION_PKG = SRC / "vision"


def test_vision_libraries_stay_inside_the_vision_package() -> None:
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if VISION_PKG == path.parent or VISION_PKG in path.parents:
            continue
        leaked = _imported_roots(path) & VISION_LIBS
        if leaked:
            offenders.append(f"{path.relative_to(SRC)} imports {sorted(leaked)}")
    assert not offenders, (
        "Capture and OCR libraries belong in aura/vision/ only: " + "; ".join(offenders)
    )


def test_ocr_library_stays_behind_the_provider_seam() -> None:
    """Only aura/vision/ocr/ may name a recogniser."""
    ocr_pkg = VISION_PKG / "ocr"
    offenders = [
        str(path.relative_to(SRC))
        for path in VISION_PKG.rglob("*.py")
        if ocr_pkg not in path.parents
        and "rapidocr_onnxruntime" in _imported_roots(path)
    ]
    assert not offenders, f"OCR backend imported outside vision/ocr/: {offenders}"
