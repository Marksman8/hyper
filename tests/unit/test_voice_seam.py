"""The §32 invariant, extended to speech (Phase 2).

Same reasoning as tests/unit/test_seam.py: the voice session loop must depend
on the three ABCs and nothing else, so a different STT model, a whisper.cpp
server, or a different TTS engine is a config change. If this fails, some
module has started depending on a specific speech backend.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "aura"
VOICE = SRC / "voice"

# Packages permitted to import a speech backend. Everything else in
# aura.voice -- the session loop, the VAD, the audio layer, the types -- must
# not know which engine is in use.
PROVIDER_DIRS = (VOICE / "wake", VOICE / "stt", VOICE / "tts")

SPEECH_LIBS = frozenset(
    {
        "openwakeword",
        "faster_whisper",
        "whisper",
        "piper",
        "pvporcupine",
        "vosk",
        "speech_recognition",
        "pyttsx3",
        "elevenlabs",
        "TTS",
    }
)


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def _non_provider_files() -> list[Path]:
    return sorted(p for p in VOICE.rglob("*.py") if not any(d in p.parents for d in PROVIDER_DIRS))


@pytest.mark.parametrize("path", _non_provider_files(), ids=lambda p: p.name)
def test_no_speech_library_outside_provider_packages(path: Path) -> None:
    leaked = _imported_roots(path) & SPEECH_LIBS
    assert not leaked, (
        f"{path.relative_to(SRC)} imports {sorted(leaked)}. "
        "Speech backends belong in aura/voice/{wake,stt,tts}/ only."
    )


def test_session_loop_names_no_concrete_backend() -> None:
    source = (VOICE / "session.py").read_text(encoding="utf-8")
    for concrete in ("OpenWakeWordDetector", "FasterWhisperSTT", "PiperTTS", "whisper"):
        assert concrete not in source, (
            f"session.py references {concrete!r}; it may only use the ABCs."
        )


def test_aura_core_does_not_import_voice() -> None:
    """AURA Core must gain nothing from Phase 2.

    Core stays a 69 MB torch-free process that restarts in under a second. If
    the API, orchestrator, memory or model layers start importing aura.voice,
    that property is gone and the two-process design has quietly collapsed.

    Checked against the import statements rather than the file text. It was a
    substring search until `core/envelope.py` arrived carrying a docstring that
    explains *why* it does not import the voice package -- and was failed for
    saying so. Prose about a boundary is not a breach of it, and a test that
    cannot tell the difference discourages the comment that would have kept the
    next person on the right side of the line.
    """
    core_dirs = ("api", "core", "memory", "models", "security", "runtime")
    offenders: list[str] = []
    for name in core_dirs:
        for path in (SRC / name).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                imported: list[str] = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
                elif isinstance(node, ast.Import):
                    imported += [a.name for a in node.names]
                if any(m == "aura.voice" or m.startswith("aura.voice.") for m in imported):
                    offenders.append(str(path.relative_to(SRC)))
    assert not offenders, f"AURA Core imports voice code in: {sorted(set(offenders))}"


def test_voice_extra_is_not_a_core_dependency() -> None:
    """The speech stack must stay an optional extra."""
    pyproject = (SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    core_block = pyproject.split("[project.optional-dependencies]")[0]
    for pkg in ("openwakeword", "faster-whisper", "piper-tts", "sounddevice"):
        assert pkg not in core_block, (
            f"{pkg} is in the core dependency list; it belongs in the voice extra."
        )


def test_no_torch_anywhere() -> None:
    """Phase 2 runs on onnxruntime and CTranslate2, not torch.

    Torch is ~2.5 GB installed and several hundred MB resident. On a machine
    with under a gigabyte free it is the difference between voice working and
    voice being unusable. It does not enter until Phase 8 (LoRA).
    """
    pyproject = (SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    deps = pyproject.split("[dependency-groups]")[0]
    assert '"torch' not in deps, "torch was added to the dependency list"
