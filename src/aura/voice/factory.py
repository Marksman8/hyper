"""Builds speech backends from config.

The voice equivalent of `aura.models.registry`: one place decides which
concrete implementation exists, so swapping Piper for Kokoro, or faster-whisper
for a whisper.cpp server, is a config value.

Imports of the concrete providers are deliberately deferred into the factory
functions. `aura.voice.factory` is imported by the CLI, which must stay usable
on a machine with no voice extra installed -- importing onnxruntime at module
scope would make `aura --help` fail.
"""

from __future__ import annotations

from aura.config import Settings, voice_models_dir
from aura.errors import ConfigError
from aura.logging import get_logger
from aura.voice.stt.base import STTProvider
from aura.voice.tts.base import TTSProvider
from aura.voice.wake.base import WakeWordDetector

log = get_logger(__name__)


class VoiceDependencyError(ConfigError):
    """The voice extra is not installed."""

    code = "voice_dependency_missing"


def _require(module: str, package: str) -> None:
    import importlib.util

    if importlib.util.find_spec(module) is None:
        raise VoiceDependencyError(f"{package} is not installed. Run: uv sync --extra voice")


def build_tts(settings: Settings) -> TTSProvider:
    cfg = settings.voice.tts
    if cfg.provider == "null":
        from aura.voice.tts.null import NullTTS

        return NullTTS()
    if cfg.provider != "piper":
        raise ConfigError(f"Unknown voice.tts.provider {cfg.provider!r}. Known: piper, null")

    _require("piper", "piper-tts")
    from aura.voice.tts.piper import PiperTTS

    return PiperTTS(
        model_path=voice_models_dir() / f"{cfg.voice}.onnx",
        length_scale=cfg.length_scale,
        volume=cfg.volume,
    )


def build_stt(settings: Settings) -> STTProvider:
    cfg = settings.voice.stt
    if cfg.provider == "null":
        from aura.voice.stt.null import NullSTT

        return NullSTT()
    if cfg.provider != "faster_whisper":
        raise ConfigError(
            f"Unknown voice.stt.provider {cfg.provider!r}. Known: faster_whisper, null"
        )

    _require("faster_whisper", "faster-whisper")
    from aura.voice.stt.faster_whisper import FasterWhisperSTT

    return FasterWhisperSTT(
        model=cfg.model,
        device=cfg.device,
        compute_type=cfg.compute_type,
        language=cfg.language,
        beam_size=cfg.beam_size,
        min_confidence=cfg.min_confidence,
    )


def build_wake(settings: Settings) -> WakeWordDetector:
    cfg = settings.voice.wake
    if cfg.model == "null":
        from aura.voice.wake.null import NullWakeWord

        return NullWakeWord()

    _require("openwakeword", "openwakeword")
    from aura.voice.wake.openwakeword import OpenWakeWordDetector

    return OpenWakeWordDetector(model=cfg.model, threshold=cfg.threshold, cooldown_s=cfg.cooldown_s)
