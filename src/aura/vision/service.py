"""The vision subsystem, assembled and shared.

**Deliberately a process-wide singleton**, which is not the usual preference in
this codebase. The reason is the safety property: "AURA, don't look at Chrome"
has to be true everywhere at once. Two policy objects — one the tool consults,
one the voice command mutates — would mean the user turned vision off and it
stayed on, which is precisely the failure spec 10 forbids. One object, one
answer.

Tests reset it explicitly rather than getting a fresh one implicitly, so a test
that forgets is a test that notices.
"""

from __future__ import annotations

from dataclasses import dataclass

from aura.config import Settings
from aura.logging import get_logger
from aura.vision.capture import ScreenCapture
from aura.vision.change import ChangeDetector
from aura.vision.ocr.base import OCRProvider
from aura.vision.privacy import DEFAULT_DENY_TITLES, PrivacyPolicy, VisionMode

log = get_logger(__name__)


def build_policy(settings: Settings) -> PrivacyPolicy:
    cfg = settings.vision
    # Configured patterns are added to the built-in credential list, never
    # substituted for it: a user editing YAML to watch their editor should not
    # be able to accidentally delete the rule protecting their password manager.
    deny_titles = DEFAULT_DENY_TITLES + tuple(cfg.extra_deny_titles)
    return PrivacyPolicy(
        mode=VisionMode(cfg.mode),
        denied_apps=set(cfg.deny_apps),
        allowed_apps=set(cfg.allow_apps),
        deny_titles=deny_titles,
        denied_monitors=set(cfg.denied_monitors),
    )


def build_ocr(settings: Settings) -> OCRProvider:
    cfg = settings.vision
    if cfg.ocr_backend == "null":
        from aura.vision.ocr.null import NullOCR

        return NullOCR()
    from aura.vision.ocr.rapid import RapidOCR

    return RapidOCR(
        min_confidence=cfg.ocr_min_confidence,
        max_lines=cfg.max_lines,
        det_side=cfg.ocr_det_side,
        threads=cfg.ocr_threads,
    )


@dataclass
class VisionService:
    """Policy, capture, OCR and change detection, wired together."""

    policy: PrivacyPolicy
    capture: ScreenCapture
    ocr: OCRProvider
    changes: ChangeDetector

    @classmethod
    def from_settings(cls, settings: Settings) -> VisionService:
        policy = build_policy(settings)
        return cls(
            policy=policy,
            capture=ScreenCapture(policy),
            ocr=build_ocr(settings),
            changes=ChangeDetector(settings.vision.change_threshold),
        )

    def set_mode(self, mode: VisionMode) -> None:
        """Change the mode and, when going dark, drop what was held.

        Turning vision off has to discard state as well as stop capture. A
        retained fingerprint from before "don't look" is a small thing, but the
        promise is that AURA does not have your screen, and holding a reduced
        copy of it is not that promise.
        """
        self.policy.set_mode(mode)
        if mode in (VisionMode.PRIVATE, VisionMode.SLEEP):
            self.changes.reset()
            self.capture.close()
            self.ocr.unload()

    def status(self) -> dict[str, object]:
        info = dict(self.policy.describe())
        info["ocr"] = self.ocr.info()
        return info

    def close(self) -> None:
        self.capture.close()
        self.ocr.unload()


_service: VisionService | None = None


def get_vision(settings: Settings) -> VisionService:
    """The one vision service for this process."""
    global _service
    if _service is None:
        _service = VisionService.from_settings(settings)
        log.info("vision.service_built", mode=str(_service.policy.mode))
    return _service


def reset_vision() -> None:
    """Drop the shared service. For tests and for a full runtime restart."""
    global _service
    if _service is not None:
        _service.close()
    _service = None
