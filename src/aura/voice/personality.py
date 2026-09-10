"""Micro-expressions — the personality layer (10B.5).

A short, fixed phrase in another language, prepended to a spoken reply: 了解
before starting work, よし on success. The subtitle carries the phrase and an
English gloss, so meaning survives when the audio cannot be heard.

**The model never generates these.** It writes the answer in English; this
layer decides whether a micro-expression fits and prepends one. That separation
is the whole safety property. Ask a model to "add personality" and foreign
phrases leak into every third response, drift in register, and occasionally
mean something other than intended — and none of it is reviewable, because the
phrase is produced fresh each time. Here the phrase set is a constant you can
read in one screen, and 10B.6 renders each one to a fixed WAV, so every sound
AURA is capable of making can be listened to in advance.

Restraint is the feature. The default frequency is LOW — roughly one eligible
utterance in eight — and the same expression never runs twice consecutively.
An assistant that says 了解 every time is not charming, it is a catchphrase
generator, and the brief warned about exactly that.

This module is pure. It holds no audio, touches no clock it was not given, and
its randomness is injected, so "one in eight" is a property a test can assert
rather than a claim in a comment.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from aura.logging import get_logger

log = get_logger(__name__)


class Moment(StrEnum):
    """What just happened, in the only terms this layer needs.

    Deliberately not `PresenceState`. That enum belongs to the overlay and has
    thirteen members describing what AURA *looks* like; this needs four,
    describing what she is *reacting to*. Coupling them would mean every new
    visual state became a personality question.
    """

    ACKNOWLEDGING = "acknowledging"  # a task was accepted
    THINKING = "thinking"  # working through something
    SUCCEEDING = "succeeding"  # it worked
    FAILING = "failing"  # it did not


class Frequency(StrEnum):
    OFF = "off"
    LOW = "low"  # ~1 in 8 eligible utterances. The default.
    MEDIUM = "medium"  # ~1 in 4. Noticeably chattier.


# Chance of firing on an eligible utterance. MEDIUM is deliberately not "half":
# a coin flip reads as constant, because a listener remembers the hits.
_CHANCE: dict[Frequency, float] = {
    Frequency.OFF: 0.0,
    Frequency.LOW: 0.125,
    Frequency.MEDIUM: 0.25,
}

# The minimum gap between any two micro-expressions, whatever the dice say.
#
# Frequency alone is not enough: a burst of three short turns in twenty seconds
# can roll three hits, and three in twenty seconds is the behaviour the brief
# asked us to avoid regardless of the long-run average.
MIN_GAP_S = 90.0


@dataclass(frozen=True, slots=True)
class MicroExpression:
    """One fixed phrase.

    `clip` names a file 10B.6 renders; nothing here loads audio. `gloss` is
    what the subtitle shows beside the phrase, and it is required — an
    untranslated phrase on screen is worse than no phrase at all for the person
    who cannot hear the audio, which is the case this whole milestone serves.
    """

    id: str
    language: str
    text: str
    gloss: str
    clip: str
    moments: tuple[Moment, ...]

    def subtitle(self) -> str:
        """Phrase and gloss, as one line."""
        return f"{self.text} ({self.gloss})"


# The complete phrase set. Small and fixed on purpose: this is the list a
# person reads to know everything AURA might say in another language.
CATALOGUE: tuple[MicroExpression, ...] = (
    MicroExpression(
        id="ryoukai",
        language="ja",
        text="了解",
        gloss="understood",
        clip="ja/ryoukai.wav",
        moments=(Moment.ACKNOWLEDGING,),
    ),
    MicroExpression(
        id="hai",
        language="ja",
        text="はい",
        gloss="yes",
        clip="ja/hai.wav",
        moments=(Moment.ACKNOWLEDGING,),
    ),
    MicroExpression(
        id="fumu",
        language="ja",
        text="ふむ",
        gloss="hmm",
        clip="ja/fumu.wav",
        moments=(Moment.THINKING,),
    ),
    MicroExpression(
        id="yoshi",
        language="ja",
        text="よし",
        gloss="good",
        clip="ja/yoshi.wav",
        moments=(Moment.SUCCEEDING,),
    ),
    MicroExpression(
        id="daijoubu",
        language="ja",
        text="大丈夫",
        gloss="it's fine",
        clip="ja/daijoubu.wav",
        moments=(Moment.FAILING,),
    ),
    MicroExpression(
        id="daccord",
        language="fr",
        text="D'accord",
        gloss="alright",
        clip="fr/daccord.wav",
        moments=(Moment.ACKNOWLEDGING,),
    ),
    MicroExpression(
        id="voyons",
        language="fr",
        text="Voyons",
        gloss="let's see",
        clip="fr/voyons.wav",
        moments=(Moment.THINKING,),
    ),
    MicroExpression(
        id="parfait",
        language="fr",
        text="Parfait",
        gloss="perfect",
        clip="fr/parfait.wav",
        moments=(Moment.SUCCEEDING,),
    ),
)


class PersonalityEngine:
    """Decides whether a micro-expression fits, and which one."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        frequency: Frequency = Frequency.LOW,
        languages: Sequence[str] = ("ja",),
        catalogue: Sequence[MicroExpression] = CATALOGUE,
        rng: random.Random | None = None,
        clock: object | None = None,
        min_gap_s: float = MIN_GAP_S,
    ) -> None:
        self._enabled = enabled
        self._frequency = frequency
        self._languages = tuple(languages)
        self._catalogue = tuple(catalogue)
        # Injected so "one in eight" is a property a test can assert on a
        # seeded sequence rather than a claim in a comment.
        # S311 is suppressed deliberately: this picks a greeting, not a key.
        # A CSPRNG here would be cargo-cult -- the thing being randomised is
        # whether AURA says "understood" in Japanese.
        self._rng = rng or random.Random()  # noqa: S311
        self._clock = clock
        self._min_gap_s = min_gap_s

        self._last_id: str | None = None
        self._last_at = 0.0
        self._considered = 0
        self._fired = 0

    @property
    def fired(self) -> int:
        return self._fired

    @property
    def considered(self) -> int:
        return self._considered

    def _now(self) -> float:
        if self._clock is not None:
            return float(self._clock())  # type: ignore[operator]
        import time

        return time.monotonic()

    def candidates(self, moment: Moment) -> list[MicroExpression]:
        """Everything that could fit, before any dice are rolled."""
        return [
            expression
            for expression in self._catalogue
            if moment in expression.moments and expression.language in self._languages
        ]

    def select(self, moment: Moment) -> MicroExpression | None:
        """A micro-expression for this moment, or None -- usually None.

        The order of the checks matters. Cheap and absolute refusals come
        first, so that switching the feature off is not a probabilistic
        statement and the rate limit cannot be bypassed by a lucky roll.
        """
        if not self._enabled or self._frequency is Frequency.OFF:
            return None

        options = self.candidates(moment)
        if not options:
            return None

        self._considered += 1

        now = self._now()
        if self._last_at and (now - self._last_at) < self._min_gap_s:
            # The rate limit is absolute, not a weighting. Frequency governs
            # the long-run average; this governs how it feels in the short run,
            # and a burst of three in twenty seconds is what people notice.
            return None

        if self._rng.random() >= _CHANCE[self._frequency]:
            return None

        # Never the same phrase twice running. With one candidate left for a
        # moment, that means staying silent rather than repeating -- silence is
        # the better failure here.
        fresh = [option for option in options if option.id != self._last_id]
        if not fresh:
            return None

        choice = self._rng.choice(fresh)
        self._last_id = choice.id
        self._last_at = now
        self._fired += 1
        log.debug(
            "personality.selected",
            expression=choice.id,
            moment=str(moment),
            language=choice.language,
        )
        return choice

    def reset(self) -> None:
        """Forget the cooldown. For tests and for a fresh session."""
        self._last_id = None
        self._last_at = 0.0


def build_engine(settings: object) -> PersonalityEngine:
    """Construct from config, tolerating a config that predates the feature."""
    cfg = getattr(getattr(settings, "voice", None), "personality", None)
    if cfg is None:
        return PersonalityEngine(enabled=False)
    return PersonalityEngine(
        enabled=bool(cfg.enabled),
        frequency=Frequency(cfg.frequency),
        languages=tuple(cfg.languages),
        min_gap_s=float(cfg.min_gap_s),
    )
