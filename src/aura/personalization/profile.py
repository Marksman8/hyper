"""Turning stored preferences into behaviour.

Phase 6 got facts into the system prompt as a list of trivia. That is knowing a
preference, not acting on one: a 4B model reading `response_length: short`
among twenty other lines is not reliably going to shorten its next answer.

Here a preference becomes two things -- a directive phrased as an instruction,
and, where it has a mechanical equivalent, a generation setting. The second is
what makes this real. A prompt asking for brevity is a suggestion a model can
ignore; a token ceiling is not.

**The asymmetry in this module is the safety decision of the phase.** Prompt
directives follow any believed fact. Generation parameters follow only
*explicit* ones -- see `Dimension.may_drive_parameters`. A wrong directive
gives a differently-toned answer, which is visible and cheap. A wrong
`max_tokens` truncates mid-sentence, and the user has no way to connect a
broken reply to something they said three weeks ago.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from aura.core.types import Fact, GenerationParams
from aura.logging import get_logger

log = get_logger(__name__)

# Never truncate below this, whatever the user asked for. A preference for
# brevity is a preference, not permission to cut answers off mid-sentence.
MIN_MAX_TOKENS = 256


@dataclass(frozen=True, slots=True)
class Dimension:
    """One thing AURA knows how to adapt."""

    name: str
    value: str
    # The fact this came from, so `aura profile` can show its evidence.
    source_key: str
    # 'explicit' or 'inferred'. Gates whether this may touch sampling.
    source: str
    confidence: float

    @property
    def may_drive_parameters(self) -> bool:
        """Inference changes how AURA writes; only a statement changes what it
        is mechanically able to say."""
        return self.source == "explicit"


# Keyword maps, not a model call. AURA already stored these words; paying a
# second generation to re-read its own note would be absurd.
#
# Order matters within each tuple: the first match wins, so the more specific
# phrasing is listed before the more general one.
_VOCABULARY: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "response_length": (
        ("short", ("short", "brief", "concise", "terse", "succinct", "to the point")),
        ("detailed", ("detailed", "thorough", "long", "in depth", "comprehensive", "verbose")),
        ("normal", ("normal", "medium", "moderate", "default")),
    ),
    "formality": (
        ("casual", ("casual", "informal", "friendly", "relaxed", "conversational")),
        ("formal", ("formal", "professional", "polite", "business")),
        ("neutral", ("neutral", "plain")),
    ),
    "detail_level": (
        ("minimal", ("just the answer", "no explanation", "skip the", "minimal", "no preamble")),
        ("high", ("explain", "reasoning", "why", "walk me through", "step by step")),
    ),
    "code_comments": (
        ("sparse", ("few comments", "no comments", "sparse comment", "minimal comment")),
        ("heavy", ("lots of comments", "well commented", "heavy comment", "comment everything")),
    ),
}

# Natural languages AURA will act on a preference for. See _read_value for why
# this is an allow-list rather than free text.
NATURAL_LANGUAGES: tuple[str, ...] = (
    "english",
    "spanish",
    "french",
    "german",
    "italian",
    "portuguese",
    "dutch",
    "russian",
    "arabic",
    "hindi",
    "malayalam",
    "tamil",
    "telugu",
    "kannada",
    "bengali",
    "urdu",
    "chinese",
    "mandarin",
    "japanese",
    "korean",
    "turkish",
    "polish",
    "swedish",
    "norwegian",
    "danish",
    "finnish",
    "greek",
    "hebrew",
    "thai",
    "vietnamese",
    "indonesian",
)

# Fact keys that map onto each dimension. A fact's key is a topic, and the
# model that proposed it will not have used our exact word for it.
_KEY_HINTS: dict[str, tuple[str, ...]] = {
    "response_length": ("response_length", "length", "answer_length", "verbosity", "brevity"),
    "formality": ("formality", "tone", "style", "register"),
    "detail_level": ("detail_level", "detail", "explanation", "depth"),
    "code_comments": ("code_comments", "comments", "code_style", "commenting"),
    "language": ("language", "reply_language", "locale"),
}


def _match_dimension(key: str) -> str | None:
    """Which dimension, if any, does this fact key describe?"""
    lowered = key.lower()
    for dimension, hints in _KEY_HINTS.items():
        if any(hint in lowered for hint in hints):
            return dimension
    return None


def _read_value(dimension: str, text: str) -> str | None:
    """Pull a known value out of the fact's text."""
    lowered = text.lower()
    if dimension == "language":
        # An allow-list, despite the obvious objection that it will be
        # incomplete. Treating this as free-form -- "whatever language was
        # named" -- turned the perfectly reasonable fact
        # `preferred_language: Python` into the instruction "Reply in Python",
        # which is nonsense AURA would then have followed every turn.
        #
        # Being incomplete is the cheaper failure: an unrecognised language
        # falls through to a plain fact, so the information is still in the
        # prompt, just not driving machinery.
        # The list validates; it does not rewrite. "British English" has to
        # survive as "British English" -- the qualifier is the preference, and
        # normalising it to "English" would throw away the part the user cared
        # about.
        if any(language in lowered for language in NATURAL_LANGUAGES):
            return text.strip()[:40]
        return None
    for value, keywords in _VOCABULARY.get(dimension, ()):
        if any(word in lowered for word in keywords):
            return value
    return None


@dataclass(slots=True)
class UserProfile:
    """What AURA should do differently for this user, and why."""

    dimensions: dict[str, Dimension] = field(default_factory=dict)
    # Believed facts that are not a known dimension. Still injected as context:
    # a closed vocabulary that silently dropped `my_editor: neovim` would make
    # AURA worse at knowing the user in exchange for being better at
    # formatting.
    other_facts: list[Fact] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.dimensions and not self.other_facts

    def get(self, name: str) -> str | None:
        dimension = self.dimensions.get(name)
        return dimension.value if dimension else None

    def describe(self) -> list[dict[str, object]]:
        """For `aura profile` and the API."""
        return [
            {
                "dimension": d.name,
                "value": d.value,
                "from_fact": d.source_key,
                "source": d.source,
                "confidence": round(d.confidence, 3),
                "drives_parameters": d.may_drive_parameters,
            }
            for d in sorted(self.dimensions.values(), key=lambda d: d.name)
        ]


def build_profile(facts: Sequence[Fact], floor: float = 0.5) -> UserProfile:
    """Derive a profile from believed facts.

    Facts below the floor are ignored entirely -- Phase 6 decided what counts
    as believed, and this phase does not get a second, looser opinion.
    """
    profile = UserProfile()
    for fact in facts:
        if fact.confidence < floor:
            continue
        if fact.permanence.value not in ("long_term", "permanent"):
            # "Be brief today" must not become how AURA always writes.
            continue

        dimension_name = _match_dimension(fact.key)
        value = _read_value(dimension_name, fact.value) if dimension_name else None
        if dimension_name and value:
            existing = profile.dimensions.get(dimension_name)
            candidate = Dimension(
                name=dimension_name,
                value=value,
                source_key=fact.key,
                source=fact.source,
                confidence=fact.confidence,
            )
            # Two facts can describe one dimension. An explicit statement wins
            # over an inference regardless of confidence, because the user
            # saying so outranks AURA noticing.
            if existing is None or _outranks(candidate, existing):
                profile.dimensions[dimension_name] = candidate
        else:
            profile.other_facts.append(fact)

    return profile


def _outranks(candidate: Dimension, existing: Dimension) -> bool:
    if candidate.may_drive_parameters != existing.may_drive_parameters:
        return candidate.may_drive_parameters
    return candidate.confidence > existing.confidence


# -- turning a profile into behaviour ---------------------------------------

# Phrased as instructions, not as facts. "Keep answers brief" is something a
# model acts on; "response_length: short" is trivia it has to interpret first.
_DIRECTIVES: dict[tuple[str, str], str] = {
    ("response_length", "short"): (
        "Keep answers brief and direct. Lead with the answer. Do not pad with "
        "preamble, restatement of the question, or a summary at the end."
    ),
    ("response_length", "detailed"): (
        "Give thorough answers. Cover the reasoning, the caveats and the "
        "alternatives, not only the conclusion."
    ),
    ("response_length", "normal"): "",
    ("formality", "casual"): "Write conversationally, as you would to a colleague you know well.",
    ("formality", "formal"): "Write in a professional register. Avoid slang and contractions.",
    ("formality", "neutral"): "",
    ("detail_level", "minimal"): (
        "Give the answer with no preamble and no explanation unless it is asked for."
    ),
    ("detail_level", "high"): "Explain your reasoning, not only your conclusion.",
    ("code_comments", "sparse"): (
        "In code, comment only what is genuinely non-obvious. Do not narrate "
        "what the code plainly says."
    ),
    ("code_comments", "heavy"): (
        "In code, comment generously, including the intent behind each step."
    ),
}


def directives(profile: UserProfile) -> list[str]:
    """The instruction lines a profile contributes to the system prompt."""
    out: list[str] = []
    for dimension in sorted(profile.dimensions.values(), key=lambda d: d.name):
        if dimension.name == "language":
            out.append(f"Reply in {dimension.value}, unless the user writes in another language.")
            continue
        text = _DIRECTIVES.get((dimension.name, dimension.value))
        if text:
            out.append(text)
    return out


def apply_parameters(profile: UserProfile, params: GenerationParams) -> GenerationParams:
    """Adjust sampling to match the profile.

    Only explicit preferences reach this function's effects; see the module
    docstring. Returns a new object -- GenerationParams is shared across a
    turn and mutating it would leak one request's personalization into the next.
    """
    length = profile.dimensions.get("response_length")
    if length is None or not length.may_drive_parameters:
        return params

    if length.value == "short":
        # Floored: brevity is a preference, not permission to truncate.
        capped = max(MIN_MAX_TOKENS, min(params.max_tokens, 512))
    elif length.value == "detailed":
        capped = max(params.max_tokens, 2048)
    else:
        return params

    if capped == params.max_tokens:
        return params

    log.info(
        "personalization.params_adjusted",
        dimension="response_length",
        value=length.value,
        max_tokens=capped,
        was=params.max_tokens,
    )
    return GenerationParams(
        temperature=params.temperature,
        top_p=params.top_p,
        max_tokens=capped,
        stop=params.stop,
        seed=params.seed,
    )
