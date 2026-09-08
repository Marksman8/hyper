"""Noticing things about the user, carefully.

Spec §16 wants AURA to learn preferences from conversation. That is also the
feature most likely to go wrong in ways nobody notices for weeks: a model
invents a fact, it gets stored, and from then on every reply is shaped by
something the user never said.

Four guards, in order of how much they matter:

1. **A proposal must quote the user.** The model has to supply the words it
   inferred from, and the quote is checked against what the user actually
   said. A fabricated preference has no quote to offer, so it is dropped
   before it is ever written. This is cheap and catches the worst failure.
2. **Inferences start below the injection floor.** One remark is never acted
   on; see memory/beliefs.py. Storage is not belief.
3. **Sensitive categories are refused outright.** Health, finances,
   credentials and the like are not things to infer quietly and keep.
4. **A cap per turn.** A model that proposes fifteen facts about one sentence
   is not observing, it is confabulating, and the whole batch is suspect.

Extraction runs *after* the reply, never in the response path: it is a second
LLM call, and paying its latency on every turn to power something the user
cannot see happen would be a bad trade.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aura.core.types import Evidence, Fact, GenerationParams, Message, Permanence, Role
from aura.logging import get_logger
from aura.memory.beliefs import INFERRED_BASE

if TYPE_CHECKING:
    from aura.memory.store import MemoryStore
    from aura.models.base import LLMProvider

log = get_logger(__name__)

# Never inferred and stored quietly. The user can still state any of these
# outright with "remember that ..." - that is an explicit, visible act - but
# AURA does not go looking for them.
SENSITIVE = (
    "password",
    "passphrase",
    "credential",
    "api key",
    "token",
    "secret",
    "bank",
    "credit card",
    "salary",
    "diagnosis",
    "medication",
    "health",
    "illness",
    "religion",
    "sexual",
    "political",
)

# More than this from a single exchange is confabulation, not observation.
MAX_PER_TURN = 3

# Cheap pre-filter, applied before spending an LLM call.
#
# Extraction is a second generation per turn. On one GPU holding one model at
# ~70 tok/s, running it on every "what time is it" doubles the machine's
# generation load to learn nothing -- and competes with the *next* reply, so
# the cost lands on the user even though extraction is asynchronous.
#
# Most turns contain no durable preference, and the ones that do tend to say
# so in recognisable ways. This is a deliberate recall/precision trade: it will
# miss a preference phrased unusually. That is an acceptable loss, because a
# genuine preference is one the user tends to repeat -- and repetition is
# exactly what section 16 requires before AURA acts on it anyway.
_PREFERENCE_HINTS = re.compile(
    r"\b("
    r"i (?:like|prefer|hate|love|always|never|usually|generally|tend to|want|need)"
    r"|i(?:'m| am) (?:a|an|the|not)"
    r"|i (?:don'?t|do not|can'?t|cannot|won'?t)"
    r"|my (?:name|timezone|job|editor|setup|preference)"
    r"|remember (?:that|this|to)"
    r"|from now on"
    r"|call me"
    r"|please (?:always|never|stop|keep|don'?t)"
    r"|stop (?:giving|doing|using|making)"
    r"|too (?:long|short|verbose|wordy|technical)"
    r")\b",
    re.IGNORECASE,
)

# Below this a message is too short to carry a preference worth a model call.
MIN_LENGTH_FOR_EXTRACTION = 20


def looks_like_preference(text: str) -> bool:
    """Is this message worth spending an extraction call on?"""
    stripped = (text or "").strip()
    if len(stripped) < MIN_LENGTH_FOR_EXTRACTION:
        return False
    return bool(_PREFERENCE_HINTS.search(stripped))

_PROMPT = """You extract durable preferences about the user from a conversation.

Return JSON only, in this exact shape:
{"facts": [{"key": "...", "value": "...", "quote": "..."}]}

Rules:
- `quote` MUST be text copied verbatim from the USER's message. If you cannot
  quote the user, do not propose the fact.
- Only durable preferences and stable facts about the user. Not things that
  are true today only, not facts about the world, not what was discussed.
- `key` is a short snake_case topic, e.g. response_length, timezone, editor.
- At most 3 facts. Prefer returning none over guessing.
- If nothing durable was said, return {"facts": []}.
"""


@dataclass(frozen=True, slots=True)
class Proposal:
    """A candidate fact the model suggested. Not yet believed."""

    key: str
    value: str
    quote: str

    def is_sensitive(self) -> bool:
        haystack = f"{self.key} {self.value}".lower()
        return any(term in haystack for term in SENSITIVE)


def _normalise_key(raw: str) -> str:
    """snake_case, bounded. Keys are the identity of a fact, so a model's
    stylistic wobble must not create `Response_Length` beside `response_length`."""
    cleaned = re.sub(r"[^0-9a-z]+", "_", (raw or "").strip().lower()).strip("_")
    return cleaned[:60]


def _quoted_from_user(quote: str, user_text: str) -> bool:
    """Is this quote really the user's words?

    Whitespace-insensitive substring match. Deliberately not fuzzy: the whole
    point is that a fabricated quote fails, and a similarity threshold would
    let a confident paraphrase through.
    """
    if not quote or not user_text:
        return False
    squash = re.compile(r"\s+")
    needle = squash.sub(" ", quote).strip().lower()
    haystack = squash.sub(" ", user_text).strip().lower()
    # A quote of one or two words proves nothing; it would match by accident.
    return len(needle) >= 8 and needle in haystack


def parse_proposals(raw: str) -> list[Proposal]:
    """Read the model's JSON, tolerating the ways models wrap it.

    Never raises. A malformed extraction response means "learned nothing this
    turn", which is a normal outcome, not an error worth surfacing.
    """
    if not raw:
        return []
    text = raw.strip()
    # Models fence JSON even when told not to.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Or wrap it in a sentence.
    if not text.startswith("{"):
        brace = text.find("{")
        if brace == -1:
            return []
        text = text[brace:]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        log.debug("memory.extraction_unparseable")
        return []

    items = payload.get("facts") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []

    out: list[Proposal] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = _normalise_key(str(item.get("key", "")))
        value = str(item.get("value", "")).strip()
        quote = str(item.get("quote", "")).strip()
        if key and value:
            out.append(Proposal(key=key, value=value[:500], quote=quote[:500]))
    return out


def screen(proposals: list[Proposal], user_text: str) -> list[Proposal]:
    """Apply the guards. Returns only what is safe to store."""
    if len(proposals) > MAX_PER_TURN:
        # Not "take the first three": a model producing this many from one
        # exchange is confabulating, and the early ones are no more trustworthy
        # than the late ones.
        log.info("memory.extraction_rejected_batch", count=len(proposals))
        return []

    kept: list[Proposal] = []
    for p in proposals:
        if p.is_sensitive():
            log.info("memory.extraction_refused_sensitive", key=p.key)
            continue
        if not _quoted_from_user(p.quote, user_text):
            log.info("memory.extraction_refused_unquoted", key=p.key)
            continue
        kept.append(p)
    return kept


class FactExtractor:
    """Proposes facts from a finished exchange."""

    def __init__(self, provider: LLMProvider, store: MemoryStore) -> None:
        self._provider = provider
        self._store = store

    async def propose(self, user_text: str, assistant_text: str) -> list[Proposal]:
        """Ask the model what it noticed. Never raises."""
        if not looks_like_preference(user_text):
            log.debug("memory.extraction_skipped")
            return []

        messages = [
            Message(role=Role.SYSTEM, content=_PROMPT),
            Message(
                role=Role.USER,
                content=f"USER said:\n{user_text}\n\nASSISTANT replied:\n{assistant_text}",
            ),
        ]
        try:
            result = await self._provider.chat(
                messages,
                # Low temperature: this is an extraction, not a creative act.
                GenerationParams(temperature=0.1, max_tokens=400),
            )
        except Exception as exc:
            # Learning is best-effort. It must never break the conversation it
            # is learning from.
            log.warning("memory.extraction_failed", error=str(exc))
            return []

        return screen(parse_proposals(result.content), user_text)

    async def learn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
    ) -> list[Fact]:
        """Propose, screen, and store as low-confidence candidates.

        The fact row is written, but at inferred confidence it stays out of the
        system prompt until corroborated. Storing is not believing.
        """
        proposals = await self.propose(user_text, assistant_text)
        learned: list[Fact] = []
        for p in proposals:
            fact = await self._store.upsert_fact(
                Fact(
                    scope="user",
                    key=p.key,
                    value=p.value,
                    confidence=INFERRED_BASE,
                    source="inferred",
                    permanence=Permanence.LONG_TERM,
                )
            )
            await self._store.add_evidence(
                Evidence(
                    fact_id=fact.id,
                    quote=p.quote,
                    session_id=session_id,
                    turn_id=turn_id,
                    source="inferred",
                )
            )
            learned.append(fact)

        if learned:
            log.info("memory.learned", keys=[f.key for f in learned])
        return learned


def build_extractor(provider: Any, store: MemoryStore) -> FactExtractor:
    return FactExtractor(provider, store)
