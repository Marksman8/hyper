"""Learning from conversation, and refusing to.

The failure mode this file exists to prevent: a model invents a preference, it
is stored, and every reply from then on is shaped by something the user never
said. The quote guard is the main defence and gets the adversarial cases.
"""

from __future__ import annotations

import json

from aura.core.types import ChatResult, Fact, GenerationParams, Message
from aura.memory.extraction import (
    FactExtractor,
    Proposal,
    looks_like_preference,
    parse_proposals,
    screen,
)
from aura.memory.store import MemoryStore

USER_SAID = "please keep your answers short, I don't have time to read essays"


class FakeLLM:
    """Returns whatever extraction response a test wants to simulate."""

    name = "fake"

    def __init__(self, payload: str | None = None, fail: bool = False) -> None:
        self.payload = payload if payload is not None else json.dumps({"facts": []})
        self.fail = fail
        self.calls = 0

    async def chat(
        self,
        messages: list[Message],
        params: GenerationParams | None = None,
        tools: object = None,
    ) -> ChatResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("model server is down")
        return ChatResult(content=self.payload, model="fake", provider="fake")


def _payload(*facts: dict[str, str]) -> str:
    return json.dumps({"facts": list(facts)})


# -- parsing ----------------------------------------------------------------


def test_parses_plain_json() -> None:
    got = parse_proposals(_payload({"key": "response_length", "value": "short", "quote": "q"}))
    assert [p.key for p in got] == ["response_length"]


def test_parses_json_in_a_code_fence() -> None:
    """Models fence JSON even when told not to."""
    raw = "```json\n" + _payload({"key": "tone", "value": "brief", "quote": "q"}) + "\n```"
    assert [p.key for p in parse_proposals(raw)] == ["tone"]


def test_parses_json_after_a_preamble() -> None:
    raw = "Sure! Here is what I found:\n" + _payload(
        {"key": "tone", "value": "brief", "quote": "q"}
    )
    assert [p.key for p in parse_proposals(raw)] == ["tone"]


def test_garbage_yields_nothing_rather_than_raising() -> None:
    """Learning nothing is a normal turn, not an error."""
    for raw in ("", "I'm not sure what you mean.", "{{{", '{"facts": "not a list"}'):
        assert parse_proposals(raw) == []


def test_keys_are_normalised() -> None:
    """Otherwise `Response Length` becomes a second fact beside
    `response_length`, and neither ever gets corroborated."""
    got = parse_proposals(_payload({"key": "Response Length!", "value": "v", "quote": "q"}))
    assert got[0].key == "response_length"


def test_entries_without_a_key_or_value_are_dropped() -> None:
    raw = _payload(
        {"key": "", "value": "x", "quote": "q"},
        {"key": "tone", "value": "", "quote": "q"},
        {"key": "ok", "value": "yes", "quote": "q"},
    )
    assert [p.key for p in parse_proposals(raw)] == ["ok"]


# -- the quote guard --------------------------------------------------------


def test_a_fact_the_user_never_said_is_refused() -> None:
    """The defence that matters most.

    A fabricated preference has no real quote to offer, so it never reaches
    storage.
    """
    invented = Proposal(
        key="profession", value="the user is a cardiologist", quote="I am a cardiologist"
    )
    assert screen([invented], USER_SAID) == []


def test_a_genuine_quote_is_kept() -> None:
    real = Proposal(key="response_length", value="short", quote="keep your answers short")
    assert screen([real], USER_SAID) == [real]


def test_quote_matching_ignores_whitespace_differences() -> None:
    spaced = Proposal(key="response_length", value="short", quote="keep your\n  answers   short")
    assert screen([spaced], USER_SAID) == [spaced]


def test_quote_matching_ignores_case() -> None:
    shouted = Proposal(key="response_length", value="short", quote="KEEP YOUR ANSWERS SHORT")
    assert screen([shouted], USER_SAID) == [shouted]


def test_a_confident_paraphrase_is_still_refused() -> None:
    """Deliberately not fuzzy matching. A similarity threshold would let a
    plausible-sounding rewrite through, which is exactly the thing to stop."""
    paraphrase = Proposal(
        key="response_length", value="short", quote="the user prefers brief responses"
    )
    assert screen([paraphrase], USER_SAID) == []


def test_a_trivially_short_quote_proves_nothing() -> None:
    """Two words would match by accident against almost any message."""
    assert screen([Proposal(key="tone", value="short", quote="short")], USER_SAID) == []


def test_a_quote_from_the_assistant_is_not_the_user() -> None:
    """The assistant saying "you seem to prefer short answers" is the model
    quoting itself into a belief about the user."""
    from_reply = Proposal(
        key="response_length", value="short", quote="I will keep my answers short from now on"
    )
    assert screen([from_reply], USER_SAID) == []


# -- the other guards -------------------------------------------------------


def test_sensitive_categories_are_refused() -> None:
    """Even with a genuine quote. These are not things to infer quietly."""
    said = "my bank password is hunter2 and my diagnosis was confirmed last week"
    proposals = [
        Proposal(key="bank_password", value="hunter2", quote="my bank password is hunter2"),
        Proposal(key="health", value="a diagnosis", quote="my diagnosis was confirmed last week"),
    ]
    assert screen(proposals, said) == []


def test_too_many_proposals_rejects_the_whole_batch() -> None:
    """Not 'take the first three'. A model producing this many from one
    exchange is confabulating, and the early ones are no better than the late
    ones."""
    many = [
        Proposal(key=f"k{i}", value="v", quote="keep your answers short") for i in range(6)
    ]
    assert screen(many, USER_SAID) == []


def test_a_full_but_legal_batch_survives() -> None:
    three = [
        Proposal(key=f"k{i}", value="v", quote="keep your answers short") for i in range(3)
    ]
    assert len(screen(three, USER_SAID)) == 3


# -- end to end -------------------------------------------------------------


async def test_learning_stores_a_candidate_that_is_not_yet_believed(
    store: MemoryStore,
) -> None:
    """Storage is not belief: the fact exists but stays out of the prompt."""
    llm = FakeLLM(
        _payload(
            {"key": "response_length", "value": "short", "quote": "keep your answers short"}
        )
    )
    learned = await FactExtractor(llm, store).learn(USER_SAID, "Understood.", turn_id="t1")

    assert [f.key for f in learned] == ["response_length"]
    stored = await store.get_fact("user", "response_length")
    assert stored is not None
    assert stored.confidence < 0.5, "one inference became a standing belief"
    assert stored.source == "inferred"


async def test_the_stored_fact_carries_its_quote(store: MemoryStore) -> None:
    """The exit criterion: AURA can say why."""
    llm = FakeLLM(
        _payload(
            {"key": "response_length", "value": "short", "quote": "keep your answers short"}
        )
    )
    learned = await FactExtractor(llm, store).learn(USER_SAID, "OK", turn_id="t1")

    evidence = await store.evidence_for(learned[0].id)
    assert evidence[0].quote == "keep your answers short"
    assert evidence[0].turn_id == "t1"


async def test_saying_it_again_later_makes_it_believed(store: MemoryStore) -> None:
    """Two separate occasions cross the injection floor -- section 16's rule,
    end to end through the extractor."""
    llm = FakeLLM(
        _payload(
            {"key": "response_length", "value": "short", "quote": "keep your answers short"}
        )
    )
    extractor = FactExtractor(llm, store)

    await extractor.learn(USER_SAID, "OK", turn_id="t1")
    below = (await store.get_fact("user", "response_length")).confidence  # type: ignore[union-attr]

    await extractor.learn(USER_SAID, "OK", turn_id="t2")
    above = (await store.get_fact("user", "response_length")).confidence  # type: ignore[union-attr]

    assert below < 0.5 <= above


async def test_repeating_in_the_same_turn_does_not_convince(store: MemoryStore) -> None:
    llm = FakeLLM(
        _payload(
            {"key": "response_length", "value": "short", "quote": "keep your answers short"}
        )
    )
    extractor = FactExtractor(llm, store)
    for _ in range(4):
        await extractor.learn(USER_SAID, "OK", turn_id="t1")

    stored = await store.get_fact("user", "response_length")
    assert stored is not None
    assert stored.confidence < 0.5


async def test_a_model_failure_never_breaks_the_conversation(store: MemoryStore) -> None:
    """Learning is best-effort. It must not take down the turn it learns from."""
    extractor = FactExtractor(FakeLLM(fail=True), store)
    assert await extractor.learn(USER_SAID, "OK", turn_id="t1") == []


async def test_an_empty_user_message_is_not_sent_to_the_model(store: MemoryStore) -> None:
    llm = FakeLLM()
    assert await FactExtractor(llm, store).learn("   ", "OK") == []
    assert llm.calls == 0, "spent a model call on nothing"


async def test_an_invented_fact_never_reaches_the_database(store: MemoryStore) -> None:
    """The whole point, end to end."""
    llm = FakeLLM(
        _payload(
            {
                "key": "profession",
                "value": "cardiologist",
                "quote": "I work as a cardiologist",
            }
        )
    )
    assert await FactExtractor(llm, store).learn(USER_SAID, "OK", turn_id="t1") == []
    assert await store.get_fact("user", "profession") is None


async def test_an_explicit_fact_outranks_an_inferred_one(store: MemoryStore) -> None:
    """`aura remember` should not be dragged down by the inference machinery."""
    await store.upsert_fact(Fact(scope="user", key="tone", value="brief"))
    stored = await store.get_fact("user", "tone")
    assert stored is not None


# -- the pre-filter ---------------------------------------------------------
#
# Extraction is a second generation per turn. On one GPU holding one model,
# running it on every message doubles generation load to learn nothing, and
# competes with the *next* reply. So most turns must not reach the model.


def test_ordinary_questions_do_not_trigger_extraction() -> None:
    for text in (
        "what is the weather like today in london",
        "can you explain how the vision privacy gate works",
        "run the tests again and show me the failures",
        "hello",
    ):
        assert not looks_like_preference(text), text


def test_preference_language_does_trigger_extraction() -> None:
    for text in (
        "please keep your answers short, I don't have time to read essays",
        "I prefer detailed technical explanations with examples",
        "remember that my timezone is Asia/Kolkata",
        "from now on use British spelling in everything you write",
        "your replies are too verbose for me to skim quickly",
    ):
        assert looks_like_preference(text), text


def test_a_message_too_short_to_carry_a_preference_is_skipped() -> None:
    assert not looks_like_preference("I prefer x")
    assert not looks_like_preference("")


async def test_a_plain_question_never_reaches_the_model(store: MemoryStore) -> None:
    """The cost guard, end to end: no LLM call is spent."""
    llm = FakeLLM(_payload({"key": "k", "value": "v", "quote": "q"}))
    await FactExtractor(llm, store).learn(
        "what is the weather like today in london", "It is sunny.", turn_id="t1"
    )
    assert llm.calls == 0
