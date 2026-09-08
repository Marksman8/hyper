"""Evidence, derived confidence, search, and complete deletion (Phase 6).

Two properties carry this file:

1. Confidence is *derived* from evidence, not incremented. So removing the
   evidence lowers the belief, and a fact can always be explained.
2. Deletion leaves nothing behind that could re-derive the fact (spec §23).
"""

from __future__ import annotations

from aura.core.types import Evidence, Fact, Role, Turn
from aura.memory.store import MemoryStore, _fts_query


async def _fact(store: MemoryStore, key: str = "tone", value: str = "brief") -> Fact:
    return await store.upsert_fact(Fact(scope="user", key=key, value=value))


# -- the id bug that evidence would have inherited --------------------------


async def test_upsert_returns_the_stored_row_not_the_argument(store: MemoryStore) -> None:
    """On conflict the stored row keeps its ORIGINAL id.

    Returning the caller's object would hand back an id that is not in the
    database, and evidence references fact ids by foreign key - so this would
    have surfaced as evidence silently attached to nothing.
    """
    first = await store.upsert_fact(Fact(scope="user", key="tone", value="brief"))
    second = await store.upsert_fact(Fact(scope="user", key="tone", value="briefer"))

    assert second.id == first.id, "upsert handed back an id that is not stored"
    assert (await store.get_fact("user", "tone")).id == first.id  # type: ignore[union-attr]


# -- derived confidence -----------------------------------------------------


async def test_one_inference_stays_below_the_injection_floor(store: MemoryStore) -> None:
    fact = await _fact(store)
    await store.add_evidence(
        Evidence(fact_id=fact.id, quote="keep it short", turn_id="t1", source="inferred")
    )

    stored = await store.get_fact("user", "tone")
    assert stored is not None
    assert stored.confidence < 0.5, "a single remark became a standing belief"


async def test_second_independent_inference_crosses_the_floor(store: MemoryStore) -> None:
    fact = await _fact(store)
    await store.add_evidence(Evidence(fact_id=fact.id, quote="keep it short", turn_id="t1"))
    await store.add_evidence(Evidence(fact_id=fact.id, quote="too long again", turn_id="t2"))

    stored = await store.get_fact("user", "tone")
    assert stored is not None
    assert stored.confidence >= 0.5
    assert stored.evidence_count == 2


async def test_repeating_within_one_turn_does_not_convince(store: MemoryStore) -> None:
    fact = await _fact(store)
    for _ in range(4):
        await store.add_evidence(
            Evidence(fact_id=fact.id, quote="short please", turn_id="t1")
        )

    stored = await store.get_fact("user", "tone")
    assert stored is not None
    assert stored.confidence < 0.5
    assert stored.evidence_count == 1, "same turn counted more than once"


async def test_explicit_statement_is_believed_at_once(store: MemoryStore) -> None:
    fact = await _fact(store)
    await store.add_evidence(
        Evidence(fact_id=fact.id, quote="remember I like short answers", source="explicit")
    )

    stored = await store.get_fact("user", "tone")
    assert stored is not None
    assert stored.confidence >= 0.9


async def test_confidence_is_recomputed_not_incremented(store: MemoryStore) -> None:
    """The distinction the whole design rests on.

    Reassessing without adding evidence must not move the number. An
    incrementing counter would drift upward every time it was touched.
    """
    fact = await _fact(store)
    await store.add_evidence(Evidence(fact_id=fact.id, quote="a", turn_id="t1"))
    first = (await store.get_fact("user", "tone")).confidence  # type: ignore[union-attr]

    for _ in range(5):
        await store.reassess_fact(fact.id)

    assert (await store.get_fact("user", "tone")).confidence == first  # type: ignore[union-attr]


# -- provenance -------------------------------------------------------------


async def test_evidence_carries_the_actual_words(store: MemoryStore) -> None:
    """The exit criterion: AURA can say WHY, not just how many."""
    fact = await _fact(store)
    await store.add_evidence(
        Evidence(
            fact_id=fact.id,
            quote="please stop giving me essays",
            session_id="s1",
            turn_id="t1",
        )
    )

    evidence = await store.evidence_for(fact.id)
    assert len(evidence) == 1
    assert evidence[0].quote == "please stop giving me essays"
    assert evidence[0].session_id == "s1"


async def test_evidence_is_newest_first(store: MemoryStore) -> None:
    fact = await _fact(store)
    await store.add_evidence(Evidence(fact_id=fact.id, quote="old", turn_id="t1", created_at=100.0))
    await store.add_evidence(Evidence(fact_id=fact.id, quote="new", turn_id="t2", created_at=200.0))

    assert [e.quote for e in await store.evidence_for(fact.id)] == ["new", "old"]


# -- deletion (spec section 23) ---------------------------------------------


async def test_deleting_a_fact_deletes_its_evidence(store: MemoryStore) -> None:
    """Nothing may survive that could re-derive the belief."""
    fact = await _fact(store)
    await store.add_evidence(Evidence(fact_id=fact.id, quote="short please", turn_id="t1"))
    assert await store.evidence_for(fact.id)

    assert await store.delete_fact("user", "tone")

    assert await store.get_fact("user", "tone") is None
    assert await store.evidence_for(fact.id) == []


# -- search -----------------------------------------------------------------


async def _conversation(store: MemoryStore) -> str:
    session = await store.create_session()
    for content in (
        "the barge in threshold was measured at 0.024",
        "lets look at the vision privacy gate instead",
        "OCR took forty two seconds on a dense page",
    ):
        await store.add_turn(Turn(session_id=session.id, role=Role.USER, content=content))
    return session.id


async def test_search_finds_a_turn_by_keyword(store: MemoryStore) -> None:
    await _conversation(store)
    hits = await store.search_turns("barge")
    assert hits
    assert "barge in threshold" in hits[0].turn.content


async def test_search_ranks_the_better_match_first(store: MemoryStore) -> None:
    await _conversation(store)
    hits = await store.search_turns("how long did OCR take")
    assert hits
    assert "OCR" in hits[0].turn.content


async def test_search_returns_nothing_for_an_unknown_word(store: MemoryStore) -> None:
    await _conversation(store)
    assert await store.search_turns("kangaroo") == []


async def test_search_can_be_scoped_to_one_session(store: MemoryStore) -> None:
    first = await _conversation(store)
    second = await store.create_session()
    await store.add_turn(
        Turn(session_id=second.id, role=Role.USER, content="barge in again elsewhere")
    )

    scoped = await store.search_turns("barge", session_id=second.id)
    assert len(scoped) == 1
    assert scoped[0].session_id == second.id
    assert len(await store.search_turns("barge")) == 2
    assert first  # both sessions exist


async def test_deleting_a_session_removes_it_from_search(store: MemoryStore) -> None:
    """Otherwise 'delete this conversation' leaves it findable - a deletion
    that did not delete."""
    session_id = await _conversation(store)
    assert await store.search_turns("barge")

    await store.delete_session(session_id)
    assert await store.search_turns("barge") == []


# -- query sanitising -------------------------------------------------------


def test_punctuation_does_not_become_fts_syntax() -> None:
    """A user's question must never be parsed as a query language.

    Bare `"` or `:` in FTS5 MATCH is a syntax error, and a colon could address
    a column the user did not know existed.
    """
    assert _fts_query('what about "stop"?') == '"what" OR "about" OR "stop"'
    assert _fts_query("content: secret") == '"content" OR "secret"'
    assert _fts_query("a AND b OR c") == '"a" OR "AND" OR "b" OR "OR" OR "c"'


def test_empty_and_punctuation_only_queries_are_empty() -> None:
    assert _fts_query("") == ""
    assert _fts_query("???  !!!") == ""


async def test_search_survives_a_punctuation_only_question(store: MemoryStore) -> None:
    await _conversation(store)
    assert await store.search_turns("???") == []
