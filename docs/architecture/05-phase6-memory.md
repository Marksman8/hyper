# Phase 6 — Memory and the user model

Status: **in progress** · Date: 2026-09-08

---

## 1. What is already here

Phase 6 does not start from nothing. `facts` has carried `confidence`,
`evidence_count`, `source` and `permanence` since v0.1, and the orchestrator
already injects long-lived, well-evidenced facts into the system prompt. So
"a stated preference changes behaviour" is half-built: `aura remember` works
and the fact reaches the model.

What is missing is the interesting half — where facts *come from*, and how AURA
justifies believing one.

## 2. The correction that shapes this phase

The roadmap said: embeddings via `/v1/embeddings`, plus `sqlite-vec` for
search. I am not building that, and the reason is worth writing down rather
than quietly skipping.

**For facts, retrieval is the wrong tool entirely.** This is one user. Realistic
scale is tens to low hundreds of facts. A hundred facts at ~15 tokens each is
~1,500 tokens — affordable in every request. Loading all of them is not a
compromise against top-k retrieval, it is *strictly better*: there is no recall
miss, no threshold to tune, and no failure mode where AURA forgets it knows
something because the query embedded oddly. Retrieval over a set that fits in
the prompt is complexity buying a downgrade.

**For conversation history, search is genuinely needed** — "what did we decide
about the barge-in threshold?" across thousands of turns. But embeddings are
still the wrong first move here:

- A chat model's `/v1/embeddings` produces poor vectors. Doing this properly
  means a *dedicated* embedding model resident alongside the LLM, on a card
  that already holds 3.07 GB of Qwen and has to leave room for OCR.
- SQLite 3.50.4 in this venv has **FTS5 compiled in** (verified, not assumed).
  BM25 over turns costs zero dependencies, zero VRAM, and zero model loads.
- The query that motivates this is *keyword-shaped*. People search their own
  history with the words they used at the time.

FTS5 loses to embeddings on pure-synonym queries — asking about "interruption"
and wanting turns that said "barge-in". That is a real limitation, and it is
the trigger for revisiting this, not a reason to pre-build the heavier thing
now. Search sits behind a `Retriever` interface so an embedding backend is a
config change, exactly like `LLMProvider` and `OCRProvider`.

## 3. The actual hard part: why does AURA believe this?

The phase exit criterion is that `aura memory` shows **why** each fact is
believed. That cannot be satisfied by the schema as it stands:

```sql
evidence_count INTEGER NOT NULL DEFAULT 1
```

A counter is not a justification. "Confidence 0.8, evidence 4" tells the user
nothing they can check, disagree with, or correct. Worse, it is unfalsifiable
by construction — there is no way to look at a wrong fact and find out which
turn produced it.

So the central schema change of this phase is a `fact_evidence` table holding
the actual utterance, the turn it came from, and when. Confidence becomes a
*derived* property of evidence rather than a number someone chose. "Why do you
think I prefer short answers?" gets answered with three quotes and their dates.

This also makes deletion meaningful: removing a fact removes its evidence, so
"forget that" leaves nothing behind that could re-derive it.

## 4. Extraction proposes; it does not decide

Spec §16: an inferred preference starts at low confidence and needs repeated
evidence; only explicit statements start high. Implemented literally:

| Source | Starting confidence | Injected into the prompt? |
|---|---|---|
| `aura remember` / "remember that…" | 0.95 | immediately |
| Inferred from conversation | 0.3 | **no** — below the 0.5 threshold |

An inferred fact needs *independent* corroboration on separate occasions to
cross the injection threshold. One offhand remark never becomes a standing
belief about the user, which is the failure mode that makes personalisation
feel creepy rather than useful.

Two safeguards on top:

- Extraction runs **after** the turn is answered, never in the response path.
  It is a second LLM call; paying its latency on every reply would be a poor
  trade for a feature the user does not see happen.
- Corroboration requires a *different session or a later turn*. A model that
  restates the same inference twice in one reply must not thereby convince
  itself.

## 5. Why not just let the model write to memory as a tool?

Considered and rejected for the default path. A `remember` tool would put fact
creation on the same footing as reading a file — but memory writes are
different in kind: they are silent, they persist past the conversation, and
they shape every future reply. A model that can write its own long-term
premises can drift without anything looking wrong at the time.

Extraction with a confidence floor and visible provenance gets the same benefit
with a correction path. The user keeps the last word through `aura memory`.

## 6. Deletion (spec §23)

Deletion must be complete and immediate: the fact row, its evidence rows, and
the FTS entries, in one transaction. "Immediate" also means the in-flight
prompt — a fact deleted mid-conversation must not survive in an already-built
system message, so assembly reads facts per turn rather than caching them.

## 7. Scope

**Built this phase:** the evidence table and provenance, FTS5 search over
turns behind a `Retriever` seam, the extraction pipeline with §16's confidence
rules, `aura memory` with justifications, and complete deletion.

**Not built:** embeddings (see §2), procedural memory (spec §15's "how the user
likes tasks done" — needs Phase 7's personalisation surface to be worth
storing), and any automatic *forgetting* beyond the existing permanence sweep.
