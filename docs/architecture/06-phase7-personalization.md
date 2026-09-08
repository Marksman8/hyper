# Phase 7 — Personalization

Status: **in progress** · Date: 2026-09-08

---

## 1. What "personalization" has to mean to be worth building

Phase 6 ended with facts reaching the system prompt as a flat list:

```
What you know about this user:
- response_length: prefers short, direct answers
```

That is technically personalization and practically close to nothing. It is a
line of trivia in a prompt, competing with everything else for the model's
attention, and a 4B model reading it is not reliably going to shorten its next
answer because of it.

Phase 7 is the difference between AURA *knowing* a preference and AURA *acting*
on one. A stored preference becomes two concrete things:

1. **A directive** — an instruction phrased as an instruction, near the end of
   the system prompt where it carries weight, not an entry in a list of facts.
2. **A generation setting**, where the preference has a mechanical equivalent.
   "Short answers" is not only a request; it is also a smaller `max_tokens`.

The second is what makes this real rather than decorative. A prompt asking for
brevity is a suggestion a model can ignore. A token ceiling is not.

## 2. The asymmetry: prompts are cheap to get wrong, parameters are not

These two paths are given **different evidence requirements**, and that is the
main safety decision of this phase.

| | Driven by | Why |
|---|---|---|
| Prompt directives | any believed fact (≥ injection floor) | A wrong directive produces a differently-toned answer. Recoverable, visible, cheap. |
| Generation parameters | **explicit facts only** | A wrong `max_tokens` truncates an answer mid-sentence. The user sees a broken reply and has no way to connect it to something they said three weeks ago. |

An inferred preference — even a well-corroborated one — never moves a sampling
parameter. Inference is good enough to change how AURA writes; it is not good
enough to change what AURA is mechanically capable of saying.

Parameters are also floored: no personalization may set `max_tokens` below a
level where ordinary answers get cut off. A preference for brevity is a
preference, not permission to truncate.

## 3. Known dimensions, unknown facts

The profile recognises a small set of dimensions it knows how to act on:

- `response_length` — short / normal / detailed
- `formality` — casual / neutral / formal
- `detail_level` — how much explanation surrounds an answer
- `code_comments` — sparse / normal / heavy
- `language` — the language to reply in

Everything else stays exactly as it is today: injected as a plain fact. This
matters. A closed vocabulary that silently discarded `my_editor: neovim`
would make AURA *worse* at knowing the user in exchange for being better at
formatting. Unknown facts are still context; they just do not drive machinery.

Dimension values are parsed from the fact's text with a small keyword map, not
by asking the model. A second LLM call to interpret a preference AURA already
stored would be paying generation cost to re-derive something it wrote down.

**`language` is an allow-list, and the reason is a bug this design had.** The
first version treated it as free text -- whatever language the fact named -- on
the grounds that enumerating languages is a list to get wrong. An existing test
caught what that actually produces: the perfectly reasonable fact
`preferred_language: Python` became the instruction *"Reply in Python"*, which
AURA would then have followed on every turn. Being incomplete is the cheaper
failure. An unrecognised language falls through to a plain fact, so the
information is still in the prompt; it simply does not drive machinery. The
list validates the value without rewriting it, so `British English` survives
intact -- the qualifier is the preference.

## 4. Visibility, because silent adaptation is a debugging nightmare

Personalization changes behaviour without the user asking, every turn. When
something reads oddly, the user needs to be able to see what is being applied
and why — otherwise "AURA is being weird today" is unfalsifiable.

So `aura profile` shows the active dimensions, their values, the fact behind
each, and the resulting parameter overrides. Every adaptation traces back
through Phase 6's evidence chain to something the user said.

`personalization.enabled: false` turns the whole thing off, and a single turn
can opt out. If adaptation cannot be switched off, it cannot be blamed, and
anything that cannot be blamed cannot be fixed.

## 5. The exit criterion, honestly

The roadmap asks for "a blind A/B against the v0.1 persona showing a preference
for the adapted behaviour on a held-out set of the user's own tasks."

**A single user rating a handful of prompts cannot produce a statistically
significant result, and I am not going to present one as if it could.** Twenty
comparisons with a 13–7 split is not evidence of anything; it is a coin.

What is buildable and honest is the *instrument*: `aura ab` runs the same
prompt through both the baseline persona and the adapted one, shows the two
answers in a random order with no labels, records which the user picked, and
keeps a tally with the ordering recorded so position bias can be checked. It
reports the count and refuses to editorialise about significance.

That makes the criterion met when the user has actually run it and the tally
leans the right way — a real answer, at the sample size that genuinely exists,
rather than a green tick from a test that only ever asked a machine.

## 6. Scope

**Built:** the profile derived from facts, prompt directives, parameter
overrides gated on explicit evidence, `aura profile`, the blind A/B harness,
and a per-turn opt-out.

**Not built:** fine-tuning of any kind (that is Phase 8, and it needs Phase 8's
evaluation harness first), and workflow adaptation beyond generation — learning
*how* the user likes tasks sequenced is procedural memory, which needs more
signal than exists yet.
