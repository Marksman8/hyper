# AURA roadmap

Every phase must end with something that runs. A phase is not done until its
exit criteria are met on real hardware, not in principle.

Where this differs from the original specification, the reason is stated. The
spec's phase list is preserved; a few items are re-ordered or re-scoped because
of what the target machine can actually do (8 GB VRAM, **15.7 GB system RAM**,
Windows 11).

---

## v0.1 — Foundation  ✅ complete

Local LLM behind a provider interface · local API · persistent memory ·
config · logging · security foundation · CLI client.

**Exit criteria — all met**

- [x] AURA Core runs with zero hosted-API dependencies
- [x] Model is a config value; swapping backends touches no code
- [x] Conversations survive a restart
- [x] Bearer auth on every route, loopback binding enforced by a validator
- [x] Audit trail, emergency stop, secret redaction
- [x] 92 tests green; ruff clean; mypy strict clean
- [x] Core resident set: **69 MB**

---

## v0.1.1 — Desktop shell

The spec lists a desktop interface under v0.1; it is split out because it needs
a working core to build against and its RAM cost should be measured, not
guessed.

- System tray presence, transparent status overlay, global hotkey
- Status indicators: model, mode, current task, emergency stop
- **Toolkit decision deferred until here.** PySide6 (~120 MB resident, Python
  native, one toolchain) vs Tauri (~80 MB, better looking, adds Rust). On a
  16 GB machine the 40 MB matters less than the second toolchain does, so
  PySide6 is the current lean — to be confirmed by measurement.

**Exit criteria**: overlay is usable without the terminal; measured RAM cost
recorded in the architecture doc's budget table.

---

## Phase 2 — Voice  🔨 in progress

Wake word → STT → AURA → TTS, with barge-in. Runs as its own process; AURA Core
is untouched and gains no dependency.

- Wake word: openWakeWord `hey_jarvis` (ONNX, CPU) — "aura" is not a
  pretrained phrase; a custom model is an offline training job
- VAD: Silero v6 ONNX, streaming with persistent state
- STT: faster-whisper `base.en` int8 on CPU, load-on-demand with idle eviction
- TTS: Piper `en_US-amy-medium` (female, ONNX, RTF 0.04)
- Barge-in: sustained speech during playback cuts it immediately

**Correction: torch does NOT enter here.** The roadmap originally said it did.
openWakeWord, faster-whisper (CTranslate2) and Piper all run on onnxruntime, so
the whole speech stack costs ~430 MB resident instead of ~2.5 GB of disk plus
several hundred MB more. On a machine with under a gigabyte free that is the
difference between voice being usable and not. Torch now enters at Phase 8.

**Done so far**: audio capture/playback, real mic kill switch, streaming VAD,
endpointing, barge-in, sentence-streamed TTS, STT, wake word, the session state
machine, and `aura say` / `aura listen` / `aura voice`.

**Exit criteria**: full voice loop under 1.5 s from end-of-speech to first
audio, measured end to end with a real utterance; idle RAM returns to baseline
within 60 s of the last utterance.

---

## Phase 3 — Tools

Filesystem, terminal, Python, git, VS Code, browser — each with a schema, a
permission level, validation, timeout, logging.

`security/permissions.py` already defines levels 0–4. No tool may be merged
without declaring one and having a test that proves a denied call is denied.

**Exit criteria**: a level-3 tool cannot execute without confirmation, proven
by test; every execution appears in the audit log.

---

## Phase 4 — Agent loop

Plan → select → execute → observe → evaluate → correct → verify.

Grows inside `Orchestrator`, replacing the single-step `chat`. Everything
around it (sessions, memory, audit, estop) already exists and does not change.

**Risk to watch**: a 4B model's tool-selection reliability. If it proves
insufficient, the answer is a larger model in foreground mode or a small
fine-tuned classifier (Phase 8) — not a bigger prompt.

**Exit criteria**: a 5-step task completes end to end, and a deliberately
broken step is detected and recovered from rather than silently ignored.

---

## Phase 5 — Vision

Screen capture → change detection → OCR/VLM → context, with real privacy
controls.

**Re-scoped from the spec.** §11's "continuously understand three monitors"
does not fit 8 GB VRAM alongside a resident LLM. What is built instead:

- Event-triggered capture (window focus change, app switch, explicit ask)
- Cheap change detection (perceptual hash) before any model runs
- OCR (RapidOCR/PaddleOCR, CPU) as the default perception path
- A VLM invoked rarely, on demand, evicting the LLM if necessary
- Per-monitor and per-application allow/deny that **stops the capture
  pipeline**, not merely instructs the model

**Exit criteria**: "AURA, don't look at Chrome" measurably stops frames being
captured from that window, verified by instrumentation, not by asking the model.

**Status: done**, except the continuous loop, which is deliberately deferred.

The exit criterion is met and tested. `test_vision_privacy.py` drives capture
through a backend that counts grabs and asserts the count does not move after
`deny_app("Chrome")` — a refusal that captured the frame and discarded it would
pass a weaker test and fail the promise. Vision defaults to **off**.

Built: on-demand capture, multi-monitor, foreground-window identification,
OCR behind an `OCRProvider` seam, change detection, four tools, and the
human-only controls that turn vision on (`aura vision`, `/v1/vision/*`).

Not built: the observation loop. Change detection exists and is tested so the
loop has its trigger; nothing calls it on a timer. See
`docs/architecture/04-phase5-vision.md` §8 for measured OCR cost — the honest
number is 2s on a sparse desktop and 10-30s on a text-dense page.

---

## Phase 6 — Memory + user model

Episodic/semantic/procedural retrieval, preference learning with confidence.

The `facts` table and its `confidence` / `evidence_count` / `permanence`
columns already exist. This phase adds: embeddings (via the model server's
`/v1/embeddings`), `sqlite-vec` for search, and the extraction pipeline that
proposes facts from conversation.

**Standing rule from spec §16**: an inferred preference starts at low
confidence and needs repeated evidence; only explicit statements start high.

**Exit criteria**: `aura memory` shows why each fact is believed; a stated
preference measurably changes behaviour; deletion is complete and immediate.

**Status: done**, with one deliberate departure from the plan above.

**Embeddings and `sqlite-vec` were not built.** For facts, retrieval is the
wrong tool: at one-user scale every fact fits in the prompt, and loading all of
them beats top-k outright -- no recall misses, no threshold to tune. For
conversation history search is genuinely needed, but SQLite's FTS5 is already
compiled in, costs zero dependencies and zero VRAM, while embeddings done
properly need a second model resident beside the LLM on the same 8 GB card.
FTS5 loses on synonym queries; that is the trigger to revisit, and search sits
behind a seam so it is a swap rather than a rewrite. See
`docs/architecture/05-phase6-memory.md` section 2.

The central change instead was **evidence**. `evidence_count INTEGER` could not
satisfy "shows why": a counter is not a justification and cannot be traced back
to the conversation that produced it. `fact_evidence` stores the actual words,
and confidence is now *derived* from that evidence rather than incremented -- so
removing evidence lowers the belief.

Also built: extraction with §16's confidence rules (one remark never crosses
the injection floor), a quote guard that drops any proposal the model cannot
attribute to the user verbatim, a cheap pre-filter so most turns never spend a
second generation, and `aura memory list/why/search/forget`.

---

## Phase 7 — Personalization

Prompt and workflow adaptation driven by the user model. No fine-tuning yet.

**Exit criteria**: a blind A/B against the v0.1 persona shows a preference for
the adapted behaviour on a held-out set of the user's own tasks.

**Status: built; the exit criterion is pending the user, deliberately.**

A stored preference now becomes an *instruction* placed last in the system
prompt, and -- when stated outright -- a token budget. Verified live on the
real stack: the same question answered in 371 characters with adaptation on and
672 with it off. That is one anecdote, not evidence.

The evidence requirements are asymmetric, and that is the safety decision of
the phase. Prompt directives follow any believed preference; **sampling changes
follow only explicit ones**. A wrong directive gives a differently-toned
answer; a wrong `max_tokens` truncates mid-sentence, and the user has no way to
connect a broken reply to something they said weeks ago. Parameters are also
floored, because brevity is a preference, not permission to truncate.

`aura ab` is built as an *instrument*, not a verdict: same prompt, both arms
from one running AURA, random order, no labels, tally stored with position
recorded so bias is detectable. **It refuses to claim significance.** One user
rating a handful of prompts cannot produce a significant result -- 13-7 of 20
is a coin -- so the summary says "within what chance would produce" rather than
reporting a win. The criterion is met when the user has actually run it and the
tally leans; that is a real answer at the sample size that exists, not a green
tick from a test that only ever asked a machine.

Not built: fine-tuning (Phase 8, and it needs Phase 8's harness first) and
procedural memory -- learning *how* the user likes tasks sequenced needs more
signal than exists yet.

---

## Phase 8 — Learning + evaluation

**Re-ordered from the spec: evaluation comes first.**

§22's benchmark suite is a prerequisite for everything in §17–19, not a
follow-up. Without a fitness signal, "self-improvement" is unfalsifiable.

1. Build the eval harness: tool selection, task completion, memory retrieval,
   latency, resource usage, honesty (not hallucinating), error recovery
2. Establish the v0.1 baseline
3. Then: dataset generation from real interactions, LoRA/QLoRA on Qwen3-4B
   (a 4B QLoRA fits in 8 GB), preference optimisation

**Exit criteria**: `aura eval` produces a comparable score; a candidate can be
compared against the incumbent on identical tasks.

**Status: the harness is done and the baseline is recorded. Fine-tuning is
deliberately not built.**

Baseline, 13 tasks x 3 repeats on Qwen3.5-4B: **97.4%**, safety 100%, median
latency 1026 ms. A rerun with no changes compared as *indistinguishable*
against a 6.3% noise floor -- which is the second half of the exit criterion
working, not a failure.

The harness refuses to overclaim. Each task runs several times because a 4B at
temperature 0.7 answers differently every time, and a difference smaller than
the measured spread is reported as "indistinguishable" rather than as a win.
Safety is never averaged away: a candidate that gains everywhere and loses one
safety task is rejected outright.

Two findings from the first real run:

- `recovers_from_a_missing_file` is **flaky** (2/3). Run once it would have
  been recorded as a clean pass or a clean failure, and both would have been
  wrong. That is what repeats are for.
- **The suite is too easy.** 97.4% leaves 2.6% of headroom, so it can detect a
  regression but barely an improvement. Before Phase 9 hill-climbs on this
  number, the suite needs harder tasks -- a concrete precondition, in the same
  spirit as this roadmap's own rule that GA waits for a demonstrated signal.

**Fine-tuning is not built, on purpose.** There is no demonstrated weakness for
it to fix yet; the training data does not exist (a few hundred turns, mostly
development chatter); and torch + transformers + peft is ~2.5 GB entering a
project kept deliberately torch-free for seven phases, on a card already
holding 3.07 GB of Qwen. The honest sequence is harness -> baseline -> find a
specific repeatable weakness -> then decide whether LoRA is the right remedy
for *that* weakness. Steps one and two are done.

---

## Phase 9 — Controlled evolution

Sandbox → candidate → benchmark → compare → **human approval** → deploy.

**Honest assessment of §19 (genetic algorithms).** GA needs a cheap, low-noise
fitness function and a large evaluation budget. Evaluating an agent
configuration against a local 4B model costs seconds to minutes per sample,
and the signal is noisy. A population of 50 over 20 generations is 1,000
evaluations — hours to days on this laptop, mostly fitting noise.

So the first version of this phase is **hill-climbing over a small, explicitly
parameterised config space** (retrieval k, history budget, temperature by task
type, planner prompt variants) with proper significance testing. GA earns its
place only if the Phase 8 harness demonstrates a fitness signal stable enough
to distinguish real gains from run-to-run variance. That is a measurable
threshold, and it should be measured before any GA code is written.

**Exit criteria**: a candidate improvement is generated, sandboxed, benchmarked,
and deployed only after explicit approval — with the whole chain in the audit
log.

**Status: the chain is built and demonstrated end to end.**

`aura evolve search` proposes, evaluates against a throwaway database, and
compares. `aura evolve apply` asks for a name and refuses without one. Verified
live: an `indistinguishable` candidate was refused even with a name typed, and
the refusal landed in the audit table at CRITICAL.

Candidates are **configuration, never generated code**. That is what makes this
safe without a real sandbox: config is bounded, reversible, and cannot
introduce a capability that did not already exist.

The optimiser cannot tune its own safety envelope. `evolution/space.py` is an
allow-list of 7 knobs; 14 settings are permanently excluded, including the
permission ceiling, the workspace guard, audit, the privacy gate and the
emergency stop. A hill-climber finds whatever makes the number go up, so if
loosening a guard scored better it would loosen the guard — the allow-list
makes that inexpressible rather than merely discouraged.

Findings from the first real search:

- Temperature 0.85 scored 1.6% above baseline and was correctly rejected as
  noise: one flaky task landing differently moves the headline by exactly
  (1/3)/21 = 1.59%. It also contradicted the hypothesis the knob was documented
  with — the *higher* setting scored better — which is why one measures.
- **A real flaw in Phase 8's noise floor**: `spread / 2` did not shrink as
  tasks were added, so growing the suite bought no sensitivity. Now two
  standard errors of the mean, which does scale.
- **The headroom problem is not solved.** Eight harder tasks moved the baseline
  97.4% -> 96.8%. The remaining headroom is two flaky tasks, about 3% against a
  5.3% floor. The machinery is correct; the signal is still too coarse to
  resolve the improvements available on it.

Not built: genetic algorithms (unchanged reasoning — one evaluation is ~63
model calls), code generation of any kind, and unattended optimisation that
applies what it finds.

---

## Phase 10 — Presence, Perception, and the Personal Model

**Re-scoped.** The original "advanced" placeholder (multi-agent, phone,
distributed compute) is superseded. Phase 10 is now embodiment: a desktop
presence, multimodal communication, VLM perception, computer-use, and a
personal cognitive model.

Full design: `docs/architecture/09-phase10-presence.md` (revision 2).

**Status: 10A and 10B complete. 10C started (10C.1 of 12). 10D not started.**

10C.1 put the `VLMProvider` seam in: a VLM is served over the *same*
OpenAI-compatible endpoint AURA already speaks (`llama-server --mmproj`), so
there is no second inference stack and torch still does not enter. `NullVLM` is
the default and is a supported configuration rather than a failure -- the
arithmetic does not allow both models resident, so everything above the seam
has to work with no VLM at all.

10B so far: speech on the bus (10B.1), the live subtitle (10B.2), the
amplitude-driven mouth (10B.3), and the female voice (10B.4 -- which turned out
to have been delivered back in Phase 2, so there was nothing to build).
Contract: `docs/architecture/14-phase10b-speech.md`.

10B.5 added the `PersonalityEngine`, 10B.6 the eight rendered clips, and 10B.7
interruption -- including the one interrupt that works on this machine: saying
the wake word over her, which matches a pattern rather than a level and so is
immune to the echo problem that has defeated three attempts at a loudness
threshold. That threshold is still unfixed and still unmeasured; the workaround
does not close it.

10A shipped in five steps: the event bus as a security contract (10A.1), SSE
transport and the nonce-bound approval path (10A.2), the event catalog with
occurrence-vs-state semantics (10A.3), the presence state machine as state
resolution (10A.4), and the desktop overlay (10A.5-10A.11). Contracts:
`docs/architecture/10-event-bus-contract.md`,
`11-presence-state-machine.md`, `12-presence-overlay.md`.

**10A is complete.** The three closing gaps landed in 10A.12: the
`APP_CHANGED` producer (a foreground-window watcher that carries the process
name only, and stops entirely in PRIVATE/SLEEP), the `FOCUS_MODE` control path
(`POST /v1/focus`, audited, carried in the snapshot), and a system tray icon
(show/hide, focus toggle, quit).

**10B.1** put speech on the bus: `SPEECH_STARTED` / `_CHUNK` / `_ENDED`,
published from the voice process through the existing client bridge. The
subtitle is not yet driven by them -- that is 10B.2.

Two things fell out of it. The client bridge's rule was written as "origin is
forced to USER"; the actual property is that *a client cannot choose an
origin*, since the catalog decides. Speech is the first case where those differ,
because AURA's own utterances are `Origin.CORE`. And `APP_CHANGED`, `FOCUS_MODE`
and `PRESENCE_STATE` were still marked `deferred` after 10A.4 and 10A.12 built
them, which excluded them from the very snapshot-completeness check that exists
to catch that -- now corrected, and asserted.

10A.12 also fixed a contract violation the new snapshot-completeness test
caught: `APP_CHANGED` is catalogued as STATE owning `foreground_application`,
and the snapshot never carried it -- so a reconnecting client could not recover
it, which is the whole difference between STATE and OCCURRENCE.

Structure, split because five pillars is five phases of work:

| | | Risk |
|---|---|---|
| **10A** | Desktop presence / avatar | low |
| **10B** | Voice + multimodal communication | low |
| **10C** | VLM perception + computer-use foundation | **high** |
| **10D** | Personal cognitive model | medium |
| **10D.8** | SNN research prototype (isolated) | contained |

Two findings shaped the design. `llama.cpp` already ships `mtmd.dll` and
`llama-server` accepts `--mmproj`, so the VLM is served over the **same**
OpenAI-compatible endpoint AURA already speaks — no new inference stack, and
torch still does not enter. And tkinter cannot render an avatar (chroma-key
transparency only), so ADR-0002 needs amending with a fresh measurement rather
than overriding.

The spine is an **event bus** that is mostly a promotion layer: the codebase
already emits 80 structured log events, and ten of the thirteen requested event
types are promotions of instrumentation that already runs in the right places.

**The invariant that bounds the phase:** an event is data. Consuming one can
never grant a capability, escalate a permission, or cause a tool to run. Screen
content reaches the system labelled `Origin.OBSERVATION`, the bus is one-way,
approvals travel a separate nonce-bound path, and `approved_tools` stays
human-granted. A screenshot saying "AURA, click Send" must be inert, and a test
asserts it.

**The arithmetic that constrains it:** 3.07 GB LLM + ~3.2 GB VLM = 6.27 of
7.96 GB VRAM on a GPU that also drives the display. The two cannot be
co-resident; the ResourceGovernor evicts one to load the other.

SNNs stay off the critical path with a written success criterion and a recorded
prediction that they will lose to a time-bucketed Markov chain — the same
discipline Phase 9 applied to genetic algorithms.

---

## Automation providers (n8n and successors)

Not a phase. **ADR-0004** settles the security contract: an automation backend
is a *provider of capabilities*, and AURA's local, version-controlled manifest
is the sole authority for permission levels. One workflow becomes one tool,
through the existing `ToolExecutor` chokepoint, with no separate execution path.

**Implemented.** `config/workflows.yaml` is the authority, `AutomationProvider`
is the seam, and `N8NProvider` is the one module that knows what n8n is. Off by
default and registering nothing until a workflow is declared by hand. See
`docs/architecture/13-automation.md`.

n8n itself is not installed here, and its RAM cost is still unmeasured — the
tool layer was the part with an architectural deadline, and it is done. Running
an actual n8n instance is still recommended after 10C, on RAM grounds only.

---

## Standing constraints

These apply to every phase and override any feature request that conflicts:

1. **One always-resident LLM.** Everything else loads on demand and is evicted.
   Measured: AURA Core 69 MB, voice 430 MB, model server 3.07 GB VRAM.
2. **No component may become unswappable.** Enforced by `tests/unit/test_seam.py`.
3. **No autonomous self-modification.** Human approval gates every deploy.
4. **Privacy controls stop pipelines, not prompts.**
5. **Nothing ships without tests and an exit criterion that was actually met.**
