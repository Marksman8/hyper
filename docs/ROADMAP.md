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

---

## Phase 7 — Personalization

Prompt and workflow adaptation driven by the user model. No fine-tuning yet.

**Exit criteria**: a blind A/B against the v0.1 persona shows a preference for
the adapted behaviour on a held-out set of the user's own tasks.

---

## Phase 8 — Learning + evaluation

**Re-ordered from the spec: evaluation comes first.**

§22's benchmark suite is a prerequisite for everything in §17–19, not a
follow-up. Without a fitness signal, "self-improvement" is unfalsifiable.

1. Build the eval harness: tool selection, task completion, memory retrieval,
   latency, resource usage, hallucination, error recovery
2. Establish the v0.1 baseline
3. Then: dataset generation from real interactions, LoRA/QLoRA on Qwen3-4B
   (a 4B QLoRA fits in 8 GB), preference optimisation

**Exit criteria**: `aura eval` produces a comparable score; a candidate can be
compared against the incumbent on identical tasks.

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

---

## Phase 10 — Advanced

Multi-agent decomposition, predictive assistance, phone integration,
distributed compute. Not planned in detail; the architecture keeps the seams
open and that is enough for now.

---

## Standing constraints

These apply to every phase and override any feature request that conflicts:

1. **One always-resident LLM.** Everything else loads on demand and is evicted.
   Measured: AURA Core 69 MB, voice 430 MB, model server 3.07 GB VRAM.
2. **No component may become unswappable.** Enforced by `tests/unit/test_seam.py`.
3. **No autonomous self-modification.** Human approval gates every deploy.
4. **Privacy controls stop pipelines, not prompts.**
5. **Nothing ships without tests and an exit criterion that was actually met.**
