# Phase 8 — Evaluation

Status: **in progress** · Date: 2026-09-08

---

## 1. Why this comes before learning

The spec put self-improvement first and evaluation later. The roadmap reversed
that, and this phase is where the reversal pays.

Without a fitness signal, "AURA got better" is unfalsifiable. Fine-tuning a
model, changing a prompt, tuning retrieval — every one of these produces a
*different* AURA, and nothing in the spec's original ordering can tell you
whether different is better. You would be optimising against a feeling.

So the deliverable is the instrument, and the exit criterion says exactly that:

> `aura eval` produces a comparable score; a candidate can be compared against
> the incumbent on identical tasks.

Note what the criterion does **not** say: it does not require LoRA. That is
deliberate, and §6 explains why I am not building it in this phase.

## 2. Variance is the adversary, not the metric

A 4B model at temperature 0.7 gives a different answer to the same question
every time. This is the fact that decides the whole design.

If the harness runs each task once and reports a score, then a "3-point
improvement" is very likely to be noise, and every conclusion drawn from it is
worthless — worse than worthless, because it *looks* like measurement. Phase 9
is supposed to hill-climb on this signal; hill-climbing on noise produces
confident nonsense.

So:

- Every task runs **`repeats` times** (default 3), and the report carries the
  mean *and the spread*.
- Comparison between two configurations **refuses to declare a winner** when
  the difference is inside the combined spread. It says "indistinguishable",
  which is a real result.
- Sampling is seeded per repeat, so a rerun of the same suite against the same
  build is reproducible even though the model is stochastic.

A harness that cannot say "I cannot tell" is not an instrument, it is a
random number generator with a progress bar.

## 3. Deterministic scoring, and the narrowness that buys

Scoring is **checkable assertions only**. No LLM judge.

The obvious alternative — ask the model to rate the answer — fails here for a
specific reason: the only model available is the same 4B being evaluated. A
model judging its own output shares its own blind spots, and its scores drift
with the same sampling noise the harness is trying to see through. It would add
variance while appearing to remove subjectivity.

What is checkable:

| Scorer | Question it answers |
|---|---|
| `contains` / `not_contains` | Did the answer state the required fact? |
| `regex` | Does it match a required shape? |
| `tool_called` / `tool_not_called` | Did it pick the right tool? (§7) |
| `refused` | Did the guard hold? (§23) |
| `steps_under` | Did it get there without wandering? |
| `uncertain` | Did it admit not knowing, instead of inventing? |

The cost is honesty about coverage: this measures a **narrow slice** of what
AURA does. It cannot tell you whether an explanation was well written. That is
the right trade — a narrow thing measured properly is worth more than a broad
thing measured badly, and a badly-measured broad score is precisely what would
mislead Phase 9.

## 4. Tasks exercise AURA, not the model

The suite runs against the real orchestrator, tools and memory, on the machine
as configured. A benchmark of the raw model would measure Qwen; the point here
is to measure *AURA* — its prompt, its tool layer, its guards, its retrieval.

The categories, matching the roadmap:

- **tool_selection** — is the right tool chosen for the request?
- **safety** — do the workspace guard and permission gate hold under pressure?
- **memory** — is a stored fact retrieved and used?
- **hallucination** — is "I don't know" preferred to invention?
- **error_recovery** — after a tool fails, does the loop adapt or repeat itself?
- **latency** and **resources** — recorded on every task, not scored, because a
  fast wrong answer is not better than a slow right one. They are reported so a
  change that trades quality for speed is *visible* rather than hidden inside
  one number.

No task requires the internet (§4).

## 5. One number, and its dangers

`aura eval` reports a headline pass rate because Phase 9 needs a scalar to
climb. That number is a genuine hazard: a single figure invites optimising the
figure, and the categories above are not equally important — a safety
regression is not worth trading for two extra tool-selection passes.

Mitigation: the headline is always printed **with the per-category breakdown**,
and safety failures are reported separately and never averaged away. A
comparison that improves the total while regressing safety is reported as a
failure, not an improvement.

## 6. Why LoRA is not in this phase

The roadmap lists fine-tuning as step 3. I am not building it yet, and the
reasoning is the same discipline the roadmap itself applied to genetic
algorithms in Phase 9.

1. **There is no demonstrated weakness for it to fix.** The baseline does not
   exist until this harness runs. Fine-tuning before measuring is choosing a
   remedy before the diagnosis.
2. **The data does not exist.** QLoRA on a 4B needs thousands of good examples.
   This install has a few hundred conversational turns, most of them me
   building AURA rather than the user using it. Training on that teaches the
   model to imitate a development log.
3. **The cost is real.** Torch, transformers, peft and bitsandbytes are ~2.5 GB
   of dependencies entering a project that has stayed deliberately torch-free
   for seven phases, on a card already holding 3.07 GB of Qwen.
4. **Catastrophic forgetting is the likely outcome.** A 4B tuned on a few
   hundred in-domain examples typically gets worse at everything else, and
   without the harness there would be no way to notice.

The honest sequence is: harness → baseline → find a specific, repeatable
weakness → decide whether fine-tuning is the right remedy for *that* weakness.
Steps one and two are this phase. That is a measurable threshold rather than a
schedule, which is what the roadmap asks for everywhere else.

## 7. Scope

**Built:** the task suite, deterministic scorers, the runner with repeats and
variance, incumbent-vs-candidate comparison that can say "indistinguishable",
`aura eval`, and a recorded baseline.

**Not built:** LoRA/QLoRA training, preference optimisation, dataset generation
from real interactions (see §6), and any LLM-as-judge scoring (see §3).

## 8. The measured baseline, and what it says about the suite

First run, 13 tasks x 3 repeats, against Qwen3.5-4B on this machine:

```
Score 97.4%  (39 runs, spread 8.9%, 32s, median latency 1026 ms)

  safety           100.0%     tool_selection   100.0%
  memory           100.0%     hallucination    100.0%
  general          100.0%     error_recovery    66.7%
```

Two things follow, and the second is uncomfortable.

**The one genuine finding.** `recovers_from_a_missing_file` is flaky: 2 passes
in 3. That is exactly what the repeats exist to surface -- run once, it would
have been recorded as either a pass or a failure, and both would have been
wrong. The model usually says the file is missing and occasionally answers
around it.

**The suite is too easy.** 97.4% is close to the ceiling, which means it can
detect a *regression* but has almost no room to detect an *improvement*.
Phase 9 hill-climbs on this number; with 2.6% of headroom, nearly every real
gain would land inside the 6.3% noise floor and be reported -- correctly -- as
indistinguishable.

That is a property of the suite, not a bug in the harness, and it is better to
say so than to celebrate a high score. A benchmark AURA passes is not a
benchmark that measures AURA. The fix is harder tasks, and the right time to
write them is when Phase 9 has a specific change it wants to evaluate: tasks
invented to fill a scoreboard tend to measure whatever was easy to assert.

**Before Phase 9 climbs on this signal, the suite needs headroom.** That is a
concrete precondition, in the same spirit as the roadmap's own rule that
genetic algorithms must wait until a stable fitness signal is demonstrated.
