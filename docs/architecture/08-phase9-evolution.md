# Phase 9 — Controlled evolution

Status: **in progress** · Date: 2026-09-08

---

## 1. The rule that bounds everything else

Spec §18: *"AURA must never autonomously rewrite its production code and
immediately deploy it."* Standing constraint 3: *"No autonomous
self-modification. Human approval gates every deploy."*

So the chain is:

```
propose → sandbox → benchmark → compare → HUMAN APPROVAL → deploy
                                              ↑
                                   nothing crosses this
                                   line without a person
```

Every arrow before the approval gate is automatic. The gate itself is not
automatable, not configurable to auto-approve, and not bypassable by a flag.
There is no `--yes`.

## 2. Candidates are configuration, not code

The single decision that makes this phase tractable and safe: **a candidate is
a set of configuration overrides, never generated source.**

Code generation would mean AURA writing Python that then runs with AURA's own
permissions — a category of risk that needs a real sandbox (process isolation,
syscall filtering, a separate filesystem) that this machine does not have and
this project does not need. Configuration is bounded, inspectable in full on
one screen, trivially reversible, and cannot introduce a new capability.

The cost is a smaller search space. That is a good trade: nearly all of AURA's
tunable behaviour *is* configuration — temperature, history budget, step
ceilings, retrieval width — and those are exactly the knobs the roadmap named.

## 3. The optimiser may not tune its own safety envelope

This is the security property of the phase, and it is an explicit allow-list,
not a deny-list.

`evolution/space.py` enumerates every parameter that may be varied, with hard
bounds on each. Anything not on that list cannot be touched, so adding a
tunable is a deliberate act by a person editing that file.

Permanently excluded, and the reason each one matters:

| Excluded | Why |
|---|---|
| `tools.max_autonomous_level` | An optimiser that can raise its own permission ceiling will, because more capability scores better. |
| `tools.deny_patterns`, `workspace_roots` | The workspace guard is the thing being protected; it cannot be a variable. |
| `security.*` | Audit and authentication are not performance knobs. |
| `vision.mode`, `vision.deny_apps` | The privacy gate is the user's decision, never a search result. |
| `app.emergency_stop` | Self-explanatory. |
| `memory.learn_from_conversation` | Turning off learning to score better on a memory-free suite is a local optimum that makes AURA worse. |

A hill-climber's whole nature is to find whatever makes the number go up. If
loosening a guard makes the number go up, it will loosen the guard, and it will
be right to — by the metric. The metric is not the point. The allow-list is
what keeps that from being expressible.

## 4. Hill-climbing, not genetic algorithms

The roadmap already made this call and the reasoning holds: GA needs a cheap,
low-noise fitness function and a large evaluation budget. One evaluation here
is 63 model calls and about a minute. A population of 50 over 20 generations
is 1,000 evaluations — most of a day, mostly fitting noise.

So: **coordinate ascent over one parameter at a time.** Propose a neighbour,
evaluate it, keep it only if the Phase 8 comparison says it is genuinely
better. Slow, explainable, and each step is a single change a person can read.

GA earns its place only if the harness demonstrates a fitness signal stable
enough to distinguish real gains from run-to-run variance. That is a
measurable threshold, and it has not been measured yet.

## 5. The precondition Phase 8 uncovered

The Phase 8 baseline was **97.4%** — 2.6% of headroom against a 6.3% noise
floor. A hill-climber on that signal cannot work: every candidate, good or bad,
lands inside the noise and is correctly reported as indistinguishable. It would
run for hours and conclude nothing.

So the first work of this phase is not the optimiser, it is **making the suite
hard enough to have room to improve in**. Tasks are added in the categories
where AURA plausibly still fails: multi-step tool chains, distinguishing
similar stored facts, resisting a leading question, recovering from a tool
error without repeating it.

This is a precondition, stated as one, in the same spirit as the roadmap's own
rule that GA waits for a demonstrated signal. An optimiser built on a
saturated benchmark is theatre.

**It did not work.** Eight harder tasks moved the baseline 97.4% -> 96.8%.
See §8: the attempt is recorded rather than quietly dropped, because the
failure is itself the finding -- AURA passes nearly everything put in front
of it, and the remaining headroom is two flaky tasks rather than any failing
one.

## 6. Deploy, and getting back

"Deploy" means writing the approved overrides into `config/local.yaml` — the
layer that already exists for machine-specific settings, above the defaults and
below the environment. Nothing else changes: no code is written, no process is
patched at runtime.

Every deploy writes a timestamped snapshot of the previous config first, so
`aura evolve rollback` is a file copy rather than a reconstruction. The whole
chain — proposal, scores, verdict, approval, deploy, rollback — goes into the
audit log with the request id, because §25 asks "why did AURA do this?" and a
config that changed itself is exactly the case where that question gets asked.

## 7. Scope

**Built:** harder eval tasks and a re-baseline, the parameter space with its
exclusions, candidate generation, the search loop, the approval gate, deploy
with snapshot and rollback, audit of the whole chain, and `aura evolve`.

**Not built:** genetic algorithms (§4), code generation of any kind (§2),
unattended optimisation runs — a search that proposes while nobody is watching
is fine, but nothing it finds is applied without a person reading it.

## 8. What the first real search found

Run against `llm.temperature`, 3 repeats, 21 tasks:

```
Baseline 96.8%.

  indistinguishable   llm.temperature: 0.7 -> 0.55   96.8% vs 96.8%
  indistinguishable   llm.temperature: 0.7 -> 0.85   98.4% vs 96.8%
```

**The 1.6% "gain" at 0.85 is the most useful thing here, because it is not a
gain.** One flaky task landing differently moves the headline by exactly
(1/3)/21 = 1.59%. A harness that reported that as an improvement would have
sent a hill-climber off optimising a coin flip, and every step would have
looked like progress. It said indistinguishable instead, and it said the same
thing on a repeat run.

The result also contradicted the hypothesis this phase was built around. The
`llm.temperature` knob is documented as "lower temperature should make flaky
tasks consistent"; the *higher* setting scored better. Which is the entire
reason for measuring rather than reasoning about it.

### A flaw the search exposed in Phase 8

The noise floor was `spread / 2`. That gave a plausible number on the suite it
was written against and was wrong in a way that mattered: **it does not shrink
as tasks are added.** Under it, doubling the suite bought no extra sensitivity,
so the only available route to detecting a real improvement was closed.

The headline is a *mean* of per-task rates, so the uncertainty that matters is
the standard error of that mean -- `spread / sqrt(tasks)` -- which does shrink.
The floor is now two standard errors; at one, roughly a third of pure-noise
comparisons would read as results, and a hill-climber makes a great many
comparisons.

### The headroom problem is not solved

Eight harder tasks moved the baseline from 97.4% to 96.8%. The 4B passes almost
everything put in front of it, and the remaining headroom is not in failing
tasks but in **two flaky ones**. That is where a real improvement would have to
show up, and it is a small target: the whole distance between "flaky" and
"consistent" across the suite is about 3%, against a 5.3% floor.

Stated plainly: **this suite still cannot resolve the improvements that are
available on it.** The machinery is correct and the signal is too coarse. The
fix is more tasks -- which now genuinely helps, since the floor scales with
task count -- and specifically tasks the model fails at rather than passes.
Writing those requires finding things AURA is bad at, which is a different kind
of work from building an optimiser.
