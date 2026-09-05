# Phase 4 — The agent loop

Status: **complete** · Date: 2026-09-05

---

## 1. What Phase 4 is

The model picks its own tools now. Phase 3 built the tools and the gate; this
closes the loop:

```
  ask the model  ->  it requests tools  ->  run them  ->  give back the results
        ^                                                          |
        +----------------------------------------------------------+
                        until it answers, or the budget runs out
```

## 2. Why there is no plan/execute/verify state machine

Spec §7 lists ten stages — plan, select, execute, observe, evaluate, correct,
continue, verify, complete. Those are real things that have to happen, but
implementing them as *AURA's* stages produces a rigid agent: one that commits
to a plan before it has seen anything, and cannot revise it after the first
surprise.

So they happen inside the loop, where the model does them, and AURA owns only
what must not be delegated:

| AURA owns | Why |
|---|---|
| Step budget and wall clock | A loop that decides when to stop is a loop that sometimes does not |
| The permission gate | The loop has no way to approve anything, including for itself |
| The emergency stop | Checked every iteration, not once at entry |
| The trace | "Why did AURA do this?" needs an answer that is not a guess |

## 3. Approval is something a person grants

`AgentLoop(approved_tools=[...])` is a list a **human** supplies, per task,
before the loop starts. The model cannot add to it. There is deliberately no
path by which a sufficiently persuasive model argument becomes an approval.

With an empty list — the default — a level-2 call comes back as:

```
REFUSED: write_file needs explicit confirmation. This action needs the user's
approval, which has not been given for this task. Do not retry it — explain to
the user what you would like to do and why.
```

The model then explains and stops, and `pending_confirmations` names the tool
so a UI can offer the approval. Verified live: asked to create a file with no
approval, AURA replied *"I attempted to create the file... but the system
requires explicit confirmation from you to proceed."*

## 4. Failure is fed back, never raised

A tool error, a refusal, a denied path, an argument that will not validate —
each becomes an observation the model reads:

- `ERROR: no such file` — the model can try a different path
- `REFUSED: ... Do not retry it` — it explains instead of looping
- `DENIED: ... find another approach` — it stops trying to escape the workspace

An exception would end the task instead. The "do not retry" wording is load
bearing: without it a model spends its entire budget re-issuing the same
rejected call.

## 5. Running out of budget

When the step budget is exhausted, the model is **told**, and asked for a final
answer with no tools offered:

> You have used all 8 available tool steps. Answer now with what you have
> found, and say plainly what is still unresolved. Do not request more tools.

Tools are withheld on that last call, or the model asks for another one and the
loop never closes. `stopped_early` and `stop_reason` reach the caller, so an
answer produced after hitting the ceiling is distinguishable from a finished
one.

## 6. Two types, not one

`core.types.ToolInvocation` is what a **model asked for** — unvalidated,
unapproved. `tools.types.ToolCall` is what AURA **decided to run**. The
conversion happens on one line in `tools/runner.py`, so the question "who
decided this was allowed?" has exactly one place to look.

It is also what keeps `aura.core` from importing `aura.tools`: the loop
declares its needs as a Protocol, and the adapter satisfies it.

## 7. Measured

Against Qwen3.5-4B on this machine:

| Task | Tool calls | Latency |
|---|---|---|
| "What is 17 times 3?" | 0 | 1.1 s |
| "What Python version does this project require?" | 1 (`read_file`) | 2.8 s |
| "How many Python files in src/aura/tools?" | 1 (`list_directory`) | 2.4 s |
| "Create a file..." (unapproved) | 1, refused | 2.0 s |
| "Create a file..." (approved) | 1 (`write_file`) | 1.5 s |

The zero-tool case matters as much as the others: a question that needs no tool
must not cost a round trip.

**An honest observation.** Asked what the smallest file in a directory *does*,
the model listed the directory and then said the file "likely contains
initialisation code" rather than reading it. The answer was not wrong, but it
guessed where it could have looked. That is prompt quality, not loop
correctness, and it is the kind of thing Phase 8's evaluation harness exists to
measure rather than argue about.

## 8. Out of scope

Streaming a tool-using turn (the loop is non-streaming; a turn that runs tools
is slow for better reasons). Parallel tool execution — the calls in one step run
sequentially, which is simpler and rarely the bottleneck at this scale.
Sub-agents and task decomposition across sessions (Phase 10).
