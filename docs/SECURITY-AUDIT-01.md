# Security audit 01 — Stage 1

Findings from the audit that produced `cdee879` ("Security 1/5"), continued.

**On the standing of this document.** Finding 1 is the one that was fixed, and
it is recorded here from its commit. Findings 2–5 were **re-derived by reading
the code**, because the original list survived only in a conversation and not
in the repository. They may not be the same four the first audit found. That is
itself the first lesson of this document: a finding that exists only in a chat
log is a finding that does not exist.

Each entry states what is wrong, how to demonstrate it, and what a fix has to
be true of. Every fix below landed with a test that was red first.

| | Finding | Severity | Status |
|---|---|---|---|
| 1 | The kill switch did not gate `ToolExecutor` | critical | **fixed** — `cdee879` |
| 2 | An approval is granted per tool *name* and spends unboundedly | high | **fixed** |
| 3 | `run_python` is outside the workspace guard and can delete the kill switch | **critical** | **fixed** — denied by default |
| 4 | The stop does not interrupt work already in flight | medium | **fixed** for subprocesses |
| 5 | Two approval systems exist; the strong one is not on the tool path | medium | **deferred** 2026-09-17 |
| 6 | `act()` raised `AttributeError`; the agent loop's stop check never ran | **critical** | **fixed** |

Finding 6 was not in the original series. It was found by `mypy` while fixing
finding 3, which is its own small lesson: the type checker had been reporting it
for five phases and the error was never read.

Findings 2 and 3 compose into a single exploit chain. Finding 5 is most of the
fix for finding 2. Read them together.

---

## 1. The kill switch stopped AURA thinking and left her acting

**Status: fixed in `cdee879`.** Recorded here so the series is complete.

`Orchestrator._guard()` refused model calls when the stop was engaged.
`ToolExecutor` — the component this architecture calls "the single chokepoint"
— had never heard of it. Demonstrated live: with the stop on, `POST /v1/chat`
was refused and `POST /v1/tools/read_file` returned the file. Every file write,
subprocess, application launch and n8n webhook stayed available.

Now checked first thing in `execute()`, ahead of resolving the tool name. Also
made file-backed so it survives a restart, and effective without AURA running.

---

## 2. An approval is granted per tool name, and then spends unboundedly

**Severity: high.** Privilege escalation by argument substitution.

### What is wrong

Consent is **collected** per call, with arguments. `executor.py::_describe`
says so explicitly, and it is right:

> "click_element" is not a decision; "click_element(name='Send')" is. The
> arguments are what make consent informed.

Consent is **enforced** per name, with the arguments discarded:

```python
# core/agent.py:236
confirmed = invocation.name in self._approved
```

`self._approved` is a `frozenset` of tool *names*. So an approval is not
permission to perform the action the human read in a dialog. It is permission
to call that tool, with any arguments, any number of times, for the rest of the
task.

### The part that makes it exploitable rather than merely loose

`desktop/app.py:185` does not resume the approved call. It **re-runs the whole
task from the beginning** with the name now pre-approved:

```python
if approved:
    log.info("desktop.approval_granted", tools=pending)
    self._submit_task(task, approve=already_approved + pending)
```

The model is re-prompted from scratch. Nothing pins the second run's arguments
to the first run's. So:

1. Model asks for `write_file(path='notes.md', content='...')`.
2. Dialog shows that call. The user reads it and approves.
3. Task restarts. `write_file` is now in `_approved`.
4. Model emits `write_file(path='<startup folder>/run.bat', content='...')`.
5. `confirmed=True`. No dialog. No second look.

Step 4 needs no malice from the model — a retry after a failure, or a
differently-sampled second run at temperature 0.7, gets there on its own. With
an injected instruction in a file or a screenshot it is deliberate.

### Fixed

An approval is now a **token** rather than a name: `name:fingerprint`, where the
fingerprint is a SHA-256 of `[name, arguments]` canonicalised with sorted keys.
`approval_token()` and `tool_name()` live in `core/agent.py`.

```python
confirmed = approval_token(invocation.name, invocation.arguments) in self._approved
```

Four properties came out of the shape:

- **A bare tool name now grants nothing.** Any caller still passing
  pre-fix names fails closed rather than silently granting everything — which
  is the right direction for a change to an authorisation path.
- **Key order does not matter.** A model re-emitting the same call with its
  arguments in a different order produces the same token, so the desktop's
  re-run does not prompt twice for one decision.
- **The name and arguments are hashed as one JSON array**, not concatenated
  with a separator — so no separator has to be chosen and none can be forged by
  a tool name containing it.
- **It is still a plain string.** `AgentRequest.approve`,
  `AgentResponse.pending_confirmations`, `TaskResult` and the desktop client
  all keep their existing `list[str]` shape. Only the contents got specific.

`ApprovalDialog` renders through `tool_name()`, so a person still reads
"write_file" and never sees a fingerprint.

Five tests, red first, in `test_agent_loop.py`: a grant for one path does not
authorise another; a grant does hold for the exact call it was given for; key
order is irrelevant; a bare name grants nothing; and `pending_confirmations`
carries the binding rather than just a name.

### Still open

Single use. A token authorises its exact call **any number of times** within the
task. Finding 5 already has the machinery for burning an approval on redemption,
and that is where this should land rather than being reimplemented here.

---

## 3. `run_python` is outside the workspace guard, and can delete the kill switch

**Severity: critical.** A documented invariant is false.

### The claim

`security/estop.py` states, as one of the three properties that make the switch
a switch:

> **AURA cannot reach it.** It lives under `var/secrets/`, which
> `tools/workspace.py` already denies to every file tool, and this module adds
> an explicit pattern for it so the protection survives someone overriding the
> deny list.

### Why it is false

The deny list is enforced by `Workspace.resolve()`, which is called by
**filesystem tools**. `run_python` does not call it, and does not need to.
Its own docstring is honest about the reason:

> A subprocess with a timeout and a working directory is **isolation, not a
> sandbox**: it runs as the user, with the user's permissions, it can reach the
> network, it can read anything the user can read, workspace or not.

Both docstrings are accurate on their own. Together they are a hole.
`Path("var/secrets/emergency_stop").unlink()` inside a snippet is four words and
passes no guard. The same snippet reads `var/secrets/api_token`, `.env`, and
`~/.ssh/`, all of which the deny list exists to protect.

### The chain with finding 2

`run_python` is `PermissionLevel.ACT` (2), which the default policy maps to
CONFIRM — so it takes one human approval. By finding 2, that approval is
granted by name and survives a task restart with new arguments. Therefore:

> Approve `run_python` once, for anything at all, and for the rest of that task
> the model may run arbitrary code as you — including code that deletes the
> emergency stop that `cdee879` was written to make reliable.

The switch is unreachable by every tool that was designed to respect it, and
reachable by the one tool that executes arbitrary code.

### Fixed, partially

`run_python` is now `PermissionLevel.CRITICAL` (4). `ToolsConfig.max_confirmable_level`
defaults to 3, so level 4 is **not reachable by confirmation at all** — the
config comment already said "LEVEL 4 stays out of reach entirely until there is
a UI that can explain what it is approving", and this tool is exactly what that
sentence was protecting against. Reaching it now requires editing
`tools.max_confirmable_level` by hand, which is a decision with a diff.

This does not make the tool safe and is not claimed to. It makes it unreachable
by default, which is the honest standing for a capability the architecture
cannot bound. The remaining half — where the switch lives, and whether
`run_python` earns a lower level — is below and is still open.

### What a full fix must be true of

At minimum the subprocess must not be able to reach `var/secrets/`. Options, in
increasing cost and order of seriousness:

- **Relocate the switch** outside any path the user's own code would naturally
  touch, and outside the project tree. Cheapest, and does not pretend to be a
  sandbox.
- **Pre-flight the snippet** for references to the secrets directory. Weak —
  string obfuscation defeats it — and probably not worth building.
- **Restricted token / job object** on Windows, or a container. The real fix,
  its own project, and explicitly out of scope for this phase per the tool's
  own docstring.
- **Raise `run_python` to `CRITICAL` (4)**, which the default policy DENIES
  outright. One line, and honest: a tool the architecture cannot bound should
  not be reachable by default.

The last option is the one to take this week. The others are decisions for
later, and should be recorded as such rather than left implied.

---

## 6. `act()` raised `AttributeError`, and the loop's stop check never ran

**Severity: critical (availability). Fixed.** Found by `mypy`, not by reading.

### What was wrong

```python
# core/orchestrator.py, since c5975a3 "Phase 4: the agent loop"
outcome = await loop.run(messages, tool_params, should_stop=lambda: self._estop)
```

`Orchestrator` has `_estop_switch`, an `emergency_stop` property and an `estop`
property. It has never had `_estop`. The lambda is evaluated on the loop's first
iteration, before any model call, so **every tool-using turn raised
`AttributeError` at step 0** — `POST /v1/agent`, the desktop overlay's task
path, and the eval runner's orchestrator alike.

Demonstrated by constructing an `Orchestrator` with a tool runner and calling
`act()`:

```
[error] turn.failed  error="'Orchestrator' object has no attribute '_estop'"
                     error_type=AttributeError
```

### Why 1,181 tests did not catch it

`tests/conftest.py` builds its orchestrator without tools, and `act()` opens
with:

```python
if self._tools is None or not self._settings.agent.enabled:
    return await self.chat(...)
```

The fallback is correct and desirable. It also meant no test ever entered the
branch below it. Every other `act()` in the suite is a stub
(`test_eval_suite.py::FakeOrchestrator`). The one path that used the real thing
was never exercised without a live model server.

### The security reading

The check that was broken *is* the agent loop's per-iteration emergency stop. So
for five phases, the stop was not being consulted between agent steps at all.

It failed **closed**, which is the whole mercy here: the turn crashed rather
than continuing with an unchecked stop. A `should_stop` that silently returned
`False` would have been the same bug with the opposite blast radius. That is
worth stating plainly, because it is an accident rather than a design.

### Fixed

`should_stop=lambda: self.emergency_stop`. Two regression tests in
`test_orchestrator.py`: one asserting `act()` reaches the model at all with
tools wired, one asserting the loop raises `EmergencyStopError` when the stop is
engaged. Both were red first.

### What this changes about the roadmap

Phase 4's exit criterion — "a 5-step task completes end to end, and a
deliberately broken step is detected and recovered from" — **cannot have been
met through `Orchestrator.act()`**. Whatever verified it did not go through this
path. That should be re-run and the roadmap entry corrected or re-affirmed; it
is not something to leave implied.

---

## 4. The stop does not interrupt work already in flight

**Severity: medium.**

The stop is checked at `ToolExecutor.execute()` entry and between agent steps
(`orchestrator.py:422`, `should_stop=lambda: self._estop`). Both are checks on
*starting* something. Nothing checks during.

`run_python` accepts `timeout_s` up to 120, with a 130 s outer backstop. A
snippet that starts one second before the stop is engaged runs for its full
budget. The pool worker cannot be killed — the executor's own comment says so:
"The worker thread may still be running: Python cannot kill one."

This does not contradict `estop.py`'s promises as flatly as finding 3 does; the
switch does stop new work, from a file manager, without AURA's cooperation. But
"engage the stop and AURA stops" is how a person will read it, and for up to two
minutes it is not true of the tool most able to do damage.

### Fixed, for the tools that can be

`ToolContext` gained `should_stop`, which `ToolExecutor` binds to its own switch
before handing the context to a tool — so a tool can ask the question while it
waits rather than only before it starts.

`run_python` now uses `Popen` with a poll loop instead of
`subprocess.run(timeout=)`, checking the stop every 200 ms and terminating the
child when it engages (escalating to `kill` after two seconds). The inner
timeout moved into the same loop, since `subprocess.run`'s own no longer
applies. Four tests: the child dies within seconds rather than running its full
30, the ordinary path is unaffected, a missing `should_stop` still works, and
the timeout still fires.

### Still open

**Threads remain uninterruptible**, and that is a property of Python rather than
of this design — the executor's own comment has always said so. The tools that
can run long enough to matter are the ones that spawn a subprocess, and those
are now covered.

`open_application` was left alone: it starts a process and returns rather than
waiting on one, so there is nothing in flight to interrupt.

What is still missing is **declaration**. A tool that cannot be interrupted
should say so in its `ToolSpec`, so the gap is visible in the registry rather
than discovered by engaging the stop and watching nothing happen. That is a
small change and it is not made here.

---

## 5. There are two approval systems, and the strong one is not on the tool path

**Severity: medium.** Architectural, and it is most of the fix for finding 2.

`security/approvals.py` is the better of the two and is genuinely good:
single-use, 256-bit nonce, `secrets.compare_digest` on both comparisons, burned
on grant and on denial but not on a bad nonce, bound to a `request_id`, with
four distinct rejection outcomes so a replay and a timeout do not look alike in
the audit log. It carries a `summary` built from the actual arguments.

`ToolExecutor` does not consult it. It accepts a bare boolean:

```python
if decision is Decision.CONFIRM and not call.confirmed:
```

`call.confirmed` arrives either from `POST /v1/tools/{name}` (`app.py:722`,
straight from the request body) or from the name-set in finding 2. Neither is
nonce-bound, neither is single-use, and neither is tied to the arguments the
human was shown.

So the property the approval registry exists to provide — that a grant refers to
the one action it was issued for — is not enforced on the path tools actually
take. The registry currently issues approvals that nothing redeems as a
precondition for execution.

### What a fix must be true of

`ToolExecutor` should take an approval token, not a boolean, and redeem it
against the registry with the request id. `confirmed: bool` then becomes
unrepresentable rather than merely discouraged — the same move
`evolution/space.py` makes with its allow-list, and for the same reason.

### Deferred, 2026-09-17

**Not because it is wrong. Because finding 2 took its urgency away.**

The escalation this would have closed is closed: an approval now binds to the
arguments it was granted for, so it cannot be re-spent on a restarted task with
a substituted path. What remains is that a token is reusable *within* its task
for the identical call, and that the two systems have not been merged.

The refactor touches `ToolExecutor.execute`, `ToolCall`, `AgentRequest`,
`ExecutorToolRunner`, the agent loop and the desktop client — five to six hours
across an authorisation path, which is the worst place to be halfway through
when a session ends. The findings that were live exploits are fixed; this is
the one that is structure.

**What revives it**, whichever comes first:

- The issue-approve-**resume** loop from 10C. Today an approval re-runs the
  whole task, which is why finding 2 existed at all. Resuming the specific call
  needs the registry's `request_id` binding, so that work and this fix are the
  same work and should not be done twice.
- A second approver. The nonce path exists because the presence overlay is not
  the only thing that might grant; the moment something other than the desktop
  dialog can approve, a bare boolean is no longer defensible.
- Any tool above LEVEL 2 becoming reachable by default.

Until one of those, `approvals.py` stays built, tested and unused on the tool
path — which is recorded here rather than left to be rediscovered.

---

## Order of work

1. ~~**Finding 3, the one-line half.**~~ **Done.** `run_python` is LEVEL 4 and
   denied by default.
2. ~~**Finding 6.**~~ **Done.** The agent loop reaches the model, and its stop
   check runs.
3. ~~**Finding 2.**~~ **Done.** Approvals bind to arguments.
4. **Finding 5.** Token instead of boolean. This is the structural change and
   it should land before finding 2's tests, because it is what makes them pass.
5. **Finding 4.** Terminate child processes on stop; declare which tools cannot
   be interrupted.
6. **Finding 3, the rest.** Decide and record where the switch lives and what
   `run_python` is allowed to be.
7. **Re-verify Phase 4's exit criterion**, which finding 6 invalidates.

Each gets a test that was red first. The suite is at 1,181 passing; a security
fix that does not add a test that would have caught the flaw is not finished.

---

## What this audit did not find

Recorded because a list of problems with no negative space reads as a list of
everything that was looked at.

- `workspace.py` resolves before checking, handles the separator boundary
  (`/home/user-secrets` against `/home/user`), normalises case for Windows, and
  compensates for `fnmatch` not spanning separators. The classic traversal holes
  are closed.
- `audit.py` fails closed when `require_audit` is set, and the executor records
  denials, timeouts and crashes, not only successes.
- `auth.py` requires a bearer token on every route with no way to disable it,
  and the reasoning about loopback not being authentication is correct.
- `privacy.py` gates capture rather than instructing the model, and the vision
  tools let the model narrow its own sight but never widen it.
- The VLM seam from 10C.1 has **no call sites outside `vision/vlm/`**. Nothing
  can invoke it yet, so it opens no hole. The gate conditions in
  `15-phase10c-vision.md` are still the right ones for when it does.
