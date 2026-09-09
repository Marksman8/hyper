# The presence state machine — 10A.4

Status: **implemented** · Date: 2026-09-09

---

## 1. What it is

A **state-resolution system**, not "latest event wins". It holds a set of
*conditions* — facts currently true — and resolves the visible state by taking
the highest-priority one.

Almost every required behaviour falls out of that one mechanism:

| Requirement | Falls out of |
|---|---|
| Priority | Resolution picks the highest-priority active condition |
| Terminal vs current | Occurrences set a condition **with a deadline**; state events set one that persists |
| Timeout / expiry | An expired condition stops being active, so resolution moves on |
| Hysteresis | Re-asserting a held condition is a no-op, so no transition is emitted |
| Reconnect | Only what is currently true is reported; occurrences are never replayed |

Two cases genuinely need more:

- **Interruption** *clears* the activity conditions rather than outranking
  them. Solving it by priority would mean LISTENING beat THINKING forever.
- **Dwell** defers a transition inside a short window — but is preemptable by
  priority, and does not apply once the current state's condition has gone.

## 2. State diagram

```
   ┌───────────────────────────────────────────────────────────┐
   │ PRIVATE   private_mode=private   suppresses everything     │
   ├───────────────────────────────────────────────────────────┤
   │ SLEEP     private_mode=sleep, or 10 min idle               │
   ├───────────────────────────────────────────────────────────┤
   │ ERROR     error(*)                              expires 8s │
   ├───────────────────────────────────────────────────────────┤
   │ WAITING   permission_required / agent_waiting(confirm)     │
   │           persists until approval or task end              │
   ├───────────────────────────────────────────────────────────┤
   │ WARNING   error(permission_denied)              expires 4s │
   │ CONFUSED  task_failed                           expires 3s │
   │ ANNOYED   task_completed(stop_reason=budget)    expires 4s │
   │ SUCCESS   task_completed                        expires 2.5s│
   │ HAPPY     task_completed(tool_calls>=2)         expires 3s │
   ├───────────────────────────────────────────────────────────┤
   │ WORKING   task_started(with_tools) / agent_waiting(tool)   │
   │ THINKING  task_started / agent_thinking                    │
   │ LISTENING user_spoke / user_interrupted         expires 6s │
   ├───────────────────────────────────────────────────────────┤
   │ IDLE      nothing else is true                             │
   └───────────────────────────────────────────────────────────┘

   The named sequences:

     task_started        task_completed         2.5s
     IDLE ──▶ THINKING ──────────────▶ SUCCESS ─────▶ IDLE

              user_interrupted  (CLEARS activity)
     THINKING ────────────────────────▶ LISTENING

              permission_required        approval / task end
     WORKING ───────────────────▶ WAITING ──────────────▶ WORKING

     any state ──── error ────▶ ERROR ──── private_mode ────▶ PRIVATE
```

## 3. Priority table

Higher wins. A **total order** — no two states share a rank, so simultaneous
conditions resolve deterministically regardless of arrival order.

| Rank | State | Why here |
|---:|---|---|
| 100 | `PRIVATE` | An explicit instruction from the user |
| 90 | `SLEEP` | Also an instruction, but less absolute |
| 80 | `ERROR` | Safety over everything cosmetic |
| 70 | `WAITING` | A human is blocked and needs to act |
| 60 | `WARNING` | Something was refused |
| 55 | `CONFUSED` | Recovering from a failure |
| 50 | `ANNOYED` | Gave up |
| 45 | `SUCCESS` | Feedback |
| 40 | `HAPPY` | Cosmetic |
| 30 | `WORKING` | Activity |
| 20 | `THINKING` | Activity |
| 10 | `LISTENING` | Activity |
| 0 | `IDLE` | Default |

**Two decisions worth arguing with rather than inheriting.**

**PRIVATE outranks ERROR.** Privacy is an instruction; an error is
information. An overlay that reappears to report a problem after the user asked
for privacy has broken the promise — and the error is not lost: it is still
logged, audited and on the bus. The overlay is a convenience; privacy is a
commitment.

**ERROR outranks every cosmetic and personality state.** SUCCESS, HAPPY and
ANNOYED must never mask a failure.

## 4. Transition rules

| Event | Effect |
|---|---|
| `USER_SPOKE` | clear SUCCESS/HAPPY/ERROR · flash LISTENING |
| `USER_INTERRUPTED` | **clear activity** · clear SUCCESS/HAPPY · flash LISTENING |
| `TASK_STARTED` | clear SUCCESS/HAPPY/ANNOYED/LISTENING · hold THINKING (+ WORKING if `with_tools`) |
| `AGENT_THINKING` | hold THINKING — a no-op if already held, which is the hysteresis |
| `AGENT_WAITING(confirmation)` | hold WAITING |
| `AGENT_WAITING(tool)` | hold WORKING |
| `PERMISSION_REQUIRED` | hold WAITING |
| `TASK_COMPLETED` | **clear activity** · flash SUCCESS, or HAPPY (≥2 tool calls), or ANNOYED (gave up) |
| `TASK_FAILED` | **clear activity** · flash CONFUSED |
| `ERROR(permission_denied)` | flash WARNING |
| `ERROR(anything else)` | flash ERROR |
| `PRIVATE_MODE` | hold or clear PRIVATE / SLEEP |
| any `OBSERVATION` | **ignored entirely** — screen content must not drive what AURA looks like she is doing |
| `PRESENCE_STATE` | ignored — it is this machine's own output, and consuming it would be a feedback loop |

Any event except `SNAPSHOT` also refreshes the activity timer and clears SLEEP.

## 5. Dwell and timeout values

| Constant | Value | Rationale |
|---|---:|---|
| `SUCCESS` | 2.5 s | The only value with a real perceptual constraint: below roughly 1.5 s a state change reads as a flicker rather than as feedback |
| `HAPPY` | 3.0 s | Slightly longer; it marks work that actually took several steps |
| `CONFUSED` | 3.0 s | Long enough to notice, short enough not to nag |
| `WARNING` | 4.0 s | A refusal deserves more attention than a success |
| `ANNOYED` | 4.0 s | As above |
| `ERROR` | 8.0 s | The one a person most needs to see |
| `LISTENING` | 6.0 s | Covers a pause mid-sentence without hanging on indefinitely |
| **`MIN_DWELL_S`** | **0.4 s** | Below this, a fast agent loop strobes between THINKING and WORKING several times a second |
| `IDLE_TO_SLEEP_S` | 600 s | Ten minutes of silence is a break, not a pause |

Dwell has **two escapes**: a higher-priority state preempts it, and it does not
apply once the current state's condition has been cleared. The second was found
by a test — waking from SLEEP, the sleep condition was gone but dwell kept
showing it, so AURA stayed asleep through being spoken to. Holding a state that
is no longer true is not damping, it is lying.

## 6. Architecture

```
Event Bus → PresenceStateMachine → PRESENCE_STATE → presentation layer
```

`PresenceDriver` subscribes, feeds the machine, ticks it for expiry, and
publishes on change. **The machine imports no UI code** — asserted by test, and
so is the whole of `aura/core`. The presentation layer is just another
subscriber, so sprites → Live2D → a terminal readout changes nothing here.

The machine is passive: fed events and ticked, with an **injected clock**, so
dwell and expiry are tested deterministically rather than with sleeps.

`PresenceDriver.start()` subscribes *before* launching its task — `create_task`
returns before the coroutine runs, and a subscription made inside it would miss
anything published in that window. A presence layer that drops the first event
of a session is the kind of bug that reads as flakiness.

## 7. Measured

| Metric | Value |
|---|---|
| Transition latency, median / p95 | **19.8 µs / 22.0 µs** |
| No-op event (condition already held) | 0.66 µs — 1.5 M/s |
| Burst throughput | **1.17 M events/s** |
| Transitions during a 200k alternating burst | **2,002**, not 200,000 |
| RSS added | **0.0 MB** — conditions are bounded by the 13 states |
| Driver tick | 0.25 s (expiry granularity, not animation rate) |

The burst figure is the interesting one: 200,000 alternating events produced
2,002 transitions, because dwell and hysteresis absorb the rest. That is the
anti-thrashing property measured rather than asserted.

## 8. Known limitations

1. **`WAITING` is cleared by task end, not by approval.** Nothing redeems an
   approval mid-task yet — the executor still refuses immediately (10A.3). The
   approve-and-resume flow is 10C, and `WAITING` already has the right shape
   for it.
2. **`stop_reason` is optional**, so ANNOYED only fires when the orchestrator
   supplies it. A turn that gives up without it shows SUCCESS.
3. **No per-state configuration.** Durations are module constants; only
   `personality` (HAPPY/ANNOYED off) is switchable.
4. **Sleep is event-based only.** Ten minutes without *events* is not the same
   as ten minutes away from the machine — no lock-screen or session-idle
   awareness.
5. **One machine per process.** Multi-session presence would need keying.
6. **`WARNING` is inferred from an error code string** (`permission_denied`)
   rather than from a typed field.
