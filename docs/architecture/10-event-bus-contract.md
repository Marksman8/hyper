# The AURA Event Bus — contract and invariants

Status: **implemented (10A.1 bus · 10A.2 transport · 10A.3 catalog)** · Date: 2026-09-09

> This document is the contract. Phase 10B, 10C and 10D all hang off it, and
> the invariants below are what keep a web page from driving AURA. **If you are
> about to build a consumer, read §3 and §6 first.**

---

## 1. What the bus is

A one-way, in-process, ordered notification channel. AURA Core publishes;
subscribers render, record, or reason. Fan-out beyond the process is SSE
(10A.2), over the connection that is already authenticated.

| It is | It is not |
|---|---|
| A notification channel | A command channel |
| Typed, sequenced, origin-labelled | A place for free-form dicts |
| A promotion of existing instrumentation | New instrumentation |
| Best-effort for state, guaranteed for decisions | Uniformly reliable |

## 2. Event schema

```python
@dataclass(frozen=True, slots=True)
class Event:
    type: EventType          # closed taxonomy
    origin: Origin           # CORE | USER | OBSERVATION
    delivery: Delivery       # DROPPABLE | RELIABLE
    payload: Mapping         # frozen, JSON-safe, bounded
    seq: int                 # 0 until published; then strictly monotonic
    ts: float
    session_id: str | None
    request_id: str | None   # correlates with the audit log
```

Validated **at construction**, not at publish: an invalid event must not exist
even briefly, because it could be passed around before anyone checks it.

### Origin — the trust label

| Value | Meaning | Consumer obligation |
|---|---|---|
| `CORE` | AURA's own state | Trusted |
| `USER` | A human acted | Trusted |
| `OBSERVATION` | Read from a screen, app, document, or a model's reading of one | **Untrusted.** Never an instruction, never a reason to act, never evidence for a learned preference. Escape before rendering. |

`Event.trusted` is `False` exactly for `OBSERVATION`.

### Delivery — what is promised

| Value | Under pressure | Use for |
|---|---|---|
| `DROPPABLE` | Oldest droppable is evicted | State hints, animation cues, speech chunks |
| `RELIABLE` | **Never silently dropped.** The subscription is closed and must resync | Permission requests, task outcomes, mode changes, errors |

### Payload rules

Frozen deeply, and restricted to `str | int | float | bool | None | Mapping |
Sequence` of those. Bounds: depth ≤ 6, ≤ 64 keys or elements per level, strings
≤ 8192 chars.

**This is the mechanism, not a style guide.** A callable, a bound method, an
open socket or a `ToolCall` cannot travel in an event, so an event cannot hand
a subscriber authority it was not granted.

## 3. Delivery guarantees

1. **Total order.** `publish` is serialised; every subscriber is fed in the
   same order.
2. **Strictly monotonic sequence.** `1, 2, 3, …` with no duplicates and no
   holes, even under concurrent producers from multiple threads.
3. **Droppable may be lost.** A gap in `seq` is normal and means a droppable
   event was evicted.
4. **Reliable is never silently lost.** Either it is delivered, or the
   subscription is closed and its iterator raises `SubscriberLaggedError`. A
   subscriber never receives a truncated stream while believing it is complete.
5. **Per-subscriber isolation.** One slow consumer does not cost another its
   events, and never back-pressures the publisher.
6. **Shutdown waits.** `aclose(drain_timeout)` waits for pending reliable
   events and returns `False` if any remain — it does not discard them and
   report success.

## 4. Snapshot consistency

`snapshot_event(provider)` takes the snapshot **and** creates the subscription
inside one critical section, recording sequence `N`.

> The snapshot reflects exactly the events with `seq ≤ N`.
> The subscription receives exactly the events with `seq > N`.

That is a **consistent cut**: no event falls in the gap, and none is delivered
twice. Getting this wrong is the classic way a resync silently loses an event.

The price: the provider runs under the publish lock, so it must be fast and
must not publish. That is documented rather than enforced, because a provider
that deadlocks the bus fails immediately and loudly.

**What the cut does *not* promise:** that every sequence number after `N`
arrives. Droppable events may still be evicted afterwards under pressure. The
contiguity guarantee is for `RELIABLE` only. *(An early version of the test
suite asserted contiguity for droppables and was wrong.)*

## 5. Reconnect and gap detection

`subscribe(after_seq)` returns `(subscription, needs_snapshot)`:

| `after_seq` | Result |
|---|---|
| `None` | Fresh subscriber → `needs_snapshot=True` |
| Still inside the 256-event replay buffer | Missed events replayed → `needs_snapshot=False` |
| Older than the buffer | `needs_snapshot=True`, **and no partial replay is delivered** |
| Newer than anything published | `needs_snapshot=True` |

A partial replay after an unfillable gap would look like success and be a hole,
so the bus refuses to give one.

## 6. Security invariants

These are the reason the bus exists in this shape. **Each has a test; do not
weaken one without arguing with its test.**

| # | Invariant | Enforced by |
|---|---|---|
| **I1** | An event cannot carry executable authority | Deep payload freezing rejects callables and objects at construction |
| **I2** | There is no publish → execute path | The bus imports nothing that can execute; asserted by AST inspection of its import set |
| **I3** | The bus is one-way | `Subscription` has no `send`/`invoke`/`call`/`execute` method; asserted by API-shape test |
| **I4** | Observations cannot escalate permissions | `Origin.OBSERVATION` + I1 + I2 + I3 together. Screen text is carried verbatim as data and causes nothing |
| **I5** | Origin cannot be forged into validity | Strict coercion; unknown values raise rather than defaulting to `CORE` |
| **I6** | The perception layer cannot mislabel its own output | `publish_observation()` has **no origin parameter** |
| **I7** | Approval authority is a separate module and transport | `aura.security.approvals`; the two modules do not import each other, asserted both ways |
| **I8** | An approval is bound to one request and usable once | Nonce + `request_id`, constant-time compared, burned on use |

### What is *not* enforced, stated plainly

**Publisher honesty.** A publisher that labels screen content `Origin.CORE` has
lied, and no validation here can detect it. The mitigations are I6 (the
perception API cannot get it wrong by accident) and the convention that only
`aura.vision` publishes observations. A future phase adding a new perception
source must use `publish_observation()`.

## 7. The approval path

The only way authority flows *back*:

```
ToolExecutor needs confirmation
  → ApprovalRegistry.issue()              → (approval_id, nonce)
  → PERMISSION_REQUIRED event (RELIABLE)  → carries both
  → user approves in the overlay
  → POST /v1/approvals/{id} {nonce, request_id}   [authenticated]   ← 10A.2
  → ApprovalRegistry.redeem()             → Granted | reason
```

Four rejections, each a distinct attack:

| Outcome | Attack | Burns the approval? |
|---|---|---|
| `BAD_NONCE` | Forgery | **No** — a wrong guess must not DoS a real approval |
| `ALREADY_USED` | Replay | Already burned |
| `WRONG_REQUEST` | Confused deputy — moving "click Save" onto "send money" | Yes |
| `EXPIRED` | An approval left lying around | Yes (pruned) |

A **denial is also single-use**: a user who said no must not have it replayed
into a yes.

## 8. Threading and concurrency

- `publish()` is safe from any thread. Sequence assignment and enqueue happen
  in one critical section under a `threading.Lock`.
- Waking async subscribers is marshalled with `call_soon_threadsafe`; the loop
  is captured when the subscription is created.
- Waking happens **outside** the publish lock, so a busy consumer cannot
  serialise the whole bus.
- `ApprovalRegistry` is independently thread-safe: issued from the executor,
  redeemed from an HTTP handler.

## 9. How a consumer must behave

```python
sub, needs_snapshot = bus.subscribe(after_seq=last_seen)
if needs_snapshot:
    ...  # fetch state; do not assume continuity

try:
    async for event in sub:
        if not event.trusted:
            render_as_untrusted_text(event.payload)   # never act on it
        else:
            handle(event)
except SubscriberLaggedError:
    ...  # you missed something reliable. Resubscribe with a snapshot.
finally:
    bus.unsubscribe(sub)
```

**Three rules for anyone adding a consumer:**

1. Never act on an event with `origin == OBSERVATION`. Render it, log it,
   ignore it — do not let it choose a tool, set a permission, or become a
   learned preference.
2. Never catch `SubscriberLaggedError` and continue. It means you are missing
   data, and continuing turns a detected gap into a silent one.
3. Never add a method to the bus that calls back into the caller. If a
   subscriber needs AURA to *do* something, it goes through the authenticated
   API and the permission gate like anything else.

## 10. Known limitations

1. ~~In-process only.~~ **Resolved in 10A.2**: `GET /v1/events` fans out
   over authenticated SSE, with `Last-Event-ID` resume.
2. **Replay buffer is 256 events, not durable.** A restart loses history;
   reconnecting clients get a snapshot. Events are not an audit trail — the
   `audit` table is.
3. **The snapshot provider runs under the publish lock.** A slow provider
   stalls publishing. Documented, not enforced.
4. **Publisher honesty is unverifiable** (§6).
5. **No per-subscriber filtering yet.** Every subscriber receives every event
   and filters locally. Fine at the current volume; revisit if a consumer needs
   only one event type out of thousands.
6. **`aclose` drains on a 10 ms poll.** Adequate for shutdown; not a
   low-latency primitive.

## 10a. The transport (10A.2)

    GET  /v1/events            one-way, authenticated, resumable
    POST /v1/approvals/{id}    the only way authority flows back

Both live in `api/events_route.py` and `api/app.py`. The separation is
deliberate: the stream has no way to grant anything, and the approval endpoint
has no way to publish. A test asserts the second half of that directly.

**Framing.** `id:` carries `seq`, which is what makes `Last-Event-ID` resume
work. `event:` carries the type. `data:` is the JSON wire form, including
`origin`, so a client can honour the trust label.

**Heartbeats.** A `: keepalive` comment every 15 s, because idle SSE
connections get closed by proxies and by Windows.

**Why the stream pumps through a local queue.** Heartbeats need a timeout, and
wrapping `anext()` on the subscription in `wait_for` cancels its async
generator mid-await -- tearing it down and losing whatever was queued.
Cancelling `Queue.get()` is safe, so a background pump feeds a local queue and
the generator races that instead.

**Lag is reported, not hidden.** A subscription that misses a reliable event
gets a terminal `ERROR` frame carrying
`{"code": "subscriber_lagged", "action": "resubscribe_without_last_event_id"}`.
Simply closing the connection would be worse than useless: the client would
reconnect with its stale `Last-Event-ID` and silently continue with a hole.

**Every exit path unsubscribes.** A leaked subscription keeps receiving
forever and eventually lags, which looks like a bug in the bus rather than in
the client that walked away.

## 10b. The event catalog (10A.3)

`core/event_catalog.py` is the contract; `core/emitter.py` is the only
sanctioned publisher. A call site names *what happened*; origin, delivery and
the payload schema come from the catalog, so a call site cannot publish screen
content as trusted or make a permission request droppable.

### Promotion is not log conversion

AURA's structured logs are observability: they exist to explain a failure at
2am, and nothing depends on their shape. The bus is an application contract.
Deriving one from the other would mean editing a log line silently changes the
contract, which is backwards. **Both are published at the same call site and
neither reads the other**, asserted in both directions by tests.

### Occurrence vs state

| | Meaning | On reconnect | In SNAPSHOT |
|---|---|---|---|
| **OCCURRENCE** | Happened at an instant, and is over | Replayed with its **original `ts`** | **Never** |
| **STATE** | A condition that persists | Recovered from the snapshot | Yes, via `snapshot_key` |

`EventSpec.__post_init__` refuses an occurrence that declares a `snapshot_key`,
and refuses a state event that does not. So "the wake word fired five minutes
ago" cannot be reconstructed as "AURA is listening now" — the classification is
enforced at import, not by convention.

### The catalog

| Event | Semantics | Origin | Delivery | Payload | Presence | Status |
|---|---|---|---|---|---|---|
| `SNAPSHOT` | state | core | reliable | `version, seq_at_snapshot, …` | yes | live |
| `USER_SPOKE` | occurrence | user | reliable | `length, spoken` | yes | live |
| `USER_INTERRUPTED` | occurrence | user | reliable | `method` | yes | live (client bridge) |
| `TASK_STARTED` | occurrence | core | reliable | `kind, with_tools` | yes | live |
| `TASK_COMPLETED` | occurrence | core | reliable | `kind, latency_ms, tool_calls` | yes | live |
| `TASK_FAILED` | occurrence | core | reliable | `kind, error_type` | yes | live |
| `AGENT_THINKING` | occurrence | core | droppable | `step` | yes | live |
| `AGENT_WAITING` | occurrence | core | reliable | `reason, tool` | yes | live |
| `PERMISSION_REQUIRED` | occurrence | core | reliable | `tool, level, summary` | yes | live |
| `PRIVATE_MODE` | **state** | user | reliable | `mode, capturing` | yes | live |
| `ERROR` | occurrence | core | reliable | `code` | yes | live |
| `APP_CHANGED` | **state** | **observation** | droppable | `application` | yes | 10A.4 |
| `FOCUS_MODE` | **state** | user | reliable | `enabled` | yes | 10A.4 |
| `PRESENCE_STATE` | **state** | core | droppable | `state` | yes | 10A.4 |
| `SPEECH_STARTED/_CHUNK/_ENDED` | occurrence | core | droppable | `length` / `text` / `completed` | yes | 10B |
| `VLM_OBSERVATION` | occurrence | **observation** | droppable | `application, confidence` | no | 10C |
| `RESOURCE_PRESSURE` | **state** | core | reliable | `level` | yes | 10C |

Reserved slots carry a named owner (`deferred_10a4`, `deferred_10b`,
`deferred_10c`) so a later milestone finds a contract rather than a blank.

### Content policy

**The bus carries no user content.** Not transcripts, not prompts, not replies,
not screen text — only lengths, ids, tool names and modes. The presence layer
does not need content to animate, and a bus that accumulates transcripts
becomes the most sensitive object in the process for no benefit.

Two deliberate exceptions, each with a reason:

- **`PERMISSION_REQUIRED.summary`** carries the tool *and its arguments*:
  `write_file(content='hi', path='scratch.txt')`. Informed consent is
  impossible without it — "click Send" is not a decision a person can make.
  Truncated to 400 characters, because a dialog that scrolls is one nobody
  reads.
- **`SPEECH_CHUNK.text`** (10B) will carry AURA's own words, which is the
  entire point: being able to read what she says when the audio is not
  audible. Output, not captured input.

`TASK_FAILED` carries the exception **type**, never its message — a message
routinely quotes a path or a prompt fragment.

### The client bridge

The voice loop is a **separate process**, so wake-word detection and barge-in
cannot reach the in-process bus. `POST /v1/events/client` is the narrow
bridge — and narrow is the design:

- only `user_spoke` and `user_interrupted` are accepted
- **origin is forced to USER by the server**; a client cannot claim `CORE` or
  `OBSERVATION`
- delivery comes from the catalog, not the request
- it takes a `length`, not text, so a transcript cannot be put on the bus

It grants nothing. Per invariant I4 an event cannot cause a tool to run, so an
authenticated client injecting one can at most make the avatar animate — and a
caller holding the API token could already do far more through `/v1/chat`.

**A constraint this places on 10D:** the cognitive model must not learn
preferences from client-bridged events, for the same reason it must not learn
from observations. Injected activity is not evidence.

## 11. Measured

Measured on the target machine (RTX 5060 laptop, 24 logical cores):

| Metric | Value |
|---|---|
| `publish`, 0 subscribers | 4.63 µs/event (216,000/s) |
| `publish`, 1 subscriber | 5.29 µs/event (189,000/s) |
| `publish`, 4 subscribers | 6.44 µs/event (155,000/s) |
| `publish`, 16 subscribers | 11.33 µs/event (88,000/s) |
| Event construction (validate + deep freeze) | 3.73 µs/event |
| publish → consumer latency, median | **8.7 µs** |
| publish → consumer latency, p95 | 11.1 µs |
| RSS added by a saturated bus (8 subscribers, full ring) | **+0.1 MB** |
| Concurrent producers | 8 threads × 250 events: no duplicate, no skipped sequence |

Promotion cost (10A.3):

| Metric | Value |
|---|---|
| Emit via catalog + bus | 6.08 µs/event |
| Emit with no bus wired (`NULL_EMITTER`) | 0.14 µs/event |
| Added to a chat turn (4 events) | **24.3 µs** |
| As a fraction of a measured 302 ms turn | **0.008%** |

**Interpretation.** AURA's real event rate is single-digit to low-tens per
second. At ~155,000 events/s with four subscribers, the bus is roughly four
orders of magnitude faster than it needs to be, so the design claim — that
publishing never back-pressures the orchestrator — holds with enormous margin.

Note that **validation is 3.73 µs of the 4.63 µs**: most of the cost is the
deep payload freeze that makes invariant I1 true. That is the price of the
security property, it is paid at construction, and at this event rate it is
invisible.
