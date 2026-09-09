# ADR-0004: AURA owns permission authority for automation providers

Status: **accepted (contract only — not implemented)** · 2026-09-09 ·
Supersedes nothing

> **The permanent decision, in one line:**
> **n8n is a provider of automation capabilities. AURA's local, version-controlled
> manifest is the sole authority for what AURA is permitted to do with them.**

This ADR settles the security contract *before* any code exists, so that the
implementation cannot accidentally introduce an authority-escalation path. No
n8n integration is built. Nothing in `src/` changed.

---

## Context

n8n is a self-hostable workflow automation tool with a large library of
third-party integrations (mail, chat, calendars, storage, issue trackers). It
would let AURA reach those services without reimplementing several hundred API
clients.

The attraction is real, and so is the hazard. n8n is edited in a browser, holds
its own OAuth credentials, and describes its own workflows. If AURA were to
take n8n's word for what a workflow *is*, n8n would be deciding what AURA is
allowed to do — and AURA's entire security model is built on the opposite
arrangement.

**Repository state at the time of writing.** Phases 1–9 are complete. Phase 10A
(event bus, SSE transport, event catalog, presence state machine, desktop
overlay) is complete. **10B, 10C and 10D are not implemented** — there is no
`perception/` package and no `tools/computer/`. This ADR is therefore written
against the architecture that exists today, not against a planned one.

## The escalation problem, stated concretely

A workflow is created and named `draft_email`. It composes a draft and saves it.
It is reasonably declared LEVEL 2.

Later — in a browser, with no involvement from AURA — it is renamed
`send_email`, or its final node is changed from "save draft" to "send". The
webhook URL is unchanged. The workflow id is unchanged.

If AURA derived permission from anything n8n reports, she would keep executing
it at LEVEL 2, without confirmation, and would now be sending mail on the user's
behalf under a permission granted for drafting.

> **n8n describes what a workflow is. AURA's local manifest determines what
> AURA is allowed to do with it.**

Nothing about a workflow's identity in n8n — name, tags, description, folder,
notes, or any other metadata — may influence its permission level.

---

# DECIDED

These are binding. An implementation that violates one of them is wrong, not
merely different.

## D1. One workflow = one tool

Each declared workflow becomes exactly one `ToolSpec` in the existing
`ToolRegistry`. It is not a category, a namespace, or a generic
`call_workflow(name)` escape hatch. A generic caller would make the workflow
name a *runtime argument*, and the model chooses runtime arguments — which
would hand permission selection to the model.

## D2. The local manifest is the only authority

Permission levels come from a version-controlled file in the repository, and
from nowhere else. AURA must not:

- discover workflows from n8n's API and register them,
- read n8n metadata to infer a level,
- trust a workflow's display name, description, or tags,
- register a capability because n8n advertises one.

**An undeclared workflow is not callable.** Changing a workflow's authority
requires editing a file under version control and is therefore reviewable in a
diff.

## D3. No separate execution path

```
AURA Agent
    ↓
ToolRegistry
    ↓
ToolSpec
    ↓
ToolExecutor
    ├── validate arguments (JSON Schema)
    ├── permission check (PermissionPolicy)
    ├── confirmation (ApprovalRegistry, nonce-bound)
    ├── execute (timeout-bounded)
    └── audit
          ↓
    N8NWorkflowTool
          ↓
    n8n webhook
```

`ToolExecutor` remains the single chokepoint, exactly as for `read_file` and
`open_application`. Automation gets no side door on the grounds of latency,
convenience, or "it's just a webhook".

## D4. Audit names the capability, not the provider

```
tool.send_slack_message        ✅
n8n workflow executed          ❌
```

The provider is an implementation detail. An audit trail that says "n8n did
something" is unreadable six months later, and it would make a native tool and
an n8n-backed tool look different in the log when they should look identical.

## D5. Destructive and third-party actions are LEVEL 4, always confirm

Anything that **sends, posts, publishes, pays, deletes, or performs a
destructive external action** is LEVEL 4 with confirmation that cannot be
pre-approved for the task. The confirmation must show the payload — "run
send_slack" is not a decision a person can make; "post 'deploying now' to
#eng" is.

This matches the rule already set for messaging in the Phase 10 design, and the
`PERMISSION_REQUIRED.summary` content exception already recorded in
`event_catalog.py`.

**A workflow may not declare itself LEVEL 1**, or otherwise reduce its own
required permission, because a workflow declares nothing (D2).

## D6. Webhook responses are untrusted input

A workflow that reads Gmail returns *content from the internet*. That reaches
AURA with the same standing as screen text under `Origin.OBSERVATION`:

- it is data, never an instruction;
- it may not escalate a permission or cause a tool to run;
- the cognitive model (10D) must not learn preferences from it.

This is the existing non-escalation invariant (ADR-adjacent, recorded in
`docs/architecture/10-event-bus-contract.md` §6, invariant I4), applied to a new
input channel rather than a new rule.

## D7. Credentials never enter AURA

n8n holds the OAuth tokens. AURA holds a webhook URL and a permission level.
Credentials must not appear in a `ToolResult`, in an event payload, in the
audit table, or in a log line. `logging.redact_keys` already exists for the
last of these and applies unchanged.

## D8. The manifest is protected from the optimiser

`config/workflows.yaml` — or wherever the manifest lands — goes on the
`FORBIDDEN` list in `evolution/space.py`, alongside `tools.max_autonomous_level`
and `tools.deny_patterns`.

Phase 9's hill-climber finds whatever makes the number go up. If lowering a
workflow's permission level removed a confirmation prompt and made a benchmark
faster, it would lower it — and by the metric it would be right. That must be
inexpressible, not merely discouraged.

## D9. n8n is not a Core dependency

AURA starts, runs, reasons and answers with n8n absent. An unreachable n8n
produces `ToolResult.failure` — a value the agent loop reads and adapts to —
never an exception crossing the tool boundary, and never a failed startup.

**Registration does not depend on liveness.** Tools are registered from the
manifest whether or not n8n is currently reachable. This differs from the
`application_tools()` precedent, which returns `[]` on non-Windows, and the
difference is deliberate: that is *platform* availability (never going to work),
this is *service* availability (transient). A tool surface that flapped with
n8n's uptime would make the model's capabilities unpredictable turn to turn.

## D10. The abstraction is "automation provider", not "n8n"

AURA's tool layer depends on an automation interface, not on n8n concepts.
Webhooks, node types and execution ids stay inside the provider. Replacing n8n
with Node-RED, Home Assistant, Windmill, or direct API tools is then a provider
swap — the same discipline as `LLMProvider`, `OCRProvider`, `TTSProvider` and
`AvatarBackend`.

## D11. Bounded event-bus participation

If automation reaches the bus at all, it does so as a small, catalogued set:

```
AUTOMATION_STARTED       occurrence · core · droppable
AUTOMATION_COMPLETED     occurrence · core · reliable
AUTOMATION_FAILED        occurrence · core · reliable
AUTOMATION_REQUIRES_APPROVAL   — NOT a new event type
```

The fourth is deliberately **not added**: `PERMISSION_REQUIRED` already exists,
already carries `tool`, `level` and `summary`, and already has an approval path
bound by nonce. A second approval-shaped event would be a second authority
path, which is precisely what the bus contract forbids.

**Never on the bus:** credentials, secrets, raw webhook payloads, whole workflow
definitions, complete external API responses, user transcripts, screen contents.
Consistent with the existing content policy — the bus carries metadata, and the
two documented exceptions stay the two documented exceptions.

## D12. Security summary

| # | Rule |
|---|---|
| 1 | n8n cannot grant itself authority |
| 2 | n8n metadata cannot assign AURA permissions |
| 3 | Undeclared workflows are not callable |
| 4 | A webhook URL does not determine permission |
| 5 | A display name does not determine permission |
| 6 | The LLM cannot modify permissions at runtime |
| 7 | Credentials stay out of the event bus |
| 8 | Credentials stay out of audit records |
| 9 | Webhook responses are untrusted (D6) |
| 10 | n8n failures fail safe, as values |
| 11 | n8n is not a Core dependency |
| 12 | AURA operates fully without n8n |
| 13 | Future discovery may surface *candidates* for a human to declare — never an executable privileged capability |

Rule 13 is the one most likely to erode. A discovery feature is genuinely
useful: "here are eight workflows n8n knows about that your manifest does not."
That is a *list for a person to read*. The moment it writes the manifest itself,
D2 is gone.

---

# FUTURE IMPLEMENTATION

Not built. Recorded so the eventual phase has a starting point rather than a
blank page.

## Manifest schema — with two additions the brief's sketch omits

```yaml
# config/workflows.yaml — trusted, version-controlled, the authority
workflows:
  - id: daily_digest
    description: >-           # REQUIRED: the model selects tools by description
      Summarise today's calendar and unread mail into a short digest.
    webhook: http://127.0.0.1:5678/webhook/digest
    parameters:               # REQUIRED: JSON Schema, as every ToolSpec has
      type: object
      properties: {}
    level: 1                  # READ
    timeout_s: 30
    idempotent: true

  - id: send_slack
    description: Post a message to a Slack channel.
    webhook: http://127.0.0.1:5678/webhook/slack
    parameters:
      type: object
      properties:
        channel: {type: string}
        text: {type: string, maxLength: 4000}
      required: [channel, text]
    level: 4                  # CRITICAL — third-party, irreversible
    confirm: always           # cannot be pre-approved for a task
    timeout_s: 20
    idempotent: false
    reversible: false
```

**`description` and `parameters` are not optional.** Without a description the
agent cannot choose the tool; without a parameter schema `ToolExecutor` cannot
validate arguments, and validation is the first stage of the chokepoint. The
brief's example omitted both; a manifest without them produces a tool that is
registered and unusable.

## Components

| Component | Responsibility |
|---|---|
| `config/workflows.yaml` | The authority. Version-controlled. |
| Manifest loader/validator | Fails at load on an unknown key, a bad level, or a missing schema — never mid-run |
| `AutomationProvider` ABC | The seam (D10) |
| `N8NProvider` | Webhook POST, timeout, response handling |
| `N8NWorkflowTool` | One per declared workflow; implements `Tool` |
| ToolSpec generation | Manifest entry → `ToolSpec` |
| Audit integration | Already free via `ToolExecutor` |
| Tests | Manifest validation, escalation refusal, failure-as-value, untrusted responses |
| Optional management UI | Read-only listing; **must not write the manifest** (D12 rule 13) |

## Deployment progression

```
Now (single machine)                Later (separated)

AURA PC                             AURA PC  ⟷  n8n server / VPS
├── AURA Core                       ├── AURA Core
├── llama-server (LLM)              ├── llama-server
├── llama-server (VLM, 10C)         ├── VLM
├── presence overlay                └── presence overlay
└── n8n
```

Moving n8n off the machine is a URL change in the manifest plus a network-egress
decision. It does not change the authority model: the manifest is still local
and still version-controlled.

## Resource note

RAM is the binding constraint on this machine — 6.4 GB free at rest, measured.
n8n plus a Node runtime is roughly 400 MB–1 GB and **has not been measured
here**. Following this project's convention, that number should be measured
before committing, not estimated. Phase 10C's LLM↔VLM eviction already has the
tightest budget in the project; adding a persistent Node runtime beforehand
competes with it.

**Recommended sequencing: after 10C.** Not an architectural objection — the tool
layer already exists, so nothing about waiting makes this harder.

---

## Consequences

**Good**

- Several hundred integrations without writing them.
- Zero new authority surface: automation is subject to the same permission
  levels, the same confirmation gate, and the same audit trail as `read_file`.
- Provider-swappable, so this is not a bet on n8n surviving.
- The manifest is a diff. "What can AURA do?" is answered by reading one file.

**Bad**

- A second secret-bearing system. n8n's credential store sits outside AURA's
  security model, and compromising n8n compromises those accounts regardless of
  what AURA permits.
- Declaring a workflow is manual by design. That friction is the security
  property, and it will feel like friction.
- RAM cost, unmeasured (see above).
- Two places to debug when a workflow misbehaves.

**The failure mode to watch for**

n8n accumulating *decisions* — "if the email looks urgent, then…". Reasoning in
two systems makes "why did AURA do that?" unanswerable from either. n8n should
do integration plumbing; AURA decides.

## Revisit when

Automation is actually needed and 10C has settled the VRAM/RAM budget; or a
provider appears that is meaningfully lighter than a Node runtime; or discovery
becomes desirable enough that rule 13 needs restating in code rather than in
prose.
