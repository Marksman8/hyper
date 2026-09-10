# n8n workflows for AURA

Three importable workflows and the manifest entries that declare them.

**Nothing here is active until you declare it.** Importing a workflow into n8n
gives AURA nothing — the manifest at `config/workflows.yaml` is the only thing
that makes a workflow callable, and that is the whole security arrangement
(ADR-0004).

| File | Level | What it is |
|---|---|---|
| `aura_ping.json` | 1 | No credentials. Proves the path works. **Start here.** |
| `daily_digest.json` | 1 | Reads. Sends nothing. |
| `post_to_team_chat.json` | 4 | Third-party, unsendable. Ships as a dry run. |
| `workflows.yaml` | — | The manifest entries. Copy into `config/workflows.yaml`. |

---

## Setup

### 1. Run n8n

```powershell
npx n8n
```

It comes up on <http://localhost:5678>. First run asks you to create a local
account — that account is n8n's, not AURA's.

> **Not measured on this machine.** n8n plus a Node runtime is roughly
> 400 MB–1 GB. This project's convention is to measure before committing, and
> that measurement is still outstanding. See ADR-0004's resource note.

### 2. Import a workflow

**Workflows → ⋯ → Import from File →** pick `aura_ping.json`.

### 3. Activate it

Toggle **Active**, top right.

This step is the one people miss, and it produces the single most common n8n
error. An inactive workflow has no production URL, so AURA gets a 404 — which
is why the provider's 404 message says *"the workflow may be inactive"* rather
than just reporting the status code.

The two URLs are not interchangeable:

| URL | Works when |
|---|---|
| `/webhook-test/aura/ping` | Only while you have clicked "Execute workflow" — one call, for debugging |
| `/webhook/aura/ping` | The workflow is **Active**. This is what belongs in the manifest |

### 4. Declare it to AURA

Copy the `aura_ping` entry from `workflows.yaml` into `config/workflows.yaml`,
then switch automation on in `config/local.yaml`:

```yaml
automation:
  enabled: true
```

### 5. Restart Core and check

```powershell
aura serve
```

Startup logs `automation.tools_registered` with the tool names and their
levels. Then:

```powershell
curl -H "Authorization: Bearer $(Get-Content var/secrets/api_token)" `
     http://127.0.0.1:8760/v1/automation
```

Or just ask her: *"run aura_ping"*.

---

## The contract a workflow has to meet

Three things, and they are all in `aura_ping.json` if you would rather read
code than prose.

### Arguments arrive under `body`

AURA POSTs the tool's arguments as the JSON request body, so in n8n they are at
`$json.body`, not the top level:

```js
const args = $input.first().json.body ?? {};
```

### Answer synchronously

Set the webhook node's **Respond** to `Last node` (or use a Respond to Webhook
node). n8n's default replies *"Workflow was started"* instantly and AURA learns
nothing — it would report success having received no result.

### Return a `summary` string

```js
return [{ json: {
  summary: "3 events today: 10:00 standup, 14:30 design review.",
  events: [...],      // structured data stays available
  count: 2
}}];
```

`summary` is what the model reads and speaks. Everything else is kept in the
tool result's `data`. Without a `summary` field the model gets the raw JSON,
which makes a poor spoken answer.

Responses are capped at 64 KB, and the model-facing summary at 4,000
characters.

---

## Adding your own

1. Build it in n8n. Give it a webhook trigger with a path under `aura/`.
2. Activate it.
3. Add an entry to `config/workflows.yaml` — `id`, `description`,
   `parameters` (JSON Schema), `webhook`, `level`.
4. Restart Core.

**Choosing a level:**

| | |
|---|---|
| 1 READ | Reads something. Changes nothing. |
| 2 ACT | A reversible change. |
| 3 MUTATE | Hard to undo. |
| 4 CRITICAL | Sends, posts, pays, deletes, or reaches a third party. **`confirm: always`.** |

Level 0 is rejected: calling an external system is never "no side effects".

**The loader refuses to start** if a workflow whose id or description contains
*send, post, publish, delete, remove, pay, purchase, transfer, deploy, email,
message* or *tweet* is declared below level 4 with confirmation. It matches
substrings and only in English, so `resend_report` trips it and a description
mentioning "unread messages" trips it too. Rename around it — a false positive
costs one rename, a false negative costs a sent message.

---

## What n8n can and cannot do

n8n **executes**. It does not decide.

- A workflow cannot set its own permission level. Renaming `draft_email` to
  `send_email` in the browser changes nothing about what AURA will do with it.
- An undeclared workflow is not callable, whatever n8n advertises.
- A webhook response is untrusted data. It is wrapped and labelled before it
  reaches the model, and it cannot cause a tool to run.
- Credentials stay in n8n. Gmail and Slack tokens never enter AURA.
- If n8n is down, AURA carries on. The tool returns a failure the agent reads
  and adapts to.

The full reasoning is in `docs/decisions/0004-automation-provider-authority.md`.

## Optional: locking the webhooks down

The webhook URLs are reachable by anything that can route to n8n. On loopback
that is a small surface, but AURA can authenticate itself:

```yaml
automation:
  auth_header: "X-Aura-Token"
  auth_token_file: "var/secrets/automation_token"
```

Put the same secret in an n8n **Header Auth** credential on each webhook node.
AURA sends it and never logs it. This is AURA proving it is AURA — it has
nothing to do with your Gmail or Slack credentials, which stay inside n8n.
