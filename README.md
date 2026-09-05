# AURA

**A**utonomous **U**nified **R**easoning **A**ssistant — a local-first,
model-agnostic personal AI runtime.

AURA runs entirely on your own machine. Its intelligence comes from open-weight
models served locally; no hosted API is required for normal operation, and the
model underneath is a configuration value, not a code dependency.

> **v0.1 scope.** Conversation, persistent memory, local API, CLI client.
> No vision, voice, tools, or autonomous action yet — see
> [the roadmap](docs/architecture/00-v0.1-architecture.md#9-explicitly-out-of-scope-for-v01).

---

## Architecture in one picture

```
  aura chat  (CLI)  ->  AURA Core :8760  ->  llama-server :8081  ->  GGUF on GPU
                          |
                          +-- orchestrator, memory (SQLite), auth, audit
```

Two processes on purpose: AURA Core restarts in under a second while the model
stays loaded, and the boundary between them is an OpenAI-compatible wire
protocol rather than a Python import — so llama.cpp, Ollama, vLLM, or a future
fine-tuned AURA model are all drop-in.

Full reasoning: [`docs/architecture/00-v0.1-architecture.md`](docs/architecture/00-v0.1-architecture.md).

## Requirements

- Windows 11 (Linux/macOS should work; only Windows is exercised)
- Python **3.12** (managed by `uv` — your system Python is untouched)
- NVIDIA GPU with 6+ GB VRAM recommended; 8 GB is the design target
- ~4 GB disk for the default model

## Setup

```powershell
# 1. Dependencies (creates .venv with Python 3.12)
uv sync

# 2. Check the machine
uv run aura doctor

# 3. Fetch the default model (~2.5 GB)
uv run python scripts/download_model.py

# 4. Fetch llama.cpp with CUDA support
powershell -ExecutionPolicy Bypass -File scripts/setup_llama_cpp.ps1
```

## Running

Two terminals.

```powershell
# Terminal 1 - the model
powershell -ExecutionPolicy Bypass -File scripts/start_model_server.ps1

# Terminal 2 - AURA
uv run aura serve
```

Then, in a third terminal:

```powershell
uv run aura chat            # interactive
uv run aura chat "hello"    # one shot
uv run aura status          # health, model, RAM/VRAM
```

**No GPU yet, or still downloading?** Set `AURA__LLM__PROVIDER=echo` in `.env`
and everything above works against a stub backend. That is the point of the
provider seam.

## The local API

Every route needs `Authorization: Bearer <token>`, including from localhost.
The token is generated on first run into `var/secrets/api_token`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | liveness (unauthenticated) |
| `GET` | `/v1/status` | model, memory, RAM/VRAM |
| `GET` | `/v1/config` | effective config, secrets redacted |
| `POST` | `/v1/chat` | AURA-native chat, `stream: true` for SSE |
| `POST` | `/v1/chat/completions` | OpenAI-compatible surface |
| `GET` | `/v1/sessions` | list conversations |
| `GET` | `/v1/sessions/{id}` | replay a conversation |
| `DELETE` | `/v1/sessions/{id}` | delete a conversation |
| `GET` `POST` | `/v1/memory` | inspect / add stored facts |
| `DELETE` | `/v1/memory/{key}` | forget a fact |
| `POST` | `/v1/control/estop` | emergency stop |
| `GET` | `/v1/audit` | why AURA did what it did |

Interactive docs at <http://127.0.0.1:8760/docs>.

## Swapping the model

Edit `active:` in [`config/models.yaml`](config/models.yaml), re-run the
download script, restart the model server. AURA Core needs no change and no
restart.

To point at something else entirely — Ollama, LM Studio, a box on your LAN —
set `AURA__LLM__BASE_URL` and you are done.

## Project layout

```
src/aura/
  api/         FastAPI surface, auth, SSE
  core/        domain types, orchestrator (the agent loop grows here)
  memory/      SQLite: sessions, turns, facts, audit
  models/      LLMProvider interface + concrete providers
  security/    permission levels, tokens, audit policy
  runtime/     resource sampling, process supervision
  clients/     reference CLI client
config/        default.yaml, models.yaml
docs/          architecture and decision records
scripts/       model download, llama.cpp setup, server launch
tests/         unit + integration
var/           runtime state - db, logs, weights, secrets (gitignored)
```

## Development

```powershell
uv run pytest              # tests
uv run ruff check src tests
uv run ruff format src tests
uv run mypy                # strict
uv run aura serve --reload
```

## Security posture

- Loopback binding is enforced by a config validator, not convention.
- Bearer auth on every route; no way to turn it off.
- Permission levels 0–4 exist now so Phase 3 tools cannot be written without
  declaring one.
- Audit is append-only and fail-closed.
- Emergency stop halts all model calls without a restart.
- Secrets are redacted on every serialization path.
