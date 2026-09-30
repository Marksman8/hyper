# AURA

**A**utonomous **U**nified **R**easoning **A**ssistant — a local-first,
model-agnostic personal AI runtime.

AURA runs entirely on your own machine. Its intelligence comes from open-weight
models served locally; no hosted API is required for normal operation, and the
model underneath is a configuration value, not a code dependency.

> **Current scope.** Phases v0.1 through 9 are complete, and Phase 10 is
> complete through 10B.
>
> Working: conversation, persistent memory with evidence, local API, CLI,
> desktop overlay and tray, the full voice loop (wake word, STT, TTS, a
> personality engine, and an interrupt that works), tools behind permission
> levels 0–4, a bounded agent loop, screen capture and OCR behind privacy
> controls that stop the pipeline rather than instruct the model, an evaluation
> harness that refuses to overclaim, config-space hill-climbing under a human
> gate, an event bus, an operator console at `/hud`, conversation recall across
> sessions, and fetching and storing web pages.
>
> In progress: 10C (VLM perception) is 2 of 12 and ends at 10C.6 by decision --
> computer use is deferred with the condition that would revive it. 10D is not
> started.
>
> See [the roadmap](docs/ROADMAP.md), [known issues](docs/KNOWN-ISSUES.md), and
> [the security audit](docs/SECURITY-AUDIT-01.md), which records one finding
> still open and why it was deferred.

---

## Architecture in one picture

```
  aura chat  (CLI) ---+
                      +--> AURA Core :8760 --> llama-server :8081 --> GGUF on GPU
  aura voice --------+       |
  (mic/speaker)              +-- orchestrator, memory (SQLite), auth, audit
```

Three processes, each replaceable. Voice is a *client*, not a subsystem: it
reaches AURA over the same authenticated HTTP API any other client uses, so
Core stays a 69 MB torch-free process that restarts in under a second.

Two processes on purpose: AURA Core restarts in under a second while the model
stays loaded, and the boundary between them is an OpenAI-compatible wire
protocol rather than a Python import — so llama.cpp, Ollama, vLLM, or a future
fine-tuned AURA model are all drop-in.

Full reasoning: [`docs/architecture/00-v0.1-architecture.md`](docs/architecture/00-v0.1-architecture.md).

## Requirements

- Windows 11 (Linux/macOS should work; only Windows is exercised)
- Python **3.12** (managed by `uv` — your system Python is untouched)
- NVIDIA GPU with 6+ GB VRAM recommended; 8 GB is the design target
- ~4 GB disk for the default model (Qwen3.5 4B, Q4_K_M)

## Setup

```powershell
# 1. Dependencies (creates .venv with Python 3.12)
uv sync

# 2. Check the machine
uv run aura doctor

# 3. Fetch the default model (~2.7 GB)
uv run python scripts/download_model.py

# 4. Fetch llama.cpp with CUDA support
powershell -ExecutionPolicy Bypass -File scripts/setup_llama_cpp.ps1
```

### Voice (Phase 2, optional)

```powershell
uv sync --extra voice                             # onnxruntime + CTranslate2, no torch
uv run python scripts/download_voice.py           # Piper voice, ~63 MB
uv run python scripts/download_vad.py             # Silero VAD, ~2 MB
uv run python scripts/download_wakeword.py        # openWakeWord models, ~10 MB
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
uv run aura chat            # interactive conversation
uv run aura chat "hello"    # one shot
uv run aura do "..."        # give it a task; it picks its own tools
uv run aura status          # health, model, RAM/VRAM
uv run aura desktop         # tray icon + status overlay + approval prompts
```

With voice installed:

```powershell
uv run aura voice           # wake word -> listen -> answer aloud
uv run aura say "hello"     # TTS only
uv run aura listen          # mic -> VAD -> STT only
uv run aura devices         # list audio devices
```

The wake word is **"hey jarvis"**. openWakeWord ships a fixed set of
pretrained phrases and "aura" is not one of them; training a custom model is
an offline job. Change it with `voice.wake.model`.

**No GPU yet, or still downloading?** Set `AURA__LLM__PROVIDER=echo` in `.env`
and everything above works against a stub backend. That is the point of the
provider seam.

## The local API

Every route needs `Authorization: Bearer <token>`, including from localhost.
The token is generated on first run into `var/secrets/api_token`.

The model server has its own separate token (`var/secrets/model_api_token`,
written by the launch script). llama-server ships with no auth and
`Access-Control-Allow-Origin: *`, so without it any web page you visit could
drive your GPU on 127.0.0.1:8081.

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
| `GET` | `/v1/tools` | what AURA can do, and what needs approval |
| `POST` | `/v1/tools/{name}` | run one tool (`confirmed: true` for level 2+) |
| `POST` | `/v1/agent` | let the model pick tools and loop until it can answer |

Interactive docs at <http://127.0.0.1:8760/docs>.

## Giving AURA a task

```powershell
uv run aura do "What Python version does this project require?"
uv run aura do --approve write_file "Write a summary of the README to notes.txt"
```

Reading happens without asking. Writing or executing needs `--approve`, named
per tool, per task — the model cannot grant that itself. When it wants a tool
it has not been given, it says so and prints the command that would allow it.

A CLI rather than a curl example on purpose: in PowerShell `curl` is an alias
for `Invoke-WebRequest`, which takes `-Headers` as a hashtable, so every
bash-shaped example fails there with a parameter-binding error that says
nothing about the real problem.

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
  voice/       wake word, VAD, STT, TTS, session loop (Phase 2)
  tools/       filesystem, Python, app launcher, permission gate (Phase 3)
  desktop/     tray, overlay, approval dialog (v0.1.1)
  clients/     reference CLI clients
config/        default.yaml, models.yaml
docs/          architecture and decision records
scripts/       model download, llama.cpp setup, server launch
tests/         unit + integration
var/           runtime state - db, logs, weights, secrets (gitignored)
```

## Development

```powershell
uv run pytest              # tests
uv run ruff check src tests scripts
uv run ruff format src tests scripts
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
