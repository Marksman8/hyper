# ADR-0001: llama.cpp as the v0.1 inference engine

Status: **accepted** · 2026-09-05 · Supersedes nothing

Required by spec §30: before selecting a major technology, state why it is
needed, what the alternatives are, and the performance, hardware and
maintainability implications.

---

## Context

AURA needs to run an open-weight model locally on:

- Windows 11 (native, not WSL)
- RTX 5060 Laptop, 8 GB VRAM, Blackwell (`sm_120`, compute capability 12.0)
- 15.7 GB system RAM
- Python 3.12

The engine must expose an API AURA can talk to over a process boundary, run
4-bit quantised weights, and stream tokens.

## Options considered

### 1. llama.cpp (`llama-server`) — **chosen**

- **Windows**: first-class. Official prebuilt CUDA binaries per release.
- **Blackwell**: supported by CUDA 12.8+ builds. Prebuilt assets exist, so no
  local compile, no cmake, no Visual Studio.
- **Quantisation**: GGUF is its native format. Q4_K_M is the well-trodden path.
- **API**: `llama-server` ships an OpenAI-compatible `/v1/chat/completions`,
  `/v1/embeddings` and `/v1/models`. Exactly the seam AURA wants.
- **Footprint**: a single ~200 MB binary. No Python dependency at all, which is
  why AURA Core can stay torch-free and start in 400 ms.
- **VRAM**: partial offload (`--n-gpu-layers N`) means a model that does not
  quite fit still runs, degraded rather than dead. Matters a lot at 8 GB.
- **Maintainability**: very active, releases multiple times a week. The risk is
  churn, not abandonment. Pinning a release tag mitigates it.

### 2. vLLM — rejected

Best-in-class throughput and the obvious choice on a Linux server. But it has
**no native Windows support**; it would require WSL2, which on a 16 GB machine
means a second memory-hungry VM alongside Windows, and a GPU passthrough setup
that is fragile across driver updates. Its architecture also targets high
concurrent batch throughput, which is worthless for one user typing one message
at a time. Revisit if AURA ever moves to a Linux host.

### 3. Hugging Face Transformers + bitsandbytes, in-process — rejected

The most flexible option and the one that makes Phase 8 (LoRA/QLoRA) easiest,
so it is genuinely tempting. Rejected for v0.1 because:

- It drags torch (~2.5 GB installed, ~500 MB resident) into AURA Core, making
  every code-reload cost a model reload — the exact coupling §2 of the
  architecture exists to avoid.
- Blackwell needs a `cu128` torch build; bitsandbytes support for `sm_120`
  lagged and is a moving target.
- 4-bit inference through bitsandbytes is meaningfully slower than a tuned
  GGUF kernel for single-stream decoding.

It is not off the table forever: it becomes a *second* provider behind the same
interface when Phase 8 needs to serve an adapter, which is precisely what the
seam is for.

### 4. Ollama — rejected as the default, kept as an option

Genuinely pleasant: one installer, automatic model management, an
OpenAI-compatible endpoint. It is a llama.cpp wrapper, so the ceiling is the
same. Rejected as the default because it interposes its own model store and
naming, which fights AURA's `config/models.yaml` being the single source of
truth, and it hides the llama.cpp flags (KV cache quantisation, layer offload,
context size) that matter most when squeezing a model into 8 GB.

Because it speaks the same protocol, `AURA__LLM__BASE_URL=http://localhost:11434/v1`
makes it work today. That is the seam earning its keep.

## Decision

Use **llama.cpp `llama-server`** as the default backend, launched as a separate
process, reached over its OpenAI-compatible HTTP API by
`aura.models.providers.openai_compat.OpenAICompatProvider`.

## Consequences

**Good**

- AURA Core has no ML dependency: 69 MB resident, sub-second restarts.
- The engine is swappable without touching AURA — verified by the fact that the
  entire test suite runs against a stub provider.
- Partial GPU offload gives a graceful degradation path at 8 GB.

**Bad**

- Two processes to start. Mitigated by `scripts/start_model_server.ps1` and, if
  it becomes annoying, `runtime.manage_model_server` (already stubbed).
- Tool-calling support in llama.cpp is newer and less battle-tested than
  vLLM's. This is the main risk to Phase 3/4 and needs to be re-evaluated
  before the agent loop is built; `--jinja` is passed for that reason.
- Pinning to prebuilt binaries means waiting on upstream for new
  architectures. Acceptable: Blackwell is already supported.

## Revisit when

- Phase 4 needs structured tool-calling and llama.cpp's implementation proves
  insufficient.
- Phase 8 needs to serve a LoRA adapter (add a Transformers provider alongside).
- The hardware changes to something with 24 GB+ VRAM on Linux (vLLM becomes
  worth its weight).
