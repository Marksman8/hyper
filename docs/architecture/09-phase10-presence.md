# Phase 10 — Presence, Perception, and the Personal Model

Status: **design proposal — not approved, not implemented** · Date: 2026-09-09

> This document is a plan. No Phase 10 code has been written. It ends with an
> implementation plan awaiting explicit authorisation.

---

## 0. The headline judgement, before the detail

Two findings from inspecting the repository change the shape of this phase, and
one scoping opinion follows from them.

**Finding 1 — the VLM needs no new inference stack.** `var/llama.cpp/` already
contains `mtmd.dll`, `llama-mtmd-cli.exe`, and a `llama-server.exe` that
accepts `--mmproj`. Vision-language models are served over the *same*
OpenAI-compatible endpoint AURA already speaks. `VLMProvider` becomes a thin
extension of the existing `OpenAICompatProvider`, the seam test keeps holding,
and **torch still does not enter the project.** This removes what would
otherwise have been the largest and riskiest piece of Phase 10.

**Finding 2 — tkinter cannot be the avatar.** ADR-0002 chose tkinter on a
measured 13 MB vs 120 MB argument, and that reasoning was right *for a status
panel*. An avatar needs per-pixel alpha (tkinter has only chroma-key
`-transparentcolor`, which fringes badly on anti-aliased art), 30–60 fps
compositing, and smooth transforms. tkinter has none of these. ADR-0002 needs a
narrow amendment, not a reversal — see §D.1.

**The scoping opinion, stated plainly.** Phase 10 as specified is five pillars,
and **each one is comparable in size to Phases 5 through 9 individually.**
Desktop presence alone (avatar, state machine, animation, speech
presentation) is a phase. Computer-use is a phase, and the highest-risk one in
the project's history. The Personal Cognitive Model is a phase. Delivered as
one unit, this would be six to ten weeks of work landing in a single
untestable lump, which is exactly the failure mode §36 of the original spec
warns against.

I recommend splitting it into **10A / 10B / 10C**, sequenced by risk and by
what unblocks what. The full argument is in §H. The design below covers all
five pillars regardless, because the interfaces have to be designed together
even if they ship apart.

---

# A. Current Architecture Assessment

## A.1 What exists

```
src/aura/
  core/           1219   orchestrator, agent loop, types
  api/            1040   FastAPI, SSE streaming, auth middleware
  tools/          2014   registry, executor, permission gate, workspace guard
  voice/          2576   wake → STT → orchestrator → TTS, all ONNX/CTranslate2
  memory/         1045   SQLite WAL, facts + evidence + FTS5
  vision/          970   privacy gate, capture, OCR behind a seam
  eval/            933   deterministic suite, repeats, noise floor
  evolution/       685   hill-climb, allow-list, human approval gate
  clients/        1395   CLI clients (chat, vision, memory, profile, eval, evolve)
  personalization/ 499   profile → directives + sampling
  desktop/         888   tkinter tray, status overlay, approval dialog
  models/          660   LLMProvider ABC + OpenAI-compatible provider
  security/        309   permission levels, audit log
  runtime/          94   CPU/RAM/VRAM snapshot
```

680 tests, ruff clean, mypy strict clean.

## A.2 What Phase 10 reuses unchanged

| Existing | Phase 10 use |
|---|---|
| `OpenAICompatProvider` | **The VLM, via `--mmproj`.** Same wire protocol, same auth, same retry logic. |
| `PermissionLevel` 0–4 + `ToolExecutor` | Every computer-use action goes through the existing chokepoint. No new permission model. |
| `Workspace` guard | Already covers file paths for computer-use file operations. |
| `AuditLog` + `audit` table | Mouse, keyboard and VLM observations are audited by the same path. |
| `PrivacyPolicy` / `VisionMode` | **Already the right shape** for §17. `PRIVATE / ASSIST / ACTIVE / SLEEP` exist and gate *before capture*. Needs one new mode (`FOCUS`) and one new consumer (the VLM). |
| `ChangeDetector` | Built in Phase 5, tested, called by nothing. **This is exactly the VLM trigger** §7 asks for. It was built for this. |
| `ScreenCapture` | Multi-monitor, foreground-window identification, gate-before-pixels. Unchanged. |
| Memory `facts` + `fact_evidence` + `beliefs.py` | The Personal Cognitive Model's *explainability* layer already exists. Evidence-backed confidence with `explain()` is precisely §12's requirement. |
| `eval/` harness | Benchmarks the SNN-vs-baseline comparison in §F. |
| SSE (`_sse` in `api/app.py`) | The event bus. Extend, do not replace — see §D.9. |
| `runtime/resources.py` | Foundation for the resource manager. Currently reports only; §16 asks it to decide. |

## A.3 What must be extended

| Component | Extension | Risk |
|---|---|---|
| `models/base.py` | Add `VLMProvider`, `TemporalModelProvider` ABCs alongside `LLMProvider` | Low |
| `models/registry.py` | Register VLM factory | Low |
| `vision/` | `perception/` subpackage: VLM observation, trigger policy | Medium |
| `tools/` | `computer/` subpackage: UIA-based interaction tools | **High** |
| `security/permissions.py` | No new levels. New *policies* for input injection | **High** |
| `desktop/` | New `presence/` process; existing shell stays or merges | Medium |
| `memory/` | `cognitive/` subpackage: behavioural events, patterns | Medium |
| `runtime/resources.py` | From reporting to arbitration (`ResourceGovernor`) | Medium |
| `config.py` | New sections: `presence`, `perception`, `computer_use`, `cognitive` | Low |
| `eval/suites/` | New suites: computer-use safety, perception accuracy | Low |

## A.4 What must NOT be changed

These are load-bearing and Phase 10 must route around them, not through them:

1. **The provider seam.** `tests/unit/test_seam.py` fails the build if an
   inference library is imported outside `models/providers/` or a vision
   library outside `aura/vision/`. The VLM and any SNN work must respect this.
   The SNN, if built, needs torch — it therefore lives behind
   `TemporalModelProvider` in its own provider package, and the seam test gains
   a new rule rather than an exception.

2. **`ToolExecutor` as the single chokepoint.** registry → validate →
   permission → execute(timeout) → audit. Computer-use tools are *tools*. They
   do not get a side door because they are latency-sensitive.

3. **The privacy gate sits before capture.** The VLM must consult
   `PrivacyPolicy.may_capture()` through `ScreenCapture.grab()`. It must not
   acquire its own capture path. A refusal must mean no pixels were read.

4. **Approval is human-granted and cannot be self-issued.** `approved_tools` is
   supplied per task; the model cannot add to it. Computer-use does not relax
   this.

5. **The evolution allow-list.** `evolution/space.py` must never gain a knob
   that touches input injection, capture scope, or permission ceilings.
   Phase 10 adds settings that are *tempting* to tune (VLM trigger threshold,
   presence intrusiveness). Any that affect what AURA may see or do go on
   `FORBIDDEN`, not `KNOBS`.

6. **Errors as values across the tool boundary.** Computer-use failures are
   `ToolResult.failure`, not exceptions.

---

# B. Phase 10 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  PRESENCE PROCESS            (new, PySide6, ~120 MB, GPU-composited)  │
│  ┌────────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────────────┐  │
│  │  Avatar    │ │ Subtitle │ │ Speech    │ │ Cursor / window      │  │
│  │  Renderer  │ │ / bubble │ │ visualiser│ │ awareness (read-only)│  │
│  └─────┬──────┘ └────┬─────┘ └─────┬─────┘ └──────────┬───────────┘  │
│        └─────────────┴─────────────┴──────────────────┘              │
│                          PresenceController                          │
│                    (subscribes; holds NO reasoning)                  │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │  SSE  /v1/events   (authenticated)
                                 │  HTTP /v1/presence/*
┌────────────────────────────────┴─────────────────────────────────────┐
│  AURA CORE                                            (existing, 69 MB)│
│                                                                       │
│   ┌───────────────┐   ┌──────────────┐   ┌─────────────────────────┐ │
│   │ Orchestrator  │──▶│  EventBus    │──▶│  PresenceStateMachine   │ │
│   │  + AgentLoop  │   │  (new, in-   │   │  IDLE/LISTENING/...     │ │
│   └───────┬───────┘   │   process)   │   └─────────────────────────┘ │
│           │           └──────┬───────┘                                │
│           │                  │                                        │
│           ▼                  ▼                                        │
│   ┌───────────────┐   ┌──────────────────┐   ┌────────────────────┐  │
│   │  ToolExecutor │   │ PersonalityEngine│   │ ResourceGovernor   │  │
│   │  ── the only  │   │  micro-expression│   │ VRAM/RAM arbiter   │  │
│   │     chokepoint│   │  selection       │   │ (extends runtime/) │  │
│   └───┬───────┬───┘   └──────────────────┘   └─────────┬──────────┘  │
│       │       │                                        │             │
│  ┌────▼───┐ ┌─▼──────────────┐                         │             │
│  │ files  │ │ computer/  NEW │                         │             │
│  │ apps   │ │  UIA locate    │                         │             │
│  │ python │ │  click / type  │◀── verified targets     │             │
│  └────────┘ └───────┬────────┘                         │             │
│                     │                                  │             │
│   ┌─────────────────▼──────────────────┐               │             │
│   │  PERCEPTION  (extends vision/)     │               │             │
│   │  PrivacyPolicy ─ gate before pixels│               │             │
│   │  ScreenCapture → ChangeDetector    │               │             │
│   │        │              │            │               │             │
│   │        ▼              ▼            │               │             │
│   │   OCRProvider    VLMProvider  NEW  │               │             │
│   │   (RapidOCR)     (llama-server     │               │             │
│   │                   --mmproj)        │               │             │
│   │        └──────┬───────┘            │               │             │
│   │        StructuredObservation       │               │             │
│   └────────────────────────────────────┘               │             │
│                                                        │             │
│   ┌────────────────────────────────────────────────────▼──────────┐  │
│   │  COGNITIVE MODEL  (extends memory/)                           │  │
│   │  BehaviouralEvent log → PatternStore → UserState              │  │
│   │  TemporalModelProvider  ── Markov | GRU | SNN (comparable)    │  │
│   │  every inference carries evidence (reuses fact_evidence)      │  │
│   └───────────────────────────────────────────────────────────────┘  │
└──────────────────────┬────────────────────────────────────────────────┘
                       │ HTTP (OpenAI-compatible)
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌───────────────────┐        ┌────────────────────┐
│ llama-server :8081│        │ llama-server :8082 │
│ Qwen3.5-4B  LLM   │        │ VLM + mmproj  NEW  │
│ 3.07 GB VRAM      │        │ on demand, evicted │
│ resident          │        │ MUTUALLY EXCLUSIVE │
└───────────────────┘        └────────────────────┘
              └──── ResourceGovernor arbitrates ────┘
```

**The one architectural rule that keeps this honest:** the presence process
holds no reasoning. It subscribes to events and renders. If the avatar process
is killed, AURA loses a face and nothing else.

---

# C. Component Breakdown

## C.1 EventBus (new, `core/events.py`)

| | |
|---|---|
| **Responsibility** | One in-process publish/subscribe point. Every subsystem that wants to say "something happened" says it here. |
| **Interface** | `publish(Event)` / `subscribe() -> AsyncIterator[Event]`. `Event` = `{type, payload, ts, session_id, request_id}`. |
| **Dependencies** | None. Deliberately the most dependency-free module in the system. |
| **Process** | AURA Core. Fan-out to the presence process over SSE. |
| **Resources** | Negligible. Bounded ring buffer (256 events); slow subscribers drop rather than back-pressure the orchestrator. |

**Why this and not ZeroMQ / named pipes:** AURA already has an authenticated
SSE endpoint, and the desktop already speaks HTTP to Core. Adding a second IPC
mechanism means a second auth story, a second failure mode, and a second thing
to debug on Windows. SSE is a text stream over an authenticated connection that
already works.

**Dropping is deliberate.** A presence UI that stalls the orchestrator because
it is repainting is worse than a presence UI that misses a frame of state.

## C.2 PresenceStateMachine (new, `core/presence.py`)

| | |
|---|---|
| **Responsibility** | Derive the visible state from real events. Never random. |
| **Interface** | Consumes `EventBus`, emits `PresenceState` transitions. |
| **Process** | AURA Core. |

States: `IDLE LISTENING THINKING WORKING SUCCESS CONFUSED WARNING ANNOYED HAPPY SLEEP PRIVATE ERROR`

Mapping from events that **already exist** in the codebase:

| Existing event | State |
|---|---|
| wake word detected (`voice/wake`) | `LISTENING` |
| `orchestrator.chat` / `stream` started | `THINKING` |
| `agent.tool_call` | `WORKING` |
| `agent.answered`, no failures | `SUCCESS` → `IDLE` |
| `tools.executed decision=deny` | `WARNING` |
| tool returned `ok=False`, loop retrying | `CONFUSED` |
| `agent.stopped_early` (budget exhausted) | `ANNOYED` |
| `vision.mode_changed → private` | `PRIVATE` |
| `orchestrator.emergency_stop` | `ERROR` |
| idle > N minutes | `SLEEP` |

**Design opinion on `ANNOYED` and `HAPPY`.** These are the two states where an
assistant becomes tiresome. `ANNOYED` should be rare and *earned* — mapped to
"I hit my step budget and could not finish", not to "the user asked something
twice". `HAPPY` should be reserved for genuinely completed multi-step work, not
every successful reply. An assistant that emotes constantly is noise; one that
emotes rarely carries information. Both should be configurable to off.

**Dwell times and hysteresis are required.** Without a minimum dwell (~400 ms)
and a transition cooldown, a fast agent loop makes the avatar strobe between
`THINKING` and `WORKING` several times a second. This is a real bug in every
naive implementation of this idea.

## C.3 Presence process (new, `presence/`)

| | |
|---|---|
| **Responsibility** | Draw the avatar, subtitle, speech visualisation. Nothing else. |
| **Interface** | Subscribes to `/v1/events`. Calls `/v1/presence/*` for user-initiated actions (click-to-talk, dismiss). |
| **Process** | **Separate OS process.** Crash isolation, independent restart, and it can be killed without touching Core. |
| **Resources** | PySide6 ~110–130 MB RAM. GPU compositing is the desktop compositor's job; the avatar is a layered window, not a 3D scene. |

**Sub-components:**

- `AvatarRenderer` — implements `AvatarBackend` (§C.4)
- `SubtitlePanel` — text beside/below the avatar, always available
- `SpeechVisualiser` — amplitude-driven mouth + a small level indicator
- `WindowWatcher` — foreground window + cursor position, **read-only**, for
  placement only. Reuses `vision/windows.py`.
- `PlacementPolicy` — where she sits, and when she is allowed to move

**On movement (§2's explicit warning).** Movement must be *event-driven*, not
continuous. My recommendation: the avatar has a small number of anchor slots
(near the taskbar corner, beside the foreground window's title bar, a
user-pinned position). She moves **only** on: foreground application change,
explicit user drag, a state transition that needs visibility (`WARNING`,
`ERROR`), or the current slot being occluded. Cursor awareness should be
*gaze only* — the eyes track, the body does not. Continuous position-following
is the single fastest way to make a desktop companion intolerable.

## C.4 AvatarBackend seam (new, `presence/avatar/base.py`)

The §3 requirement: the backend must never depend on a specific avatar
technology. Same discipline as `LLMProvider`.

```python
class AvatarBackend(ABC):
    @abstractmethod
    def info(self) -> AvatarInfo: ...
    @abstractmethod
    def set_state(self, state: PresenceState, *, immediate: bool = False) -> None: ...
    @abstractmethod
    def set_mouth(self, openness: float) -> None:        # 0.0–1.0, lip sync
    @abstractmethod
    def look_at(self, dx: float, dy: float) -> None:     # normalised gaze
    @abstractmethod
    def render(self, dt_ms: int) -> Image: ...           # RGBA frame
    def preload(self) -> None: ...
    def unload(self) -> None: ...
```

Implementations, in the order I would build them:

1. `SpriteAvatar` — PNG sprite sheets per state. **Start here.** Zero GPU cost,
   trivially replaceable, and it forces the interface to be honest.
2. `Live2DAvatar` — Cubism. **Licensing must be checked before any work**: the
   Cubism SDK for Native is free only below a revenue threshold and requires
   accepting a publication licence. For a personal tool this is very likely
   fine, but it is a legal question, not a technical one, and it should be
   answered before code is written.
3. `ModelAvatar` — 3D/VRM. Costs real VRAM, which is the scarcest resource in
   this project. Not recommended until the VRAM budget in §E has slack.

## C.5 VLMProvider (new, `models/base.py` + `models/providers/`)

```python
class VLMProvider(ABC):
    @abstractmethod
    async def observe(
        self,
        image: Image,
        question: str,
        schema: dict | None = None,
    ) -> Observation: ...
```

Backed by `OpenAICompatVLMProvider`, which posts an OpenAI-style message with
an `image_url` content block containing a base64 data URI — exactly what
`llama-server --mmproj` accepts. **This reuses the existing provider almost
entirely.**

## C.6 Perception (new, `vision/perception/`)

The pipeline §7 asks for, wired to parts that already exist:

```
ScreenCapture.grab()          ← existing, gate already before pixels
      ↓
ChangeDetector.check()        ← existing, built in Phase 5, tested, unused
      ↓
TriggerPolicy                 ← NEW: decides if this is worth a VLM call
      ↓
OCRProvider (cheap)  ──────┐
VLMProvider (expensive) ───┴──▶ StructuredObservation
```

**`TriggerPolicy` is the component that makes this affordable.** Triggers, in
increasing cost order:

| Trigger | Perception used |
|---|---|
| Foreground window changed | window title only (free) |
| Change score above threshold | OCR (~1–2 s CPU) |
| Agent explicitly asks | OCR, then VLM if OCR insufficient |
| User asks about the screen | VLM |
| Computer-use verification | VLM (targeted crop, not full screen) |

**A hard rate limit is required**, not optional: minimum interval between VLM
calls, a per-hour ceiling, and a hard stop when `ResourceGovernor` reports
pressure. Without it, a screen that changes continuously (a video, a build log)
triggers unbounded inference.

`StructuredObservation` is the §8 requirement — the agent must not be coupled
to raw VLM prose:

```python
@dataclass(frozen=True)
class StructuredObservation:
    application: str
    window_title: str
    state: str                     # "normal" | "error" | "dialog" | "loading"
    text: list[str]
    elements: list[UIElement]       # from UIA where available, VLM otherwise
    summary: str
    confidence: float
    source: str                     # "uia" | "ocr" | "vlm" | "uia+vlm"
    captured_at: float
```

`source` matters: a consumer must be able to tell a UIA-derived element (exact,
from the OS) from a VLM-derived one (inferred, fallible).

## C.7 Computer use (new, `tools/computer/`) — **highest risk in the project**

**The core design decision: UI Automation first, coordinates last.**

Windows exposes UI Automation (UIA), which gives named, typed, addressable
elements — `Button "Save"`, `Edit "Search"` — with their bounding boxes,
directly from the OS. This is dramatically safer and more reliable than asking
a VLM to guess pixel coordinates:

| | UIA | Vision + coordinates |
|---|---|---|
| Accuracy | Exact, from the app itself | Inferred, drifts with DPI/theme/scroll |
| Verifiable | Element identity is checkable before and after | Only by looking again |
| Auditable | "clicked Button 'Save' in Notepad" | "clicked (817, 442)" |
| Cost | Microseconds, CPU | Seconds, GPU |
| Coverage | Most native + Electron apps | Everything, including canvas/games |

So: **UIA is the primary path; vision is the fallback and the verifier.** The
audit log records a semantic action, which is the difference between a log
someone can review and a log of coordinates nobody can interpret.

Proposed tools, all through the existing `ToolExecutor`:

| Tool | Level | Notes |
|---|---|---|
| `list_ui_elements` | 1 READ | UIA tree of foreground window. Gated by `PrivacyPolicy`. |
| `find_ui_element` | 1 READ | Locate by name/role. Returns a handle, not coordinates. |
| `click_element` | 3 MUTATE | Takes a **handle from `find_ui_element`**, re-verified immediately before the click. |
| `type_text` | 3 MUTATE | Into a named, verified field. Never blind keystrokes. |
| `press_keys` | 3 MUTATE | Allow-listed chords only (Ctrl+S, Ctrl+C…). Not arbitrary. |
| `wait_for_element` | 1 READ | The verify half of OBSERVE→ACT→VERIFY. |
| `screenshot_region` | 1 READ | Bounded crop for verification. |

**Non-negotiable guards, and why each exists:**

1. **No raw coordinate clicking in the default tool set.** A model that can
   click anywhere can click "Confirm delete" it never saw. Coordinates are a
   separate, explicitly-enabled, LEVEL 4 escape hatch.
2. **Re-verify the target immediately before acting.** Same class of bug as the
   window-switch race already handled in `vision/capture.py`: focus moves
   between decide and act.
3. **Password and credential fields are refused.** UIA exposes `IsPassword`.
   `type_text` into such a field is denied outright, at any permission level.
   This mirrors the existing `DEFAULT_DENY_TITLES` in `vision/privacy.py`.
4. **Destructive-affordance detection.** Buttons matching a deny list
   (`Delete`, `Uninstall`, `Format`, `Send`, `Pay`, `Confirm purchase`) require
   confirmation *even when* `click_element` is pre-approved for the task.
   Approving "you may click" is not approving "you may click Pay".
5. **A real emergency stop.** A global hotkey that halts input injection
   synchronously — not a flag checked at the top of the next loop iteration.
   The existing `emergency_stop` is checked between steps, which is too coarse
   when the thing being stopped is a keystroke stream.
6. **Input injection is audited at CRITICAL** regardless of nominal level.

**On messaging apps specifically** (§9's WhatsApp example). Sending a message
is irreversible, attributable to the user, and visible to a third party. It
should be **LEVEL 4 and confirmation-required by default**, with the message
text shown in the confirmation dialog. "Send Rahul: I'll reach in 20 minutes"
is a lovely demo and a genuinely bad thing to get wrong.

## C.8 Personal Cognitive Model (new, `memory/cognitive/`)

```
BehaviouralEvent  →  EventLog  →  PatternStore  →  UserState  →  prompt context
   (app focus,        (SQLite,      (interpretable   (current
    tool use,          append-       patterns +       activity,
    correction,        only)         confidence)      focus, mood)
    time of day)
```

`BehaviouralEvent` is deliberately coarse and non-content-bearing:

```python
@dataclass(frozen=True)
class BehaviouralEvent:
    kind: str          # "app_focus" | "tool_used" | "correction" | "session_start"
    subject: str       # "Code.exe" | "read_file" | "response_length"
    ts: float
    session_id: str | None
    # NO screen content, NO message text, NO file contents.
```

**Why coarse:** this log will accumulate for months. If it contains content, it
becomes the most sensitive file on the machine and every privacy question gets
harder. Storing *that* an app was focused, not *what was in it*, keeps the
cognitive model useful and the blast radius small.

**Explainability is not bolted on — it reuses Phase 6.** `fact_evidence`
already stores the utterance behind a belief, and `beliefs.py` already derives
confidence from evidence with an `explain()` sentence. Patterns get the same
treatment: a pattern is believed because of *N* observed transitions, and
`aura cognitive why <pattern>` shows them. §12's four questions map directly:

| §12 asks | Mechanism |
|---|---|
| "What have you learned about me?" | `aura cognitive list` — extends `aura memory list` |
| "Why did you think I wanted this?" | Evidence rows, as Phase 6 |
| "Forget that preference." | `delete` cascades to evidence, as Phase 6 |
| "Stop learning this type of behavior." | Per-`kind` opt-out in config — **new** |

## C.9 ResourceGovernor (extends `runtime/`)

§16 asks the resource manager to *decide*, not just report.

```python
class ResourceGovernor:
    def may_load(self, model: ModelSpec) -> Decision: ...
    async def acquire(self, model: ModelSpec) -> Lease: ...   # may evict
    def pressure(self) -> Pressure: ...      # NORMAL | TIGHT | CRITICAL
    def register_activity(self, kind: str) -> None:   # "gaming", "training"
```

**The rule it enforces:** the LLM and the VLM are **mutually exclusive in
VRAM** on this machine (§E shows the arithmetic). Acquiring the VLM evicts the
LLM and vice versa, with the swap cost paid explicitly and logged. Pretending
both can be resident is the single most likely way for Phase 10 to produce a
system that thrashes.

---

# D. Technology Evaluation

## D.1 Desktop UI framework — **PySide6, for the presence process only**

| Option | RAM | Per-pixel alpha | Animation | Verdict |
|---|---|---|---|---|
| **tkinter** (current) | +13 MB | ✗ chroma-key only | Poor | Keep for status/approval. **Cannot do the avatar.** |
| **PySide6** | ~120 MB | ✓ `WA_TranslucentBackground` | ✓ QPropertyAnimation, QML | **Recommended** |
| Win32 layered window (ctypes + Pillow) | ~25 MB | ✓ `UpdateLayeredWindow` | Manual | Viable; ~800 lines of custom compositing |
| Electron / Tauri+web | 150–400 MB | ✓ | ✓ | Rejected: RAM, and a second language runtime |
| Dear PyGui | ~60 MB | Partial | ✓ | GPU context always alive; awkward for a click-through overlay |

**Recommendation: PySide6 in a separate process, LGPL, Windows-native.**
ADR-0002's 13-vs-120 MB argument stands for a status panel and does not apply
to an avatar — the requirement changed. The measured free RAM at rest on this
machine is **6.4 GB** (measured just now: 9.34/15.71 GB used with llama-server
down), so 120 MB is ~2% of headroom. That is affordable; the ADR should be
*amended with the new measurement*, not silently overridden.

**The honest alternative:** if the 120 MB is unacceptable, the Win32 layered
window route gets per-pixel alpha for ~25 MB at the cost of writing the
compositor. I would not start there, but it is a real fallback and the
`AvatarBackend` seam means switching later is contained.

## D.2 Avatar rendering — **sprite sheets first, Live2D later**

| Option | Cost | Quality | Risk |
|---|---|---|---|
| **PNG sprite sheets** | ~10 MB RAM | Good if the art is good | **Lowest. Start here.** |
| Live2D Cubism | ~40 MB + SDK | Excellent | **Licence review required first** |
| VRM / 3D | 200 MB+ VRAM | Excellent | VRAM is the scarce resource |
| Generated per-frame (diffusion) | Prohibitive | — | Rejected outright |

Sprite sheets force the `AvatarBackend` interface to be honest, cost almost
nothing, and can ship while better art is commissioned.

## D.3 VLM — **Qwen2.5-VL-3B GGUF via the existing llama-server**

| Model | Q4 size | +mmproj | Quality on UI/text | Verdict |
|---|---|---|---|---|
| **Qwen2.5-VL-3B-Instruct** | ~2.2 GB | ~1.0 GB | Strong on UI, OCR, structured output | **Recommended** |
| Moondream2 (1.8B) | ~1.6 GB | ~0.5 GB | Decent; weaker on dense UI | Fallback if VRAM is tighter than modelled |
| Qwen2.5-VL-7B | ~4.7 GB | ~1.3 GB | Best | **Does not fit** alongside anything |
| MiniCPM-V 2.6 (8B) | ~5.5 GB | ~1 GB | Excellent | Does not fit |
| Florence-2-base | ~0.5 GB | — | Grounding/detection, not conversation | Interesting *complement*, not a replacement |

All served by the **existing** `llama-server.exe --mmproj` on a second port.
No torch, no transformers, no new provider architecture.

## D.4 Screen capture — **keep `mss`**

Already integrated, 0.1 MB, measured at 31–69 ms for 1920×1080 in Phase 5.
Windows Graphics Capture API would be faster and can exclude windows at the OS
level, but it needs a native binding. Revisit only if capture latency becomes
the bottleneck, which it currently is not — OCR is.

## D.5 STT / TTS — **keep faster-whisper and Piper**

Both already work, both are ONNX/CTranslate2, both are behind seams. The Phase
10 addition is not a new engine but **speech events on the bus** so the
presence process can render subtitles.

**Female voice**: Piper has multiple English female voices (e.g.
`en_GB-alba-medium`, `en_US-amy-medium`). This is a config change, not
engineering.

## D.6 Speech synchronisation — **amplitude envelope, not phonemes**

| Option | Effort | Quality |
|---|---|---|
| **RMS envelope → mouth openness** | Low | Surprisingly convincing for 2D |
| Phoneme/viseme timing from espeak-ng | High | Better, but requires plumbing Piper's phonemizer |
| Neural viseme prediction | Very high | Overkill |

Piper already produces PCM in chunks; computing RMS per 20 ms frame and mapping
to `set_mouth(0..1)` is a few dozen lines. **This is 90% of the perceived
quality for 5% of the work**, and the `AvatarBackend` interface allows visemes
later without changing callers.

## D.7 Computer-use automation — **UIA via `uiautomation` or `pywinauto`**

| Option | Notes |
|---|---|
| **`uiautomation`** (pure Python, ctypes) | Light, direct UIA access. **Recommended.** |
| `pywinauto` | Richer, heavier, more magic. Good fallback. |
| Raw `pyautogui` | **Rejected as a primary path.** Coordinate-only; no element identity, nothing to audit or verify. |
| Playwright (browser only) | **Recommended specifically for browsers** — real DOM selectors beat any pixel approach. Adds ~300 MB installed, browser process on demand. |

`pyautogui`/`pydirectinput` remain available *underneath* the UIA layer for the
actual input synthesis, but never as the addressing mechanism.

## D.8 SNN framework — see §F. Short answer: `snnTorch`, and only in a
sandboxed prototype, only after baselines exist.

## D.9 Event bus / IPC — **extend the existing SSE**

| Option | Verdict |
|---|---|
| **SSE on `/v1/events`** | **Recommended.** Auth, framing, reconnect and client code already exist. |
| WebSocket | Bidirectional, but presence→core is low-frequency and HTTP POST covers it |
| ZeroMQ / named pipes | Faster, and a second auth story, second failure mode, second thing to debug |

Latency: SSE over loopback is sub-millisecond. The bottleneck is repaint, not
transport.

---

# E. VRAM / RAM Budget

Measured values are marked ✓; the rest are estimates to be validated in 10.0.

**Machine:** RTX 5060 Laptop, **7.96 GB usable VRAM** ✓ · 15.71 GB RAM ✓ ·
**6.4 GB RAM free at rest** ✓ (measured 2026-09-09, llama-server down)

| Component | VRAM | RAM | CPU | Latency | Residency |
|---|---|---|---|---|---|
| AURA Core | — | 69 MB ✓ | low | — | always |
| llama-server (Qwen3.5-4B Q4) | 3.07 GB ✓ | ~0.4 GB | — | 60–76 tok/s ✓ | resident |
| Voice client (wake+STT+TTS) | — | 430 MB ✓ | med bursts | 919–1150 ms to audio ✓ | on demand |
| RapidOCR | — | ~250 MB | 8 threads | 2 s sparse / 10–30 s dense ✓ | on demand |
| **Presence (PySide6)** | ~30 MB shared | **~120 MB** | low | 60 fps | always (when shown) |
| **VLM (Qwen2.5-VL-3B Q4 + mmproj)** | **~3.2 GB** | ~0.5 GB | — | est. 3–8 s / image | **event only, evicts LLM** |
| **UIA automation** | — | ~40 MB | low | ms | on demand |
| **Cognitive model (statistical)** | — | ~20 MB | negligible | ms | always |
| **SNN prototype (torch)** | ~0.5 GB | **~2.5 GB** | — | ms | **prototype only, never resident** |

## The arithmetic that decides the architecture

```
VRAM:  7.96 total
       3.07  LLM
       3.20  VLM
       ────
       6.27  + display/compositor/browser  →  NO SAFE MARGIN
```

**Conclusion: the LLM and VLM cannot be co-resident.** Not "should not" —
cannot, with any confidence, on a laptop GPU that also drives the display.

Three options, and my recommendation:

1. **Sequential with eviction (recommended).** `ResourceGovernor` unloads one
   to load the other. Cost: ~3–6 s swap. Acceptable because VLM use is
   event-driven and rare by design.
2. **Small VLM co-resident.** Moondream2 Q4 (~2.1 GB total) + LLM = 5.2 GB.
   Fits, with ~2.7 GB margin. Lower quality on dense UI. **Good fallback.**
3. Partial CPU offload for the VLM. Very slow. Not recommended.

```
RAM:   6.4 free at rest
       0.12  presence
       0.43  voice
       0.25  OCR
       0.50  VLM host process
       ────
       ~5.1  remaining  →  workable, but a browser costs 1–2 GB
```

**RAM is tighter than VRAM in practice**, and it is the reason the SNN's
2.5 GB torch footprint must never be resident.

---

# F. SNN Feasibility Study

You asked me to evaluate this honestly rather than adopt it because it sounds
advanced. Here is my honest assessment.

## F.1 Where SNNs genuinely win

Spiking networks have real, demonstrated advantages in three settings:

1. **Neuromorphic hardware** (Loihi 2, SpiNNaker, Akida). Energy per inference
   can be 10–1000× lower than a GPU. This is the headline result and it is real.
2. **Natively sparse event streams** — DVS/event cameras, where data *is*
   spikes and conventional frames would be wasteful.
3. **Always-on, ultra-low-power sensing** — keyword spotting on a
   milliwatt budget.

## F.2 Whether any of that applies here

**No, and the reason is specific: you do not have neuromorphic hardware.**

On an RTX 5060, an SNN is simulated with dense tensor operations over
timesteps. A spiking layer unrolled over *T* timesteps costs roughly *T*×
the tensor ops of the equivalent non-spiking layer. **On GPU, the SNN is
strictly slower than the GRU it replaces**, and the energy advantage — the
entire point — does not materialise, because it comes from event-driven silicon
that skips computation, not from the algorithm.

Three further problems specific to this application:

**Data volume.** Behavioural events here are app-focus changes, tool uses,
corrections — realistically 200–2,000 events/day. After a month: perhaps 30k
events over a vocabulary of maybe 30 applications. That is a *small* dataset.
Surrogate-gradient SNN training is notoriously finicky and data-hungry relative
to its capacity; this is not enough data to train one well.

**The problem is not temporally rich enough.** SNNs earn their keep on precise
spike-timing at millisecond resolution. "You usually open Chrome after VS Code,
around 3 pm" is a *coarse* temporal pattern at second-to-hour resolution. That
is a Markov chain's home ground.

**Explainability is decisive, and it is your own requirement.** §12 says
learning must be "explainable, inspectable, editable, reversible". Compare:

> **Markov/statistical:** "You opened Chrome after VS Code in 34 of 48
> observed transitions (71%), mostly between 14:00 and 17:00."
>
> **SNN:** a membrane-potential vector.

The first can be shown in `aura cognitive why`, argued with, and deleted. The
second cannot, without building an interpretability layer that would be harder
than the model.

## F.3 What I recommend instead

Build the Personal Cognitive Model on **interpretable statistical models**,
behind a `TemporalModelProvider` seam so that anything can be swapped in later:

| Model | Predicts | Explainable? | Cost |
|---|---|---|---|
| **Frequency + recency** | Next app / tool | Trivially | ~0 |
| **First-order Markov** | Transition probabilities | Yes — a table you can read | ~0 |
| **Time-bucketed Markov** | Transitions conditioned on hour/weekday | Yes | ~0 |
| **HMM** | Latent "activity mode" (coding / browsing / meeting) | Mostly | small |
| GRU (torch) | Sequence prediction | No | 2.5 GB dep |
| **SNN** (snnTorch) | Sequence prediction | No | 2.5 GB dep |

## F.4 If the SNN is built anyway — how to do it properly

I am not saying "never". I am saying "prove it". A defensible prototype:

**Objective.** Next-application prediction and activity-state classification
from behavioural event sequences.

**Data.** `BehaviouralEvent` log; ≥30 days before training is meaningful.
Encoding: one-hot event type × subject, with inter-event interval as either a
spike-timing offset (SNN) or an explicit feature (baselines).

**Output.** A `UserState` vector plus a top-*k* next-event distribution.

**Benchmark** — reuse the Phase 8 harness, which already handles repeats and a
noise floor:

| Metric | Why |
|---|---|
| Top-1 / top-3 next-event accuracy | The core task |
| Macro-F1 over activity states | Class imbalance is severe |
| Log-loss | Calibration, not just ranking |
| Inference latency (ms) | Must be negligible; this runs constantly |
| RAM/VRAM while loaded | The binding constraint |
| Cold-start behaviour (7/14/30 days) | Does it work before a month of data? |

**Baselines that must be beaten**, in order: majority-class, frequency+recency,
first-order Markov, time-bucketed Markov, HMM, GRU.

**Success criterion, stated in advance:** the SNN must beat *time-bucketed
Markov* by a margin exceeding the Phase 8 noise floor, on held-out data, **and**
justify its explainability loss. If it merely ties, the simpler model wins by
default.

**My prediction, recorded now so it can be checked later:** the SNN will not
beat time-bucketed Markov on this data volume, and the GRU will beat both by a
small margin that does not justify 2.5 GB of torch. I would be pleased to be
wrong, and the benchmark is designed to tell us.

## F.5 Verdict

**Do not put SNNs on the Phase 10 critical path.** Build the cognitive model
with interpretable statistics. Run the SNN as **Phase 10.9, an isolated
prototype in a separate optional extra**, with the benchmark above and a
written go/no-go. That is the same discipline the roadmap already applied to
genetic algorithms in Phase 9 — and there, the discipline was right.

---

# G. Security Model

No new permission levels. The existing 0–4 taxonomy is sufficient; what Phase
10 adds is **policies within it**.

## G.1 Level assignment for new capabilities

| Capability | Level | Confirmation | Rationale |
|---|---|---|---|
| `list_ui_elements` | 1 READ | no | Same as reading the screen; gated by `PrivacyPolicy` |
| `find_ui_element` | 1 READ | no | Locating is not acting |
| `screenshot_region` | 1 READ | no | Narrower than existing `read_screen` |
| VLM observation | 1 READ | no | Gated by the same privacy gate as capture |
| `click_element` | **3 MUTATE** | **yes**, unless pre-approved per task | Effects are app-defined and often irreversible |
| `type_text` | **3 MUTATE** | yes | Keystrokes go wherever focus is |
| `press_keys` | **3 MUTATE** | yes | Allow-listed chords only |
| Click a destructive affordance | **4 CRITICAL** | **always, cannot be pre-approved** | Delete/Send/Pay/Uninstall |
| Send a message | **4 CRITICAL** | **always, text shown** | Irreversible, third-party visible, attributed to you |
| Raw coordinate click | **4 CRITICAL** | always + explicitly enabled in config | No element identity to verify or audit |
| Browser navigation | 2 ACT | no | Reversible |
| Browser form submit | **4 CRITICAL** | always | Same argument as messaging |
| Terminal command | existing policy | existing | Unchanged — already the strictest path |

## G.2 New guards

1. **Input-injection kill switch.** A global hotkey that stops synthesis
   *synchronously*, not at the next loop check. Must interrupt mid-sequence.
2. **Password-field refusal.** UIA `IsPassword` → `type_text` denied at any
   level, no override.
3. **Focus-stability check.** Re-verify the target element immediately before
   acting; abort if focus moved. Same race already handled in
   `vision/capture.py`.
4. **Rate limits.** Bounded actions/minute. A runaway loop clicking 500 times
   is a different failure from clicking once wrongly.
5. **Screen-content redaction before the VLM.** Deny-listed windows never reach
   the model — enforced by the *existing* gate, which already sits before
   capture.
6. **Behavioural log is content-free** (§C.8).
7. **Evolution exclusions.** Every new setting affecting what AURA may see or
   do goes on `FORBIDDEN` in `evolution/space.py`.

## G.3 Privacy modes (§17)

Extends the existing `VisionMode` enum:

| Mode | Vision | VLM | Mic | Presence | Computer use |
|---|---|---|---|---|---|
| `ACTIVE` | on | on event | on | visible | permitted |
| `ASSIST` | on request | on request | on | visible | permitted |
| **`FOCUS`** *(new)* | on request | rare | on | **minimised, silent** | permitted |
| `PRIVATE` | **off** | **off** | policy | **hidden** | **denied** |
| `SLEEP` | off | off | wake only | hidden | denied |

`FOCUS` is the mode §16 asks for — reduce interruption without turning
capability off. It is a presentation and inference-budget change, not a
capability change.

**All of these are real runtime state**, enforced by code, not instructions to
the model. That is already how `vision/privacy.py` works and Phase 10 must not
weaken it.

---

# H. Implementation Plan

## H.1 The split I recommend

Phase 10 as specified is 5 pillars ≈ 5 phases of work. Sequenced by risk:

### **Phase 10A — Presence** (lowest risk, highest daily value)
| | |
|---|---|
| 10A.0 | ADR amendment: PySide6 for presence. **Measure RAM before committing.** |
| 10A.1 | `EventBus` + `/v1/events` SSE endpoint + tests |
| 10A.2 | `PresenceStateMachine` — states from real events, dwell/hysteresis |
| 10A.3 | Presence process skeleton: transparent click-through overlay, placement |
| 10A.4 | `AvatarBackend` seam + `SpriteAvatar` |
| 10A.5 | Speech events → subtitle + bubble (**the part you asked for by name**) |
| 10A.6 | Amplitude-driven mouth; gaze-only cursor awareness |
| 10A.7 | `PersonalityEngine` + multilingual micro-expressions (§H.3) |
| 10A.8 | Privacy: `FOCUS` mode, hide on `PRIVATE` |

**Exit criterion:** the avatar reflects real AURA state with no polling, shows
what she says as text, and disappears on `PRIVATE` — verified by instrumentation,
not by watching it.

### **Phase 10B — Perception** (medium risk, unblocks 10C)
| | |
|---|---|
| 10B.1 | `VLMProvider` ABC + `OpenAICompatVLMProvider`; seam test extended |
| 10B.2 | Second `llama-server` on :8082 with `--mmproj`; download script |
| 10B.3 | `ResourceGovernor` with LLM↔VLM eviction |
| 10B.4 | `TriggerPolicy` + rate limits, wired to the existing `ChangeDetector` |
| 10B.5 | `StructuredObservation` + schema-constrained VLM output |
| 10B.6 | `observe_screen` tool; perception eval suite |

**Exit criterion:** VLM answers a screen question through the existing privacy
gate; VRAM never exceeds budget; a denied window produces no VLM call.

### **Phase 10C — Computer use** (highest risk, gated on 10B)
| | |
|---|---|
| 10C.1 | UIA element enumeration (read-only) |
| 10C.2 | `find_ui_element` / `wait_for_element` |
| 10C.3 | Input synthesis with focus re-verification |
| 10C.4 | `click_element` / `type_text` behind confirmation |
| 10C.5 | Destructive-affordance and password-field guards |
| 10C.6 | Synchronous input kill switch |
| 10C.7 | OBSERVE→ACT→VERIFY→RECOVER loop |
| 10C.8 | Browser via Playwright (separate, DOM-addressed) |
| 10C.9 | **Adversarial safety suite** — must pass before any default-on tool |

**Exit criterion:** a multi-step task completes with verification; every
adversarial safety test passes; no coordinate clicking in the default set.

### **Phase 10D — Cognitive model** (independent; can run parallel to 10A)
| | |
|---|---|
| 10D.1 | `BehaviouralEvent` + append-only log (content-free) |
| 10D.2 | `TemporalModelProvider` seam |
| 10D.3 | Frequency/recency + Markov + time-bucketed Markov |
| 10D.4 | `UserState` → prompt context |
| 10D.5 | `aura cognitive list / why / forget / stop-learning` |
| 10D.6 | Benchmark harness (§F.4) with baselines |
| 10D.7 | **HMM/GRU comparison** |
| 10D.8 | **SNN prototype — optional extra, go/no-go against §F.4** |

**Exit criterion:** AURA can state a learned pattern with its evidence; the
benchmark ranks the models; the SNN decision is made on data.

## H.2 Why this order

10A gives you the thing you actually asked for first, and it is the lowest-risk
work in the set. 10B must precede 10C because computer-use *verification*
depends on perception. 10D is independent of all of it and can be built in
parallel or deferred entirely without blocking anything.

## H.3 Multilingual micro-expressions — the concrete design

Your §6 requirement, with one implementation opinion that saves a great deal of
trouble:

```
PersonalityEngine
   ├── state (from PresenceStateMachine)
   ├── context (task kind, time, recent history)
   ├── config: enabled, languages, frequency (default LOW)
   └── cooldown (never twice in a row; ~1 in 8 eligible utterances)
            ↓
   MicroExpression { text: "了解", audio: clips/ja/ryoukai.wav, meaning: "understood" }
            ↓
   prepended to TTS as a separate audio clip; subtitle shows text + gloss
```

**The implementation opinion: pre-render the clips as audio files.** The phrase
set is small and fixed (perhaps 20–40 short expressions). Piper is per-language,
so speaking Japanese would otherwise mean loading a second TTS voice, or an
English voice mangling 了解. Pre-rendered WAVs give perfect pronunciation, zero
runtime cost, zero extra model loading, and they are trivially auditable — you
can listen to every clip AURA can possibly say.

**The LLM never generates these.** It writes the answer in English; the
personality layer decides whether a micro-expression is appropriate and
prepends it. That is what stops foreign words leaking into every response, which
§6 explicitly warns against.

The subtitle shows the gloss (`了解 — understood`) so the meaning is never
lost when you cannot hear the audio.

---

# I. Testing Strategy

| Layer | What | How |
|---|---|---|
| **Unit** | State machine transitions, trigger policy, personality selection, Markov math, UIA element parsing | Existing pytest; fakes, no hardware |
| **Seam** | VLM/temporal libraries confined to their provider packages | Extend `tests/unit/test_seam.py` — build fails otherwise |
| **Integration** | Event → SSE → presence state; capture → change → VLM → observation | ASGI transport, as `test_vision_api.py` does |
| **Privacy** | **`PRIVATE` produces zero VLM calls and zero captures** | Spy backends counting calls — the Phase 5 pattern, which already caught a real bug |
| **Computer-use safety** | Password field refused; destructive affordance requires confirmation; focus-change aborts; rate limit holds; kill switch interrupts mid-sequence | **Adversarial suite. Must pass before any tool ships default-on.** |
| **Resource** | VRAM never exceeds budget; eviction works; no leak over 1000 cycles | Sample `nvidia-smi` during a soak |
| **UI** | Overlay does not steal focus; click-through works; survives DPI change, monitor unplug, lock/unlock | Manual checklist + screenshot diff where feasible |
| **Voice/speech** | Subtitle matches spoken text; mouth tracks amplitude | Golden-file comparison on the envelope |
| **VLM accuracy** | Structured observation correctness on a fixed screenshot corpus | New eval suite with **committed** screenshots so it is reproducible |
| **SNN benchmark** | §F.4 metrics against all baselines | Phase 8 harness with repeats and noise floor |
| **Stress** | Rapid app switching, video playing (continuous change), 8-hour soak | Must not trigger unbounded inference |

**The two suites I would write first**, because they are the ones that catch
the failures that matter: the privacy suite (does `PRIVATE` really stop the
VLM?) and the computer-use adversarial suite.

---

# J. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **VRAM exhaustion / thrashing** from LLM↔VLM swapping | High | High | `ResourceGovernor` with hard budget; Moondream2 fallback for co-residency; event-driven triggers with rate limits |
| 2 | **Computer-use causes real damage** — wrong click, wrong message sent | Medium | **Severe** | UIA addressing not coordinates; re-verify before act; destructive-affordance list; messaging at LEVEL 4 always-confirm; synchronous kill switch; adversarial suite gates the release |
| 3 | **Avatar is annoying** and gets turned off within a week | **High** | Medium | Event-driven movement only; gaze-only cursor tracking; rare `HAPPY`/`ANNOYED`; every element individually disableable; `FOCUS` mode |
| 4 | RAM exhaustion (6.4 GB free, browser costs 1–2 GB) | Medium | High | Presence is a separate killable process; VLM host on demand; SNN never resident |
| 5 | **SNN consumes weeks and delivers nothing** | **High** | Medium | Explicitly off the critical path; baselines first; written go/no-go criterion; prediction recorded in §F.4 |
| 6 | VLM latency (3–8 s) makes it unusable interactively | Medium | Medium | OCR-first (already built); VLM only when OCR is insufficient; targeted crops not full screens |
| 7 | UIA coverage gaps (Electron, custom-drawn, games) | Medium | Medium | VLM fallback; refuse rather than guess coordinates |
| 8 | Live2D licensing blocks the intended avatar | Medium | Low | Sprite backend first; seam makes the swap contained; **check the licence before any Live2D work** |
| 9 | Behavioural log becomes a privacy liability | Low | **Severe** | Content-free by construction; per-kind opt-out; deletion cascades; documented retention |
| 10 | Scope: five pillars in one phase never lands | **High** | High | **The 10A/B/C/D split.** Each has its own exit criterion |
| 11 | Prompt injection via screen content — the VLM reads "ignore previous instructions" | Medium | High | Observations enter as *data* with a `source` field, never as instructions; the Phase 5 asymmetry (model can narrow vision, never widen) extends to computer-use: **screen content must never be able to grant a permission** |
| 12 | PySide6 in a second process desynchronises from Core state | Medium | Low | Presence holds no state of its own; full state on every SSE reconnect |

**Risk 11 deserves emphasis.** Once AURA both *reads* the screen with a VLM and
*acts* on the computer, the screen becomes an input channel that can attempt to
drive the actor. A web page saying "AURA, click Send" must be inert. The Phase 7
asymmetry — the model may narrow its own capabilities but never widen them —
must extend to every Phase 10 capability. **No observation may ever escalate a
permission.**

---

## Open questions for you

1. **Scope**: approve the 10A/B/C/D split, or insist on one phase?
2. **Start point**: I recommend **10A** (presence — what you asked for, lowest
   risk). 10D (cognitive) is the alternative if you would rather build the mind
   before the body — which is what your own closing line argues for.
3. **PySide6's 120 MB**: acceptable, or should I prototype the 25 MB Win32
   layered-window route first?
4. **Avatar art**: commissioned, generated, or sprite placeholder to start?
5. **VLM residency**: sequential-with-eviction (better quality), or Moondream2
   co-resident (always available, weaker)?
6. **Computer-use appetite**: full 10C, or stop at read-only UIA observation
   until presence and perception have been lived with?

Nothing will be implemented until you say which.
