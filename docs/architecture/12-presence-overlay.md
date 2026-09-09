# The presence overlay — 10A.5 to 10A.11

Status: **implemented** · Date: 2026-09-09

---

## 1. What it is

A separate process that draws AURA on the desktop and does nothing else.

```
AURA Core ──/v1/events──▶ CoreEventClient ──Qt signal──▶ PresenceController
                                                              │
                                        ┌─────────────────────┼──────────────────┐
                                        ▼                     ▼                  ▼
                                 PresenceOverlay      PlacementPolicy    AvatarBackend
                                 (frameless, masked)  (four triggers)    (silhouette)
```

**No reasoning lives here.** Every decision about what state AURA is in was
made by `core/presence.py` and arrived as a `PRESENCE_STATE` event. This
process renders the answer. Kill it and AURA loses a face and nothing else —
verified: Core keeps serving, the bus keeps publishing, the state machine keeps
resolving.

Two tests hold that line: the package may not import the orchestrator, the
agent loop, the tool layer or memory; and no file in it may reference
`ToolExecutor`, `/v1/tools` or `/v1/approvals`.

## 2. Not interfering with the desktop

Three specific flags, not an emergent effect:

| Property | Mechanism |
|---|---|
| Not in taskbar or Alt-Tab | `Qt.Tool` |
| Never steals focus | `WA_ShowWithoutActivating` |
| Clicks pass through | `setMask()` — the window does not exist outside the mask |
| Per-pixel alpha | `WA_TranslucentBackground` |

The mask is a rectangle around the avatar rather than its exact silhouette. A
per-pixel input region would be more precise and materially harder to get
right; the cost is a small transparent margin that is draggable, which is what
a user expects of this kind of overlay anyway.

## 3. Movement — mostly refusal

**The four triggers, and nothing else:**

1. foreground application changed
2. the user dragged her
3. a state transition that needs to be seen — `WARNING`, `ERROR`, `WAITING`
4. the anchor became unusable — display unplugged, resolution changed

Plus a **3-second rate limit**: two applications fighting over focus must not
turn the avatar into a metronome. A burst of 60 focus changes produces exactly
one move (tested).

Two deliberate exemptions: a **user drag** is never rate-limited, because
refusing to follow a drag feels broken; and an **invalid anchor** bypasses it,
because staying put is not an option when "put" is a monitor that no longer
exists.

**Cursor awareness is gaze-only and off by default.** The eyes track; the body
never does. Continuous body-following is the fastest way to make a desktop
companion intolerable.

She sits in one of a small set of **anchor slots** rather than anywhere: an
avatar that can be anywhere ends up somewhere unhelpful, and a bounded set is
something a person can predict.

## 4. The avatar seam

`AvatarBackend` is the same discipline as `LLMProvider` and `OCRProvider`.

```python
info()      -> AvatarInfo
set_state(state, *, immediate)
advance(dt_ms) -> bool      # True only if something actually moved
paint(painter, w, h)        # untyped painter: the seam must not import Qt
set_mouth(openness)         # 10B
look_at(dx, dy)             # gaze
```

`advance()` returning False is what makes an idle overlay free: the paint timer
*asks* whether anything changed, and does nothing when the answer is no.

`paint` takes an untyped painter deliberately. Typing it as `QPainter` would
make the interface — and everything that imports it — depend on Qt.

### The placeholder is drawn, not drawn *from assets*

`SilhouetteAvatar` renders a professional silhouette with vector primitives:
head, shoulders, collar lines, eyes, and a state-coloured accent ring. There
are no PNGs to version, license or half-commit to, and the real AURA character
later is a **new backend**, not an edit to this file.

State is expressed by **colour and one animated element**, never by a different
pose. A placeholder that implied a character design would quietly become the
design.

Most states do not animate at all — only THINKING, WORKING, LISTENING, WAITING
and ERROR have a pulse. A companion that is always moving is one that gets
closed.

## 5. Privacy and focus

| Mode | Behaviour |
|---|---|
| `PRIVATE` | Hidden. **No cross-fade** — a graceful 260 ms transition means AURA is still on screen a quarter-second after being told to stop |
| `SLEEP` | Hidden, same mechanism |
| `FOCUS` | Dimmed to 45% opacity. Focus reduces intrusion; it does not turn presence off |
| Core unreachable | Dimmed to 35%. Showing a confident IDLE avatar would be claiming to know something we do not |

## 6. Connection handling

`CoreEventClient` is a `QThread`. Qt signals across threads are queued and
delivered on the GUI thread — a genuine advantage over the tkinter shell, where
a cross-thread widget call corrupts Tcl state and crashes somewhere unrelated
later.

- **Reconnects with `Last-Event-ID`**, so a brief drop does not lose the state.
- **Lag means resync.** On the terminal `subscriber_lagged` frame it reconnects
  *without* the resume id and takes a fresh snapshot. Reconnecting with the
  stale id would silently continue with a hole — the bus contract's rule 2 for
  consumers.
- Read timeout 45 s, comfortably above Core's 15 s heartbeat.

## 7. Measured

Live, on the running stack:

| Condition | RSS | CPU |
|---|---|---|
| Visible, idle | **15.3 MB** | **0.000%** of total |
| Hidden (PRIVATE) | 21.5 MB | 0.000% |
| Animating, 62 fps (10A.0 bench) | 50.6 MB peak | **0.23%** of total |

RSS varies with Windows working-set trimming: 50.6 MB was measured on a fresh,
continuously-animating process in 10A.0; 15 MB is the trimmed steady state. The
honest range is **15–51 MB**, against a 150 MB budget.

The animating CPU figure comes from the 10A.0 benchmark rather than the live
run — the live sample window landed after the turn had finished, so it measured
idle. Reported as such rather than as a better number than was actually taken.

## 8. Verified live

| Check | Result |
|---|---|
| Launches and connects to Core | ✅ `presence.connected` |
| Reflects real AURA events | ✅ `private → idle → listening → thinking → success → idle` |
| PRIVATE hides her | ✅ screenshot: gone entirely |
| Does not steal focus | ✅ typing continued uninterrupted |
| Clicks pass through | ✅ terminal remained usable underneath |
| Survives Core restart | ✅ reconnects and resumes |

## 9. Known limitations

1. **Subtitle is static.** `set_subtitle()` exists and renders, but nothing
   drives it — `SPEECH_*` events are 10B.
2. ~~No system tray icon.~~ **Closed in 10A.12** — show/hide, focus toggle,
   quit. Hiding is local presentation only and deliberately does not touch
   privacy mode: conflating them would let a cosmetic click silently change a
   security setting. Showing is refused while PRIVATE or SLEEP.
3. ~~`APP_CHANGED` is never published.~~ **Closed in 10A.12** —
   `vision/watcher.py` polls at 1 Hz, carries the process name only, and stops
   entirely in PRIVATE/SLEEP rather than polling and withholding.
4. ~~`FOCUS_MODE` is never published.~~ **Closed in 10A.12** — `POST /v1/focus`,
   audited, in the snapshot. Set from the tray by authenticated POST, never by
   publishing to the bus.
5. **The mask is rectangular**, so a small transparent margin around the avatar
   is draggable rather than click-through.
6. **Single monitor assumed for placement.** `QApplication.primaryScreen()` is
   used throughout; a second display is not an anchor option.
7. **Anchor is a CLI flag, not persisted.** A drag pins her for the session and
   is forgotten on restart.
8. **No DPI-change test.** The flags are correct in principle; the behaviour
   has not been verified across a live DPI switch.
