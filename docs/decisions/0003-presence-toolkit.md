# ADR-0003: PySide6 for the presence overlay

Status: **accepted** · 2026-09-09 · Phase 10A.0

Supersedes ADR-0002 **for the avatar only**. ADR-0002 stands unchanged for the
status/approval shell.

---

## Context

ADR-0002 chose tkinter for the desktop shell on a measured argument: 13 MB
against PySide6's estimated 120 MB, on a machine where free RAM had been
observed between 0.6 GB and 5.8 GB. That reasoning was correct for a status
panel.

It also wrote down the condition for revisiting:

> **Revisit when** the overlay becomes something the user interacts with for
> minutes at a time rather than glances at, or a feature needs compositing
> tkinter cannot do.

Phase 10A is both of those. The avatar is looked at continuously, and it needs
compositing tkinter genuinely cannot do:

- tkinter offers only `-transparentcolor`, a **chroma key**. Any anti-aliased
  edge blends toward the key colour and fringes visibly. There is no per-pixel
  alpha.
- No compositing of overlapping translucent elements (avatar + speech bubble).
- No animation primitives; smooth transitions must be hand-rolled on `after()`.

This is not a preference. A chroma-keyed avatar looks broken on any background
that is not flat.

## Measurement

The estimate in the Phase 10 design was ~120 MB. **That estimate was wrong.**
Measured on this machine, same window shape as the real overlay — frameless,
`WA_TranslucentBackground`, always-on-top, `Qt.Tool`, click-through:

```
baseline interpreter                            21.2 MB
+ import PySide6 (QtCore/QtGui/QtWidgets)       37.8 MB   (+16.6)
+ QApplication                                  45.9 MB   (+8.1)
+ overlay widget constructed                    46.2 MB   (+0.3)
+ shown (translucent, topmost, click-through)   50.0 MB   (+3.8)
+ after 5 s animating at 62 fps                 50.9 MB   (+0.9)
+ after 5 s idle                                50.6 MB   (-0.3)

TOTAL process RSS                               50.6 MB
Qt delta over the bare interpreter              29.7 MB
```

CPU, measured with a real Qt event loop (the first attempt spun
`processEvents()` in a while-loop and measured busy-waiting instead — the
numbers below are from the corrected run):

```
ANIMATING   62.4 fps   5.6% of one core   0.23% of 24 cores
IDLE        0 repaints  0.0%              0.00%
```

**50.6 MB, not 120 MB.** The difference is `PySide6-Essentials`, which ships
QtCore/QtGui/QtWidgets without QtWebEngine, Qt3D, Multimedia, Charts and the
rest of the full distribution.

For comparison, against ADR-0002's own numbers:

| | RSS | Per-pixel alpha | Animation |
|---|---|---|---|
| tkinter overlay | ~29.9 MB | ✗ chroma key | poor |
| **PySide6-Essentials overlay** | **50.6 MB** | **✓** | **✓** |

The real cost of the avatar is **~21 MB more than the tkinter shell**, not
~90 MB more. That is roughly 0.3% of this machine's 15.7 GB, and 0.8% of the
6.4 GB free at rest.

## Decision

**Use PySide6-Essentials for the presence overlay**, in a **separate process**
from AURA Core, behind the `presence` extra.

- `uv sync` without `--extra presence` still gets a server with no GUI
  dependency. Core remains a 69 MB process that starts fast.
- The presence process holds **no reasoning and no state of its own**. Killing
  it costs AURA a face and nothing else.
- The `AvatarBackend` seam (Phase 10 design §D.2) means the renderer itself —
  sprite, Live2D, 3D — is replaceable without touching this decision.

**The Win32 layered-window fallback is not built.** It would save ~25 MB at the
cost of writing a compositor by hand, and 25 MB is not worth that on the
measured numbers. Recorded as a known option, not a planned one.

## Consequences

**Good**
- Per-pixel alpha, so the avatar composites correctly over any background.
- Idle costs literally nothing measurable: repaint-on-change means 0 repaints
  and 0.0% CPU when the state is static.
- Qt's threading model is friendlier than tkinter's — signals/slots marshal to
  the GUI thread properly, rather than tkinter's "touch a widget off-thread and
  crash somewhere unrelated later".
- Animation, easing and transforms are built in.

**Bad**
- +21 MB over the tkinter shell, and a second GUI toolkit in the repository
  while both exist.
- PySide6 is LGPL. Fine for this project; worth knowing if it is ever
  distributed as a closed binary.
- A larger install (~100 MB on disk) behind an optional extra.

**A trap this measurement exposed.** `uv sync --extra presence` *removes* extras
not named on the command line — it syncs the environment to exactly what is
asked for. The voice, vision and desktop extras were silently uninstalled and
had to be restored with `uv sync --all-extras`. Use `--all-extras` on this
project, or name every extra.

## Revisit when

The avatar needs 3D/VRM rendering (which costs VRAM, the scarcest resource
here), or the presence process's RSS grows past ~150 MB, at which point the
Win32 layered-window route becomes worth its complexity.
