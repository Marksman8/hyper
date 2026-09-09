# ADR-0002: tkinter for the desktop shell

Status: **accepted** · 2026-09-06 · Deferred from the v0.1 architecture doc

The v0.1 architecture listed this as an open decision, to be made "when there
is a working core to build against; RAM cost measured, not guessed." Here is
the measurement.

---

## Context

Spec §13 asks for a desktop presence: transparent overlay, status indicators,
current task, notifications, system tray, emergency stop — and says explicitly
that it "should NOT consume unnecessary resources."

That last clause is the constraint that decides this. Free RAM on this machine
has been observed between **0.6 GB and 5.8 GB** depending on what else is
running. A UI that costs 120 MB is a UI that competes with Whisper.

## Options

### 1. tkinter — **chosen**

Measured on this machine, same process shape as the real overlay (borderless,
alpha-blended, always-on-top):

```
baseline python      17.0 MB
+ tkinter window     29.9 MB   (delta +13.0 MB)
```

- **In the standard library.** No new dependency for the window itself, which
  spec §30 asks for directly.
- 13 MB is close enough to free that the overlay can stay open permanently
  without anyone thinking about it.
- Does everything §13 needs: `overrideredirect` for a borderless window,
  `-alpha` for transparency, `-topmost` for always-on-top.

The honest cost: tkinter looks like tkinter. A flat dark panel with good
spacing is achievable; a polished modern UI is not. For a status overlay that
is mostly glanced at, that trade is worth 100 MB.

### 2. PySide6 (Qt) — rejected

The better toolkit by a distance: real styling, proper compositing, a mature
widget set. Rejected on cost. The wheel chain (PySide6 → PySide6-Essentials →
shiboken6) is a few hundred MB installed and roughly **120 MB resident** — an
order of magnitude more than tkinter for a window that displays six lines of
text.

Revisit if the overlay grows into something people look *at* rather than
*past*.

### 3. Tauri — rejected

Smallest binary of the three and the best-looking result, but it adds a Rust
toolchain and a second build system to a Python project, and the IPC to AURA
Core would be a third process boundary. Wrong shape for the size of the thing.

## Decision

**tkinter** for the window, plus `pystray` (≈50 KB) and `Pillow` (7 MB
download) for the system tray icon, as an optional `desktop` extra so a
headless install pays nothing.

## Consequences

**Good**
- The overlay is affordable enough to leave running.
- No new toolchain, no second build system.
- `uv sync` without `--extra desktop` gets a server with no GUI dependencies.

**Bad**
- tkinter's threading model is unforgiving: all widget calls must happen on the
  thread running `mainloop()`. HTTP therefore runs on a worker thread and
  results are marshalled back through a queue drained by `root.after()`. That
  pattern is written down in `desktop/app.py` because getting it wrong produces
  crashes that look random.
- Styling is limited. Accepted.

## Revisit when

The overlay becomes something the user interacts with for minutes at a time
rather than glances at, or a feature needs compositing tkinter cannot do.

---

## Revisited: 2026-09-09 (Phase 10A.0)

Both conditions above were met by the Phase 10 avatar, which is looked at
continuously and needs per-pixel alpha that tkinter cannot provide -- it offers
only chroma-key `-transparentcolor`, which fringes on every anti-aliased edge.

**This ADR still stands for the status/approval shell.** The avatar moved to
PySide6-Essentials in a separate process; see **ADR-0003**.

Worth recording: the 120 MB figure this ADR used for PySide6 was an estimate,
and it was wrong. Measured, the same window shape costs **50.6 MB** total --
about 21 MB more than the tkinter shell, not 90 MB more. The decision made here
was still right for a status panel; the number it was made against was too
pessimistic.
