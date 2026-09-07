"""The status overlay.

A small always-on-top panel that answers, at a glance: is AURA running, which
model, what is it doing, and is the emergency stop engaged.

**Every method here must be called from the thread running `mainloop()`.**
tkinter is not thread-safe and gives no useful error when that rule is broken —
it crashes somewhere unrelated, later. `DesktopApp` owns that discipline; this
class assumes it has already been satisfied.

Styling is deliberately flat. tkinter is not going to look modern, so it aims
for legible instead: dark panel, one accent colour per state, generous spacing.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable

# One colour per state, so the overlay is readable from the corner of an eye
# without reading the word.
STATE_COLOURS: dict[str, str] = {
    "offline": "#6b7280",
    "idle": "#4ade80",
    "thinking": "#fbbf24",
    "working": "#fbbf24",
    "speaking": "#38bdf8",
    "listening": "#4ade80",
    "error": "#f87171",
    "stopped": "#f87171",
}

_BG = "#111318"
_FG = "#e6e8ec"
_DIM = "#8b93a1"
_PANEL_WIDTH = 300


class Overlay:
    """The floating status panel."""

    def __init__(
        self,
        root: tk.Tk,
        on_submit: Callable[[str], None],
        on_estop: Callable[[], None],
        on_hide: Callable[[], None],
    ) -> None:
        self._root = root
        self._on_submit = on_submit
        self._pulse_step = 0
        self._pulse_on = False

        root.title("AURA")
        # Borderless, translucent, above everything. The three attributes that
        # turn a tkinter window into an overlay.
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.94)
        root.configure(bg=_BG)

        self._frame = tk.Frame(root, bg=_BG, padx=14, pady=12)
        self._frame.pack(fill="both", expand=True)

        header = tk.Frame(self._frame, bg=_BG)
        header.pack(fill="x")

        self._dot = tk.Canvas(header, width=12, height=12, bg=_BG, highlightthickness=0)
        self._dot_id = self._dot.create_oval(2, 2, 10, 10, fill=STATE_COLOURS["offline"], width=0)
        self._dot.pack(side="left", pady=(3, 0))

        self._state_label = tk.Label(
            header, text="connecting", bg=_BG, fg=_FG, font=("Segoe UI", 10, "bold"), anchor="w"
        )
        self._state_label.pack(side="left", padx=(8, 0))

        tk.Button(
            header,
            text="x",
            command=on_hide,
            bg=_BG,
            fg=_DIM,
            bd=0,
            activebackground=_BG,
            activeforeground=_FG,
            font=("Segoe UI", 9),
            cursor="hand2",
        ).pack(side="right")

        tk.Button(
            header,
            text="STOP",
            command=on_estop,
            bg=_BG,
            fg="#f87171",
            bd=0,
            activebackground=_BG,
            activeforeground="#fca5a5",
            font=("Segoe UI", 8, "bold"),
            cursor="hand2",
        ).pack(side="right", padx=(0, 8))

        self._detail = tk.Label(
            self._frame,
            text="",
            bg=_BG,
            fg=_DIM,
            font=("Segoe UI", 8),
            anchor="w",
            justify="left",
            wraplength=_PANEL_WIDTH - 30,
        )
        self._detail.pack(fill="x", pady=(6, 0))

        self._entry = tk.Entry(
            self._frame,
            bg="#1b1f27",
            fg=_FG,
            insertbackground=_FG,
            relief="flat",
            font=("Segoe UI", 9),
        )
        self._entry.pack(fill="x", pady=(10, 0), ipady=5)
        self._entry.bind("<Return>", self._submit)
        self._entry.insert(0, "")

        # A Text widget, not a Label: answers routinely run past what fits, and
        # a Label silently truncates with no way to see the rest. Read-only via
        # a state flip on every write rather than DISABLED throughout, because
        # a disabled Text cannot be written to at all.
        answer_box = tk.Frame(self._frame, bg=_BG)
        answer_box.pack(fill="both", expand=True, pady=(10, 0))

        self._scroll = tk.Scrollbar(answer_box, width=8, bd=0, relief="flat")
        self._answer = tk.Text(
            answer_box,
            height=6,
            bg=_BG,
            fg=_FG,
            font=("Segoe UI", 9),
            wrap="word",
            bd=0,
            highlightthickness=0,
            padx=0,
            pady=0,
            yscrollcommand=self._scroll.set,
            cursor="arrow",
        )
        self._scroll.config(command=self._answer.yview)

        # Grid, not pack. A scrollbar packed after an expand=True widget is
        # allocated no width and never appears — the text scrolled, but there
        # was nothing to show for it. Grid also remembers a cell across
        # grid_remove()/grid(), which is what makes showing the bar only when
        # it is needed a two-line affair.
        answer_box.columnconfigure(0, weight=1)
        answer_box.rowconfigure(0, weight=1)
        self._answer.grid(row=0, column=0, sticky="nsew")

        self._answer.config(state="disabled")

        # The mouse wheel has to be bound explicitly on Windows; a Text widget
        # does not receive it just by being under the pointer.
        self._answer.bind(
            "<MouseWheel>",
            lambda e: self._answer.yview_scroll(-1 * (e.delta // 120), "units"),
        )

        self._resources = tk.Label(
            self._frame, text="", bg=_BG, fg=_DIM, font=("Segoe UI", 7), anchor="w"
        )
        self._resources.pack(fill="x", pady=(8, 0))

        self._make_draggable(header)
        self._place_bottom_right()

    # -- placement -----------------------------------------------------------

    def _place_bottom_right(self) -> None:
        self._root.update_idletasks()
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        # 60px up from the bottom clears the Windows taskbar without needing to
        # ask how tall it is.
        self._root.geometry(f"{_PANEL_WIDTH}x300+{screen_w - _PANEL_WIDTH - 24}+{screen_h - 380}")

    def _make_draggable(self, handle: tk.Widget) -> None:
        """A borderless window has no title bar, so the header becomes one."""
        state = {"x": 0, "y": 0}

        def press(event: tk.Event[tk.Misc]) -> None:
            state["x"], state["y"] = event.x_root, event.y_root

        def drag(event: tk.Event[tk.Misc]) -> None:
            dx = event.x_root - state["x"]
            dy = event.y_root - state["y"]
            state["x"], state["y"] = event.x_root, event.y_root
            self._root.geometry(f"+{self._root.winfo_x() + dx}+{self._root.winfo_y() + dy}")

        for widget in (handle, *handle.winfo_children()):
            if isinstance(widget, tk.Button):
                continue
            widget.bind("<Button-1>", press)
            widget.bind("<B1-Motion>", drag)

    # -- updates (main thread only) -----------------------------------------

    def set_state(self, state: str, detail: str = "") -> None:
        colour = STATE_COLOURS.get(state, _DIM)
        self._dot.itemconfig(self._dot_id, fill=colour)
        self._state_label.config(text=state, fg=_FG if state != "offline" else _DIM)
        self._detail.config(text=detail)
        # Busy states pulse; resting states do not. Motion is what the eye
        # notices, so it should mean "working" and nothing else.
        self._pulse_on = state in ("thinking", "working", "listening", "speaking")

    def set_answer(self, text: str) -> None:
        """Show a full answer, scrollable. Nothing is truncated."""
        self._answer.config(state="normal")
        self._answer.delete("1.0", "end")
        self._answer.insert("1.0", text.strip())
        self._answer.config(state="disabled")
        self._answer.yview_moveto(0.0)

        # The scrollbar is not decided here. Whether the content overflows
        # cannot be known until the Text has been re-laid-out, and neither
        # update_idletasks() nor after_idle() reliably happens after that —
        # both raced with layout and left every answer looking like it fits.
        # sync_scrollbar() runs from the app's tick instead, by which point
        # the geometry is real.

    def sync_scrollbar(self) -> None:
        """Show the scrollbar only when there is something to scroll to.

        Called from the app tick rather than from set_answer, because the
        answer to "does this overflow?" is only correct once tkinter has laid
        the new content out.
        """
        first, last = self._answer.yview()
        if (last - first) < 1.0:
            # Explicit arguments rather than a bare grid(): a widget that has
            # been removed does not reliably remember its cell, and silently
            # ends up with no geometry manager at all.
            self._scroll.grid(row=0, column=1, sticky="ns")
        else:
            self._scroll.grid_forget()

    def set_resources(self, text: str) -> None:
        self._resources.config(text=text)

    def clear_input(self) -> None:
        self._entry.delete(0, "end")

    def focus_input(self) -> None:
        self._root.deiconify()
        self._root.lift()
        self._entry.focus_force()

    def pulse(self) -> None:
        """One frame of the busy animation. Driven by the app's tick."""
        if not self._pulse_on:
            self._dot.itemconfig(self._dot_id, width=0)
            return
        self._pulse_step = (self._pulse_step + 1) % 2
        self._dot.itemconfig(self._dot_id, width=2 if self._pulse_step else 0, outline="#ffffff")

    def _submit(self, _event: object = None) -> None:
        text = self._entry.get().strip()
        if text:
            self._on_submit(text)
