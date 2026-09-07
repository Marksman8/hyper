"""The approval dialog.

This is the reason the desktop shell exists. Until now the answer to "may I
write this file?" was *re-run the command with a flag* — a CLI shape, not an
assistant shape. Here it is a question with two buttons.

What it does **not** do is change who decides. The dialog gathers a human's
answer and passes it to the same `approve` list the CLI uses; the permission
gate in `tools/executor.py` is untouched, and nothing about a prettier prompt
makes an approval easier for the model to obtain.

Approving re-runs the task from the start rather than resuming a paused loop.
That repeats a little work — measured at one to three seconds for the tasks
this handles — and in exchange there is no long-lived suspended state on the
server, no resumption protocol, and no way for an approval to be applied to a
different call than the one the user saw. When tasks get long enough for the
repetition to hurt, that is the moment to build the pause/resume machinery, not
before.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont

_BG = "#111318"
_FG = "#e6e8ec"
_DIM = "#8b93a1"
_WARN = "#fbbf24"

# What each tool actually does, in plain language. The tool's own description is
# written for a model; a person approving an action deserves the consequence,
# not the API summary.
_PLAIN_ENGLISH: dict[str, str] = {
    "write_file": "create or overwrite a file on disk",
    "run_python": "execute Python code on your machine",
    "delete_file": "permanently delete a file",
    "open_application": "open an application on your desktop",
}


class ApprovalDialog:
    """A modal asking whether AURA may use a tool it has been refused."""

    def __init__(
        self,
        parent: tk.Tk,
        tools: list[str],
        task: str,
        explanation: str = "",
    ) -> None:
        self.approved = False

        self._win = tk.Toplevel(parent)
        self._win.title("AURA needs your approval")
        self._win.configure(bg=_BG)
        self._win.attributes("-topmost", True)
        self._win.resizable(False, False)
        # Modal: the question is blocking, so the window should be too.
        self._win.transient(parent)
        self._win.grab_set()

        body = tk.Frame(self._win, bg=_BG, padx=18, pady=16)
        body.pack(fill="both", expand=True)

        tk.Label(
            body,
            text="AURA wants to do something that needs your permission",
            bg=_BG,
            fg=_WARN,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            justify="left",
            wraplength=380,
        ).pack(fill="x")

        tk.Label(
            body,
            text=f"Task: {task}",
            bg=_BG,
            fg=_DIM,
            font=("Segoe UI", 8),
            anchor="w",
            justify="left",
            wraplength=380,
        ).pack(fill="x", pady=(8, 0))

        listing = tk.Frame(body, bg="#1b1f27", padx=12, pady=10)
        listing.pack(fill="x", pady=(12, 0))
        for name in tools:
            plain = _PLAIN_ENGLISH.get(name, "perform an action outside the read-only set")
            row = tk.Frame(listing, bg="#1b1f27")
            row.pack(fill="x")
            tk.Label(
                row,
                text=name,
                bg="#1b1f27",
                fg=_FG,
                font=("Consolas", 9, "bold"),
                anchor="w",
            ).pack(side="left")
            tk.Label(
                row,
                text=f"  -  {plain}",
                bg="#1b1f27",
                fg=_DIM,
                font=("Segoe UI", 8),
                anchor="w",
            ).pack(side="left")

        if explanation:
            tk.Label(
                body,
                text=explanation,
                bg=_BG,
                fg=_DIM,
                font=("Segoe UI", 8),
                anchor="w",
                justify="left",
                wraplength=380,
            ).pack(fill="x", pady=(10, 0))

        tk.Label(
            body,
            text=("Approving allows this for this task only. The task runs again from the start."),
            bg=_BG,
            fg=_DIM,
            font=("Segoe UI", 7),
            anchor="w",
            justify="left",
            wraplength=380,
        ).pack(fill="x", pady=(12, 0))

        buttons = tk.Frame(body, bg=_BG)
        buttons.pack(fill="x", pady=(14, 0))

        bold = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        # Deny is the wider, leftmost, default-focused button. The safe answer
        # should be the easy one, and Escape should always mean no.
        deny = tk.Button(
            buttons,
            text="Don't allow",
            command=self._deny,
            bg="#272c36",
            fg=_FG,
            bd=0,
            padx=18,
            pady=6,
            activebackground="#333a47",
            activeforeground=_FG,
            font=bold,
            cursor="hand2",
        )
        deny.pack(side="left")

        tk.Button(
            buttons,
            text="Allow once",
            command=self._approve,
            bg=_WARN,
            fg="#111318",
            bd=0,
            padx=18,
            pady=6,
            activebackground="#fcd34d",
            activeforeground="#111318",
            font=bold,
            cursor="hand2",
        ).pack(side="right")

        self._win.bind("<Escape>", lambda _e: self._deny())
        # Closing the window is a refusal, not a dismissal that leaves the
        # question open.
        self._win.protocol("WM_DELETE_WINDOW", self._deny)
        deny.focus_set()

        self._centre(parent)

    def _centre(self, parent: tk.Tk) -> None:
        self._win.update_idletasks()
        w, h = self._win.winfo_width(), self._win.winfo_height()
        x = parent.winfo_screenwidth() // 2 - w // 2
        y = parent.winfo_screenheight() // 3 - h // 2
        self._win.geometry(f"+{x}+{y}")

    def _approve(self) -> None:
        self.approved = True
        self._win.destroy()

    def _deny(self) -> None:
        self.approved = False
        self._win.destroy()

    def ask(self) -> bool:
        """Block until answered. Returns True only on an explicit approval."""
        self._win.wait_window()
        return self.approved
