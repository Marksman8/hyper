# Phase 3 — Tools

Status: **in progress** · Date: 2026-09-05

---

## 1. What Phase 3 is

The layer that lets AURA *do* things instead of only talking, and the controls
that make that safe.

Phase 3 builds the tool layer and its permission system. It does **not** hand
the model the keys — wiring tools into the model's own reasoning loop is Phase
4. Building it in this order is deliberate: a permission system retrofitted
around code that already assumed it was absent has a hole at every call site
that forgot to ask.

## 2. The pipeline

Every tool call, without exception, takes this path:

```
  caller (API now; the agent loop in Phase 4)
      |
      v
  ToolRegistry        does this tool exist? what is its schema?
      |
      v
  Validation          arguments against the schema, then the tool's own checks
      |
      v
  Workspace guard     is this path / command inside what AURA may touch?
      |
      v
  PermissionPolicy    LEVEL 0-4: allow / confirm / deny
      |
      v
  Execution           in a subprocess where it can be, with a hard timeout
      |
      v
  Audit               always, including the denials and the failures
      |
      v
  ToolResult          structured; errors are values, not exceptions
```

The order matters. Validation before permission, because a malformed call
should be rejected on its own terms rather than prompting the user to approve
something incoherent. Permission before execution, obviously. Audit after
everything, including — especially — the paths that refused.

## 3. Tools are values, not functions

A tool is an object carrying a `ToolSpec`: name, description, JSON schema,
permission level, timeout, and whether it can be undone. That makes the tool
surface *data*, which matters for three things Phase 4 depends on:

- The schema is what gets sent to the model. One definition, no drift between
  what the model is told and what actually runs.
- The permission level is declared, not inferred. A tool cannot be written
  without answering "how dangerous is this?".
- The registry can be filtered per session, so a restricted context genuinely
  cannot see the tools it may not use.

## 4. Errors are return values

`ToolResult.ok` is False and `error` is populated; nothing raises across the
tool boundary. Two reasons:

1. In Phase 4 the model reads the failure and adapts. An exception that
   propagates out of the loop ends the task; a structured error is an
   observation the model can act on.
2. A tool that raises past the executor skips the audit record. Errors as
   values means the audit path is unconditional.

## 5. The workspace guard

The single most important piece of Phase 3, because path handling is where
filesystem tools actually go wrong.

- **Allow-list, not deny-list.** `tools.workspace_roots` lists what AURA may
  touch. Everything else is refused. A deny-list is a promise that you thought
  of every dangerous path in advance.
- **Resolve before checking.** `Path.resolve()` collapses `..` and follows
  symlinks, so containment is checked against the real target. Checking the
  string before resolution is the classic traversal hole: `project/../../.ssh`
  looks contained and is not.
- **Deny patterns run after containment**, for things that are inside the
  workspace but still must not be read: `.env`, `var/secrets/**`, `**/*.pem`,
  `**/id_rsa*`. AURA's own API tokens live in the workspace, and a "read this
  file" tool that can hand them to a model is a credential leak.
- **Windows case-insensitivity is real.** Containment compares normalised
  case, or `C:\Users\...` versus `c:\users\...` becomes an escape.

## 6. Permission levels, now with something to guard

Defined in v0.1, assigned here:

| Level | Meaning | Tools in this phase |
|---|---|---|
| 0 OBSERVE | AURA's own state | — |
| 1 READ | read the user's data | `read_file`, `list_directory`, `file_info` |
| 2 ACT | reversible side effects | `write_file`, `run_python` |
| 3 MUTATE | hard to reverse | `delete_file` (not yet built) |
| 4 CRITICAL | system-level, irreversible | — |

The default policy allows up to LEVEL 1 autonomously and requires explicit
confirmation above it. So `write_file` and `run_python` exist, are tested, and
**refuse to run** unless the caller passes an explicit confirmation — which in
Phase 3 means a human, and in Phase 4 means a human via the desktop UI.

## 7. Executing Python

`run_python` runs a **subprocess**, never `exec()` in-process. In-process
execution shares AURA's memory, its open database handle, and its API tokens
with whatever the model wrote.

Honest about what this is not: a subprocess with a timeout and a working
directory is **isolation, not a sandbox**. It runs as the user, can reach the
network, and can read anything the user can. Real sandboxing on Windows means
a container or a job object with restricted tokens, which is its own project.
Until then `run_python` sits at LEVEL 2 behind confirmation, and the docstring
says plainly what it does not protect against.

## 8. Out of scope for Phase 3

Browser control · VS Code integration · git · database · clipboard · keyboard
and mouse · window management · the model choosing tools (Phase 4).
