# Agent rules for `events/`

This directory is Person 1's sound event detection stage. The root
[AGENTS.md](../../../AGENTS.md) still applies in full; these rules add to it.

## Before you start

1. Read [plan.md](plan.md) for the roadmap and current status.
2. Read [references/](references/README.md): the stack, past decisions, known pitfalls and
   tuning results. Check `pitfalls.md` before touching model loading, scoring or
   post-processing.
3. Before editing files in any directory under `events/`, read every `AGENTS.md` from that
   directory up to this one.

## Keep the references up to date

Update `references/` in the **same commit** as the change that makes it stale:

- A model, library or runtime requirement changes → `stack.md`.
- A design choice is made, revisited or reversed → `decisions.md` (update the entry and
  say why; don't silently delete it).
- Something non-obvious breaks or surprises you → `pitfalls.md`. Remove entries whose
  cause is gone.
- An eval run informs a decision → `tuning-log.md`, with the numbers.

The references are **not a mirror of the code**. Record the why, the constraints, the
measured results and the traps. Don't describe functions, signatures, file layouts or
control flow, and don't copy docstrings. If reading the code answers it, leave it out.
Keep entries short.

Also keep [plan.md](plan.md) current: update the status and milestones when work lands
or the plan changes.

## Nested agent files

If a subdirectory of `events/` has something an agent must know **before touching it**
(a constraint, an ordering rule, a trap that isn't visible from the code), add an
`AGENTS.md` there, plus a `CLAUDE.md` containing only `@AGENTS.md`. Keep them short and
specific to that directory. Update them when the directory changes and delete them when
they no longer say anything useful. General knowledge goes in `references/`, not in nested
agent files.
