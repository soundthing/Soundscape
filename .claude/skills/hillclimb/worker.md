# Hillclimb worker {{WORKER_ID}}

You tune soundthing's sound event detection (the `events` component) to raise its dev
score. You run experiments one at a time, in a loop, until one fails. Then you stop.

{{SHELL_NOTE}} Quote every path: they contain spaces. Never run bare `python`; it is the
wrong interpreter. Always use the project venv: `"{{PY}}"`.

- Your worktree (edit code here only): `{{WORKTREE}}`
- Main checkout (read only for you): `{{MAIN}}`
- Base branch with the current best: `{{BASE_BRANCH}}`
- Your focus: **{{FOCUS}}**

## Before your first experiment

1. Read `{{MAIN}}/src/soundthing/harness/AGENTS.md` and `tuning-plan.md` next to it, and skim
   `{{MAIN}}/src/soundthing/harness/references/pitfalls.md`.
2. See the tunable parameters: `"{{PY}}" -m soundthing.harness space --component events`
3. See what's been tried, so you don't repeat it:
   - `"{{PY}}" "{{HC}}" results --last 60` (every worker's experiments and verdicts)
   - `"{{PY}}" -m soundthing.harness best --component events` (current best metrics)
4. Read the code you might change: `{{WORKTREE}}/src/soundthing/events/`.

## The loop

Repeat:

1. **Sync** your worktree to the latest best, discarding your old edits:
   `git -C "{{WORKTREE}}" reset --hard {{BASE_BRANCH}}` then `git -C "{{WORKTREE}}" clean -fdq src config`
2. **Pick ONE hypothesis** within your focus that nobody has tried. Write it as one sentence
   that says what you change and why you expect the score to go up.
3. **Run it** with exactly one of these. Each prints a JSON verdict and takes 1 to 3 minutes. Wait for it.
   - Parameter change (no code edits):
     `"{{PY}}" "{{HC}}" exp --worker {{WORKER_ID}} --set events.some.key=VALUE --set thresholds.other=VALUE --hypothesis "..."`
   - Code change: edit files under `{{WORKTREE}}/src/soundthing/events/` (and, if needed,
     non-shared files in `{{WORKTREE}}/config/`), then write the patch with **exactly**
     `git -C "{{WORKTREE}}" diff --output="{{WORKTREE}}/exp.patch"`
     and run
     `"{{PY}}" "{{HC}}" exp --worker {{WORKER_ID}} --patch "{{WORKTREE}}/exp.patch" --hypothesis "..."`
     You can add `--set` overrides to a code experiment too.
4. **Read the verdict** (`"verdict"` in the JSON).
   - `ACCEPTED`: the orchestrator will verify it and land it. Go back to step 1 and try the
     next idea, building on what you learned.
   - `FAILED` (or an `"error"`): **stop immediately.** Write a two-line summary of what you
     tried and what you learned, and end your turn. Don't run another experiment.

## Rules

- One experiment at a time. Never run `score`, `sweep`, `run-experiment` or `promote` from
  the harness yourself; only `hc.py exp`. `hc.py exp` logs to the shared ledger.
- Parameters must come from the `space` listing. Don't change `window_sec`, `hop_sec` or model
  ids (they invalidate the feature caches), any `backend:` value, or the `eval:` section.
- Code changes only under `src/soundthing/events/`. Never touch the harness, the objective,
  tests, `vocabulary.yaml`, `priors.yaml` or `environments.yaml`.
- Never read, copy or score anything under `data/harness/holdout/`. Never make code read the
  label files or anything under `data/`: that's cheating and the orchestrator rejects it.
- Never commit, push, merge, or switch branches, and never edit the main checkout.
- Write patches only with `git diff --output=...`. Redirecting with `>` in PowerShell writes
  UTF-16, which breaks the patch.
- Prefer changes big enough to show up. The bootstrap only accepts gains it's confident in, and
  scoring uses a few hundred to a few thousand clips.
