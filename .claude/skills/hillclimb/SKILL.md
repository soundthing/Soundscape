---
name: hillclimb
description: Start, check or stop the soundthing hillclimb, a looping pool of agy and claude worker agents that tune the events component, run by an Opus 5.5 orchestrator that verifies and lands wins. Use for "/hillclimb", "start hillclimbing", "how's the hillclimb", "stop the hillclimb".
---

# /hillclimb

A detached driver process (`hc.py drive`) starts an orchestrator, `claude -p --model
claude-opus-5-5 --effort medium`, about once a minute, whenever something changed. Each run is
one tick of [orchestrator.md](orchestrator.md):
- verify and land accepted experiments on a `hillclimb/<timestamp>` branch;
- end workers whose experiment failed, and spawn replacements;
- clean up worktrees.

Workers are headless `agy` or `claude` processes that follow [worker.md](worker.md) in their
own worktree under `../.wt/hc-<id>`. All state is in `data/hillclimb/`. Everything mechanical
goes through `hc.py`. Run it from the main checkout:

```bash
.venv/Scripts/python.exe .claude/skills/hillclimb/hc.py <command>
```

## Arguments

- `/hillclimb`, or a roster like `/hillclimb 5 agy 5 claude`: start (below).
- `/hillclimb status`: run `hc.py status` and `hc.py results --last 20`, plus
  `tail -n 20 data/hillclimb/journal.md`, and summarise the best score, the wins landed, and
  the running workers.
- `/hillclimb stop`: run `hc.py stop`. It ends the driver and every worker and cleans up.
  Report the branch and the wins landed (see "When it's stopped").

## Start

1. **Ask how many agents**, unless the arguments already say. Use AskUserQuestion with a free-text
   answer, for example "5 agy 5 claude". Parse each group into
   `{"kind", "count", "model", "effort"}`:
   - **agy** defaults to Gemini 3.8 Flash, high: model `gemini-3.8-flash-high`. agy puts the
     effort in the model id, so "agy flash medium" means `gemini-3.8-flash-medium`. Run
     `agy models` for the valid ids (other models, such as `claude-opus-4-6-thinking`, are
     listed there).
   - **claude** defaults to `claude-opus-5-5` with effort `medium`. "claude sonnet high" means
     model `claude-sonnet-5-5`, effort `high`.
   - Use `null` for model or effort to get these defaults.
2. **Preflight.** Stop and report anything that fails:
   - `.venv/Scripts/python.exe` exists;
   - `agy --version` and `claude --version` work, for the kinds in the roster;
   - `git status --porcelain --untracked-files=no` is empty;
   - `.venv/Scripts/python.exe -m soundthing.harness rebaseline --status` shows cached clips > 0;
   - `data/hillclimb/STOP` is absent, or the user wants a fresh start;
   - no driver is already running (`driver_pid` in `data/hillclimb/state.json` is alive).
3. **Start:**
   ```bash
   .venv/Scripts/python.exe .claude/skills/hillclimb/hc.py start --roster '[{"kind":"agy","count":5,"model":null,"effort":null},{"kind":"claude","count":5,"model":null,"effort":null}]'
   ```
   This switches the main checkout to a new `hillclimb/<timestamp>` branch, unless it's
   already on one. It rebaselines the best on the clips cached so far and launches the
   driver, detached, so it survives this session. Workers start on the orchestrator's
   first tick.
4. **Tell the user:**
   - the branch;
   - how to watch: `/hillclimb status`, `data/hillclimb/journal.md`, `data/hillclimb/driver.log`;
   - that it loops until `/hillclimb stop`.

## When it's stopped

Wins are commits on the `hillclimb/...` branch, never on `main`. They were checked on the
dev clips cached at the time, not on the holdout, because promotion is the holdout gate.
Tell the user to:
- review the branch;
- once the holdout caches are built, check it against the holdout before merging to main
  (`promote`/`promote-code`, or an equivalent trusted holdout score of the branch vs `main`).

Merging into `main` is the user's call.

## agy and claude quirks (tested)

These are what `hc.py` and the briefs rely on. Keep them in mind when changing either.

- **Headless runs need `--dangerously-skip-permissions`.** Without it, agy `-p` denies every
  tool, prints the reason only on stderr, and still exits 0.
- **agy's shell tool is PowerShell, and file writes add a UTF-8 BOM.** Redirecting with
  `git diff > file` writes UTF-16LE with CRLF, which `git apply` rejects as "unrecognized
  input". Workers must use `git diff --output=FILE`. `hc.py exp` also decodes UTF-16 and BOM
  patches and normalises CRLF.
- **Bare `python` is the system 3.14, not the venv.** Briefs always give the venv's absolute
  path. Paths contain spaces ("Projects 2"), so quote them. Forward slashes work in both shells.
- **agy model ids carry the effort** (`gemini-3.8-flash-high`). `--effort` exists but isn't
  needed for Gemini.
- **agy moves long shell commands to the background.** Before the turn ends, it waits up to
  30 minutes for them ("root agent idle; waiting up to 30m0s..."), so a 1 to 3 minute
  experiment is fine.
- **agy reads AGENTS.md** in its cwd, and it can write outside its cwd once permissions are
  skipped.
- **Parallel agy and claude processes don't interfere.** `--output-format json` gives agy's
  `status`, `conversation_id`, `response` and claude's `result`, `session_id`.
- **Killing a worker:** `taskkill /T /F /PID <pid>` takes down agy or claude and its children,
  with nothing left over.
- **Trusted scoring runs the committed harness.** `run-experiment` scores the base branch's
  HEAD with the patch on top, so wins only count once they're committed to the base
  branch. `hc.py apply` does that.
