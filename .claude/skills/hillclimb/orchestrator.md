# Hillclimb orchestrator: one tick

You run soundthing's hillclimb: a pool of worker agents that each test ideas for the
`events` component. A driver loop starts you every minute or so with a fresh context. Each
time, you do **one tick**: look at the state, act, write one journal line, and end your turn.
The driver keeps looping until the user stops it, so never stop it or end workers to "finish".

All commands run from the main checkout with the venv. In bash:

```bash
PY=".venv/Scripts/python.exe"; HC=".claude/skills/hillclimb/hc.py"
```

## 1. Read the state

- `$PY $HC status`: base branch, current best, clip set (cached vs active), running workers,
  each worker's last result and `unreviewed_wins`.
- `tail -n 40 data/hillclimb/journal.md`: your notes from earlier ticks.
- `$PY $HC results --last 60`: recent experiments across workers. Use it to steer focus.
- `data/hillclimb/state.json` → `roster`: how many workers of each kind, model and effort.

## 2. Verify and land wins (at most two per tick, oldest first)

For each unreviewed win (`verdict: ACCEPTED`, no `reviewed`):

1. **Review it.** For a code change, read the saved patch (`patch_file` in the result). Reject it
   with `$PY $HC reject <worker> <run_id> --reason "..."` if it:
   - reads labels, anything under `data/`, or the holdout;
   - special-cases clip ids;
   - edits outside `src/soundthing/events/` or non-shared `config/`;
   - games the objective instead of improving detection;
   - is broken or nonsense.
   For parameters, check that the values are sane.
2. **Verify and land it:** `$PY $HC apply <worker> <run_id>`. This re-runs the experiment
   against the current best on the current base. Only if it's accepted again does it commit
   the change to the base branch and rebaseline the best. A win that fails verification is
   marked rejected; that's normal when another win landed first.
3. Only one `apply` at a time, and never edit or commit in the main checkout yourself.

## 3. Replace workers

A worker is done when its last verdict is `FAILED`, or its process isn't `alive`. For each:

1. `$PY $HC kill <worker> --reason "<why>"`. This ends the process and removes its worktree.
2. Spawn a replacement of the **same kind, model and effort** so the pool matches the roster:
   `$PY $HC spawn --kind <agy|claude> [--model M] [--effort E] --focus "<focus>"`.
   Leave out `--model`/`--effort` for the roster defaults (null in the roster).

A worker whose last verdict is `ACCEPTED` and that is still alive keeps running.

If fewer workers are running than the roster asks for (for example on the first tick), spawn
the missing ones.

**Choosing a focus.** Give every worker a different, concrete focus, and steer away from
directions that keep failing in `results`. Examples:
- per-class thresholds for frequent classes, or for safety classes;
- postprocess smoothing, merging and min-duration;
- VAD and speech suppression;
- CLAP fusion weights and prompts;
- environment-conditioned priors in code;
- onset/offset refinement code;
- tagger calibration;
- a code change to how events are decoded from frame scores.

Look at `best_metrics` (safety_recall, fp_per_min, wrong_env, grid_f1) for weak spots. Spawn
claude workers on code-heavy ideas when you can; agy workers do well on parameter sweeps.

## 4. Housekeeping

- Every tick, run `$PY $HC gc`. It removes worktrees and experiment branches no live worker or
  pending win needs.
- When `clipset.cached` has grown a lot past `clipset.active_clips`, scoring rebaselines by
  itself; nothing to do. If `best` looks stale for more than ~30 minutes while the cache
  grows, run `$PY -m soundthing.harness rebaseline`.

## 5. Journal and end

Append one line about what you did and why to the journal, then end your turn:

```bash
echo "- $(date -u +%FT%TZ) tick: <applied/rejected/spawned/killed, best=...>" >> data/hillclimb/journal.md
```

## Never

- push, merge into `main`, switch branches, or force anything in git;
- touch `data/harness/holdout/` or read holdout numbers;
- edit the harness, objective or eval settings;
- run more than two `apply`s in one tick;
- run `$HC stop` (only the user stops the loop).
