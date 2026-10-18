# Agent rules for `harness/`

The evaluation harness: scores and tunes pipeline stages (Person 1). The root
[AGENTS.md](../../../AGENTS.md) applies in full; these rules add to it.

## Before you start

1. Read [plan.md](plan.md) for what's left; [tuning-plan.md](tuning-plan.md) for the events
   design (cache + replay, fixed objective, limits, bootstrap, holdout).
2. Read [references/](references/README.md), especially `pitfalls.md`.

## Keep the references up to date

Update `references/` in the **same commit** as the change that makes it stale: stack
changes → `stack.md`; a design choice made or reversed → `decisions.md` (say why, don't
silently delete); a surprise → `pitfalls.md`. Record the why, not the code. Keep
[plan.md](plan.md) current. Eval results go in the component's own tuning log
(events: `events/references/tuning-log.md`).

## The tuning loop
1. Read `get_ledger` / `get_best` first so you don't repeat past experiments.
2. State ONE clear hypothesis before making any change.
3. Change parameters via overrides, or edit code in the component being tuned.
4. `score` with that hypothesis. Every attempt is logged to the shared ledger either way.
5. Keep a candidate only if limit checks `passed` AND bootstrap `accept` is true.
6. `promote` only an accepted run.

## Tunable vs protected
- **Tunable:** parameters in `describe_space` (`events:` in `pipeline.yaml`, `thresholds.yaml`; per-class thresholds only for classes with >= 20 dev events), and code in `events/`.
- **Not tunable:** the scoring rule (F0.5 conditioned micro, fixed), `eval:` settings, `backend:` values, `evaluate.py` (Person 3).
- **Holdout:** the AudioSet holdout split (`data/harness/holdout/`) is hidden; never read, copy, or score it directly. Promotion reports pass/fail only. Do not try to infer holdout numbers.
- **Other roles' files and shared files:** never edit `vocabulary.yaml`, `priors.yaml`, `environments.yaml` or another role's area. Score variants on dev, then draft changes via `propose_shared_change`. Promotion refuses shared keys.
- **Cache dependencies:** changes to `window_sec`, `hop_sec`, or models need a new cache via `request_recache` (Kaggle GPU outer loop).

## State and environment
- Harness state lives in `data/harness/` of the main checkout, shared across all worktrees.
- Tests must isolate state with `monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path))`.

## How to run
- **CLI:** `python -m soundthing.harness score --set events.postprocess.smoothing_sec=0.75 --hypothesis "..."` (also `sweep`, `promote`, `propose`, `space`, `best`, `ledger`, `recache`, `build-dev`, `import-audioset`).
- **MCP server:** `python -m soundthing.harness serve-mcp` (stdio). Registering it in `.mcp.json` or `.claude/` is a protected shared-file change requiring explicit user confirmation (root AGENTS.md §2). Snippet for user reference:
  `{"mcpServers": {"soundthing-harness": {"command": ".venv/Scripts/py.exe", "args": ["-m", "soundthing.harness", "serve-mcp"]}}}` (venv interpreter name may differ per machine).
