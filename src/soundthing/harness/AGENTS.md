# Agent rules for `harness/`

The evaluation harness: scores and tunes pipeline stages. The root
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
3. Change parameters via overrides, or write a patch against the component being tuned (`code_paths` only).
4. For parameter changes, `score` with that hypothesis. For code experiments, call `run_experiment` with one hypothesis and the patch; never edit the harness, objective, or eval settings in an experiment. Every attempt is logged to the shared ledger either way.
5. Keep a candidate only if limit checks `passed` AND bootstrap `accept` is true. Unaccepted experiment worktrees and branches are automatically removed.
6. `promote` only an accepted parameter run. Only `promote_code` accepted code experiment runs; review branches (`harness/review/<run_id>`) are created for human review and must be merged by a human (never auto-merge or push).

## Tunable vs protected
- **Tunable:** parameters in each component's `describe_space` (`events:`, `environment:`, `captions:` in `pipeline.yaml`, `thresholds.yaml`; per-class thresholds only for classes with >= 20 dev events), and code in the component's `code_paths`.
- **Not tunable:** the scoring rule (F0.5 conditioned micro, fixed), `eval:` settings, `backend:` values, `evaluate.py`.
- **Holdout:** the AudioSet holdout split (`data/harness/holdout/`) is hidden; never read, copy, or score it directly. Promotion reports pass/fail only. Do not try to infer holdout numbers.
- **Shared files:** never edit `vocabulary.yaml`, `priors.yaml` or `environments.yaml` (root AGENTS.md §1). Score variants on dev, then draft changes via `propose_shared_change`. Promotion refuses shared keys.
- **Cache dependencies:** changes to `window_sec`, `hop_sec`, or models need a new cache via `request_recache` (Kaggle GPU outer loop).

## State and environment
- Harness state lives in `data/harness/` of the main checkout, shared across all worktrees.
- Tests must isolate state with `monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path))`.

## How to run
- **CLI:** `python -m soundthing.harness score --component events --set events.postprocess.smoothing_sec=0.75 --hypothesis "..."` (also `sweep`, `promote`, `propose`, `space`, `best`, `ledger`, `list-components`, `recache`, `build-dev`, `build-cache`, `import-audioset`, `run-experiment`, `promote-code`; `--component NAME` selects the stage component, `score --labels DIR` scores another labels dir).
- **MCP server:** `python -m soundthing.harness serve-mcp` (stdio). Registering it in `.mcp.json` or `.claude/` is a protected shared-file change requiring explicit user confirmation (root AGENTS.md §1). Snippet for user reference:
  `{"mcpServers": {"soundthing-harness": {"command": ".venv/Scripts/py.exe", "args": ["-m", "soundthing.harness", "serve-mcp"]}}}` (venv interpreter name may differ per machine).
