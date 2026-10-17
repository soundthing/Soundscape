# Agent rules for `events/tuning/`

Rules for agents using or modifying the events tuning harness. Root and `events/` AGENTS.md apply.

## The tuning loop
1. Read `get_ledger` / `get_best` first so you don't repeat past experiments.
2. State ONE clear hypothesis before making any change.
3. Modify parameters via overrides or edit code in `events/`.
4. `score` with that hypothesis. Every attempt is logged to the shared ledger either way.
5. Keep a candidate only if limit checks `passed` AND bootstrap `accept` is true.
6. `promote` only an accepted run.

## Tunable vs protected
- **Tunable:** Parameters in `describe_space` (`events:` in `pipeline.yaml`, `thresholds.yaml`; per-class thresholds only for classes with >= 20 dev events), and code in `events/`.
- **Not tunable:** Scoring rule (F0.5 conditioned micro, fixed), `eval:` settings, `backend:` values, `evaluate.py` (Person 3).
- **Holdout:** Hand-labeled clips are hidden; never read, copy, or score them directly. Promotion reports pass/fail only. Do not try to infer holdout numbers.
- **Shared files:** Never edit `vocabulary.yaml`, `priors.yaml`, or `environments.yaml`. Score variants on dev, then draft changes via `propose_shared_change`. Promotion refuses shared keys.
- **Cache dependencies:** Changes to `window_sec`, `hop_sec`, or models require a new cache via `request_recache` (Kaggle GPU outer loop).

## State and environment
- Harness state lives in `data/tuning/` of the main checkout, shared across all worktrees.
- Tests must isolate state with `monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))`.

## How to run
- **CLI:** `python -m soundthing.events.tuning score --set events.postprocess.smoothing_sec=0.75 --hypothesis "..."` (also `sweep`, `promote`, `propose`, `space`, `best`, `ledger`, `recache`, `build-dev`).
- **MCP server:** `python -m soundthing.events.tuning serve-mcp` (stdio). Registering it in `.mcp.json` or `.claude/` is a protected shared-file change requiring explicit user confirmation (root AGENTS.md §2). Snippet for user reference:
  `{"mcpServers": {"soundthing-tuning": {"command": ".venv/Scripts/py.exe", "args": ["-m", "soundthing.events.tuning", "serve-mcp"]}}}` (venv interpreter name may differ per machine).
