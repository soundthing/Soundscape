# Harness plan: a standalone tuning component

Goal: turn the events tuning harness into its own component that can score and
hill-climb **any** stage (events, environment, captions, end to end), through config
overrides **and** code changes made by agents. Supersedes the component-scoping parts of
[tuning-plan.md](tuning-plan.md); the design there (cache + replay, fixed objective, limits
vs best, bootstrap, hidden holdout, ledger, MCP) still holds.

## Target layout

| Where | What |
|---|---|
| `src/soundthing/events/cache.py`, `replay.py` | Stay with events: production `detect` uses them. Moved out of `tuning/`. |
| `src/soundthing/harness/` | Core, component-agnostic: `paths`, `ledger`, `overrides`, `bootstrap`, `sweep`, `promote`, `proposals`, `holdout`, `kaggle_io` + `kaggle/`, `mcp_server`, `runner` (new), `__main__`, `AGENTS.md`. |
| `src/soundthing/harness/components/` | `base.py` (Component protocol), `events.py` (today's objective, score, space, datasets), `environment.py`, `captions.py`, `pipeline.py` (end to end). |
| `data/harness/` | State (was `data/tuning/`); `SOUNDTHING_HARNESS_HOME`, old env var kept as an alias. |
| `tests/test_harness_*.py` | Was `tests/test_harness_*.py`. |
| CLI | `python -m soundthing.harness <cmd> --component events` |

## Component protocol

Each component declares: `name`, the config keys it tunes, `cache_key(cfg)`,
`build_cache(clip, cfg)` (slow model stage), `predict(cache, cfg, context)` (fast replay),
`objective(preds, labels) -> per-clip counts + scalar`, `limits(candidate, best)` and
`space(cfg, dev_stats)`. Core score/sweep/bootstrap/promote/ledger work on any component.

| Component | Objective (fixed) | Cached model stage | Tunable by replay |
|---|---|---|---|
| events | micro F0.5 conditioned, coarse half credit (exists) | tagger outputs, VAD probs, per-window RMS (exists) | VAD, post-processing, thresholds |
| environment | time-weighted env accuracy + macro accuracy over envs | per-segment SigLIP + CLAP embeddings, shot boundaries | fusion weights, temperature, confidence threshold, prompts (score-only), segmentation min/max |
| captions | cue-level F0.5 on rendered cues vs labeled events | none (cheap) | min/max cue durations, merge rules |
| pipeline | caption F0.5 with predicted (not labeled) environment | union of the above | everything, jointly |

## Safety

- **Promotion:** writes any tuned config key (`pipeline.yaml`, `thresholds.yaml`). Shared
  files (`vocabulary.yaml`, `priors.yaml`, `environments.yaml`) only get proposals.
- **Trusted scorer:** scoring always runs the harness from a pinned commit of the main
  checkout and imports only the candidate's stage code from the experiment worktree, so
  an agent can't improve its number by editing the objective.
- **Holdout:** unchanged: pass/fail only at promotion, stored outside the repo.

## Status

Done (2026-10-18): component protocol + registry (`components/`), events/environment/
captions/pipeline components, generic score/sweep/bootstrap/promote/ledger with
`--component`, per-component best, promotion writes any component's config keys,
trusted scorer (`trusted.py`), experiment runner and code promotion to review branches
(`runner.py`), MCP tools `list_components`, `run_experiment`, `promote_code`.

## First runs (2026-10-18, current config, 42 ESC-50 dev mixes, audio only)

| Component | Objective | Notes |
|---|---|---|
| events | F0.5 0.085 | baseline 0.099, **wrong env 0.220**, safety recall 0.20, 23 FP/min |
| environment | 0.119 | audio-only CLAP; 14 envs, so barely above chance (0.07) |
| captions | cue F0.5 0.042 | safety recall 0.11, mean cue 2.0 s |
| pipeline | cue F0.5 0.071 | baseline 0.051, 12.8 FP/min, env time acc 0.119 |

Conditioning on the labeled environment scores *below* a deliberately wrong one, so the
priors or their application likely hurt; investigate before tuning anything else.
Reproduce: `build-dev --esc50 data/esc50 --n-per-env 3 --out data/harness/dev-esc`, then
`build-cache --component environment ...` and `score --component C --labels data/harness/dev-esc/labels --no-log`.

## Work left

- Build dev caches at scale (Kaggle) for AudioSet dev; only the ESC-50 mixes are cached locally.
- Environment component on clips with video (dev mixes are audio-only, so visual fusion is untested on real data).
- Captions merge rules: `captions.py` has only min/max cue duration to tune.

## Open decisions

1. ~~Ownership~~ Dropped (2026-10-18): no role ownership; anyone edits anything, harness writes all config keys.
2. ~~End-to-end objective~~ Caption-cue F0.5 with predicted environment; env accuracy is a diagnostic.
