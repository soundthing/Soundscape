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

Each component declares: `name`, owning role, the config keys it owns, `cache_key(cfg)`,
`build_cache(clip, cfg)` (slow model stage), `predict(cache, cfg, context)` (fast replay),
`objective(preds, labels) -> per-clip counts + scalar`, `limits(candidate, best)` and
`space(cfg, dev_stats)`. Core score/sweep/bootstrap/promote/ledger work on any component.

| Component | Objective (fixed) | Cached model stage | Tunable by replay |
|---|---|---|---|
| events | micro F0.5 conditioned, coarse half credit (exists) | tagger outputs, CLAP audio embeddings, VAD probs (exists) | VAD, post-processing, thresholds, CLAP calibration, combine rule |
| environment | time-weighted env accuracy + macro accuracy over envs | per-segment SigLIP + CLAP embeddings, shot boundaries | fusion weights, temperature, confidence threshold, prompts (score-only), segmentation min/max |
| captions | cue-level F0.5 on rendered cues vs labeled events | none (cheap) | min/max cue durations, merge rules |
| pipeline | caption F0.5 with predicted (not labeled) environment | union of the above | everything, jointly |

## Ownership and safety

- **Owner:** the harness becomes its own row in root AGENTS.md (needs confirmation; see below).
- **Cross-role promotion:** promote writes config only for components owned by the
  current user's role. A winner for another role's component becomes a proposal for that
  owner (same path as `vocabulary.yaml`/`priors.yaml` today).
- **Trusted scorer:** scoring always runs the harness from a pinned commit of the main
  checkout and imports only the candidate's stage code from the experiment worktree, so
  an agent can't improve its number by editing the objective.
- **Holdout:** unchanged: pass/fail only at promotion, stored outside the repo.

## Work left

- Move to the layout above (after the in-flight AudioSet work is committed).
- Component protocol; core score/sweep/promote take `--component`; events becomes the first component.
- Trusted scorer; ledger stores each candidate's patch.
- Experiment runner (worktree per experiment, score, keep/discard) and code promotion to a review branch.
- Environment, captions and end-to-end components.
- MCP tools `list_components` and `run_experiment`; agent loop rules in `harness/AGENTS.md`.
- First real runs on ESC-50 mixes, AudioSet strong labels and env-labeled clips.

## Open decisions

1. Who owns `harness/` (Person 1, or a new shared row everyone may edit)?
2. May the harness ever write another role's config directly, or proposals only (default)?
3. End-to-end objective: caption F0.5 only, or also weigh environment accuracy and modifiers?
