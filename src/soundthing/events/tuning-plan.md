# Tuning harness plan (Person 1)

A harness that scores the events stage with one number, so agents can hill-climb it:
change something, re-score, keep the change only if it really helps.

## Key design

- **Score models once, replay the rest.** Model outputs are cached per clip; everything
  after them replays in milliseconds. The replay is the same function production
  `detect` calls, so tuned settings behave identically in the real pipeline.
- **One fixed objective.** Micro F0.5 (precision-weighted) on conditioned output, pooled
  over dev clips. A prediction that matches only at the coarse level gets 0.5 credit.
- **Limits relative to the current best, no hand-picked constants.** A candidate is
  rejected if it lowers safety recall, raises FP/min, scores below baseline, or collapses
  when the environment is wrong.
- **Agents tune everything except the scoring rule.** All `events:` parameters,
  `thresholds.yaml`, and code in `events/`.
- **Overfitting guards.** Tune on public data. The hand-labeled clips are a hidden test
  set that is checked only on promotion. A change must pass a paired bootstrap over clips
  (lower confidence bound of the gain > 0). Per-class thresholds are allowed only for
  classes with enough dev events.
- **Shared ledger.** Every experiment is logged (git SHA, parameter diff, scores), so
  parallel agents don't repeat each other.
- **Shared files stay human-approved.** Variants of `vocabulary.yaml` (CLAP prompts,
  AudioSet mapping) or `priors.yaml` are scored, then written up as proposals; the
  harness never writes them.

## Cache

| Cached per clip | Makes tunable without a rerun |
|---|---|
| All 527 tagger outputs | AudioSet → vocabulary mapping |
| CLAP audio embeddings | CLAP prompts (re-embed text only), calibration, combine rule, small heads |
| Per-frame VAD probability | VAD threshold and suppression |
| (replay) | smoothing, merging, thresholds, collapse, priority filter, `onset_shift_sec` |

Window, hop and model choice need a rerun (a slow outer loop via `request_recache`).
The cache key is a hash of model, window, hop and code version. An equivalence test
checks that replay matches `detect` exactly. The cache is about 1 MB per minute of audio.

## Metrics

The harness imports `match_events`/`score_events` from `evaluate.py` (Person 3), so its
numbers agree with `soundthing eval`. F0.5 and grid-based F1 (1 s slices) live in the
harness, not in `evaluate.py`; they can be offered to Person 3 later.

Diagnostics (shown, not optimized): per-class TP/FP/FN, signed onset error, grid F1, and
top false positives with timestamps.

Environment input for conditioned mode: the labeled environment at about 0.7 confidence,
a deliberately wrong environment, and Person 2's real `env_segments.json` once it exists.

## Data

| Split | Source | Use |
|---|---|---|
| dev | FSD50K / ESC-50 synthetic per-environment mixes (exact onsets, known environment) | tuning, both modes |
| dev | AudioSet strong-labeled eval split, if its audio is obtainable on Kaggle | tuning, baseline mode |
| holdout | hand-labeled clips in `labels/` | promotion only; stored outside the repo |

Converted dataset labels go in `data/` (gitignored), in `clip_labels` format.

## Compute: Kaggle

Model scoring runs in a Kaggle notebook on a free T4 or P100 (about 10–20 min per cache
build). The notebook attaches ESC-50 and FSD50K from Kaggle mirrors and saves the cache as
a private Kaggle dataset. The harness downloads it with the `kaggle` CLI and checks its
config hash. Hill-climbing runs locally on CPU.

## Agent interface: MCP

Agents use the harness only through an MCP server
(`src/soundthing/events/tuning/mcp_server.py`):

- `describe_space`: tunable parameters, bounds, current values
- `get_best`, `get_ledger`: the current best result and past experiments
- `score(overrides)`: score the agent's working tree; returns F0.5, limit checks, CI,
  per-class results, top false positives
- `sweep(params, strategy)`: coordinate descent or Optuna
- `promote(run_id)`: hidden test set (pass/fail only), then write config and
  `references/tuning-log.md`
- `propose_shared_change`: draft a `vocabulary.yaml` / `priors.yaml` change for a human
- `request_recache(window, hop, model)`: push a Kaggle run via its API

The server enforces the rules: fixed scoring, a pass/fail-only holdout, every call
logged, no writes to shared files. MCP doesn't stop agents reading files directly, so
holdout labels live outside the repo and its worktrees.

Agent loop: read the ledger, state one hypothesis, change parameters or code, `score`,
keep it if it passes, log it either way.

## Milestones

1. **M0:** split `detect` into a cached model stage and a replayable post-model stage
   (no behavior change), plus the equivalence test.
2. **M1:** cache format, `score`, objective and limits, ledger.
3. **M2:** Kaggle cache notebook, dataset converters, holdout split.
4. **M3:** `sweep`, bootstrap acceptance, `promote`.
5. **M4:** MCP server, agent rules (`events/tuning/AGENTS.md`), first automated pass.

## Open

- Whether AudioSet strong-label audio can be obtained on Kaggle.
- Registering the MCP server in `.mcp.json` / `.claude/` is a shared-file change and needs
  explicit approval.
