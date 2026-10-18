# Decisions

Why the harness works the way it does. Moved from `events/references` when the harness
became its own component; entries tagged `events` apply to the events component.

## Tuning objective is micro F0.5 with coarse-only matches at half credit — events
The fixed tuning objective uses micro-pooled F0.5 because precision is prioritized over recall
(false-positive captions are more jarring than missed non-safety events). Coarse-only matches
receive half credit (0.5 TP + 0.5 FP + 0.5 FN), rewarding coarse detections while preserving
an incentive for fine-grained accuracy.

## Safety recall includes coarse-level matches — events
For safety-critical classes, coarse captions (such as "alarm" for a siren) still alert the
viewer to danger. Safety recall therefore counts coarse matches toward true positives.

## Tuning matching replicates evaluate greedy semantics — events
Harness matching uses the identical greedy nearest-onset iteration and tie-breaking order
(<= distance favoring the later truth) as soundthing.evaluate so tuning metrics directly
predict official evaluation results.

## Tuning acceptance uses limits vs current best plus paired bootstrap — events
Acceptance requires passing limits vs the current best (safety recall, FP/min, above baseline, wrong-environment robustness) plus a paired bootstrap over clips where the lower 95% confidence bound of F0.5 gain is strictly positive (> 0). Single-number metric improvements on a few dozen clips are mostly noise and risk promoting random fluctuations.

## AudioSet strong labels are the eval set; no hand labels — events
Hand-labeling 10–20 clips was dropped: it is slow and too small to tune or promote on. The AudioSet strong-label eval split gives ~6k clips with real onsets for our vocabulary. It is split ~80/20 into dev and holdout by a hash of the YouTube id, so reruns are stable and no video lands in both. Trade-offs: no environments (baseline mode only; conditioning still needs the synthetic mixes), clips are only 10 s, and five classes have no examples (ball_bounce, kettle_whistle, mouse_click, seatbelt, turn_signal).

## Holdout test set is pass/fail only and checked only at promotion — events
AudioSet holdout clips evaluate candidate vs current config on objective, safety recall, and FP/min only during promotion, returning a binary pass/fail result with no metric numbers revealed. Keeping holdout scores confidential prevents agents from hill-climbing against the holdout set.

## Per-class thresholds require at least 20 dev events — events
Per-class threshold tuning is restricted to classes with at least 20 dev events (`MIN_CLASS_EVENTS`) to limit overfitting to sparse sample outliers. Classes with fewer occurrences fall back to default thresholds.

## Promotion edits YAML by line and routes shared files via proposals — events
Promotion updates `config/pipeline.yaml` (`events:` section) and `config/thresholds.yaml` via line-by-line splicing rather than whole-file re-dumping, preserving comments, formatting, and ownership notes. Shared files (`vocabulary.yaml`, `priors.yaml`, `environments.yaml`) cannot be promoted directly; variants must be submitted as human-reviewed proposals.

## Every MCP and CLI call is logged to the shared ledger — events
All tool and CLI operations are logged to the shared ledger (`ledger.jsonl`) under `data/harness/`, so parallel agents across worktrees have full visibility into past hypotheses, parameters, and results without repeating experiments.

## Kaggle kernel ships the working tree instead of cloning — harness
The kernel used to clone the (private) repo at a pushed sha, which needed a GitHub token as a Kaggle secret and a commit+push before every recache. Cache building needs only `src/soundthing`, `config/` and `pyproject.toml` (~160 KB base64), so the pusher embeds them in the script and records their sha256 in the cache meta. Trade-off: provenance is a bundle hash rather than a commit.

## Component protocol and dynamic registry — harness
The harness core is decoupled from event-specific logic via a runtime_checkable `Component` protocol (`base.py`) and dynamic string-based registry (`REGISTRY`). Components (`events`, `environment`, `captions`, `pipeline`) implement their own search spaces, stat objectives, limits, and scoring routines. Unimplemented components are reported as unavailable by `list-components` without failing imports.

## Component ledger separation and backward compatibility — harness
Best runs for the legacy `events` component remain stored at `data/harness/best.json`, while other components write to `best-<component>.json`. Entries in `ledger.jsonl` record their `component`, and reading/filtering defaults to `events` to preserve behavior for existing tools and tests.

## Promotion writes every component's config; no role routing — harness
Reversed 2026-10-18. Promotion used to write only keys owned by the current role
(`SOUNDTHING_ROLE`) and turn the rest into proposals. The repo dropped role ownership, so
promotion now writes any tunable key in `pipeline.yaml` / `thresholds.yaml`. Proposals
remain only for the shared contract files (`vocabulary.yaml`, `priors.yaml`), which still
need explicit confirmation under root AGENTS.md §1.

## Environment tuning uses raw logits caching with pure replay function — environment
`environment.classifier.classify` was refactored to extract `segments_from_logits`, separating slow inference (audio CLAP embeddings, visual SigLIP embeddings, PySceneDetect) from fast post-processing (temperature scaling, late fusion, visual similarity floor, thresholding, and adjacent same-env segment merging). `EnvCache` stores pre-softmax logits per shot span. Replay evaluates candidates in milliseconds without model re-runs.

## Environment objective balances time accuracy with macro accuracy — environment
The fixed environment objective is `0.5 * time_acc + 0.5 * macro_acc`. Time accuracy pools total matching seconds across all clips against total labeled seconds, prioritizing dominant scenes. Macro accuracy averages per-environment accuracy across only those environments present in the ground truth labels, preventing frequent background environments from completely overshadowing rarer scenes. Per-clip stats store `[correct_sec, total_sec]` for each environment in config order, enabling exact objective recomputation in paired bootstrap resampling.
## End-to-end pipeline objective is caption-cue F0.5 with predicted environment — pipeline
The pipeline component evaluates end-to-end performance using the same caption-cue micro F0.5 (with coarse half-credit) as the captions component, but conditioning events replay on the predicted environment (`predict_segments`) rather than ground truth labels. Baseline metric evaluates unconditioned events, and environment time accuracy (`env_time_acc`) is tracked as a diagnostic.

## Cue matching evaluates onset tolerance and duration-scaled offset tolerance — captions
Captions and pipeline evaluation match predicted caption cues against labeled events using greedy nearest-onset matching order (identical to events evaluation), requiring matching class (fine or coarse group for half-credit), onset error <= `eval.onset_tolerance_sec`, and offset error <= `max(eval.onset_tolerance_sec, 0.5 * event.duration)`. Diagnostics include a 5-bin signed offset error histogram.

## Pipeline search space is the de-duplicated union of constituent components — pipeline
The pipeline component search space aggregates parameters from the events, environment, and captions stages by querying `space()` on each component and dropping duplicate parameter entries.

## Trusted scorer overlay isolates candidate code — harness
Scoring candidate code changes uses an overlay directory created by archiving the pinned commit (default HEAD of main checkout) for `src`, `config`, and `pyproject.toml`. Only the component's declared `code_paths` and non-shared `config/` files are copied from the candidate worktree, with the `eval:` section in `pipeline.yaml` strictly restored from the pinned commit and shared yaml files (`environments.yaml`, `vocabulary.yaml`, `priors.yaml`) preserved from pinned. This guarantees that candidate code cannot alter the objective, harness, or evaluation parameters.

## Code experiments run in short worktrees and prune unaccepted branches — harness
Code experiments run in dedicated git worktrees located under `<main_checkout parent>/.wt/soundthing-exp-<id>` to mitigate Windows MAX_PATH limitations. If the candidate fails limit checks or paired bootstrap acceptance against current best, its worktree and branch are automatically removed to prevent repo clutter. Accepted runs keep their branch for promotion.

## Code promotion targets review branches with tamper verification — harness
`promote_code` verifies that the experiment was accepted, validates the patch SHA against the recorded ledger entry to detect any tampering, enforces holdout gating on events, and creates a local `harness/review/<run_id>` branch. It never automatically merges into user branches or pushes to remotes, keeping human review as the final gate for code promotion.
