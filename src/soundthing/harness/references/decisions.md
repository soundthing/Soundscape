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
