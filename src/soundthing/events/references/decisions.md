# Decisions

Each entry: the decision, why, and what was rejected. Mark entries **agreed** (team-level,
from `project.md` / `docs/interfaces.md`) or **events** (made within this stage). When a
decision is reversed, update the entry and say why instead of deleting it.

## Sounds are detected from audio only — agreed
Visuals only inform the environment. Many important sounds happen off-screen.

## Precision over recall, except safety — agreed
A wrong caption is worse than a missed one. Safety classes get their own, separately tuned
thresholds because they bypass conditioning and would otherwise false-alarm more.

## Conditioning weakens, never deletes — agreed
The priors floor (0.25) means a misclassified environment lowers a score instead of
zeroing it. Low environment confidence blends toward `general`, which weakens conditioning
smoothly.

## Tagger and CLAP combine by max — events
A class takes the higher of its tagger and CLAP scores. Simple, and lets either model
detect a class on its own. Revisit if CLAP adds false positives on classes the tagger
already covers well (options: CLAP only for classes without AudioSet labels, or per-class
weights). Decide from eval numbers.

## `score_window` stays a pure single-window function — agreed
So a streaming wrapper can reuse it later. Offline speed comes from batching around it,
not from changing it.

## Models and raw scores are shared across baseline and conditioned — events
Both modes see the same audio, scoring and speech masking; only conditioning differs.
Loading and scoring once per run halves the cost.

## Post-processing may use future context — events
Processing is offline, so smoothing is two-sided. A streaming mode would need causal
variants.

## Speech masking scales scores down, doesn't remove them — agreed
Non-safety scores are multiplied by `suppress_under_speech` inside speech; safety classes
are untouched. Loud events during dialogue can still surface.

## VAD regions derived from cached per-chunk probabilities — events
VAD regions are now derived from cached Silero per-chunk probabilities (our hysteresis + min_speech + pad/merge) in BOTH production and replay, replacing get_speech_timestamps, so VAD threshold and suppression are tunable from the cache. Differs from get_speech_timestamps: no internal Silero speech_pad/min_silence; our pad/merge covers it. Raw spans are padded by `pad_sec`, clipped to [0, duration], and merged across gaps under `merge_gap_sec`.

## Merge runs before dropping short events — events
Runs within merge_gap_sec are merged before filtering by min_event_sec so that
split detections of a single continuous sound are not erased as isolated blips.

## Caption modifiers prioritize precision over recall — events
A wrong spatial or loudness modifier is worse than none. Left/right, approaching/receding,
and loud/quiet use conservative physical thresholds on RMS balance, dB trends, and clip
median levels; when audio is mono, short, or ambiguous, no modifier is attached.

## Wrong-environment experiment uses lowest prior cosine similarity — events
To stress-test conditioning under an adversarial environment without hard-coding
environment pairs, the wrong environment is selected per segment as the one whose prior
weight vector over vocabulary classes has lowest cosine similarity to the true environment's.

## Synthetic dev mixes keep folds isolated, sample by prior, and gate onsets — events
Each mix draws sources from a single dataset fold so cross-validation splits remain
leak-free. Foreground event classes are sampled proportional to the environment's prior
weights to mirror real test distributions without hard-coding ids. Onsets are trimmed
with an energy gate to ensure tight ground truth timestamps rather than arbitrary clip
padding. Background beds use unmapped categories so the bed provides realistic texture
without leaking vocabulary sound events.

## FSD50K sources require exactly one mapped class; AudioSet strong labels are baseline-only — events
FSD50K clips are multi-labeled and ontology-propagated; admitting clips mapping to multiple
vocabulary classes would inject unintended overlapping events into synthetic mixes, so sources
must reach exactly one distinct mapped class. AudioSet strong-label clips have unknown environments,
so their annotations are generated with empty segments and evaluated exclusively in baseline mode.
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

## Split detection into cached model stage and replay — events
Detection is split into a cached model stage (`ClipCache`) and a replayable post-model stage (`replay`). Known accepted behaviour changes: tagger and CLAP audio embeddings pass through float16 to conserve memory (~1 MB/min); VAD regions come from per-chunk probs. Production detector and tuning harness share the exact same replay path. In-process memo in `detector.detect` caches the single most recent `ClipCache` by audio path, mtime, size, cache_key and models_cache_key so conditioned runs on the same file avoid recomputation.

## Tuning acceptance uses limits vs current best plus paired bootstrap — events
Acceptance requires passing limits vs the current best (safety recall, FP/min, above baseline, wrong-environment robustness) plus a paired bootstrap over clips where the lower 95% confidence bound of F0.5 gain is strictly positive (> 0). Single-number metric improvements on a few dozen clips are mostly noise and risk promoting random fluctuations.

## Holdout test set is pass/fail only and checked only at promotion — events
Hand-labeled holdout clips evaluate candidate vs current config on objective, safety recall, and FP/min only during promotion, returning a binary pass/fail result with no metric numbers revealed. Keeping holdout scores confidential prevents agents from hill-climbing against the holdout set.

## Per-class thresholds require at least 20 dev events — events
Per-class threshold tuning is restricted to classes with at least 20 dev events (`MIN_CLASS_EVENTS`) to limit overfitting to sparse sample outliers. Classes with fewer occurrences fall back to default thresholds.

## Promotion edits YAML by line and routes shared files via proposals — events
Promotion updates `config/pipeline.yaml` (`events:` section) and `config/thresholds.yaml` via line-by-line splicing rather than whole-file re-dumping, preserving comments, formatting, and ownership notes. Shared files (`vocabulary.yaml`, `priors.yaml`, `environments.yaml`) cannot be promoted directly; variants must be submitted as human-reviewed proposals.

## Every MCP and CLI call is logged to the shared ledger — events
All tool and CLI operations are logged to the shared ledger (`ledger.jsonl`) under `data/tuning/`, so parallel agents across worktrees have full visibility into past hypotheses, parameters, and results without repeating experiments.
