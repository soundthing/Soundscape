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

## Merge runs before dropping short events — events
Runs within merge_gap_sec are merged before filtering by min_event_sec so that
split detections of a single continuous sound are not erased as isolated blips.

