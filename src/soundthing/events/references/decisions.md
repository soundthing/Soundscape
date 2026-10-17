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

## VAD spans are padded and merged before masking — events
Raw Silero spans are padded by `pad_sec`, clipped to [0, duration], and merged across gaps
under `merge_gap_sec` so brief pauses in dialogue don't leak false events.

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

## Raw-score cache keeps only the most recent file — events
Sharing masked scores across baseline and conditioned modes halves scoring and VAD cost.
Caching only the most recent entry bounded by audio path, mtime, size, and config keys
prevents unbounded memory growth during batch runs.

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
