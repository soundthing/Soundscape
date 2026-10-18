# Decisions

Each entry: the decision, why, and what was rejected. Mark entries **agreed** (team-level,
from `project.md` / `docs/interfaces.md`) or **events** (made within this stage). When a
decision is reversed, update the entry and say why instead of deleting it.

## Sounds are detected from audio only — agreed
Visuals only inform the environment. Many important sounds happen off-screen.

## Precision over recall, except safety — agreed
A wrong caption is worse than a missed one. Safety classes get their own, separately tuned
thresholds because they bypass conditioning and would otherwise false-alarm more.

## Conditioning keeps every sound possible — agreed, revised with user approval
The old multiplicative floor could prevent unlikely classes reaching threshold.
The fitted table instead uses positive odds coefficients with configurable runtime bounds (default 0.25–4): relevant
sounds can be boosted and unlikely ones downweighted, while strong enough audio
can always qualify. Legacy configs retain multiplication. Low scene confidence
still blends toward neutral `general`, and safety classes bypass the table.

## PANNs-only event classification — events, reverses max fusion
Previously each class took the maximum of PANNs and an untuned CLAP sigmoid score. The user requested removing event CLAP after reviewing how ambiguous similarities could override PANNs or supply unsupported specific labels. Remove the event scorer rather than leave a flag that can reactivate it; environment CLAP is independent and retained. This is a requested simplification, not a measured accuracy improvement. The user separately approved removing the five unmapped ids and shared prompt fields; schema version 2 prevents reuse of stale CLAP-generated outputs.

## `score_window` stays a pure single-window function — agreed
So a streaming wrapper can reuse it later. Offline speed comes from batching around it,
not from changing it.

## Models and raw scores are shared across baseline and conditioned — events
Both modes see the same audio, scoring and speech masking; only conditioning differs.
Loading and scoring once per run halves the cost.

## Temporal smoothing and display suppression removed — events, user-requested reversal

Previously, offline median smoothing removed isolated score spikes, a 60-second
cooldown hid repeated sustained events, and a concurrent-caption cap discarded
overlapping events. The user requested removing all three to retain qualifying
sounds and recurrences. Their implementations and default settings are removed;
legacy copies of those settings are ignored. More eligible captions can now appear,
but caption accuracy has not been measured. Thresholds, minimum duration, gap
merging, hierarchy suppression and readability timing remain separate decisions.
Existing `sustained` vocabulary metadata stays accepted for compatibility and no
longer suppresses recurrences.

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

## Split detection into cached model stage and replay — events
Detection is split into a cached model stage (`ClipCache`) and a replayable post-model stage (`replay`). Known accepted behaviour changes: tagger outputs pass through float16 to conserve memory (~1 MB/min); VAD regions come from per-chunk probs. Production detector and tuning harness share the exact same replay path. In-process memo in `detector.detect` caches the single most recent `ClipCache` by audio path, mtime, size, cache_key and models_cache_key so conditioned runs on the same file avoid recomputation.

## Evaluation harness is its own component — events
Tuning started inside `events/`, but the same loop (score on a fixed objective, accept by
limits + bootstrap, promote, log) applies to environment, captions and end to end, so the
harness moved to `soundthing/harness/` (2026-10-17). The model cache and replay stay in
`events/` because production detection runs through them; they import the harness only
lazily for the default cache location. Tuning decisions now live in
[harness/references](../../harness/references/decisions.md).

## Source separation is opt-in and gated — events
What: rescore windows where speech or music masks sound events using an effects stem (mix minus vocals) from a source separator. Separation runs only on time spans gated by AudioSet speech/music labels or VAD, padded and crossfaded. Separated scores fuse with raw scores by max by default, with a separate VAD suppression factor (`suppress_under_speech`) on masked windows.
Why: overlapping speech and music mask foreground sound events. Full-clip separation is too slow on CPU (~3x real time), so gating limits separation to relevant spans. Fusing by max preserves raw detections where separation might introduce artifacts.
Rejected: separating every clip (untenable latency on CPU); replacing raw scores outright (causes false negatives on unmasked events); a DnR (dialogue/music/effects) model for now (packaging and dependency risk; can be added later as an alternative backend). Note: shipped without the masked-mix experiment; enable only after a tuning run with promote shows a gain (precision-first).

## Keep the deepest qualified sound during overlap — events
AudioSet parent/child predictions often describe one source. Keep qualified
specific labels and subtract their intervals from qualifying ancestors, preserving
unrelated sounds and the general label's remaining intervals. Use all ontology
paths rather than only one parent. This does not separate two physical sources.
Readability padding must obey the same relationship or duplicate parents reappear.

## Every PANNs output has an independent class — events
A broad alias must not directly score a specific class: a high `Dog` score does not
prove barking. Retain the original ids for primary labels, give remaining AudioSet
labels machine-id-based ids. Five legacy CLAP-only classes were originally preserved separately; the user subsequently approved retiring them with event CLAP. The remaining 527 AudioSet mappings and coefficients are preserved.
This expands coverage without replacing the model or claiming new accuracy.

## Dataset enrichment is a weak association estimate — events
Sound-VECaps provides visual context for AudioSet records, allowing scene-word
matching without downloading another audio corpus or treating event labels as scene
labels. Its generated descriptions and AudioSet's incomplete positive annotations
limit what can be inferred. Exclude uncertain/multiple scene matches, hold out video
ids, smooth toward global frequencies, bound relative enrichment, and keep
unsupported cases neutral. Held-out annotation loss is not sound-event F1.

## Positive odds coefficients keep unlikely sounds possible — events
Multiplying by a small environment weight can make a class unable to reach its
threshold even with perfect acoustic evidence. Positive bounded odds adjustment
preserves scores at 0 and 1, changes the evidence required in between, and supports
both enrichment and downweighting. Existing configurations retain multiplication
unless they explicitly select the new application. Safety and general remain neutral.


## Preserve estimates separately from the runtime minimum — events, user-approved

Baking 0.25 into the table erased the distinction between 3,045 fitted associations.
The optional unfloored representation restores those estimates from the original
counts; the shared `floor` setting alone controls the runtime minimum. Bound each
environment before mixing so default uncertain-scene behavior stays unchanged.
Legacy bounded configs keep their original validation. Lowering the minimum does
not establish better accuracy, and zero positive scene annotations remain unknown
evidence rather than verified absence. The maximum cap and safety bypass remain.

## Specificity changes requirements, not confidence — events, user-approved

A precise source or subtype makes a stronger claim than a general acoustic
description. Explicit, editable tiers raise the evidence required for stronger
claims without rewriting model scores or treating taxonomy depth as accuracy.
Independently scored parents still need their own evidence. Existing safety
thresholds remain exact, and non-safety overrides remain minimums. These initial
settings are a policy proposal, not measured precision; labeled evaluation is
still needed before calling any tier reliable.

The user subsequently lowered fine to 0.65, with specific at 0.55 and broad at
0.45, to allow more captions through. Safety requirements and per-class minimums
stay unchanged. This tuning request does not establish improved accuracy.

## Generic inference uses maximum support and an activity gate — events

Without new model outputs, a broad fallback can only summarize existing evidence.
Taking the maximum descendant score avoids adding correlated predictions and
inventing higher confidence. Restrict inference to explicitly configured coarse
groups; independently scored parents keep their own outputs. A recording-level
RMS gate prevents sound guesses in digital silence but does not establish listener
loudness or source accuracy. Missing settings disable this optional path. The
proposed new groups and group-parent contract remain subject to shared approval.
