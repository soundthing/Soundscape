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
The fitted table instead uses positive bounded odds coefficients (0.25–4): relevant
sounds can be boosted and unlikely ones downweighted, while strong enough audio
can always qualify. Legacy configs retain multiplication. Low scene confidence
still blends toward neutral `general`, and safety classes bypass the table.

## PANNs-only event classification — events, reverses max fusion
Previously each class took the maximum of PANNs and an untuned CLAP sigmoid score. The user requested removing event CLAP after reviewing how ambiguous similarities could override PANNs or supply unsupported specific labels. Remove the event scorer rather than leave a flag that can reactivate it; environment CLAP is independent and retained. This is a requested simplification, not a measured accuracy improvement. The shared legacy unmapped ids remain silent until their separately approved cleanup.

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

## Keep the deepest qualified sound during overlap — events
AudioSet parent/child predictions often describe one source. Keep qualified
specific labels and subtract their intervals from qualifying ancestors, preserving
unrelated sounds and the general label's remaining intervals. Use all ontology
paths rather than only one parent. This does not separate two physical sources.
Readability padding must obey the same relationship or duplicate parents reappear.

## Every PANNs output has an independent class — events
A broad alias must not directly score a specific class: a high `Dog` score does not
prove barking. Retain the original ids for primary labels, give remaining AudioSet
labels machine-id-based ids. Five legacy CLAP-only classes were originally preserved separately; after event CLAP removal they remain configured but cannot fire pending shared-interface cleanup.
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
