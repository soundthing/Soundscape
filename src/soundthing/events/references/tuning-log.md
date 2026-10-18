# Tuning log

One entry per eval run that informed a decision, newest first. Record the change, the
clips used, and the numbers from `soundthing eval` (fine/coarse F1, FP/min) for both
baseline and conditioned. Note what was kept or reverted and why.

Entries titled `promote <run_id>` are written automatically by the tuning harness's `promote`
(holdout reported as pass only). The format below is for hand-run evals.

Format:

```
## YYYY-MM-DD — <what changed>
Clips: <which / how many>    Config: <relevant settings>
Baseline:    F1 fine x.xx / coarse x.xx, FP/min x.x
Conditioned: F1 fine x.xx / coarse x.xx, FP/min x.x
Outcome: <kept / reverted, and why>
```

## 2026-10-18 — specificity policy runtime verification

Clip: first 24 seconds of cached YouTube `_ZsEyMpvbCk`, using real PANNs and
Silero VAD, plus the already computed environment segments. Priors remain at
floor 0.25; no model scores or original cached outputs were changed. Baseline and
conditioned modes share a 93-window × 527-class raw-score cache.

Legacy thresholds: 3 baseline and 3 conditioned events (biting, chewing,
chopping). Tiered thresholds: 1 baseline and 1 conditioned event (chopping).
Baseline chopping score remains 0.953; lower-scoring biting (0.645) and chewing
(0.556) miss their new 0.80/0.65 requirements. JSON and VTT outputs validate.

Outcome: confirms real execution and unchanged scores, not accuracy improvement.
There are no human labels, fine/coarse F1 or FP/min measurements for this check.
The initial specificity settings remain tunable policy values.

## 2026-10-17 — remove event CLAP at user request

The event scorer now uses PANNs alone; environment CLAP remains. The user also
approved retiring five unmapped classes, their prompts and priors, with schema
version 2 so stale CLAP-generated runs require regeneration.

Runtime check: the same short local clip used for the earlier expanded smoke
completed real environment CLAP/SigLIP plus PANNs inference. Baseline: 0 events;
conditioned: 1 vehicle event. Both VTT tracks were written. The old smoke's
seatbelt/turn-signal/kettle/ball-bounce labels are no longer eligible. No human
labels were used, so these counts do not establish precision or recall.

Validation: 213 tests and check-config pass. Retained class definitions,
coefficients, environment code and environment prompts are unchanged. The original
fitting report remains historical evidence rather than rewriting its diagnostics.

## 2026-10-17 — dataset-derived environment enrichment

Data: 1,315,263 joined training records, 329,391 held-out
records by video id; 22,394 held-out clips matched a single environment. All 527
PANNs classes have positives. Config: `application: odds`, floor 0.25, max gain 4,
Beta shrinkage 200, minimum class support 20, minimum scene support 50.

Held-out annotation log loss (weighted across matched environments):
Baseline/global frequencies: 0.016746. Environment-adjusted: 0.014739.
All 13 matched environments improved this diagnostic. This is annotation association,
not sound-event accuracy: missing labels are not verified negatives and context
captions are generated. No fine/coarse F1 or FP/min is available without hand labels.

Runtime check: a 63.616-second cached YouTube video (`iTHtXgnhqfM`) completed real
extraction, environment classification, PANNs/CLAP, hierarchy filtering, and VTT
writing. Baseline: 34 events; conditioned: 43 events / 42 readable cues, including
11 newly exposed AudioSet-class events. These counts do not establish precision.
Stereo/surround direction remains independent (this video had verified 5.1).

Outcome: keep bounded coefficients and explicit uncertainty documentation; validate
caption quality with labeled clips before changing thresholds or CLAP calibration.
Full source hashes/counts: `docs/research/sound-priors-report.json`.
