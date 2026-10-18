# Dataset-derived environment priors

The live `config/priors.yaml` table is fitted from the
[Sound-VECaps 1.0 visual-enhanced context captions](https://zenodo.org/records/12606207)
and [Google's AudioSet training annotations](https://research.google.com/audioset/download.html).
The AudioSet event labels are human annotated; the context descriptions are
**generated**, and environments are matched using explicit scene phrases. This is
a weak estimate of annotated sound/scene association, not measured real-world
occurrence probabilities or a calibrated posterior.

Sound-VECaps was chosen because it supplies context for AudioSet-scale data without
downloading or relabeling videos. AudioSet alone lacks our fine scene labels.
[MACS](https://zenodo.org/records/5114771) has human scene/context annotation but
only a few scenes and a small sound-label set; [FSD50K](https://zenodo.org/records/4060432)
covers only 200 AudioSet classes. Neither supplies a complete 527-class table for
our 13 specific environments. The generated-context compromise is explicit here.

## Evidence and split

Joined **1,644,654** records: **1,315,263** training and **329,391** held-out records. All 527 PANNs labels have training positives.

- Exclude official AudioSet evaluation ids and AudioCaps validation/test ids.
- Hold out 20% by SHA-256 of video id; duplicate captions contribute only once.
- Reject mismatched multi-segment video ids, uncertain context words, and captions
  matching multiple environments. Match environments only from context phrases;
  a frying label alone does not make a clip a kitchen.
- Expand positive labels through the ontology so a known child also counts toward
  its ancestors. AudioSet does not exhaustively label every audible sound.

| Environment | Matched training clips | Matched held-out clips |
|---|---:|---:|
| street | 12,634 | 3,163 |
| kitchen | 16,930 | 4,276 |
| office | 10,120 | 2,521 |
| nature | 23,484 | 5,838 |
| sports_venue | 8,423 | 2,104 |
| vehicle_interior | 6,477 | 1,623 |
| lecture_hall | 124 | 29 |
| restaurant | 2,028 | 521 |
| cafe | 1,610 | 436 |
| public_transportation | 1,000 | 263 |
| factory | 919 | 251 |
| construction | 2,138 | 507 |
| classroom | 3,672 | 862 |

Lecture-hall evidence is particularly limited. General/unknown remains neutral.
The full source SHA-256 hashes, class counts, and per-environment counts are in
[`research/sound-priors-report.json`](research/sound-priors-report.json).
No source captions or downloaded audio are redistributed in the repository.
The report preserves the original fit, including five zero-support classes later
retired with event CLAP. The live table retains only the 527 AudioSet columns.
The 3,045 coefficients originally clamped to 0.25 have been recovered from its
recorded counts; all other stored coefficients are unchanged. With the default
runtime minimum, effective coefficients still match the original table. The
diagnostic losses describe the original fit.

## Coefficients

For each sound, let `p0 = (global_positive_count + 0.5)/(training_clips + 1)`.
For an environment with `n` matched clips and `k` positives,
`pe = (k + 200*p0)/(n + 200)`. The enrichment is `pe/p0`; its log is shrunk by
`global_positive_count/(global_positive_count + 20)` and capped at 4.
Fewer than 20 class positives or fewer than 50 environment clips gives neutral
weight 1. Safety sounds remain neutral. The five former CLAP-only classes have
been removed from the live vocabulary and priors. Absent
annotations are not verified acoustic negatives; these are dataset associations.

Examples from this fit: kitchen/sizzling, office/typing, nature/birdsong,
sports/cheering, and street/traffic have coefficient 4. Kitchen/train horn has
stored fitted coefficient about 0.012356, with effective coefficient 0.25 under
the default minimum. These are relative evidence adjustments, not event probabilities.

### Adjust the runtime minimum in one place

The live `config/priors.yaml` contains:

```yaml
coefficient_format: unfloored
floor: 0.25  # Change this minimum; no coefficient refit is needed.
```

For example, `floor: 0.05` permits lower effective coefficients while leaving all
stored estimates intact. Each environment coefficient is bounded before mixing
environment probabilities. Changing the minimum does not change the fitted
estimates. The duplicate fitter `floor` setting in `pipeline.yaml` was removed;
unfloored fitting reads this shared setting, including when an older settings file
still contains a floor. Restart the backend and regenerate a video's cached
captions to observe a changed minimum. New-format fitting reports record
`coefficient_format` and the resolved floor; their held-out diagnostic uses the
effective coefficients that runtime inference would apply.

All 3,045 former 0.25 entries were reconstructed using the original smoothing,
support rules, and counts, with estimates ranging from about 0.009001 to 0.249735.
Of these pairs, 2,331 had no positive scene annotation; their nonzero values come
from smoothing. These are dataset-derived estimates, not measured real-world rates.
The upper cap, safety exceptions, and unsupported-case neutral defaults are unchanged.

At runtime, bound coefficients using the configurable minimum/maximum, then mix
them using the full environment probability distribution.
Then `adjusted_score = score*w/(1-score+score*w)`. A score of 0 stays 0; a score
of 1 stays 1. Positive coefficients ensure every sound can cross its threshold
in any environment, given sufficiently strong audio. No environment blacklists a
sound. Safety classes always use 1; unknown/general also uses 1.

## Validation limits

Held-out annotation log loss, weighted by matched clip count, changed from **0.016746** to **0.014739**. Each of the 13 matched environments improved on this diagnostic.

This tests prediction of *dataset annotations*, treating unannotated labels as absent
for the diagnostic. It is not an event-detection accuracy result: generated captions
may misdescribe a location, omitted annotations can hide real sounds, and the corpus
is selected rather than a random sample of real-world environments. The thresholds
still need labeled-clip evaluation (fine/coarse F1, FP/min). Event CLAP has been removed; environment CLAP remains.

## Reproduction

Keep the public metadata (about 500 MiB, no audio) in `data/audioset/`:

- `ontology.json`: [AudioSet ontology](https://raw.githubusercontent.com/audioset/ontology/master/ontology.json)
- `balanced_train_segments.csv`, `unbalanced_train_segments.csv`, `eval_segments.csv`:
  [official AudioSet download page](https://research.google.com/audioset/download.html)
- `audiocaps_val.csv`, `audiocaps_test.csv`:
  [AudioCaps annotations](https://github.com/cdjkim/audiocaps/tree/master/dataset)
- `Sound-VECaps_full.csv`: [publisher's metadata](https://zenodo.org/records/12606207)
  (publisher MD5: `e975cb5bbf37538962a5c6434ba97c8f`).

Scene phrases and fitting parameters live in the Person 1 section
`events.prior_estimation` of `config/pipeline.yaml`; the minimum lives only in
`config/priors.yaml`. To generate separate proposals:

```bash
.venv/bin/python -m soundthing.events.prior_fit \
  --config config --data data/audioset \
  --labels ~/panns_data/class_labels_indices.csv \
  --output /tmp/priors-proposal.yaml --report /tmp/priors-report.json
```

The CLI refuses to overwrite that configuration's live priors. Applying a proposal
requires the shared-file confirmation in `AGENTS.md`. Environment ids and event JSON
are unchanged. Regenerate cached runs after changing vocabulary, hierarchy, or priors.
