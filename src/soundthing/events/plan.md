# Sound events roadmap (Person 1)

The events stage takes the audio and Person 2's environment segments and produces the
final caption events (`events.{baseline,conditioned}.json`) for Person 3 to render.

Background knowledge (stack details, decisions, pitfalls, tuning results) lives in
[references/](references/README.md).

## Responsibilities

1. **Detection:** score every vocabulary class over short sliding audio windows.
2. **Speech masking:** turn down non-safety guesses where people are talking.
3. **Conditioning:** reweight likely events by environment; the safety set bypasses this.
4. **Post-processing:** turn raw scores into clean, display-ready captions.
5. **Vocabulary and thresholds:** all 527 AudioSet labels, hierarchy filtering, and per-class thresholds. The five former CLAP-only classes and their configuration were retired with user approval.
6. **Baseline vs conditioned:** both outputs every run, to show the environment layer helps.

Also: hand-label a share of the eval clips. Prior changes go through Person 2.

## Stack

| Layer | Choice |
|---|---|
| Language / core | Python 3.10+, numpy, soundfile |
| ML runtime | PyTorch + torchaudio (resampling) |
| AudioSet tagger | PANNs CNN14 via `panns-inference` (BEATs / YAMNet as alternatives) |
| Speech detection | Silero VAD via `silero-vad` |
| Post-processing | plain numpy (no scipy) |
| Testing / eval | pytest; `soundthing eval` against hand labels |

Model libraries come from the `events` install extra, so the stub path works without
them. Details and rationale: [references/stack.md](references/stack.md).

## Status

**Done (code, tested with fake models):** prior conditioning, AudioSet → vocabulary
mapping, tagger (PANNs CNN14) loading and window scoring, Silero VAD speech
masking, all post-processing steps, modifiers, the end-to-end `detect` flow with models
and raw scores shared across both modes, and the wrong-environment experiment
(`python -m soundthing.events.experiments`).

**Expanded detection:** all 527 PANNs labels now map independently into exactly
527 caption classes. The original expansion retained 61 ids and included five
CLAP-only classes; those five have since been retired. Hierarchy suppression and cue-padding
protection have synthetic integration tests. Real environment + PANNs + CLAP inference was run on a 63.616-second cached
YouTube video: 34 baseline and 43 conditioned events. The 532 × 527 mapping,
VTT output, ancestor suppression, and new AudioSet-class outputs were verified. There are still no labeled-clip F1 / FP-per-minute measurements.
That run predates removal of event CLAP and does not validate PANNs-only caption quality. Thresholds now have an initial specificity policy, with no labeled calibration yet. CLAP is now used only by environment detection.

**PANNs-only verification:** 213 tests pass. A short local real-model run generated
both schema-version-2 VTT tracks, with zero baseline events and one conditioned
vehicle event, while real environment CLAP and SigLIP inference remained active.
These counts verify execution, not caption accuracy.

**Data-derived prior proposal:** Sound-VECaps context captions joined with AudioSet
training annotations cover all 527 labels. Estimation excludes official evaluation
and AudioCaps validation/test ids, splits remaining records by video id, and shrinks
sparse cases toward neutral. These are weak annotation associations, not measured
real-world occurrence probabilities or detector-accuracy improvements.

**Prior storage:** the optional unfloored format restores all 3,045 estimates
previously clamped to 0.25, with the shared `floor` as the sole adjustable runtime
minimum. Default effective weights remain unchanged; old bounded configs retain
their validation and behavior. No JSON schema change is required.

**Specificity policy:** all 527 classes have explicit, editable broad/specific/fine
assignments, with current thresholds 0.45/0.55/0.65 (lowered from the initial
0.50/0.65/0.80 at user request). Existing safety requirements
are preserved and per-class overrides remain minimums. Legacy configs without
tiers keep their lookup. Scores are unchanged; precision/recall effects have not
been measured on labeled clips. Generic fallback groups are a separate proposal.

**Generic fallback preparation:** optional inference is implemented and tested
against fake groups under the current contract. It takes maximum descendant
support, uses a measured activity gate on real audio, and leaves native model
scores unchanged. Default config leaves this disabled. Exact new live group ids
and optional coarse-group ancestry await their separate interface approval.

**Post-processing simplification:** the user requested removing temporal score
smoothing, the 60-second sustained-event cooldown and the concurrent-caption
limit. All qualifying unrelated events and repeated sustained events now survive
those former filters. Legacy settings cannot reactivate them. Thresholds, minimum
duration, gap merging, coarse labels, hierarchy suppression, modifiers and the
caption writer's existing duration limits remain.

**Next:** label clips and measure precision/recall and false positives before
tuning thresholds. The expanded conditioned path has completed a real-video run.

## Milestones

1. **Baseline detection:** load the tagger, score windows with it alone, threshold into
   events. Get the first event F1 and false-positives-per-minute numbers.
   *Code done; PANNs is now the sole event scorer. Labeled accuracy numbers pending.*
2. **Event CLAP removed:** the user requested PANNs-only sound classification after reviewing the untuned zero-shot conversion and max-fusion behavior. Environment CLAP remains. The five unmapped classes were removed from the vocabulary and priors; schema-version-1 runs require regeneration.
3. **Speech masking:** add VAD so dialogue stops triggering false events. *Code done.*
4. **Post-processing:** merging, minimum event duration, coarse fallback when unsure
   between similar sounds, hierarchy-aware suppression and cue-padding protection.
   *Done. Smoothing, sustained-repeat suppression and the caption-count limit were
   subsequently removed at user request.*
5. **Conditioned vs baseline:** compare both modes on the eval clips, including clips
   where the environment is deliberately wrong. *Tooling done (`soundthing eval` plus the
   wrong-env experiment); needs labeled clips.*
6. **Tuning:** calibrate thresholds and parameters on the labeled clips without
   overfitting a small set. *Planned as an agent hill-climbing harness:
   [tuning-plan.md](tuning-plan.md).*
7. **Go live:** switch `events.backend` to `real` once the stage runs end to end
   (use a local `--config` copy until then).
8. **Modifiers (nice to have):** left/right, approaching/receding, loud/quiet. *Done;
   thresholds untuned.*
9. **Stretch:** few-shot custom sounds. This needs an interface discussion first.

## Approach

- **Detection:** only PANNs scores sound-event windows. Models load once per run and are shared by both modes. Old local zero-shot settings cannot reactivate the removed scorer.
- **Speech masking:** VAD finds speech spans; non-safety scores inside them are scaled
  down.
- **Conditioning:** already implemented; applies Person 2's environment probabilities
  through the priors table.
- **Post-processing:** preserve qualifying detections without temporal smoothing,
  sustained-repeat suppression or a caption-count cap. Retain per-class thresholds,
  event duration checks, gap merging and hierarchy suppression.
- **Tuning:** agents hill-climb one fixed objective through an MCP harness over cached
  model scores ([tuning-plan.md](tuning-plan.md)); results are recorded in
  [references/tuning-log.md](references/tuning-log.md).
- **Tests:** post-processing and scoring are tested with synthetic scores and fake models;
  real-model tests skip when the `events` extra isn't installed.

## Open checks

- Verify the AudioSet names in `vocabulary.yaml` against the tagger's label list
  (`soundthing check-config --audioset <csv>`).
- `vocabulary.yaml` is a shared interface file: each edit needs explicit confirmation.
