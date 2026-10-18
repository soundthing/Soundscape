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
5. **Vocabulary and thresholds:** all 527 AudioSet labels plus five CLAP-only classes, hierarchy filtering, and per-class thresholds.
6. **Baseline vs conditioned:** both outputs every run, to show the environment layer helps.

Also: hand-label a share of the eval clips. Prior changes go through Person 2.

## Stack

| Layer | Choice |
|---|---|
| Language / core | Python 3.10+, numpy, soundfile |
| ML runtime | PyTorch + torchaudio (resampling) |
| AudioSet tagger | PANNs CNN14 via `panns-inference` (BEATs / YAMNet as alternatives) |
| Zero-shot scoring | LAION CLAP (`laion/clap-htsat-unfused`) via Hugging Face `transformers` |
| Speech detection | Silero VAD via `silero-vad` |
| Post-processing | plain numpy (no scipy) |
| Testing / eval | pytest; `soundthing eval` against hand labels |

Model libraries come from the `events` install extra, so the stub path works without
them. Details and rationale: [references/stack.md](references/stack.md).

## Status

**Done (code, tested with fake models):** prior conditioning, AudioSet → vocabulary
mapping, tagger (PANNs CNN14) and CLAP loading and window scoring, Silero VAD speech
masking, all post-processing steps, modifiers, the end-to-end `detect` flow with models
and raw scores shared across both modes, and the wrong-environment experiment
(`python -m soundthing.events.experiments`).

**Expanded detection:** all 527 PANNs labels map independently into 532 caption
classes, with the 61 original ids retained. Hierarchy suppression and cue-padding
protection have synthetic integration tests. Real environment + PANNs + CLAP inference was run on a 63.616-second cached
YouTube video: 34 baseline and 43 conditioned events. The 532 × 527 mapping,
VTT output, ancestor suppression, and new AudioSet-class outputs were verified. There are still no labeled-clip F1 / FP-per-minute measurements.
CLAP calibration and the default thresholds remain untuned placeholders.

**Data-derived prior proposal:** Sound-VECaps context captions joined with AudioSet
training annotations cover all 527 labels. Estimation excludes official evaluation
and AudioCaps validation/test ids, splits remaining records by video id, and shrinks
sparse cases toward neutral. These are weak annotation associations, not measured
real-world occurrence probabilities or detector-accuracy improvements.

**Next:** label clips and measure precision/recall and false positives before
tuning thresholds. The expanded conditioned path has completed a real-video run.

## Milestones

1. **Baseline detection:** load the tagger, score windows with it alone, threshold into
   events. Get the first event F1 and false-positives-per-minute numbers.
   *Code done (`events.clap.enabled: false` gives tagger-only); numbers pending.*
2. **CLAP:** add zero-shot scores so classes AudioSet covers poorly (or not at all) get
   detected. Calibrate them to sit alongside tagger scores. Re-measure.
   *Code done; calibration and measurement pending.*
3. **Speech masking:** add VAD so dialogue stops triggering false events. *Code done.*
4. **Post-processing:** smoothing, merging, coarse fallback when unsure between similar
   sounds, once-only captions for sustained sounds, and a cap on captions on screen.
   *Done; hierarchy-aware ancestor suppression and cue-padding protection added.*
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

- **Detection:** the tagger and CLAP each score every window; a class takes the higher of
  the two. Models load once per run and are shared by both modes.
- **Speech masking:** VAD finds speech spans; non-safety scores inside them are scaled
  down.
- **Conditioning:** already implemented; applies Person 2's environment probabilities
  through the priors table.
- **Post-processing:** offline, so it can look at future context. Precision over recall:
  when in doubt, don't caption, except for safety sounds.
- **Tuning:** agents hill-climb one fixed objective through an MCP harness over cached
  model scores ([tuning-plan.md](tuning-plan.md)); results are recorded in
  [references/tuning-log.md](references/tuning-log.md).
- **Tests:** post-processing and scoring are tested with synthetic scores and fake models;
  real-model tests skip when the `events` extra isn't installed.

## Open checks

- Verify the AudioSet names in `vocabulary.yaml` against the tagger's label list
  (`soundthing check-config --audioset <csv>`).
- `vocabulary.yaml` is a shared interface file: each edit needs explicit confirmation.
