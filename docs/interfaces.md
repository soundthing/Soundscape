# Interfaces

The contracts between the three components. Source of truth is
[`src/soundthing/contracts.py`](../src/soundthing/contracts.py) (dataclasses + validation);
examples live in [`docs/examples/`](examples/). Changing anything here needs agreement
from everyone.

```
video ──► [P3 extract] ──inputs.json──┬──► [P2 environment] ──env_segments.json──┐
                                      │                                          ▼
                                      └──────────────────────────────► [P1 events] ──events.<mode>.json──► [P3 captions / eval / demo]
```

Every JSON file is wrapped in an envelope so stale files fail loudly:

```json
{ "schema_version": 1, "kind": "<kind>", "data": ... }
```

Loaders also accept the bare `data` payload, so hand-written fakes can skip the envelope.

## 1. `extracted_inputs` — Person 3 → everyone (`runs/<id>/inputs.json`)

| field | type | notes |
|---|---|---|
| `video_id` | str | file stem, `[A-Za-z0-9_-]` |
| `video_path` | str | original video |
| `audio_path` | str | 16 kHz mono 16-bit WAV — model input |
| `audio_stereo_path` | str \| null | original-rate stereo WAV, for left/right modifiers |
| `frames_dir` | str | `frame_000000.jpg …`; frame *i* is at *t = i / fps* |
| `fps` | float | ~1 |
| `duration` | float | seconds |
| `sample_rate` | int | of `audio_path` |

Paths are relative to the JSON file on disk and resolved to absolute by `load_inputs`.

## 2. `env_segments` — Person 2 → Person 1 (`runs/<id>/env_segments.json`)

List of `{start, end, env, confidence, probs}`.

- Sorted, contiguous, cover `[0, duration]` exactly (`validate_segments` checks this).
- `probs`: distribution over environment ids from `config/environments.yaml`; sums to 1.
- `confidence`: the classifier's max probability over real environments.
- **Fallback:** if `confidence < confidence_threshold` → `env = "general"` and
  `probs` is mixed: `p' = conf·p + (1 − conf)·onehot(general)`. Downstream
  conditioning reads `probs`, never just `env`, so low confidence smoothly weakens
  conditioning toward none (`general` has prior 1.0 for every class).

## 3. `caption_events` — Person 1 → Person 3 (`runs/<id>/events.{baseline,conditioned}.json`)

List of `{start, end, label, class_id, confidence, modifiers}`.

- `label`: caption text without brackets (`"door slams"`), from `vocabulary.yaml`
  `caption` or a coarse group's `caption`.
- `class_id`: vocabulary id, or coarse group id when the detector collapsed to a coarse
  label (`"impact"`). Eval uses this, not `label`.
- `confidence` in [0, 1].
- `modifiers` ⊆ `left, right, distant, nearby, approaching, receding, loud, quiet, continues`.
  Rendered as `[door slams, right]`.
- These are **final, display-ready** events: thresholding, merging, coarse collapsing and
  priority filtering already applied. Person 3 only clamps cue durations for readability.

`baseline` = audio-only, no environment conditioning (`segments=None`).
`conditioned` = with Person 2's segments.

## 4. `clip_labels` — everyone → Person 3 (`labels/<video_id>.json`)

`{video_id, duration, labeler, segments: [{start, end, env}], events: [{start, end, class_id, note?}]}`
with fine class ids only. See [`labels/README.md`](../labels/README.md).

## Python stage interfaces ([`stages.py`](../src/soundthing/stages.py))

```python
EnvironmentClassifier = (inputs: ExtractedInputs, cfg: Config) -> list[EnvSegment]          # Person 2
EventDetector = (inputs: ExtractedInputs, segments: list[EnvSegment] | None, cfg) -> list[CaptionEvent]  # Person 1
```

`config/pipeline.yaml` `environment.backend` / `events.backend` pick `stub` or `real`.
Stubs produce valid, deterministic fake output so the whole path runs from day one.

Inside Person 1's component, the model core is
`events.core.score_window(audio_window, sr, models) -> (C,) scores` — keep it pure so a
streaming wrapper can reuse it.

## Shared config ([`config/`](../config/))

| file | owner | what |
|---|---|---|
| `environments.yaml` | P2 | environment ids + CLIP/CLAP prompts; must include `general` |
| `vocabulary.yaml` | P1 | caption classes, AudioSet mapping, CLAP prompts, coarse groups, `safety`, `sustained` |
| `priors.yaml` | **P2 (single owner)** | `P(event \| env)` weights; semantics below |
| `thresholds.yaml` | P1 | per-class thresholds |
| `pipeline.yaml` | per section | window/hop, models, fusion weights, post-processing, eval tolerance |

Run `soundthing check-config` after any edit (add `--audioset class_labels_indices.csv`
to verify AudioSet names).

### Prior semantics ([`events/conditioning.py`](../src/soundthing/events/conditioning.py))

```
w[e][c]          = priors.environments[e].classes[c]  or  priors.environments[e].default
weight(c, t)     = Σ_e probs_t[e] · w[e][c]          (safety classes: 1)
conditioned(c,t) = raw(c,t) · (floor + (1 − floor) · weight(c,t))
```

Then per-class thresholds from `thresholds.yaml` apply to the conditioned score.

## Evaluation ([`evaluate.py`](../src/soundthing/evaluate.py))

- Event P/R/F1: greedy match, same class, `|onset_pred − onset_true| ≤ 0.5 s`.
  Reported `fine` (exact class) and `coarse` (both sides mapped to coarse group).
- False positives per minute of labeled audio.
- Environment accuracy: time-weighted agreement of `env` with labeled `env`.
