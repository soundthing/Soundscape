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

## Direction track — Person 3 analyzer → server/extension (`runs/<id>/direction.json`)

Independent of event/environment models. Uses the existing schema-version-1 envelope
with `kind: "direction"`. This additive artifact does not change existing payloads.

`DirectionTrack = {duration: float, mode: str, channels: list[str], frames: list[DirectionFrame]}`;
`DirectionFrame = {time: float, levels: dict[str, float]}`.

- `mode`: `stereo`, `surround51`, or `unavailable`.
- Channels in order: stereo `[FL, FR]`; surround `[FL, FR, FC, SL, SR]`;
  unavailable `[]`, with no frames. LFE never contributes direction.
- `5.1` back and `5.1(side)` side feeds normalize to `SL`/`SR`. These are nominal
  5.1 playback feeds, not measured source angles; use center 0°, fronts ±30°,
  surrounds ±110° for the top-down display. Stereo indicates lateral balance only.
- Duration is finite and nonnegative. Available frames start at zero, have strictly
  increasing finite times below duration, and contain exactly the declared channels.
- Levels are finite `[0,1]` strengths: RMS over 50 ms, linearly map -60..0 dBFS to
  0..1, then smooth with 50 ms attack / 200 ms release. No per-channel normalization.
- Mono/unknown layouts and analysis failures use `unavailable`. Existing captions
  remain usable independently. The analyzer reads original media, not downmixed WAVs.
- `GET /api/videos/{id}/direction` returns the validated **bare track** (no envelope)
  when present; 404 when absent. The extension retries independently of captions.
- Stored tracks cache analysis, including unavailable results. Missing tracks in older
  YouTube runs trigger queued analysis; the downloader checks available audio formats
  and may reacquire a channel-preserving source under `uploads/youtube/`.

See `docs/examples/direction.json` for a deterministic fake track usable without
audio tools or real models. Use `load_direction` / `save_direction` for on-disk IO.

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

### Legacy prior semantics ([`events/conditioning.py`](../src/soundthing/events/conditioning.py))

```
w[e][c]          = priors.environments[e].classes[c]  or  priors.environments[e].default
weight(c, t)     = Σ_e probs_t[e] · w[e][c]          (safety classes: 1)
conditioned(c,t) = raw(c,t) · (floor + (1 − floor) · weight(c,t))
```

This is the backward-compatible `multiplicative` mode. The current fitted table
selects `odds`; see “Prior application modes” below. In either mode, per-class
thresholds from `thresholds.yaml` apply to the conditioned score.

## Evaluation ([`evaluate.py`](../src/soundthing/evaluate.py))

- Event P/R/F1: greedy match, same class, `|onset_pred − onset_true| ≤ 0.5 s`.
  Reported `fine` (exact class) and `coarse` (both sides mapped to coarse group).
- False positives per minute of labeled audio.
- Environment accuracy: time-weighted agreement of `env` with labeled `env`.

## Optional sound hierarchy

A sound class in `vocabulary.yaml` may include `parents: [class_id, ...]`.
The default is `[]`, so existing config and schema-version-1 artifacts remain valid.
Parents must reference vocabulary classes or coarse groups. Multiple parents are
allowed; unknown ids, duplicate parents and cycles are rejected by `check-config`.
Existing `coarse` membership also makes that group an ancestor for display purposes.
`Config.hierarchy()` resolves all ancestors through intermediate nodes.

Person 1 suppresses a qualifying ancestor only during intervals covered by a
qualifying descendant. Parent and child meet their own thresholds independently;
siblings and unrelated simultaneous events remain eligible. Apply this after
thresholds/minimum event duration and before the concurrent-caption limit.
Person 3 uses the same hierarchy when extending cue durations, so readability
padding cannot reintroduce parent/child duplicates. The event JSON fields and
schema version do not change; Person 2's environment outputs are unchanged.
See `docs/examples/hierarchy.yaml` for a configuration-only example.

### Full AudioSet vocabulary

The live vocabulary contains 527 independently scored AudioSet labels plus five
existing CLAP-only classes. Every PANNs output maps to exactly one caption class.
Existing class ids are retained for their first (primary) AudioSet label. Former
aliases become independent classes; this prevents a broad label such as `Dog`
from directly scoring the existing `dog_bark` class. New ids are
`audioset_` plus the AudioSet machine id with slashes replaced by underscores
(e.g. `/m/09x0r` becomes `audioset_m_09x0r`). No existing id is renamed.
`parents` comes from all paths in the AudioSet ontology, projected through any
unscored intermediate nodes to the nearest scored ancestors. Multiple inheritance
is retained. Existing labels and stored JSON stay loadable; regenerating runs is
necessary to see expanded detection and hierarchy filtering.

## Prior application modes

`priors.yaml` optionally selects `application: multiplicative | odds` and
`max_gain: float >= 1` (default 4). Old configurations omit these fields and keep
their existing multiplication behavior and coefficient range `[0,1]`.
In `odds` mode `floor` must be positive, coefficients/defaults must be finite and
in `[floor, max_gain]`, and the general environment must have neutral weight 1.
Safety classes always have effective weight 1 regardless of the table.

Person 1 mixes coefficients with Person 2's existing `probs` distribution:
`w = sum_e probs[e] * coefficient[e][class]`. Odds mode applies
`adjusted = score*w / (1-score+score*w)`, with `w` bounded by `floor`/`max_gain`.
Unlike multiplying a probability by a small coefficient, positive odds weights
permit every class to meet its threshold when acoustic evidence is strong enough.
A score of 0 stays 0 and a score of 1 stays 1. General leaves scores unchanged.
The resulting unit-range score is not a calibrated real-world probability.

The fitted table replaces manually chosen scene coefficients. It estimates weak
annotation enrichment from Sound-VECaps context descriptions joined with AudioSet
training annotations, with video-id holdout, excluded evaluation ids, shrinkage and
neutral unsupported cases. It covers all 527 PANNs labels; the five CLAP-only
classes have no dataset label mapping and remain neutral. Person 2 owns this table;
Person 1 consumes it, and Person 3 receives the same schema-version-1 event JSON.
No environment id or stored JSON field changes. See `docs/environment-priors.md`,
`docs/research/sound-priors-report.json`, and `docs/examples/priors-odds.yaml`.
