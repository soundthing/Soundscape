# soundthing: non-speech captions for video

## What we're building

soundthing watches a video and writes captions for meaningful non-speech sounds, such as
`[door slams]`, `[footsteps approaching]`, `[glass shatters]` or `[dog barking, distant]`.
The output is a timed WebVTT/SRT track. Plenty of tools already caption speech. Non-speech
sound is mostly left out, and it carries information that deaf and hard-of-hearing viewers
miss. The core product is the model pipeline that produces these captions.

**Key idea.** Sound event detection gets noisy when it tries to recognize too many classes
at once. We cut that noise by first classifying the *environment* of each part of the video
(street, kitchen, office, nature, sports venue, vehicle interior) from both audio and
frames. That environment then reweights which sound events are likely. A small
**safety set** is always active whatever the environment: sirens, alarms, horns, glass
breaking, screams, gunshots, explosions and a baby crying.

### Scope

**In scope (MVP)**
- Offline processing of MP4/MOV/WebM files. There is no real-time requirement.
- A `.vtt` track of non-speech events, with an optional merge into an existing speech
  subtitle file.
- A multimodal environment classifier (audio + frames sampled at about 1 fps).
- Environment-conditioned sound event detection over a curated vocabulary of about 60
  classes.
- A web demo: upload a video, then play it back with baseline and conditioned captions side
  by side.
- Eval data: the AudioSet strong-label eval split (~6k 10 s clips with event onsets mapped to our vocabulary). It has no environment labels, so it scores baseline mode. No hand labeling.

**Out of scope (for now):** live streams (but the model core is designed so it can be
reused for them), speech transcription (a VAD only masks speech regions), music or speaker
identification, emotion or tone of speech, and training large models. We use pretrained
models and fine-tune only small heads.

### Design principles
- **Precision over recall** for display. A wrong caption is worse than a missed one,
  except for safety-critical sounds.
- **Pure model core.** `score_window(audio, sr, models) -> scores` has no state, so a
  streaming wrapper can reuse it later.
- **Config, not code.** Everything specific to an environment or a class lives in
  `config/`.
- **Always compare against the baseline.** Every run produces both the audio-only,
  unconditioned output and the conditioned output, so we can show that the environment
  layer helps.

---

## Architecture

```
video file
  ├─ ffmpeg → audio.wav (16 kHz mono) + audio_stereo.wav
  └─ ffmpeg → frames/frame_%06d.jpg (~1 fps)
                │
                ▼
 ┌──────────── Person 2: environment ───────────┐
 │ 1. Segmentation    PySceneDetect shot spans    │
 │ 2. Audio branch    CLAP vs env prompts         │
 │    Visual branch   CLIP/SigLIP vs env prompts  │
 │    Fusion          weighted late fusion,       │
 │                    low confidence → "general"  │
 └──────────────── env_segments.json ────────────┘
                │
                ▼
 ┌──────────── Person 1: sound events ──────────┐
 │ 3. Sliding windows (1 s / 0.25 s hop)          │
 │    AudioSet tagger → vocabulary + CLAP scores  │
 │    VAD speech masking                          │
 │ 4. Conditioning    soft priors P(event|env),   │
 │                    safety set bypasses         │
 │ 5. Post-process    smoothing, thresholds,      │
 │                    merging, coarse fallback,   │
 │                    priority filter, modifiers  │
 └──────────── events.{baseline,conditioned}.json ┘
                │
                ▼
 ┌──────────── Person 3: product & eval ────────┐
 │ 6. VTT/SRT writer, speech subtitle merge       │
 │    FastAPI demo, side-by-side player           │
 │    Eval: event F1, FP/min, env accuracy        │
 └───────────────────────────────────────────────┘
```

Visuals are used **only** for environment context. Sound events are always detected from
audio, because many important sounds come from off-screen.

---

## Components

| | Covers | Code |
|---|---|---|
| **Person 1: sound events** | tagger, CLAP extra labels, AudioSet→vocabulary mapping, thresholds, safety set, VAD, post-processing, applying priors | `src/soundthing/events/`, `config/vocabulary.yaml`, `config/thresholds.yaml` |
| **Person 2: environment** | shot segmentation, visual and audio branches, fusion, confidence and `general` fallback, environment list, priors table | `src/soundthing/environment/`, `config/environments.yaml`, `config/priors.yaml` |
| **Person 3: pipeline, product, eval** | ffmpeg extraction, orchestration/CLI, VTT/SRT and subtitle merge, FastAPI and player, eval harness, baseline vs conditioned comparison, pitch | `extract.py`, `pipeline.py`, `captions.py`, `evaluate.py`, `server/` |
| **Labels** | hand-labeled eval clips | `labels/` |

Shared and changed only by agreement: `contracts.py`, `config.py`, `stages.py`,
`docs/interfaces.md`.

---

## Interfaces

The full spec is in [docs/interfaces.md](docs/interfaces.md). The source of truth is
[src/soundthing/contracts.py](src/soundthing/contracts.py), and valid examples are in
[docs/examples/](docs/examples/). Every JSON file uses the envelope
`{"schema_version": 1, "kind": ..., "data": ...}`, so a stale or mismatched file fails
loudly. Loaders also accept the bare payload for hand-written fakes.

| Contract | Direction | File | Shape |
|---|---|---|---|
| `extracted_inputs` | P3 → all | `runs/<id>/inputs.json` | `{video_id, video_path, audio_path, audio_stereo_path, frames_dir, fps, duration, sample_rate}` |
| `env_segments` | P2 → P1 | `runs/<id>/env_segments.json` | `[{start, end, env, confidence, probs}]`, contiguous and covering `[0, duration]` |
| `caption_events` | P1 → P3 | `runs/<id>/events.<mode>.json` | `[{start, end, label, class_id, confidence, modifiers}]`, final and ready to display |
| `clip_labels` | all → P3 | `labels/<id>.json` | `{video_id, duration, labeler, segments[{start,end,env}], events[{start,end,class_id}]}` |

**Stage interfaces** ([stages.py](src/soundthing/stages.py)):

```python
EnvironmentClassifier(inputs, cfg) -> list[EnvSegment]                  # Person 2
EventDetector(inputs, segments | None, cfg) -> list[CaptionEvent]       # Person 1; None = baseline
```

`backend: stub | real` in `config/pipeline.yaml` chooses which implementation runs.

**Agreed semantics:**
- **`general` fallback.** If the classifier's max probability is below
  `confidence_threshold`, then `env = "general"` and `probs` is mixed as
  `conf·p + (1−conf)·onehot(general)`. Conditioning reads `probs`, so low confidence
  smoothly weakens conditioning toward none.
- **Priors.** `weight(c,t) = Σ_e probs_t[e]·w[e][c]`, and safety classes always get 1.
  The conditioned score is `raw · (floor + (1−floor)·weight)`. The floor (0.25) means a
  misclassified environment weakens events instead of deleting them.
- **`class_id` vs `label`.** `label` is the caption text and `class_id` is the vocabulary
  id, or a coarse group id such as `impact` when the detector isn't sure. Eval uses
  `class_id`.

---

## Shared config (`config/`)

| File | Contents |
|---|---|
| `environments.yaml` | 7 environments (`general` + 6), each with CLIP visual prompts and CLAP audio prompts |
| `vocabulary.yaml` | 61 classes (10 safety) with caption text, AudioSet label mapping, CLAP prompts, coarse group, `safety` and `sustained` flags; 7 coarse groups |
| `priors.yaml` | per-environment `default` + class weights, plus `floor` |
| `thresholds.yaml` | default 0.5, safety default 0.6, per-class overrides (e.g. gunshot 0.75) |
| `pipeline.yaml` | backends, extraction rate/fps, segmentation, model names, fusion weights, window/hop, VAD suppression, post-processing params, caption cue limits, eval tolerance |

Adding an environment means one entry in `environments.yaml` plus one block in
`priors.yaml`, with no code changes. Run `soundthing check-config` after any edit. It
catches unknown ids, missing prior blocks, out-of-range values and id collisions. It also
warns about non-safety classes that no environment lists. Add `--audioset <csv>` to check
the AudioSet names against the checkpoint's label list.

---

## What is set up today

### Working now
- **Contracts and validation:** dataclasses, JSON load/save, and checks for span ordering,
  probability sums, segment coverage and allowed modifiers.
- **Config loading and cross-validation**, plus the `check-config` command.
- **Stub backends** for the environment classifier and the event detector. They're
  deterministic for each `video_id` and produce valid output. The event stub applies the
  real priors, so baseline and conditioned output already differ.
- **End-to-end CLI:** `soundthing run` (a video, or an existing `inputs.json`) →
  segments → both event sets → both VTT files.
- **ffmpeg extraction:** mono and stereo WAV, frames at 1 fps, duration via ffprobe.
- **VTT/SRT writer** with cue duration clamping.
- **Fusion and `general` fallback** (`environment/fusion.py`).
- **Prior conditioning** (`events/conditioning.py`), the AudioSet → vocabulary mapping
  matrix, the speech-mask helper, and the sliding-window scoring loop.
- **Eval harness:** greedy onset-tolerance matching (0.5 s), precision/recall/F1 at
  fine and coarse levels, FP per minute, and time-weighted environment accuracy, via
  `soundthing eval`.
- **FastAPI demo:** upload, background processing, status polling, video, VTT and segment
  endpoints. The page has two synced players (baseline vs conditioned) and shows the
  current environment.
- **Tests:** 16 pytest tests covering contracts, config, conditioning, fusion, eval
  matching, caption formatting, an end-to-end stub run, and a perfect-score eval
  self-check.

### Not yet verified
- The upload → ffmpeg path, because ffmpeg isn't installed on the dev machine where this
  was set up.
- The AudioSet names in `vocabulary.yaml` haven't been checked against a real label CSV.

### Open work (marked `TODO(Person N)`, raises `NotImplementedError`)

| Person | Where | Task |
|---|---|---|
| 1 | `events/tagger.py` `load_tagger` | load the PANNs/BEATs checkpoint and its label list |
| 1 | `events/clap.py` | CLAP loading, per-class text embeddings, calibrated window scores |
| 1 | `events/core.py` `score_window` | combine tagger and CLAP scores into one window score |
| 1 | `events/detector.py` `load_models` | build `EventModels` |
| 1 | `events/vad.py` `speech_regions` | Silero VAD |
| 1 | `events/postprocess.py` | `smooth`, `threshold_to_events`, `collapse_confusable`, `priority_filter`, `add_modifiers` |
| 2 | `environment/segmentation.py` | PySceneDetect spans with min/max length |
| 2 | `environment/visual.py` | CLIP/SigLIP frame pooling and prompt similarity |
| 2 | `environment/audio.py` | CLAP segment pooling and prompt similarity |
| 3 | `captions.py` `merge_with_speech` | parse and interleave the speech subtitle track |

---

## Milestones

1. **Data and vocabulary:** finish the environment list, vocabulary and safety set, then
   import the AudioSet strong-label eval set. *(Done.)*
2. **Baseline:** audio-only detection over the full vocabulary → VTT. Measure event F1 and
   false positives per minute.
3. **Environment classifier:** audio-only, then video-only, then fused. Measure accuracy
   for each (`classify(use_audio=..., use_visual=...)`).
4. **Conditioning:** priors plus the safety set, compared against the baseline, including
   cases where the environment is misclassified.
5. **Post-processing and caption quality:** smoothing, coarse collapsing, priority
   filtering.
6. **Demo:** upload → process → side-by-side playback.
7. **Stretch:** few-shot custom sounds (P1), and a streaming mode that reuses
   `score_window` (P3).

### How the team works in parallel
- **Kickoff:** confirm the environment list, vocabulary and contracts (drafted in this
  repo).
- **Parallel build:** P1 swaps `events.backend` to `real` once the baseline works. P2 does
  the same for `environment.backend`. P3 keeps the stub path working end to end the whole
  time, so it's the fallback if a model component runs late.
- **Integration and tuning:** run `soundthing eval` early and often. It settles "did this
  change help?" quickly.
- **Priors table** is the shared hotspot: every component reads it.

---

## Repo layout

```
config/            shared YAML config (see above)
docs/
  interfaces.md    full contract spec
  examples/        valid example JSON for each contract
labels/            optional `clip_labels` JSON for pipeline runs (eval data: labels/README.md)
src/soundthing/
  contracts.py     shared types + JSON IO + validation
  config.py        config loading + cross-validation
  stages.py        stage interfaces, stub/real selection
  environment/     Person 2: segmentation, visual, audio, fusion, classifier, stub
  events/          Person 1: core, tagger, clap, vad, conditioning, postprocess, detector, stub
  extract.py       Person 3: ffmpeg
  captions.py      Person 3: VTT/SRT
  evaluate.py      Person 3: metrics
  pipeline.py      Person 3: orchestration + CLI
  server/          Person 3: FastAPI app + static/index.html
tests/
runs/ uploads/ data/   generated or media (gitignored)
```

## Quick start

```bash
python -m venv .venv
```

```bash
.venv/Scripts/pip install -e ".[server,dev]"
```

```bash
soundthing check-config
```

```bash
soundthing run data/clips/clip01.mp4
```

```bash
soundthing eval --runs runs --labels labels
```

```bash
soundthing serve
```

Add the `events` or `environment` extras to get the model dependencies for your component.
