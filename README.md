# soundthing

Captions for meaningful non-speech sounds in video — `[door slams]`, `[siren approaching]` —
as a WebVTT track. Sound event detection is conditioned on the environment of each part
of the video (classified from audio + frames), with a safety set that is always active.

## Setup

Requires Python ≥ 3.10 and `ffmpeg`/`ffprobe` on `PATH`.

```bash
python -m venv .venv
```

```bash
.venv/Scripts/pip install -e ".[server,dev]"
```

Add `events` and/or `environment` extras for the model components you work on.

Agent rules for everyone (Claude Code, Codex, Antigravity) are in [AGENTS.md](AGENTS.md).

## Usage

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

`run` writes `runs/<video_id>/` with `inputs.json`, `env_segments.json`,
`events.{baseline,conditioned}.json` and `captions.{baseline,conditioned}.vtt`.
Pass an existing `runs/<id>/inputs.json` instead of a video to skip extraction.
Both model components default to `backend: stub` in `config/pipeline.yaml`, so the
whole path works with fake output until the real models land.

## Layout

```
config/                 shared config — see docs/interfaces.md
  environments.yaml  vocabulary.yaml  priors.yaml  thresholds.yaml  pipeline.yaml
docs/interfaces.md      the JSON contracts between components  (+ docs/examples/)
labels/                 optional clip labels for pipeline runs (eval data: labels/README.md)
src/soundthing/
  contracts.py          shared dataclasses + JSON IO + validation      (everyone)
  config.py             config loading + cross-validation              (everyone)
  stages.py             stage interfaces, stub/real backend selection  (everyone)
  environment/          Person 2: segmentation, visual/audio branches, fusion
  events/               Person 1: tagger, CLAP, VAD, conditioning, post-processing
  extract.py            Person 3: ffmpeg extraction
  captions.py           Person 3: VTT/SRT writer, speech subtitle merge
  evaluate.py           Person 3: event F1, FP/min, env accuracy
  pipeline.py           Person 3: orchestration + CLI
  server/               Person 3: FastAPI demo + side-by-side player
tests/
```

Open work is marked `TODO(Person N)` and raises `NotImplementedError`.
