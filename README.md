# soundthing

Captions for meaningful non-speech sounds in video — `[door slams]`, `[siren approaching]` —
as a WebVTT track. Sound event detection is conditioned on the environment of each part
of the video (classified from audio + frames), with a safety set that is always active.

## Setup

You need Python 3.10 or later and `ffmpeg`/`ffprobe`. On Windows run
`winget install Gyan.FFmpeg`, on macOS `brew install ffmpeg`, on Linux `sudo apt install ffmpeg`.

The quick path is the demo with stub backends:

```bash
python -m venv .venv
```

```bash
.venv/Scripts/pip install -e ".[server,dev]"
```

### Full development workspace

Git doesn't track model weights, datasets or harness data (`data/`, `runs/`).
`soundthing setup` rebuilds them. Each step skips whatever already exists, so if one
fails, fix it and run it again.

```bash
.venv/Scripts/pip install -e ".[events,environment,harness,server,dev]"
```

```bash
.venv/Scripts/soundthing setup
```

```bash
.venv/Scripts/soundthing setup check
```

`setup` runs these steps in order. You can also run any step on its own:

| Step | What it does | Where it puts it |
|---|---|---|
| `setup models` | PANNs CNN14 + AudioSet class list, CLAP and SigLIP (ids from `config/pipeline.yaml`), Silero VAD. `--separation` adds demucs | `~/panns_data`, HF and torch caches |
| `setup data` | ESC-50 from GitHub (about 600 MB) and AudioSet strong-eval clips from Kaggle (about 1.3 GB) | `data/esc50`, `data/audioset_strong` |
| `setup harness` | AudioSet dev/holdout split (hash-based, so it comes out the same on every machine) and ESC-50 dev mixes. `--caches` also builds feature caches, which takes hours on CPU | `data/harness/` |
| `setup check` | Reports what's present and what's missing, with the command that fixes each gap | — |

The AudioSet download needs the Kaggle CLI (`pip install kaggle`), an API token in
`~/.kaggle` (kaggle.com → Settings → API), and access to the
`wwilliamx07/soundthing-audioset-eval` dataset. Set `SOUNDTHING_KAGGLE_AUDIOSET` to use a
different copy. Without Kaggle, pass `--skip-audioset`.

### Moving to a new machine

Some state can't be rebuilt: the experiment ledger, the local `--config` copies
(`data/config-*`), the test clip, `runs/`, and the exact ESC-50 subset the recorded dev
mixes came from. Feature caches can be rebuilt, but that's slow. To carry all of it over:

```bash
soundthing setup pack --with-caches --archive soundthing-workspace.zip
```

Copy the zip to the new machine and run:

```bash
soundthing setup --archive soundthing-workspace.zip
```

This unpacks the zip first (it never overwrites existing files unless you pass `--force`),
then fetches whatever is still missing. `setup unpack` restores the zip without doing anything else.

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
  workspace.py          `soundthing setup`: models, datasets, harness data, pack/unpack
  server/               Person 3: FastAPI demo + side-by-side player
tests/
```

Open work is marked `TODO(Person N)` and raises `NotImplementedError`.
