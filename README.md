# soundthing

Captions for meaningful non-speech sounds in video, such as `[door slams]`,
`[siren approaching]` or `[glass shatters]`, written as a timed WebVTT track.

Speech captioning is a solved problem. Non-speech sound mostly isn't captioned at all, and
it carries information deaf and hard-of-hearing viewers miss. Generic sound event
detectors are too noisy to caption from directly, because they try to recognize hundreds
of classes everywhere. soundthing first works out *where* each part of the video takes
place (street, kitchen, office, vehicle...) and uses that to reweight which sounds are
likely. A safety set (sirens, alarms, gunshots, screams, a baby crying...) is always
active whatever the environment. Precision beats recall: a wrong caption is worse than a
missed one.

## How it works

```
video ─ ffmpeg ─┬─ audio (16 kHz) ──────────────────────────┐
                └─ frames (~1 fps) ─┐                        │
                                    ▼                        ▼
                     environment classifier  ──►  sound event detector  ──►  VTT
                     (audio + frames, per shot)   (audio only, conditioned
                                                   on the environment)
```

1. **Extraction** (`extract.py`): ffmpeg pulls mono/stereo audio and frames.
2. **Environment** (`environment/`): PySceneDetect splits shots, then SigLIP (frames) and
   CLAP (audio) score each shot against environment prompts. Late fusion combines them,
   and low confidence falls back to `general`.
3. **Events** (`events/`): PANNs CNN14 tags sliding 1 s windows, Silero VAD masks speech,
   and per-environment priors reweight class scores. Post-processing smooths, thresholds,
   merges and filters into final caption events.
4. **Captions and product** (`captions.py`, `server/`, `extension/`): a VTT/SRT writer,
   a local FastAPI backend, and a Chrome extension whose popup controls synchronized
   captions and soundtrack direction waves on YouTube.
5. **Evaluation** (`evaluate.py`, `harness/`): event F1 and false positives per minute on
   the AudioSet strong-label eval set, plus a harness that tunes each stage.

Every run produces both a **baseline** (unconditioned) and a **conditioned** output, so the
effect of the environment layer is always measurable.

**Key parts of the stack:** Python 3.10+, PyTorch, Hugging Face `transformers` (SigLIP,
CLAP), `panns-inference`, Silero VAD, PySceneDetect, ffmpeg, FastAPI. Stages talk through
versioned JSON contracts ([docs/interfaces.md](docs/interfaces.md)), and everything
environment- or class-specific lives in `config/*.yaml`, not code.

More detail: [project.md](project.md). Rules for coding agents: [AGENTS.md](AGENTS.md).

## Setup

Needs Python 3.10+ and ffmpeg (`winget install Gyan.FFmpeg`, `brew install ffmpeg` or
`sudo apt install ffmpeg`).

```bash
python -m venv .venv
```

```bash
.venv/Scripts/pip install -e ".[events,environment,harness,server,dev]"
```

Model weights, datasets and harness data aren't in git. `soundthing setup` fetches them;
each step skips what already exists, so rerun it after fixing a failure.

```bash
.venv/Scripts/soundthing setup
```

```bash
.venv/Scripts/soundthing setup check
```

| Step | What it does |
|---|---|
| `setup models` | PANNs CNN14, CLAP, SigLIP, Silero VAD (`--separation` adds demucs) |
| `setup data` | ESC-50 (~600 MB) and AudioSet strong-eval clips from Kaggle (~1.3 GB) |
| `setup harness` | Deterministic dev/holdout split and ESC-50 mixes (`--caches` builds feature caches; hours on CPU) |
| `setup check` | Reports what's missing and the command that fixes it |

The AudioSet download needs the Kaggle CLI, an API token in `~/.kaggle`, and access to
`wwilliamx07/soundthing-audioset-eval` (override with `SOUNDTHING_KAGGLE_AUDIOSET`).
Without it, pass `--skip-audioset`.

To move a workspace to another machine, including state that can't be rebuilt (ledger,
`data/config-*`, `runs/`):

```bash
soundthing setup pack --with-caches --archive soundthing-workspace.zip
```

```bash
soundthing setup --archive soundthing-workspace.zip
```

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

`serve` starts the local API used by the Chrome extension. The toolbar popup is the
app interface; there is no separate browser workspace.

`run` writes `runs/<video_id>/` with the extracted inputs, environment segments, and
baseline/conditioned events and VTT files. The Chrome extension is documented in
[extension/README.md](extension/README.md).
