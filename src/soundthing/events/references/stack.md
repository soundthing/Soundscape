# Stack

## Models

| Role | Model | Why | Expects |
|---|---|---|---|
| AudioSet tagger | PANNs CNN14 (`panns-inference`) | Strong AudioSet tagger, simple pip install, covers most of the vocabulary directly | 32 kHz mono; outputs 527 AudioSet probabilities |
| Zero-shot scorer | LAION CLAP `laion/clap-htsat-unfused` (`transformers`) | Scores classes AudioSet lacks or covers poorly from text prompts alone | 48 kHz mono; scores are cosine similarities, not probabilities |
| Speech detection | Silero VAD (`silero-vad`) | Small, fast, accurate on CPU | 16 kHz mono |
| Source separation | Demucs `htdemucs` (`demucs`) | Optional source separation; extracts effects stem (mix minus vocals) to rescore speech/music masked windows | 44.1 kHz stereo; we feed 16 kHz mono upsampled and duplicated, so separation quality above 8 kHz is unavailable |

Extracted audio is 16 kHz mono, so the tagger and CLAP need resampling (torchaudio).

**Alternatives kept open:** BEATs (stronger tagger, heavier setup) and YAMNet (lighter,
TensorFlow) as tagger options, selectable via `events.tagger.model`. Whisper timestamps as
a VAD alternative if Silero misses speech under music.

## Libraries

- **Base (always installed):** numpy, pyyaml, soundfile. Post-processing uses only these;
  there's no scipy.
- **`events` extra:** torch, torchaudio, panns-inference, transformers, silero-vad. Import
  these lazily so the stub path and CI run without them.
- **`separation` extra:** torch, torchaudio, demucs. Optional; imported lazily by the separator backend.
- **`tuning` extra:** mcp (2.x server SDK), optuna (TPE for sweeps), kaggle CLI. Imported
  lazily so the base package and standard pipeline run without them. Note that mcp 2.x is
  required by the server code (pyproject still says `mcp>=1.2`; the floor should be raised
  to `>=2.0`).
- **Testing:** pytest. Real-model tests skip when the extra isn't installed.

## Runtime

Offline only. CPU must work; CUDA is used when available. Model weights download on first
use (PANNs checkpoint ~300 MB, CLAP ~600 MB; Demucs ~80 MB from Hugging Face). CPU ~3x real
time per separated second on this machine (measured on a 6 s clip), so separation runs only
on gated spans.

Dev mixes resample with torchaudio (events extra), keeping scipy out.

## Compute

- **Kaggle cache runner:** Kaggle (kaggle 1.7 CLI/API, free T4/P100) builds dev caches; the kernel installs the repo at a pushed git sha.
