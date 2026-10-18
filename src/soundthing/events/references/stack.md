# Stack

## Models

| Role | Model | Why | Expects |
|---|---|---|---|
| AudioSet tagger | PANNs CNN14 (`panns-inference`) | Strong AudioSet tagger, simple pip install, covers most of the vocabulary directly | 32 kHz mono; outputs 527 AudioSet probabilities |
| Speech detection | Silero VAD (`silero-vad`) | Small, fast, accurate on CPU | 16 kHz mono |
| Source separation | Demucs `htdemucs` (`demucs`) | Optional source separation; extracts effects stem (mix minus vocals) to rescore speech/music masked windows | 44.1 kHz stereo; we feed 16 kHz mono upsampled and duplicated, so separation quality above 8 kHz is unavailable |

Extracted audio is 16 kHz mono, so PANNs needs resampling to 32 kHz (torchaudio). CLAP remains an environment-stage dependency only.

**Alternatives kept open:** BEATs (stronger tagger, heavier setup) and YAMNet (lighter,
TensorFlow) as tagger options, selectable via `events.tagger.model`. Whisper timestamps as
a VAD alternative if Silero misses speech under music.

## Libraries

- **Base (always installed):** numpy, pyyaml, soundfile. Post-processing uses only these;
  there's no scipy.
- **`events` extra:** torch, torchaudio, panns-inference, silero-vad. Import
  these lazily so the stub path and CI run without them.
- **`separation` extra:** torch, torchaudio, demucs. Optional; imported lazily by the separator backend.
- **Evaluation harness:** its extra and Kaggle compute are in
  [harness/references/stack.md](../../harness/references/stack.md).
- **Testing:** pytest. Real-model tests skip when the extra isn't installed.

## Runtime

Offline only. CPU must work; CUDA is used when available. Model weights download on first
use (PANNs checkpoint ~300 MB; Demucs ~80 MB from Hugging Face). Event detection no longer
loads a CLAP model. CPU ~3x real time per separated second on this machine (measured on a
6 s clip), so separation runs only on gated spans.

Dev mixes resample with torchaudio (events extra), keeping scipy out.
