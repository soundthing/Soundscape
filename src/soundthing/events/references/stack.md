# Stack

## Models

| Role | Model | Why | Expects |
|---|---|---|---|
| AudioSet tagger | PANNs CNN14 (`panns-inference`) | Strong AudioSet tagger, simple pip install, covers most of the vocabulary directly | 32 kHz mono; outputs 527 AudioSet probabilities |
| Speech detection | Silero VAD (`silero-vad`) | Small, fast, accurate on CPU | 16 kHz mono |

Extracted audio is 16 kHz mono, so PANNs needs resampling to 32 kHz (torchaudio). CLAP remains an environment-stage dependency only.

**Alternatives kept open:** BEATs (stronger tagger, heavier setup) and YAMNet (lighter,
TensorFlow) as tagger options, selectable via `events.tagger.model`. Whisper timestamps as
a VAD alternative if Silero misses speech under music.

## Libraries

- **Base (always installed):** numpy, pyyaml, soundfile. Post-processing uses only these;
  there's no scipy.
- **`events` extra:** torch, torchaudio, panns-inference, silero-vad. Import
  these lazily so the stub path and CI run without them.
- **Testing:** pytest. Real-model tests skip when the extra isn't installed.

## Runtime

Offline only. CPU must work; CUDA is used when available. Model weights download on first
use (PANNs checkpoint ~300 MB). Event detection no longer loads a CLAP model.
