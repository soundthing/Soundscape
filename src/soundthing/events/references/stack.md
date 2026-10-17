# Stack

## Models

| Role | Model | Why | Expects |
|---|---|---|---|
| AudioSet tagger | PANNs CNN14 (`panns-inference`) | Strong AudioSet tagger, simple pip install, covers most of the vocabulary directly | 32 kHz mono; outputs 527 AudioSet probabilities |
| Zero-shot scorer | LAION CLAP `laion/clap-htsat-unfused` (`transformers`) | Scores classes AudioSet lacks or covers poorly from text prompts alone | 48 kHz mono; scores are cosine similarities, not probabilities |
| Speech detection | Silero VAD (`silero-vad`) | Small, fast, accurate on CPU | 16 kHz mono |

Extracted audio is 16 kHz mono, so the tagger and CLAP need resampling (torchaudio).

**Alternatives kept open:** BEATs (stronger tagger, heavier setup) and YAMNet (lighter,
TensorFlow) as tagger options, selectable via `events.tagger.model`. Whisper timestamps as
a VAD alternative if Silero misses speech under music.

## Libraries

- **Base (always installed):** numpy, pyyaml, soundfile. Post-processing uses only these;
  there's no scipy.
- **`events` extra:** torch, torchaudio, panns-inference, transformers, silero-vad. Import
  these lazily so the stub path and CI run without them.
- **Testing:** pytest. Real-model tests skip when the extra isn't installed.

## Runtime

Offline only. CPU must work; CUDA is used when available. Model weights download on first
use (PANNs checkpoint ~300 MB, CLAP ~600 MB).
