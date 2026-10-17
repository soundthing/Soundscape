"""AudioSet tagger (PANNs/BEATs/YAMNet) and mapping to the caption vocabulary."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import Config


def build_tagger_map(cfg: Config, audioset_names: list[str]) -> np.ndarray:
    """(C, A) 0/1 matrix: row c has 1s at the AudioSet labels listed for class c."""
    index = {n: i for i, n in enumerate(audioset_names)}
    m = np.zeros((len(cfg.classes), len(audioset_names)), dtype=np.float32)
    for ci, c in enumerate(cfg.classes):
        for name in c.audioset:
            if name not in index:
                raise KeyError(f"class {c.id}: AudioSet label {name!r} not in tagger labels")
            m[ci, index[name]] = 1.0
    return m


def map_to_vocab(audioset_scores: np.ndarray, tagger_map: np.ndarray) -> np.ndarray:
    """(..., A) AudioSet probs -> (..., C) class scores: max over mapped labels
    (0 for classes with no AudioSet labels)."""
    masked = audioset_scores[..., None, :] * tagger_map  # (..., C, A)
    return masked.max(axis=-1)


class PannsTagger:
    """Callable wrapper around PANNs AudioTagging model.

    Resamples input audio to 32 kHz with torchaudio and returns clipwise probabilities.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Score audio batch (B, N) at sample rate `sr` -> (B, A) clipwise probabilities."""
        if sr != 32000:
            import torch
            import torchaudio

            t = torch.as_tensor(audio, dtype=torch.float32)
            t = torchaudio.functional.resample(t, sr, 32000)
            audio = t.cpu().numpy()

        clipwise_output, _ = self.model.inference(audio)
        return np.asarray(clipwise_output, dtype=np.float32)


def load_tagger(cfg: Config) -> tuple[PannsTagger, list[str]]:
    """Return (model, audioset_names). Model maps (B, N) audio at sr -> (B, A) probs.

    Resamples sr -> 32000 with torchaudio.functional.resample.
    Only 'panns_cnn14' is supported; other values raise NotImplementedError.
    """
    model_name = cfg.pipeline.get("events", {}).get("tagger", {}).get("model")
    if model_name != "panns_cnn14":
        raise NotImplementedError(
            f"Unsupported tagger model: {model_name!r}. Only 'panns_cnn14' is supported."
        )

    import torch
    from panns_inference import AudioTagging, labels

    device = "cuda" if torch.cuda.is_available() else "cpu"
    at = AudioTagging(checkpoint_path=None, device=device)
    return PannsTagger(at), list(labels)
