"""AudioSet tagger (PANNs/BEATs/YAMNet) and mapping to the caption vocabulary."""

from __future__ import annotations

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


def load_tagger(cfg: Config):
    """Return (model, audioset_names). Model maps (B, N) audio at 16 kHz -> (B, A) probs."""
    raise NotImplementedError("TODO(Person 1): load PANNs/BEATs checkpoint")
