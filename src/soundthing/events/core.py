"""The model core: a pure function from one audio window to vocabulary scores.

Keep this stateless (models are passed in, not loaded here) so a future
real-time wrapper can call it on a live ring buffer unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .tagger import map_to_vocab


@dataclass
class EventModels:
    """Loaded AudioSet tagger and vocabulary mapping. Built once by detector.load_models."""

    tagger: Any
    class_ids: list[str]
    tagger_map: np.ndarray  # (C, A) 0/1 matrix: vocabulary class <- AudioSet label


def score_window(audio: np.ndarray, sr: int, models: EventModels) -> np.ndarray:
    """audio: (N,) float32 mono at `sr`. Returns (C,) scores in [0, 1] in
    models.class_ids order, using only mapped AudioSet tagger scores."""
    tagger_raw = models.tagger(audio[None, :], sr)[0]
    scores = map_to_vocab(tagger_raw, models.tagger_map)

    return np.clip(scores, 0.0, 1.0).astype(np.float32)
