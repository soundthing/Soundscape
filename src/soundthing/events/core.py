"""The model core: a pure function from one audio window to vocabulary scores.

Keep this stateless (models are passed in, not loaded here) so a future
real-time wrapper can call it on a live ring buffer unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EventModels:
    """Loaded models + precomputed text embeddings. Built once by detector.load_models."""

    tagger: Any
    clap: Any
    class_ids: list[str]
    tagger_map: np.ndarray  # (C, A) 0/1 matrix: vocabulary class <- AudioSet label
    clap_text_emb: np.ndarray | None  # (C, D); zero rows for classes without prompts


def score_window(audio: np.ndarray, sr: int, models: EventModels) -> np.ndarray:
    """audio: (N,) float32 mono at `sr`. Returns (C,) scores in [0, 1] in
    models.class_ids order: max(tagger mapped score, CLAP score) per class."""
    raise NotImplementedError("TODO(Person 1): tagger + CLAP window scoring")
