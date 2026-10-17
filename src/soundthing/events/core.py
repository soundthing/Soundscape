"""The model core: a pure function from one audio window to vocabulary scores.

Keep this stateless (models are passed in, not loaded here) so a future
real-time wrapper can call it on a live ring buffer unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


from .clap import clap_scores
from .tagger import map_to_vocab


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
    tagger_raw = models.tagger(audio[None, :], sr)[0]
    scores = map_to_vocab(tagger_raw, models.tagger_map)

    if models.clap is not None and models.clap_text_emb is not None:
        clap_s = clap_scores(models.clap, audio, sr, models.clap_text_emb)
        scores = np.maximum(scores, clap_s)

    return np.clip(scores, 0.0, 1.0).astype(np.float32)
