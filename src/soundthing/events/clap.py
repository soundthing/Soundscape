"""CLAP zero-shot scoring for vocabulary classes AudioSet doesn't cover well."""

from __future__ import annotations

import numpy as np

from ..config import Config


def load_clap(cfg: Config):
    raise NotImplementedError("TODO(Person 1): load CLAP")


def text_embeddings(clap, cfg: Config) -> np.ndarray:
    """(C, D) mean-of-prompts embedding per class (zero row if no `clap` prompts)."""
    raise NotImplementedError("TODO(Person 1)")


def clap_scores(clap, audio: np.ndarray, sr: int, text_emb: np.ndarray) -> np.ndarray:
    """(C,) calibrated scores in [0, 1] for one window. Calibrate similarities
    (e.g. sigmoid with per-class offset) so they are comparable to tagger probs."""
    raise NotImplementedError("TODO(Person 1)")
