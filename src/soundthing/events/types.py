"""Internal (Person 1) intermediate type: per-window class scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrameScores:
    times: np.ndarray  # (T,) window centre times in seconds
    class_ids: list[str]  # (C,) vocabulary ids, column order of `scores`
    scores: np.ndarray  # (T, C) in [0, 1]
    hop: float  # seconds between windows

    def __post_init__(self) -> None:
        assert self.scores.shape == (len(self.times), len(self.class_ids)), self.scores.shape

    def copy_with(self, scores: np.ndarray) -> "FrameScores":
        return FrameScores(self.times, self.class_ids, scores, self.hop)

    def column(self, class_id: str) -> np.ndarray:
        return self.scores[:, self.class_ids.index(class_id)]
