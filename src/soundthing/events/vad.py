"""Speech masking (Silero VAD or Whisper timestamps)."""

from __future__ import annotations

import numpy as np

from ..config import Config


def speech_regions(audio: np.ndarray, sr: int, cfg: Config) -> list[tuple[float, float]]:
    """Return sorted (start, end) speech spans in seconds."""
    raise NotImplementedError("TODO(Person 1): Silero VAD")


def speech_mask(times: np.ndarray, regions: list[tuple[float, float]]) -> np.ndarray:
    """(T,) bool: True where window centre falls inside speech."""
    mask = np.zeros(len(times), dtype=bool)
    for s, e in regions:
        mask |= (times >= s) & (times < e)
    return mask
