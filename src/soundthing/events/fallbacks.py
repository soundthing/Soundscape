"""Conservative scores for configured, unscored generic hierarchy groups."""

from __future__ import annotations

import numpy as np

from ..config import Config
from .types import FrameScores


def window_rms_dbfs(audio: np.ndarray, sr: int, times: np.ndarray, window_sec: float) -> np.ndarray:
    """Recording-level RMS over the same centred, zero-padded windows as scoring."""
    win = max(1, int(window_sec * sr))
    starts = np.rint(times * sr - win / 2).astype(np.int64)
    ends = np.clip(starts + win, 0, len(audio))
    starts = np.clip(starts, 0, len(audio))
    energy = np.empty(len(audio) + 1, dtype=np.float64)
    energy[0] = 0
    np.square(audio, out=energy[1:])
    np.cumsum(energy[1:], out=energy[1:])
    mean_square = np.maximum((energy[ends] - energy[starts]) / win, 0)
    return 10 * np.log10(np.maximum(mean_square, 1e-24))


def add_generic_scores(scores: FrameScores, rms_dbfs: np.ndarray, cfg: Config) -> FrameScores:
    """Append generic scores without altering any independently scored class.

    Each configured coarse group inherits the strongest eligible descendant's
    final score, gated by recording activity. No probability summing or boosting.
    Normal thresholding and hierarchy suppression decide which labels display.
    """
    settings = cfg.pipeline.get('events', {}).get('generic_fallbacks', {})
    if not settings.get('enabled', False):
        return scores
    if np.shape(rms_dbfs) != (len(scores.times),):
        raise ValueError('generic fallback RMS must match score windows')

    hierarchy = cfg.hierarchy()
    excluded = set(settings.get('exclude_classes', []))
    eligible = [(i, cid) for i, cid in enumerate(scores.class_ids)
                if cid not in excluded and cid not in cfg.coarse_groups]
    active = np.asarray(rms_dbfs) >= settings.get('min_rms_dbfs', -60.)
    groups, columns = [], []
    for group in settings.get('groups', []):
        if group in scores.class_ids:
            continue
        indices = [i for i, cid in eligible if group in hierarchy.get(cid, ())]
        support = (scores.scores[:, indices].max(axis=1) if indices
                   else np.zeros(len(scores.times), dtype=scores.scores.dtype))
        groups.append(group)
        columns.append(np.where(active, support, 0))
    if not columns:
        return scores
    return FrameScores(scores.times, [*scores.class_ids, *groups],
                       np.column_stack([scores.scores, *columns]), scores.hop)
