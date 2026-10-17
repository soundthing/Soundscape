"""Environment conditioning: apply P(event | environment) soft priors.

This is the consumer side of Person 2's env_segments.json and the single place
that defines what the numbers in config/priors.yaml mean.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..contracts import EnvSegment, env_probs_at
from .types import FrameScores


def prior_weights(cfg: Config, env_probs: dict[str, float], class_ids: list[str]) -> np.ndarray:
    """(C,) weight(c) = sum_e p(e) * w[e][c]; safety classes -> 1."""
    safety = cfg.safety_ids()
    w = np.zeros(len(class_ids))
    for env_id, p in env_probs.items():
        ep = cfg.priors.environments[env_id]
        w += p * np.array([ep.weight(c) for c in class_ids])
    w[[i for i, c in enumerate(class_ids) if c in safety]] = 1.0
    return w


def apply_priors(scores: FrameScores, segments: list[EnvSegment], cfg: Config) -> FrameScores:
    floor = cfg.priors.floor
    out = scores.scores.copy()
    for t_idx, t in enumerate(scores.times):
        w = prior_weights(cfg, env_probs_at(segments, float(t)), scores.class_ids)
        out[t_idx] *= floor + (1.0 - floor) * w
    return scores.copy_with(out)
