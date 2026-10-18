"""Environment conditioning: apply configured soft environment priors.

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


def adjust_scores(scores, weights, *, application="multiplicative", floor=.25, max_gain=4.):
    """Legacy multiplication or positive bounded odds ratios, without hard vetoes."""
    scores = np.asarray(scores)
    weights = np.asarray(weights)
    if application == "multiplicative":
        return scores * (floor + (1. - floor) * weights)
    if application != "odds":
        raise ValueError(f"Unknown prior application: {application}")
    if not 0 < floor <= 1 <= max_gain:
        raise ValueError("Odds priors require a positive floor and max_gain >= 1")
    ratios = np.clip(weights, floor, max_gain)
    numerator = scores * ratios
    return numerator / (1. - scores + numerator)


def apply_priors(scores: FrameScores, segments: list[EnvSegment], cfg: Config) -> FrameScores:
    floor = cfg.priors.floor
    out = scores.scores.copy()
    for t_idx, t in enumerate(scores.times):
        w = prior_weights(cfg, env_probs_at(segments, float(t)), scores.class_ids)
        out[t_idx] = adjust_scores(out[t_idx], w, floor=floor,
                                   application=getattr(cfg.priors, "application", "multiplicative"),
                                   max_gain=getattr(cfg.priors, "max_gain", 4.))
    return scores.copy_with(out)
