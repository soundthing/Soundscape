"""Fake event detector: valid output, no models. Deterministic per video_id.

When env segments are given, applies the configured multiplicative/odds priors so baseline vs
conditioned output differ visibly in the demo even before real models exist.
"""

from __future__ import annotations

import zlib

import numpy as np

from ..config import Config
from ..contracts import CaptionEvent, EnvSegment, ExtractedInputs, env_probs_at
from .conditioning import prior_weights, adjust_scores
from .hierarchy import suppress_ancestors


def detect_stub(
    inputs: ExtractedInputs, segments: list[EnvSegment] | None, cfg: Config
) -> list[CaptionEvent]:
    cfg.hierarchy()  # Exercise the same optional hierarchy contract as real configs.
    rng = np.random.default_rng(zlib.crc32(inputs.video_id.encode()))
    events: list[CaptionEvent] = []
    t = float(rng.uniform(0.5, 2.0))
    while t < inputs.duration - 0.5:
        c = cfg.classes[int(rng.integers(len(cfg.classes)))]
        score = float(rng.uniform(0.5, 0.95))
        if segments is not None:
            w = prior_weights(cfg, env_probs_at(segments, t), [c.id])[0]
            score = float(adjust_scores(score, w, floor=cfg.priors.floor,
                                        application=cfg.priors.application,
                                        max_gain=cfg.priors.max_gain))
        end = min(inputs.duration, t + float(rng.uniform(0.4, 2.5)))
        if score >= cfg.threshold(c.id) and end > t:
            events.append(
                CaptionEvent(start=round(t, 2), end=round(end, 2), label=c.caption, class_id=c.id,
                             confidence=round(score, 3))
            )
        t = end + float(rng.uniform(0.5, 3.0))
    return suppress_ancestors(events, cfg.hierarchy())
