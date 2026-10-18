"""Fake event detector: valid output, no models. Deterministic per video_id.

When env segments are given, applies the configured multiplicative/odds priors so baseline vs
conditioned output differ visibly in the demo even before real models exist.
Unfloored coefficients use the same runtime minimum and mixing as real detection.
Specificity requirements use Config.threshold, including its safety exceptions;
the randomly generated stub scores themselves are not adjusted by the tiers.
"""

from __future__ import annotations

import zlib

import numpy as np

from ..config import Config
from ..contracts import CaptionEvent, EnvSegment, ExtractedInputs, env_probs_at
from .conditioning import prior_weights, adjust_scores
from .hierarchy import suppress_ancestors
from .fallbacks import add_generic_scores
from .types import FrameScores


def detect_stub(
    inputs: ExtractedInputs, segments: list[EnvSegment] | None, cfg: Config
) -> list[CaptionEvent]:
    cfg.hierarchy()  # Exercise the same optional hierarchy contract as real configs.
    classes = [c for c in cfg.classes if c.audioset]
    rng = np.random.default_rng(zlib.crc32(inputs.video_id.encode()))
    events: list[CaptionEvent] = []
    t = float(rng.uniform(0.5, 2.0))
    while t < inputs.duration - 0.5:
        c = classes[int(rng.integers(len(classes)))]
        score = float(rng.uniform(0.5, 0.95))
        if segments is not None:
            w = prior_weights(cfg, env_probs_at(segments, t), [c.id])[0]
            score = float(adjust_scores(score, w, floor=cfg.priors.floor,
                                        application=cfg.priors.application,
                                        max_gain=cfg.priors.max_gain))
        end = min(inputs.duration, t + float(rng.uniform(0.4, 2.5)))
        # Stub activity is synthetic too: nominal 0 dBFS, not measured input audio.
        # Reuse the real generic support/exclusion rule, without loading models.
        probe = FrameScores(np.array([(t + end) / 2]), [c.id], np.array([[score]]), end - t)
        probe = add_generic_scores(probe, np.array([0.]), cfg)
        for i, cid in enumerate(probe.class_ids):
            value = float(probe.scores[0, i])
            if value >= cfg.threshold(cid) and end > t:
                events.append(
                    CaptionEvent(start=round(t, 2), end=round(end, 2), label=cfg.caption_for(cid), class_id=cid,
                                 confidence=round(value, 3))
                )
        t = end + float(rng.uniform(0.5, 3.0))
    return suppress_ancestors(events, cfg.hierarchy())
