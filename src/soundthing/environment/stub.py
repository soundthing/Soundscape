"""Fake environment classifier: valid output, no models. Deterministic per video_id."""

from __future__ import annotations

import zlib

import numpy as np

from ..config import Config
from ..contracts import GENERAL_ENV, EnvSegment, ExtractedInputs
from .fusion import to_segments


def classify_stub(inputs: ExtractedInputs, cfg: Config) -> list[EnvSegment]:
    env_ids = [e for e in cfg.env_ids if e != GENERAL_ENV]
    max_len = cfg.pipeline["environment"]["segmentation"]["max_segment_sec"]
    n = max(1, int(np.ceil(inputs.duration / max_len)))
    edges = np.linspace(0.0, inputs.duration, n + 1)
    spans = list(zip(edges[:-1].tolist(), edges[1:].tolist()))

    # One "true" environment per clip, with a confident-ish distribution.
    main = zlib.crc32(inputs.video_id.encode()) % len(env_ids)
    probs = np.full((n, len(env_ids)), 0.3 / max(1, len(env_ids) - 1))
    probs[:, main] = 0.7
    return to_segments(spans, probs, env_ids, cfg.pipeline["environment"]["confidence_threshold"])
