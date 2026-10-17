"""Late fusion of per-modality environment probabilities, and conversion to the
EnvSegment contract (including the `general` fallback)."""

from __future__ import annotations

import numpy as np

from ..contracts import GENERAL_ENV, EnvSegment


def softmax(logits: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def fuse(
    audio_probs: np.ndarray | None,
    visual_probs: np.ndarray | None,
    audio_weight: float,
    visual_weight: float,
) -> np.ndarray:
    """Weighted late fusion of (S, E) probability matrices. Either may be None
    (single-modality runs for the per-modality accuracy comparison)."""
    parts = [(p, w) for p, w in ((audio_probs, audio_weight), (visual_probs, visual_weight)) if p is not None]
    if not parts:
        raise ValueError("need at least one modality")
    total = sum(w for _, w in parts)
    fused = sum(p * (w / total) for p, w in parts)
    return fused / fused.sum(axis=1, keepdims=True)


def to_segments(
    spans: list[tuple[float, float]],
    probs: np.ndarray,
    env_ids: list[str],
    confidence_threshold: float,
) -> list[EnvSegment]:
    """Build contract segments from (S, E) probs over non-general environments.

    Fallback rule: if max prob < confidence_threshold, env = general and the
    distribution is mixed with general: p' = conf * p + (1 - conf) * onehot(general).
    Downstream conditioning then naturally weakens toward "no conditioning".
    """
    assert probs.shape == (len(spans), len(env_ids)), (probs.shape, len(spans), len(env_ids))
    segments: list[EnvSegment] = []
    for (start, end), p in zip(spans, probs):
        i = int(np.argmax(p))
        conf = float(p[i])
        dist = {e: float(v) for e, v in zip(env_ids, p)}
        if conf < confidence_threshold:
            dist = {e: v * conf for e, v in dist.items()}
            dist[GENERAL_ENV] = dist.get(GENERAL_ENV, 0.0) + (1.0 - conf)
            env = GENERAL_ENV
        else:
            env = env_ids[i]
        total = sum(dist.values())
        dist = {e: round(v / total, 6) for e, v in dist.items()}
        segments.append(EnvSegment(start=float(start), end=float(end), env=env, confidence=conf, probs=dist))
    return segments
