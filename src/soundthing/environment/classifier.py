"""Real environment classifier (backend: real)."""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..contracts import GENERAL_ENV, EnvSegment, ExtractedInputs, validate_segments
from .audio import audio_env_logits
from .fusion import fuse, softmax, to_segments
from .segmentation import detect_spans
from .visual import visual_env_logits


def segments_from_logits(
    spans: list[tuple[float, float]],
    audio_logits: np.ndarray | None,
    visual_logits: np.ndarray | None,
    env_ids: list[str],
    cfg: Config,
    duration: float,
) -> list[EnvSegment]:
    if audio_logits is None and visual_logits is None:
        raise ValueError("need at least one environment modality")
    ecfg = cfg.pipeline["environment"]
    temp = ecfg["fusion"]["temperature"]

    a = softmax(audio_logits, temp) if audio_logits is not None else None
    v = softmax(visual_logits, temp) if visual_logits is not None else None
    probs = fuse(a, v, ecfg["fusion"]["audio_weight"], ecfg["fusion"]["visual_weight"])

    # Softmax must choose among the configured environments even for unrelated
    # graphics. If no visual prompt has positive cosine similarity, treat the
    # real-environment distribution as uninformative so the existing confidence
    # fallback can select general without changing the output contract.
    if visual_logits is not None:
        floor = float(ecfg["fusion"].get("visual_similarity_floor", 0.0))
        if not np.isfinite(floor) or not -1.0 <= floor <= 1.0:
            raise ValueError("visual_similarity_floor must be finite and in [-1, 1]")
        unsupported = visual_logits.max(axis=1) < floor
        probs[unsupported] = 1.0 / len(env_ids)

    # Scene cuts and the maximum window length define places to reassess the
    # environment. They should not force a boundary in the final output when
    # adjacent windows have the same predicted environment.
    candidates = to_segments(spans, probs, env_ids, ecfg["confidence_threshold"])
    merged_spans = []
    merged_probs = []
    merged_envs = []
    for (start, end), row, candidate in zip(spans, probs, candidates):
        span_duration = end - start
        if merged_spans and candidate.env == merged_envs[-1]:
            previous_start, previous_end = merged_spans[-1]
            previous_duration = previous_end - previous_start
            merged_probs[-1] = (merged_probs[-1] * previous_duration + row * span_duration) / (previous_duration + span_duration)
            merged_spans[-1] = (previous_start, end)
        else:
            merged_spans.append((start, end))
            merged_probs.append(row.copy())
            merged_envs.append(candidate.env)
    segments = to_segments(merged_spans, np.stack(merged_probs), env_ids, ecfg["confidence_threshold"])
    validate_segments(segments, duration)
    return segments


def classify(
    inputs: ExtractedInputs,
    cfg: Config,
    use_audio: bool = True,
    use_visual: bool = True,
) -> list[EnvSegment]:
    if not use_audio and not use_visual:
        raise ValueError("need at least one environment modality")
    envs = [e for e in cfg.environments if e.id != GENERAL_ENV]
    env_ids = [e.id for e in envs]

    spans = detect_spans(inputs.video_path, inputs.duration, cfg)
    audio_logits = audio_env_logits(inputs, spans, envs, cfg) if use_audio else None
    if use_visual and not inputs.frame_paths():
        if not use_audio:
            raise FileNotFoundError(f"no extracted frames in {inputs.frames_dir}")
        use_visual = False
    visual_logits = visual_env_logits(inputs, spans, envs, cfg) if use_visual else None
    return segments_from_logits(spans, audio_logits, visual_logits, env_ids, cfg, inputs.duration)
