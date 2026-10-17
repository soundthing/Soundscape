"""Real environment classifier (backend: real)."""

from __future__ import annotations

from ..config import Config
from ..contracts import GENERAL_ENV, EnvSegment, ExtractedInputs, validate_segments
from .audio import audio_env_logits
from .fusion import fuse, softmax, to_segments
from .segmentation import detect_spans
from .visual import visual_env_logits


def classify(
    inputs: ExtractedInputs,
    cfg: Config,
    use_audio: bool = True,
    use_visual: bool = True,
) -> list[EnvSegment]:
    ecfg = cfg.pipeline["environment"]
    envs = [e for e in cfg.environments if e.id != GENERAL_ENV]
    env_ids = [e.id for e in envs]
    temp = ecfg["fusion"]["temperature"]

    spans = detect_spans(inputs.video_path, inputs.duration, cfg)
    a = softmax(audio_env_logits(inputs, spans, envs, cfg), temp) if use_audio else None
    v = softmax(visual_env_logits(inputs, spans, envs, cfg), temp) if use_visual else None
    probs = fuse(a, v, ecfg["fusion"]["audio_weight"], ecfg["fusion"]["visual_weight"])

    segments = to_segments(spans, probs, env_ids, ecfg["confidence_threshold"])
    validate_segments(segments, inputs.duration)
    return segments
