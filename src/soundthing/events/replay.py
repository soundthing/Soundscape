"""Replay cached model outputs into final CaptionEvents."""

from __future__ import annotations

from dataclasses import replace
import numpy as np

from soundthing.config import Config
from soundthing.contracts import CaptionEvent, EnvSegment
from soundthing.events.conditioning import apply_priors
from soundthing.events.fallbacks import add_generic_scores
from soundthing.events.hierarchy import suppress_ancestors
from soundthing.events import postprocess as pp
from soundthing.events.tagger import build_tagger_map, map_to_vocab
from soundthing.events.types import FrameScores
from soundthing.events import vad
from soundthing.events.cache import ClipCache

DEFAULT_SEP_FUSION: str = "max"
DEFAULT_SEP_WEIGHT: float = 0.5
DEFAULT_SEP_SUPPRESS_UNDER_SPEECH: float = 1.0


def _vocab_scores(tagger_raw: np.ndarray, tagger_map: np.ndarray) -> np.ndarray:
    """Map raw tagger probs to vocabulary scores."""
    scores = map_to_vocab(tagger_raw.astype(np.float32), tagger_map)
    return np.clip(scores, 0.0, 1.0).astype(np.float32)


def class_scores(cache: ClipCache, cfg: Config) -> FrameScores:
    """Compute per-window vocabulary scores from a ClipCache."""
    tagger_map = build_tagger_map(cfg, cache.audioset_names)
    ecfg = cfg.pipeline.get("events") or {}
    sep_cfg = ecfg.get("separation") or {}

    sep_active = (
        bool(sep_cfg.get("enabled", False))
        and cache.sep_mask is not None
        and cache.tagger_sep is not None
    )

    if sep_active:
        if cache.tagger_sep.shape != cache.tagger_raw.shape:
            raise ValueError(
                f"tagger_sep shape {cache.tagger_sep.shape} does not match tagger_raw shape {cache.tagger_raw.shape}"
            )
        if cache.sep_mask.ndim != 1 or len(cache.sep_mask) != len(cache.times):
            raise ValueError(
                f"sep_mask length {len(cache.sep_mask)} does not match T={len(cache.times)}"
            )

        fusion = sep_cfg.get("fusion") or DEFAULT_SEP_FUSION
        if fusion not in ("max", "sep", "weighted"):
            raise ValueError(f"Unknown separation fusion mode: {fusion!r}")

        raw_scores = _vocab_scores(cache.tagger_raw, tagger_map)
        sep_scores = _vocab_scores(cache.tagger_sep, tagger_map)

        scores = raw_scores.copy()
        sep_mask = cache.sep_mask.astype(bool)
        if fusion == "max":
            scores[sep_mask] = np.maximum(raw_scores[sep_mask], sep_scores[sep_mask])
        elif fusion == "sep":
            scores[sep_mask] = sep_scores[sep_mask]
        elif fusion == "weighted":
            sep_weight_val = sep_cfg.get("sep_weight")
            sep_weight = float(DEFAULT_SEP_WEIGHT if sep_weight_val is None else sep_weight_val)
            scores[sep_mask] = (
                sep_weight * sep_scores[sep_mask] + (1.0 - sep_weight) * raw_scores[sep_mask]
            ).astype(np.float32)
    else:
        scores = _vocab_scores(cache.tagger_raw, tagger_map)

    # VAD speech suppression; speech labels in vad.exempt_classes are never
    # suppressed by the speech detector itself.
    regions = vad.speech_regions_from_probs(
        cache.vad_probs,
        cache.vad_chunk_sec,
        cache.duration,
        cfg,
    )
    mask = vad.speech_mask(cache.times, regions)
    safety = cfg.safety_ids() | set((ecfg.get("vad") or {}).get("exempt_classes") or [])
    suppress = float((ecfg.get("vad") or {}).get("suppress_under_speech", 0.3))
    factor = np.array(
        [1.0 if c in safety else suppress for c in cfg.class_ids],
        dtype=np.float32,
    )

    if sep_active:
        sep_suppress_val = sep_cfg.get("suppress_under_speech")
        sep_suppress = float(
            DEFAULT_SEP_SUPPRESS_UNDER_SPEECH if sep_suppress_val is None else sep_suppress_val
        )
        sep_factor = np.array(
            [1.0 if c in safety else sep_suppress for c in cfg.class_ids],
            dtype=np.float32,
        )
        sep_mask = cache.sep_mask.astype(bool)
        scores[mask & ~sep_mask] *= factor
        scores[mask & sep_mask] *= sep_factor
    else:
        scores[mask] *= factor

    return FrameScores(
        times=cache.times,
        class_ids=list(cfg.class_ids),
        scores=scores,
        hop=cache.hop,
    )


def replay(
    cache: ClipCache,
    segments: list[EnvSegment] | None,
    cfg: Config,
) -> list[CaptionEvent]:
    """Replay post-model pipeline from a ClipCache."""
    scores = class_scores(cache, cfg)
    evidence = scores

    if segments is not None:  # None = unconditioned baseline
        scores = apply_priors(scores, segments, cfg)

    rms_dbfs = cache.rms_dbfs
    if rms_dbfs is None:  # cache built without RMS: treat every window as active
        rms_dbfs = np.zeros(len(cache.times), dtype=np.float32)
    scores = add_generic_scores(scores, rms_dbfs, cfg)
    events = pp.threshold_to_events(scores, cfg, evidence)
    events = pp.collapse_confusable(events, scores, cfg)
    events = suppress_ancestors(events, cfg.hierarchy(), keep=frozenset(cfg.safety_ids()), drop_tails=True)

    pcfg = cfg.pipeline.get("events", {}).get("postprocess", {})

    onset_shift = float(pcfg.get("onset_shift_sec", 0.0))

    final_events: list[CaptionEvent] = []
    for ev in events:
        start = max(0.0, round(ev.start + onset_shift, 3))
        if start >= cache.duration:
            continue
        end = min(round(ev.end + onset_shift, 3), cache.duration)
        if end <= start:
            continue
        shifted_ev = replace(
            ev,
            start=start,
            end=end,
            modifiers=[],
        )
        shifted_ev.validate()
        final_events.append(shifted_ev)

    final_events.sort(key=lambda ev: (ev.start, ev.end))
    return final_events
