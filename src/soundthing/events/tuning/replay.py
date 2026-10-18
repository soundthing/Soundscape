"""Replay cached model outputs into final CaptionEvents."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from ...config import Config
from ...contracts import CaptionEvent, EnvSegment
from ..clap import calibrate
from ..conditioning import apply_priors
from ..core import combine_scores
from .. import postprocess as pp
from ..tagger import build_tagger_map, map_to_vocab
from ..types import FrameScores
from .. import vad
from .cache import ClipCache
from . import paths

_text_emb_memo: dict[tuple, np.ndarray] = {}


def _clear_text_emb_memo() -> None:
    _text_emb_memo.clear()


_clear_memo = _clear_text_emb_memo


def text_embeddings_for(cfg: Config) -> np.ndarray | None:
    """Return (C, D) text embeddings for current config, cached in-process and on disk."""
    ecfg = cfg.pipeline.get("events", {})
    clap_cfg = ecfg.get("clap", {})
    if not clap_cfg.get("enabled", True):
        return None

    clap_model = clap_cfg.get("model", "laion/clap-htsat-unfused")
    prompts_tuple = tuple(tuple(c.clap) for c in cfg.classes)
    memo_key = (clap_model, prompts_tuple)

    if memo_key in _text_emb_memo:
        return _text_emb_memo[memo_key]

    key_payload = {"model": clap_model, "prompts": [list(c.clap) for c in cfg.classes]}
    key_canonical = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
    key_hash = hashlib.sha1(key_canonical.encode("utf-8")).hexdigest()[:12]

    disk_path = paths.cache_root() / "text_emb" / f"{key_hash}.npy"
    if disk_path.exists():
        emb = np.load(disk_path)
        _text_emb_memo[memo_key] = emb
        return emb

    from .. import clap

    loaded_clap = clap.load_clap(cfg)
    emb = clap.text_embeddings(loaded_clap, cfg)

    disk_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(disk_path, emb)
    _text_emb_memo[memo_key] = emb
    return emb


DEFAULT_SEP_FUSION: str = "max"
DEFAULT_SEP_WEIGHT: float = 0.5
DEFAULT_SEP_SUPPRESS_UNDER_SPEECH: float = 1.0


def _vocab_scores(
    tagger_raw: np.ndarray,
    clap_emb: np.ndarray | None,
    tagger_map: np.ndarray,
    cfg: Config,
    text_emb: np.ndarray | None,
) -> np.ndarray:
    """Map raw tagger probs and optional CLAP embeddings to vocabulary scores."""
    tagger_f32 = tagger_raw.astype(np.float32)
    scores = map_to_vocab(tagger_f32, tagger_map)

    ecfg = cfg.pipeline.get("events") or {}
    clap_cfg = ecfg.get("clap") or {}
    clap_enabled = bool(clap_cfg.get("enabled", True))

    if clap_emb is not None and clap_enabled:
        if text_emb is None:
            text_emb = text_embeddings_for(cfg)

        if text_emb is not None:
            calib = clap_cfg.get("calibration") or {}
            center = float(calib.get("center", 0.2))
            scale = float(calib.get("scale", 20.0))

            sims = clap_emb.astype(np.float32) @ text_emb.T
            clap_s = calibrate(sims, text_emb, center, scale)

            combine_mode = clap_cfg.get("combine", "max")
            has_audioset = np.any(tagger_map > 0, axis=-1)
            scores = combine_scores(
                scores,
                clap_s,
                has_audioset=has_audioset,
                combine=combine_mode,
            )

    return np.clip(scores, 0.0, 1.0).astype(np.float32)


def class_scores(
    cache: ClipCache,
    cfg: Config,
    text_emb: np.ndarray | None = None,
) -> FrameScores:
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

        clap_cfg = ecfg.get("clap") or {}
        if bool(clap_cfg.get("enabled", True)) and text_emb is None:
            if cache.clap_emb is not None or cache.clap_emb_sep is not None:
                text_emb = text_embeddings_for(cfg)

        raw_scores = _vocab_scores(cache.tagger_raw, cache.clap_emb, tagger_map, cfg, text_emb)
        sep_clap = cache.clap_emb_sep if cache.clap_emb_sep is not None else cache.clap_emb
        sep_scores = _vocab_scores(cache.tagger_sep, sep_clap, tagger_map, cfg, text_emb)

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
        scores = _vocab_scores(cache.tagger_raw, cache.clap_emb, tagger_map, cfg, text_emb)

    # VAD speech suppression
    regions = vad.speech_regions_from_probs(
        cache.vad_probs,
        cache.vad_chunk_sec,
        cache.duration,
        cfg,
    )
    mask = vad.speech_mask(cache.times, regions)
    safety = cfg.safety_ids()
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
    text_emb: np.ndarray | None = None,
) -> list[CaptionEvent]:
    """Replay post-model pipeline from a ClipCache."""
    scores = class_scores(cache, cfg, text_emb=text_emb)

    if segments is not None:
        scores = apply_priors(scores, segments, cfg)

    pcfg = cfg.pipeline.get("events", {}).get("postprocess", {})
    smoothing_sec = float(pcfg.get("smoothing_sec", 0.5))
    scores = pp.smooth(scores, smoothing_sec)
    events = pp.threshold_to_events(scores, cfg)
    events = pp.collapse_confusable(events, scores, cfg)
    events = pp.priority_filter(events, cfg)

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
