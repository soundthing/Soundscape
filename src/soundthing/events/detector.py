"""Real event detector (backend: real)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..contracts import CaptionEvent, EnvSegment, ExtractedInputs
from . import postprocess as pp
from .conditioning import apply_priors
from .clap import load_clap, text_embeddings
from .core import EventModels, score_window
from .tagger import build_tagger_map, load_tagger
from .types import FrameScores
from .vad import speech_mask, speech_regions

_models_cache: dict[tuple, EventModels] = {}
_raw_scores_cache: tuple[tuple, FrameScores] | None = None


def _freeze(val: Any) -> Any:
    """Recursively convert unhashable types (dicts, lists) into hashable tuples."""
    if isinstance(val, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in val.items()))
    if isinstance(val, (list, tuple)):
        return tuple(_freeze(v) for v in val)
    return val


def _models_cache_key(cfg: Config) -> tuple:
    ecfg = cfg.pipeline.get("events", {})
    tagger_model = ecfg.get("tagger", {}).get("model")
    clap_cfg = ecfg.get("clap", {})
    clap_model = clap_cfg.get("model")
    clap_enabled = clap_cfg.get("enabled", True)
    calibration = _freeze(clap_cfg.get("calibration", {}))
    return (tagger_model, clap_model, clap_enabled, calibration, tuple(cfg.class_ids))


def get_models(cfg: Config) -> EventModels:
    """Return cached EventModels for this configuration, loading only once per process."""
    key = _models_cache_key(cfg)
    if key not in _models_cache:
        _models_cache[key] = load_models(cfg)
    return _models_cache[key]


def _clear_caches() -> None:
    """Clear models and raw scores caches (useful between tests)."""
    global _raw_scores_cache
    _models_cache.clear()
    _raw_scores_cache = None


def load_models(cfg: Config) -> EventModels:
    tagger, audioset_names = load_tagger(cfg)
    tagger_map = build_tagger_map(cfg, audioset_names)

    clap_cfg = cfg.pipeline.get("events", {}).get("clap", {})
    clap_enabled = clap_cfg.get("enabled", True)

    clap = None
    clap_text_emb = None
    if clap_enabled:
        clap = load_clap(cfg)
        clap_text_emb = text_embeddings(clap, cfg)

    return EventModels(
        tagger=tagger,
        clap=clap,
        class_ids=list(cfg.class_ids),
        tagger_map=tagger_map,
        clap_text_emb=clap_text_emb,
    )


def score_audio(audio: np.ndarray, sr: int, models: EventModels, cfg: Config) -> FrameScores:
    ecfg = cfg.pipeline["events"]
    win = int(ecfg["window_sec"] * sr)
    hop = int(ecfg["hop_sec"] * sr)
    starts = range(0, max(1, len(audio) - win + 1), hop)
    rows, times = [], []
    for s in starts:
        chunk = audio[s : s + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        rows.append(score_window(chunk, sr, models))
        times.append((s + win / 2) / sr)
    return FrameScores(np.array(times), list(models.class_ids), np.stack(rows), ecfg["hop_sec"])


def detect(
    inputs: ExtractedInputs, segments: list[EnvSegment] | None, cfg: Config
) -> list[CaptionEvent]:
    import soundfile as sf

    global _raw_scores_cache
    ecfg = cfg.pipeline["events"]

    audio_path = Path(inputs.audio_path).resolve()
    stat = audio_path.stat()
    m_key = _models_cache_key(cfg)
    vad_key = _freeze(ecfg.get("vad", {}))
    win_hop_key = (ecfg.get("window_sec"), ecfg.get("hop_sec"))
    raw_key = (str(audio_path), stat.st_mtime, stat.st_size, m_key, vad_key, win_hop_key)

    if _raw_scores_cache is not None and _raw_scores_cache[0] == raw_key:
        scores = _raw_scores_cache[1].copy_with(_raw_scores_cache[1].scores.copy())
    else:
        models = get_models(cfg)
        audio, sr = sf.read(inputs.audio_path, dtype="float32")
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1 if audio.shape[1] <= audio.shape[0] else 0)
        elif audio.ndim > 2:
            audio = np.mean(audio, axis=tuple(range(1, audio.ndim)))
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        scores = score_audio(audio, sr, models, cfg)

        # Speech masking: suppress non-safety classes under dialogue.
        mask = speech_mask(scores.times, speech_regions(audio, sr, cfg))
        safety = cfg.safety_ids()
        factor = np.array(
            [1.0 if c in safety else ecfg["vad"]["suppress_under_speech"] for c in scores.class_ids],
            dtype=np.float32,
        )
        s = scores.scores.copy()
        s[mask] *= factor
        masked_scores = scores.copy_with(s)

        _raw_scores_cache = (raw_key, masked_scores)
        scores = masked_scores.copy_with(masked_scores.scores.copy())

    if segments is not None:  # None = unconditioned baseline
        scores = apply_priors(scores, segments, cfg)

    pcfg = ecfg["postprocess"]
    scores = pp.smooth(scores, pcfg["smoothing_sec"])
    events = pp.threshold_to_events(scores, cfg)
    events = pp.collapse_confusable(events, scores, cfg)
    events = pp.priority_filter(events, cfg)

    stereo_sr = getattr(inputs, "sample_rate", 16000)
    if inputs.audio_stereo_path is not None and Path(inputs.audio_stereo_path).exists():
        stereo, stereo_sr = sf.read(inputs.audio_stereo_path, always_2d=True, dtype="float32")
        events = pp.add_modifiers(events, stereo, stereo_sr, cfg)
    else:
        events = pp.add_modifiers(events, None, stereo_sr, cfg)

    # Clip every event's end to inputs.duration (drop events whose start >= duration),
    # then call .validate() on each event before returning. Return sorted by start.
    final_events: list[CaptionEvent] = []
    for ev in events:
        if ev.start >= inputs.duration:
            continue
        clipped_end = min(ev.end, inputs.duration)
        if clipped_end <= ev.start:
            continue
        clipped_ev = replace(ev, end=clipped_end)
        clipped_ev.validate()
        final_events.append(clipped_ev)

    final_events.sort(key=lambda ev: (ev.start, ev.end))
    return final_events
