"""Real event detector (backend: real)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..contracts import CaptionEvent, EnvSegment, ExtractedInputs
from . import postprocess as pp
from . import separation
from .core import EventModels
from .tagger import build_tagger_map, load_tagger
from .cache import ClipCache, build_cache, cache_key
from .replay import replay

_models_cache: dict[tuple, EventModels] = {}
_clip_cache_memo: tuple[tuple, ClipCache] | None = None


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
    key = (tagger_model, tuple((c.id, tuple(c.audioset)) for c in cfg.classes))
    if separation.is_enabled(cfg):
        sep_cfg = separation.sep_config(cfg)
        return key + (True, sep_cfg.get("model"))
    return key


def get_models(cfg: Config) -> EventModels:
    """Return cached EventModels for this configuration, loading only once per process."""
    key = _models_cache_key(cfg)
    if key not in _models_cache:
        _models_cache[key] = load_models(cfg)
    return _models_cache[key]


def _clear_caches() -> None:
    """Clear models and clip cache memos (useful between tests)."""
    global _clip_cache_memo
    _models_cache.clear()
    _clip_cache_memo = None


def load_models(cfg: Config) -> EventModels:
    tagger, audioset_names = load_tagger(cfg)
    tagger_map = build_tagger_map(cfg, audioset_names)

    separator = None
    if separation.is_enabled(cfg):
        separator = separation.load_separator(cfg)

    return EventModels(
        tagger=tagger,
        class_ids=list(cfg.class_ids),
        tagger_map=tagger_map,
        audioset_names=audioset_names,
        separator=separator,
    )


def detect(
    inputs: ExtractedInputs, segments: list[EnvSegment] | None, cfg: Config
) -> list[CaptionEvent]:
    import soundfile as sf

    global _clip_cache_memo

    audio_path = Path(inputs.audio_path).resolve()
    stat = audio_path.stat()
    m_key = _models_cache_key(cfg)
    c_key = cache_key(cfg)
    memo_key = (str(audio_path), stat.st_mtime, stat.st_size, c_key, m_key)

    models = get_models(cfg)

    if _clip_cache_memo is not None and _clip_cache_memo[0] == memo_key:
        cache = _clip_cache_memo[1]
    else:
        audio, sr = sf.read(inputs.audio_path, dtype="float32")
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1 if audio.shape[1] <= audio.shape[0] else 0)
        elif audio.ndim > 2:
            audio = np.mean(audio, axis=tuple(range(1, audio.ndim)))
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        cache = build_cache(audio, sr, inputs.video_id, cfg, models=models)
        _clip_cache_memo = (memo_key, cache)

    events = replay(replace(cache, duration=inputs.duration), segments, cfg)

    stereo_sr = getattr(inputs, "sample_rate", 16000)
    if inputs.audio_stereo_path is not None and Path(inputs.audio_stereo_path).exists():
        stereo, stereo_sr = sf.read(inputs.audio_stereo_path, always_2d=True, dtype="float32")
        events = pp.add_modifiers(events, stereo, stereo_sr, cfg)
    else:
        events = pp.add_modifiers(events, None, stereo_sr, cfg)

    events.sort(key=lambda ev: (ev.start, ev.end))
    return events
