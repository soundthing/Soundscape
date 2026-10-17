"""Real event detector (backend: real)."""

from __future__ import annotations

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

    ecfg = cfg.pipeline["events"]
    audio, sr = sf.read(inputs.audio_path, dtype="float32")
    models = load_models(cfg)

    scores = score_audio(audio, sr, models, cfg)

    # Speech masking: suppress non-safety classes under dialogue.
    mask = speech_mask(scores.times, speech_regions(audio, sr, cfg))
    safety = cfg.safety_ids()
    factor = np.array([1.0 if c in safety else ecfg["vad"]["suppress_under_speech"] for c in scores.class_ids])
    s = scores.scores.copy()
    s[mask] *= factor
    scores = scores.copy_with(s)

    if segments is not None:  # None = unconditioned baseline
        scores = apply_priors(scores, segments, cfg)

    pcfg = ecfg["postprocess"]
    scores = pp.smooth(scores, pcfg["smoothing_sec"])
    events = pp.threshold_to_events(scores, cfg)
    events = pp.collapse_confusable(events, scores, cfg)
    events = pp.priority_filter(events, cfg)
    return events
