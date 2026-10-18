"""Events component for sound event detection stage tuning."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from soundthing.config import Config
from soundthing.contracts import (
    GENERAL_ENV,
    CaptionEvent,
    ClipLabels,
    EnvSegment,
    load_labels,
    validate_segments,
)
from soundthing.events import experiments
from soundthing.events.cache import (
    ClipCache,
    cache_key as get_cache_key,
    cache_path,
    load_cache,
)
from soundthing.events.replay import replay as replay_events, text_embeddings_for
from soundthing.harness.bootstrap import _vectorized_f_beta
from soundthing.harness.components.base import Clip, Evaluation
from soundthing.harness.objective import (
    ClipMatch,
    diagnostics,
    f_beta,
    grid_f1,
    match_clip,
)


def load_event_caches(cfg: Config, labels: list[ClipLabels]) -> list[ClipCache]:
    """Load cached model outputs for given clip labels.

    Raises FileNotFoundError naming the clip and cache key if any cache is missing.
    """
    key = get_cache_key(cfg)
    caches: list[ClipCache] = []
    for lbl in labels:
        cpath = cache_path(lbl.video_id, key)
        if not cpath.exists():
            raise FileNotFoundError(
                f"Cache missing for clip {lbl.video_id!r} with key {key!r} at {cpath}. "
                f"caches must be built/recached for this window/hop/model."
            )
        caches.append(load_cache(cpath))
    return caches


def text_emb_for(cfg: Config, caches: list[ClipCache]) -> np.ndarray | None:
    """Return text embeddings if CLAP is enabled and at least one cache contains CLAP features."""
    clap_cfg = cfg.pipeline.get("events", {}).get("clap", {})
    clap_enabled = bool(clap_cfg.get("enabled", True))
    has_clap_cache = any(c.clap_emb is not None for c in caches)

    if clap_enabled and has_clap_cache:
        return text_embeddings_for(cfg)
    return None


def labeled_env_segments(
    labels: ClipLabels,
    cfg: Config,
    confidence: float = 0.7,
) -> list[EnvSegment]:
    """Convert ground truth labeled environment segments into full-coverage EnvSegments."""
    for s in labels.segments:
        if s.env not in cfg.env_ids:
            raise KeyError(
                f"Unknown environment {s.env!r} in labels for clip {labels.video_id!r}"
            )

    if not labels.segments:
        result = [
            EnvSegment(
                start=0.0,
                end=float(labels.duration),
                env=GENERAL_ENV,
                confidence=1.0,
                probs={GENERAL_ENV: 1.0},
            )
        ]
        validate_segments(result, labels.duration)
        return result

    sorted_segs = sorted(labels.segments, key=lambda s: (s.start, s.end))
    result: list[EnvSegment] = []
    cur_time = 0.0

    for s in sorted_segs:
        if s.start - cur_time > 1e-6:
            result.append(
                EnvSegment(
                    start=cur_time,
                    end=float(s.start),
                    env=GENERAL_ENV,
                    confidence=1.0,
                    probs={GENERAL_ENV: 1.0},
                )
            )
            cur_time = float(s.start)

        seg_start = cur_time
        seg_end = float(s.end)
        if s.env == GENERAL_ENV:
            result.append(
                EnvSegment(
                    start=seg_start,
                    end=seg_end,
                    env=GENERAL_ENV,
                    confidence=1.0,
                    probs={GENERAL_ENV: 1.0},
                )
            )
        else:
            conf = float(confidence)
            result.append(
                EnvSegment(
                    start=seg_start,
                    end=seg_end,
                    env=s.env,
                    confidence=conf,
                    probs={s.env: conf, GENERAL_ENV: float(1.0 - conf)},
                )
            )
        cur_time = seg_end

    if float(labels.duration) - cur_time > 1e-6:
        result.append(
            EnvSegment(
                start=cur_time,
                end=float(labels.duration),
                env=GENERAL_ENV,
                confidence=1.0,
                probs={GENERAL_ENV: 1.0},
            )
        )
    elif result:
        result[-1].end = float(labels.duration)

    validate_segments(result, labels.duration)
    return result


def event_counts(matches: list[ClipMatch]) -> dict[str, Any]:
    """Compute safety_recall, fp_per_min, and raw counts from clip matches."""
    total_safety_tp = sum(m.safety_tp for m in matches)
    total_safety_total = sum(m.safety_total for m in matches)
    safety_recall = (
        float(total_safety_tp / total_safety_total)
        if total_safety_total > 0
        else 1.0
    )

    total_fp = sum(m.fp for m in matches)
    total_minutes = sum(m.minutes for m in matches)
    fp_per_min = (
        float(total_fp / total_minutes)
        if total_minutes > 0.0
        else 0.0
    )

    return {
        "safety_recall": safety_recall,
        "fp_per_min": fp_per_min,
        "safety_tp": total_safety_tp,
        "safety_total": total_safety_total,
        "fp": total_fp,
        "minutes": total_minutes,
    }


class EventsComponent:
    name: str = "events"
    sections: tuple[str, ...] = ("events.", "thresholds.")
    code_paths: tuple[str, ...] = ("src/soundthing/events",)
    tuning_log: str = "src/soundthing/events/references/tuning-log.md"
    n_stats: int = 4

    def cache_key(self, cfg: Config) -> str:
        return get_cache_key(cfg)

    def build_cache(self, video_id: str, audio_path: Path, cfg: Config) -> Path | None:
        import soundfile as sf
        from soundthing.events import cache as cache_mod

        key = self.cache_key(cfg)
        out_path = cache_mod.cache_path(video_id, key)
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        c = cache_mod.build_cache(data, sr, video_id, cfg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cache_mod.save_cache(c, out_path)
        return out_path

    def load_clips(self, cfg: Config, labels_dir: Path) -> list[Clip]:
        ldir = Path(labels_dir)
        label_files = sorted(ldir.glob("*.json"))
        if not label_files:
            raise ValueError(f"No label files found in {ldir}")

        labels = [load_labels(p) for p in label_files]
        labels.sort(key=lambda l: l.video_id)
        caches = load_event_caches(cfg, labels)

        clips: list[Clip] = []
        for lbl, cache in zip(labels, caches):
            if cache.duration != lbl.duration:
                cache = replace(cache, duration=lbl.duration)
            clips.append(Clip(video_id=lbl.video_id, labels=lbl, cache=cache))
        return clips

    def evaluate(self, clips: list[Clip], cfg: Config) -> Evaluation:
        caches = [c.cache for c in clips]
        labels = [c.labels for c in clips]
        text_emb = text_emb_for(cfg, caches)
        tol = float(cfg.pipeline.get("eval", {}).get("onset_tolerance_sec", 0.5))

        cond_matches: list[ClipMatch] = []
        base_matches: list[ClipMatch] = []
        wrong_matches: list[ClipMatch] = []

        cond_preds_by_clip: dict[str, list] = {}
        labels_by_clip: dict[str, ClipLabels] = {}
        per_clip: dict[str, dict[str, Any]] = {}

        for lbl, cache in zip(labels, caches):
            cond_segs = labeled_env_segments(lbl, cfg)
            wrong_segs = experiments.wrong_env_segments(cond_segs, cfg)

            pred_cond = replay_events(cache, cond_segs, cfg, text_emb=text_emb)
            pred_base = replay_events(cache, None, cfg, text_emb=text_emb)
            pred_wrong = replay_events(cache, wrong_segs, cfg, text_emb=text_emb)

            m_cond = match_clip(pred_cond, lbl, cfg, tol)
            m_base = match_clip(pred_base, lbl, cfg, tol)
            m_wrong = match_clip(pred_wrong, lbl, cfg, tol)

            cond_matches.append(m_cond)
            base_matches.append(m_base)
            wrong_matches.append(m_wrong)

            vid = lbl.video_id
            cond_preds_by_clip[vid] = pred_cond
            labels_by_clip[vid] = lbl

            per_clip[vid] = {
                "stats": [
                    float(m_cond.tp_fine),
                    float(m_cond.tp_coarse),
                    float(m_cond.fp),
                    float(m_cond.fn),
                ],
                "conditioned": {
                    "tp_fine": int(m_cond.tp_fine),
                    "tp_coarse": int(m_cond.tp_coarse),
                    "fp": int(m_cond.fp),
                    "fn": int(m_cond.fn),
                    "minutes": float(m_cond.minutes),
                    "safety_tp": int(m_cond.safety_tp),
                    "safety_total": int(m_cond.safety_total),
                },
                "baseline": {
                    "tp_fine": int(m_base.tp_fine),
                    "tp_coarse": int(m_base.tp_coarse),
                    "fp": int(m_base.fp),
                    "fn": int(m_base.fn),
                    "minutes": float(m_base.minutes),
                    "safety_tp": int(m_base.safety_tp),
                    "safety_total": int(m_base.safety_total),
                },
                "wrong_env": {
                    "tp_fine": int(m_wrong.tp_fine),
                    "tp_coarse": int(m_wrong.tp_coarse),
                    "fp": int(m_wrong.fp),
                    "fn": int(m_wrong.fn),
                    "minutes": float(m_wrong.minutes),
                    "safety_tp": int(m_wrong.safety_tp),
                    "safety_total": int(m_wrong.safety_total),
                },
            }

        objective = f_beta(cond_matches, 0.5)
        baseline = f_beta(base_matches, 0.5)
        wrong_env = f_beta(wrong_matches, 0.5)

        counts = event_counts(cond_matches)
        safety_recall = counts["safety_recall"]
        fp_per_min = counts["fp_per_min"]

        gf1 = grid_f1(cond_preds_by_clip, labels_by_clip)
        diag = diagnostics(cond_matches, cfg)

        metrics = {
            "objective": float(objective),
            "baseline": float(baseline),
            "wrong_env": float(wrong_env),
            "safety_recall": float(safety_recall),
            "fp_per_min": float(fp_per_min),
            "grid_f1": float(gf1),
        }

        return Evaluation(
            objective=float(objective),
            metrics=metrics,
            per_clip=per_clip,
            diagnostics=diag,
        )

    def limits(self, metrics: dict[str, float], best: dict[str, Any] | None) -> dict[str, bool]:
        objective = metrics["objective"]
        baseline = metrics["baseline"]
        wrong_env = metrics["wrong_env"]
        safety_recall = metrics["safety_recall"]
        fp_per_min = metrics["fp_per_min"]

        if best is None:
            return {
                "safety_recall": True,
                "fp_per_min": True,
                "above_baseline": bool(objective >= baseline),
                "wrong_env": True,
            }

        return {
            "safety_recall": bool(safety_recall >= float(best["safety_recall"]) - 1e-9),
            "fp_per_min": bool(fp_per_min <= float(best["fp_per_min"]) + 1e-9),
            "above_baseline": bool(objective >= baseline - 1e-9),
            "wrong_env": bool(wrong_env >= float(best["wrong_env"]) - 1e-9),
        }

    def space(self, cfg: Config, labels_dir: Path | None = None) -> list[dict[str, Any]]:
        from soundthing.harness import space as space_mod

        counts = space_mod.dev_event_counts(labels_dir)
        return space_mod.describe_space(cfg, counts)

    def stat_objective(self, totals: np.ndarray) -> np.ndarray:
        return _vectorized_f_beta(totals, beta=0.5)


COMPONENT: EventsComponent = EventsComponent()
