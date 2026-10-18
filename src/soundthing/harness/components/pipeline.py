"""Pipeline (end-to-end) tuning component (Track C).

Scores caption-cue generation with events conditioned on predicted environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from soundthing.config import Config
from soundthing.contracts import ClipLabels
from soundthing.captions import events_to_cues
from soundthing.harness.components.base import Clip, Component, Evaluation
from soundthing.harness.components.captions import CaptionCue, match_cues


@dataclass
class PipelineClipCache:
    events_cache: Any
    env_cache: Any

    def __getitem__(self, index: int | str) -> Any:
        if index == 0 or index == "events_cache":
            return self.events_cache
        elif index == 1 or index == "env_cache":
            return self.env_cache
        raise KeyError(index)


def _extract_caches(cache: Any) -> tuple[Any, Any]:
    if hasattr(cache, "events_cache") and hasattr(cache, "env_cache"):
        return cache.events_cache, cache.env_cache
    if isinstance(cache, tuple) and len(cache) == 2:
        return cache[0], cache[1]
    if isinstance(cache, dict):
        ev = cache.get("events_cache", cache.get("events"))
        env = cache.get("env_cache", cache.get("env"))
        return ev, env
    raise TypeError(f"Unrecognized pipeline cache structure: {cache!r}")


def _vectorized_f_beta(totals: np.ndarray, beta: float = 0.5) -> np.ndarray:
    tp_fine = totals[:, 0]
    tp_coarse = totals[:, 1]
    fp = totals[:, 2]
    fn = totals[:, 3]

    credit = tp_fine + 0.5 * tp_coarse
    fp_prime = fp + 0.5 * tp_coarse
    fn_prime = fn + 0.5 * tp_coarse

    prec_denom = credit + fp_prime
    rec_denom = credit + fn_prime

    p = np.zeros_like(credit, dtype=np.float64)
    valid_p = prec_denom > 0
    p[valid_p] = credit[valid_p] / prec_denom[valid_p]

    r = np.zeros_like(credit, dtype=np.float64)
    valid_r = rec_denom > 0
    r[valid_r] = credit[valid_r] / rec_denom[valid_r]

    b2 = beta ** 2
    f_denom = b2 * p + r
    f = np.zeros_like(p, dtype=np.float64)
    valid_f = (p + r > 0) & (f_denom > 0)
    f[valid_f] = (1.0 + b2) * p[valid_f] * r[valid_f] / f_denom[valid_f]
    return f


class PipelineComponent:
    name: str = "pipeline"
    owner: str = "person3"
    sections: tuple[str, ...] = ("events.", "thresholds.", "environment.", "captions.")
    code_paths: tuple[str, ...] = ("src/soundthing/events", "src/soundthing/environment", "src/soundthing/captions.py")
    tuning_log: str = "src/soundthing/harness/references/pipeline-tuning-log.md"
    n_stats: int = 4

    def cache_key(self, cfg: Config) -> str | None:
        """Hash of events key + '+' + environment key."""
        try:
            import soundthing.harness.components.events as ev_comp
            if hasattr(ev_comp, "COMPONENT"):
                ev_key = ev_comp.COMPONENT.cache_key(cfg)
            elif hasattr(ev_comp, "cache_key"):
                ev_key = ev_comp.cache_key(cfg)
            else:
                raise AttributeError
        except (ImportError, AttributeError):
            from soundthing.events.cache import cache_key as ev_cache_key
            ev_key = ev_cache_key(cfg)

        try:
            import soundthing.harness.components.environment as env_comp
            if hasattr(env_comp, "COMPONENT"):
                env_key = env_comp.COMPONENT.cache_key(cfg)
            elif hasattr(env_comp, "cache_key"):
                env_key = env_comp.cache_key(cfg)
            else:
                raise AttributeError
        except (ImportError, AttributeError):
            from soundthing.harness.components import get
            env_key = get("environment").cache_key(cfg)

        return f"{ev_key}+{env_key}"

    def build_cache(self, video_id: str, audio_path: Path, cfg: Config) -> Path | None:
        """No dedicated pipeline cache (built from events + environment)."""
        return None

    def load_clips(self, cfg: Config, labels_dir: Path) -> list[Clip]:
        """Load labels and both events and environment caches."""
        labels_path = Path(labels_dir)
        label_files = sorted(labels_path.glob("*.json"))
        if not label_files:
            raise ValueError(f"No label files found in {labels_dir}")
        from soundthing.contracts import load_labels
        all_labels = [load_labels(p) for p in label_files]
        all_labels.sort(key=lambda l: l.video_id)

        # 1. Events caches
        try:
            import soundthing.harness.components.events as ev_comp
            if hasattr(ev_comp, "load_event_caches"):
                ev_caches = ev_comp.load_event_caches(cfg, all_labels)
            else:
                raise AttributeError
        except (ImportError, AttributeError):
            from soundthing.events.cache import cache_key, cache_path, load_cache
            ekey = cache_key(cfg)
            ev_caches = []
            for lbl in all_labels:
                cpath = cache_path(lbl.video_id, ekey)
                if not cpath.exists():
                    raise FileNotFoundError(
                        f"Events cache missing for clip {lbl.video_id!r} with key {ekey!r} at {cpath}."
                    )
                ev_caches.append(load_cache(cpath))

        # 2. Environment caches
        try:
            import soundthing.harness.components.environment as env_comp
            if hasattr(env_comp, "load_env_cache"):
                load_env = env_comp.load_env_cache
            else:
                raise AttributeError
        except (ImportError, AttributeError):
            raise FileNotFoundError("Environment component or load_env_cache not available")

        env_caches = []
        for lbl in all_labels:
            env_caches.append(load_env(lbl.video_id, cfg))

        return [
            Clip(
                video_id=lbl.video_id,
                labels=lbl,
                cache=PipelineClipCache(events_cache=ev_c, env_cache=env_c),
            )
            for lbl, ev_c, env_c in zip(all_labels, ev_caches, env_caches)
        ]

    def evaluate(self, clips: list[Clip], cfg: Config) -> Evaluation:
        """Score caption cues conditioned on predicted environment vs ground truth."""
        captions_cfg = cfg.pipeline.get("captions") or {}
        min_cue_sec = float(captions_cfg.get("min_cue_sec", 1.0))
        max_cue_sec = float(captions_cfg.get("max_cue_sec", 6.0))
        if min_cue_sec > max_cue_sec:
            raise ValueError(
                f"captions.min_cue_sec ({min_cue_sec}) cannot exceed captions.max_cue_sec ({max_cue_sec})"
            )

        try:
            import soundthing.harness.components.environment as env_comp
            predict_segments = getattr(env_comp, "predict_segments")
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("Environment predict_segments is required for pipeline evaluation") from exc

        try:
            import soundthing.harness.components.events as ev_comp
            replay_events = getattr(ev_comp, "replay_events", None)
            text_emb_for = getattr(ev_comp, "text_emb_for", None)
        except (ImportError, AttributeError):
            replay_events = None
            text_emb_for = None

        if replay_events is None:
            from soundthing.events.replay import replay as replay_events

        ev_caches = [_extract_caches(c.cache)[0] for c in clips]
        if text_emb_for is not None:
            text_emb = text_emb_for(cfg, ev_caches)
        else:
            clap_cfg = cfg.pipeline.get("events", {}).get("clap", {})
            clap_enabled = bool(clap_cfg.get("enabled", True))
            has_clap = any(getattr(c, "clap_emb", None) is not None for c in ev_caches)
            if clap_enabled and has_clap:
                from soundthing.events.replay import text_embeddings_for
                text_emb = text_embeddings_for(cfg)
            else:
                text_emb = None

        matches: list[Any] = []
        base_matches: list[Any] = []
        per_clip: dict[str, dict[str, Any]] = {}
        tol = float(cfg.pipeline.get("eval", {}).get("onset_tolerance_sec", 0.5))

        pred_envs: list[tuple[Clip, list[Any]]] = []

        for clip in clips:
            ev_cache, env_cache = _extract_caches(clip.cache)
            lbl = clip.labels
            if hasattr(ev_cache, "duration") and ev_cache.duration != lbl.duration:
                from dataclasses import replace
                ev_cache = replace(ev_cache, duration=lbl.duration)

            pred_segs = predict_segments(env_cache, cfg)
            pred_envs.append((clip, pred_segs))

            events = replay_events(ev_cache, pred_segs, cfg, text_emb)
            sorted_events = sorted(events, key=lambda e: e.start)
            cues = events_to_cues(sorted_events, min_cue=min_cue_sec, max_cue=max_cue_sec)
            pred_cues = [
                CaptionCue(start=c.start, end=c.end, class_id=ev.class_id)
                for c, ev in zip(cues, sorted_events)
            ]
            m = match_cues(pred_cues, lbl, cfg, tol)
            matches.append(m)

            base_events = replay_events(ev_cache, None, cfg, text_emb)
            sorted_base = sorted(base_events, key=lambda e: e.start)
            base_cues = events_to_cues(sorted_base, min_cue=min_cue_sec, max_cue=max_cue_sec)
            pred_base_cues = [
                CaptionCue(start=c.start, end=c.end, class_id=ev.class_id)
                for c, ev in zip(base_cues, sorted_base)
            ]
            m_base = match_cues(pred_base_cues, lbl, cfg, tol)
            base_matches.append(m_base)

            per_clip[lbl.video_id] = {
                "stats": [float(m.tp_fine), float(m.tp_coarse), float(m.fp), float(m.fn)],
                "tp_fine": m.tp_fine,
                "tp_coarse": m.tp_coarse,
                "fp": m.fp,
                "fn": m.fn,
                "minutes": m.minutes,
                "safety_tp": m.safety_tp,
                "safety_total": m.safety_total,
                "baseline_stats": [float(m_base.tp_fine), float(m_base.tp_coarse), float(m_base.fp), float(m_base.fn)],
            }

        total_stats = np.sum(
            [per_clip[c.labels.video_id]["stats"] for c in clips],
            axis=0,
        ) if clips else np.zeros(4, dtype=np.float64)
        obj = float(self.stat_objective(total_stats))

        base_stats = np.sum(
            [per_clip[c.labels.video_id]["baseline_stats"] for c in clips],
            axis=0,
        ) if clips else np.zeros(4, dtype=np.float64)
        baseline = float(self.stat_objective(base_stats))

        total_safety_tp = sum(m.safety_tp for m in matches)
        total_safety_total = sum(m.safety_total for m in matches)
        safety_recall = (
            float(total_safety_tp / total_safety_total)
            if total_safety_total > 0
            else 1.0
        )

        total_fp = sum(m.fp for m in matches)
        total_minutes = sum(m.minutes for m in matches)
        fp_per_min = float(total_fp / total_minutes) if total_minutes > 0.0 else 0.0

        total_agree = 0.0
        total_env_time = 0.0
        for clip, pred_segs in pred_envs:
            if clip.labels.segments:
                for ls in clip.labels.segments:
                    dur = ls.end - ls.start
                    total_env_time += dur
                    for ps in pred_segs:
                        if ps.env == ls.env:
                            total_agree += max(0.0, min(ls.end, ps.end) - max(ls.start, ps.start))

        env_time_acc = float(total_agree / total_env_time) if total_env_time > 0.0 else 0.0

        metrics = {
            "objective": obj,
            "baseline": baseline,
            "safety_recall": safety_recall,
            "fp_per_min": fp_per_min,
            "env_time_acc": env_time_acc,
        }

        diagnostics = {
            "env_time_acc": env_time_acc,
        }

        return Evaluation(
            objective=obj,
            metrics=metrics,
            per_clip=per_clip,
            diagnostics=diagnostics,
        )

    def limits(self, metrics: dict[str, float], best: dict[str, Any] | None) -> dict[str, bool]:
        """Check limits vs best ledger entry."""
        above_baseline = bool(metrics["objective"] >= metrics["baseline"] - 1e-9)
        if best is None:
            return {"above_baseline": above_baseline}
        best_safety = float(best.get("safety_recall", best.get("metrics", {}).get("safety_recall", 0.0)))
        best_fpm = float(best.get("fp_per_min", best.get("metrics", {}).get("fp_per_min", 0.0)))
        return {
            "safety_recall": bool(metrics["safety_recall"] >= best_safety - 1e-9),
            "fp_per_min": bool(metrics["fp_per_min"] <= best_fpm + 1e-9),
            "above_baseline": above_baseline,
        }

    def space(self, cfg: Config, labels_dir: Path | None = None) -> list[dict[str, Any]]:
        """Union of events, environment, and captions search spaces, dropping duplicates."""
        from soundthing.harness.components import get
        entries: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for comp_name in ("events", "environment", "captions"):
            try:
                comp = get(comp_name)
            except (ImportError, AttributeError, KeyError):
                continue
            comp_space = comp.space(cfg, labels_dir)
            for item in comp_space:
                name = item.get("name")
                if name not in seen_names:
                    seen_names.add(name)
                    entries.append(item)
        return entries

    def stat_objective(self, totals: np.ndarray) -> np.ndarray | float:
        """totals: (B, 4) or (4,) [tp_fine, tp_coarse, fp, fn] -> micro F0.5."""
        arr = np.asarray(totals, dtype=np.float64)
        if arr.ndim == 1:
            f = _vectorized_f_beta(arr.reshape(1, -1), beta=0.5)
            return float(f[0])
        return _vectorized_f_beta(arr, beta=0.5)


COMPONENT = PipelineComponent()
