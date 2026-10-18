"""Captions tuning component (Track C).

Scores caption-cue generation against labeled events with min_cue_sec and max_cue_sec clamping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from soundthing.config import Config
from soundthing.contracts import CaptionEvent, ClipLabels, LabeledEvent, load_labels
from soundthing.captions import events_to_cues
from soundthing.harness.components import events as ev_comp
from soundthing.harness.components.base import Clip, Component, Evaluation


@dataclass
class CaptionCue:
    start: float
    end: float
    class_id: str


@dataclass
class CueMatch:
    video_id: str
    minutes: float
    tp_fine: int
    tp_coarse: int
    fp: int
    fn: int
    pairs: list[tuple[Any, LabeledEvent]] = field(default_factory=list)
    false_pos: list[Any] = field(default_factory=list)
    missed: list[LabeledEvent] = field(default_factory=list)
    safety_tp: int = 0
    safety_total: int = 0
    cue_durations: list[float] = field(default_factory=list)
    offset_errors: list[float] = field(default_factory=list)


def _coarse_key(cid: str, cfg: Config) -> str:
    """Return the coarse group id or cid if no coarse group exists."""
    if cid in cfg.coarse_groups:
        return cid
    try:
        return cfg.coarse_of(cid) or cid
    except (KeyError, ValueError):
        return cid


def _cue_info(c: Any) -> tuple[float, float, str]:
    if hasattr(c, "start") and hasattr(c, "end") and hasattr(c, "class_id"):
        return float(c.start), float(c.end), str(c.class_id)
    if isinstance(c, tuple):
        if len(c) == 2:
            cue_obj, cid = c
            return float(cue_obj.start), float(cue_obj.end), str(cid)
        elif len(c) == 3:
            return float(c[0]), float(c[1]), str(c[2])
    raise TypeError(f"Cannot extract (start, end, class_id) from cue item: {c!r}")


def match_cues(
    cues: list[Any],
    labels: ClipLabels,
    cfg: Config,
    tol: float | None = None,
) -> CueMatch:
    """Greedy cue-level onset matching with coarse credit and offset tolerance."""
    if tol is None:
        tol = float(cfg.pipeline.get("eval", {}).get("onset_tolerance_sec", 0.5))

    truth = labels.events
    parsed_cues = [_cue_info(c) for c in cues]
    cue_durations = [end - start for start, end, _ in parsed_cues]

    used_cue = [False] * len(parsed_cues)
    used_truth = [False] * len(truth)

    sorted_cue_indices = sorted(
        range(len(parsed_cues)),
        key=lambda i: (parsed_cues[i][0], parsed_cues[i][2]),
    )

    fine_pairs: list[tuple[Any, LabeledEvent]] = []
    fine_offset_errors: list[float] = []

    for i in sorted_cue_indices:
        start, end, cid = parsed_cues[i]
        best_j: int | None = None
        best_d = tol
        for j, t_ev in enumerate(truth):
            if not used_truth[j] and t_ev.class_id == cid:
                dur = t_ev.end - t_ev.start
                offset_tol = max(tol, 0.5 * dur)
                if abs(end - t_ev.end) <= offset_tol:
                    d = abs(start - t_ev.start)
                    if d <= best_d:
                        best_j = j
                        best_d = d
        if best_j is not None:
            used_truth[best_j] = True
            used_cue[i] = True
            fine_pairs.append((cues[i], truth[best_j]))
            fine_offset_errors.append(end - truth[best_j].end)

    coarse_pairs: list[tuple[Any, LabeledEvent]] = []
    coarse_offset_errors: list[float] = []
    leftover_indices = [i for i in range(len(parsed_cues)) if not used_cue[i]]
    sorted_coarse_indices = sorted(
        leftover_indices,
        key=lambda i: (parsed_cues[i][0], _coarse_key(parsed_cues[i][2], cfg)),
    )

    for i in sorted_coarse_indices:
        start, end, cid = parsed_cues[i]
        p_key = _coarse_key(cid, cfg)
        best_j = None
        best_d = tol
        for j, t_ev in enumerate(truth):
            if not used_truth[j] and _coarse_key(t_ev.class_id, cfg) == p_key:
                dur = t_ev.end - t_ev.start
                offset_tol = max(tol, 0.5 * dur)
                if abs(end - t_ev.end) <= offset_tol:
                    d = abs(start - t_ev.start)
                    if d <= best_d:
                        best_j = j
                        best_d = d
        if best_j is not None:
            used_truth[best_j] = True
            used_cue[i] = True
            coarse_pairs.append((cues[i], truth[best_j]))
            coarse_offset_errors.append(end - truth[best_j].end)

    tp_fine = len(fine_pairs)
    tp_coarse = len(coarse_pairs)
    fp = len(parsed_cues) - tp_fine - tp_coarse
    fn = len(truth) - tp_fine - tp_coarse
    pairs = fine_pairs + coarse_pairs
    offset_errors = fine_offset_errors + coarse_offset_errors

    false_pos = [cues[i] for i in range(len(cues)) if not used_cue[i]]
    missed = [truth[j] for j in range(len(truth)) if not used_truth[j]]
    minutes = labels.duration / 60.0

    safety_ids = cfg.safety_ids()
    safety_total = sum(1 for t in truth if t.class_id in safety_ids)
    safety_tp = sum(1 for _, t in pairs if t.class_id in safety_ids)

    return CueMatch(
        video_id=labels.video_id,
        minutes=minutes,
        tp_fine=tp_fine,
        tp_coarse=tp_coarse,
        fp=fp,
        fn=fn,
        pairs=pairs,
        false_pos=false_pos,
        missed=missed,
        safety_tp=safety_tp,
        safety_total=safety_total,
        cue_durations=cue_durations,
        offset_errors=offset_errors,
    )


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


class CaptionsComponent:
    name: str = "captions"
    sections: tuple[str, ...] = ("captions.",)
    code_paths: tuple[str, ...] = ("src/soundthing/captions.py",)
    tuning_log: str = "src/soundthing/harness/references/captions-tuning-log.md"
    n_stats: int = 4

    def cache_key(self, cfg: Config) -> str | None:
        """Hash of events settings (reuses events cache)."""
        return ev_comp.COMPONENT.cache_key(cfg)

    def build_cache(self, video_id: str, audio_path: Path, cfg: Config) -> Path | None:
        """No cache of its own."""
        return None

    def load_clips(self, cfg: Config, labels_dir: Path) -> list[Clip]:
        """Load labels and events caches."""
        labels_path = Path(labels_dir)
        label_files = sorted(labels_path.glob("*.json"))
        if not label_files:
            raise ValueError(f"No label files found in {labels_dir}")
        all_labels = [load_labels(p) for p in label_files]
        all_labels.sort(key=lambda l: l.video_id)
        caches = ev_comp.load_event_caches(cfg, all_labels)

        return [
            Clip(video_id=lbl.video_id, labels=lbl, cache=c)
            for lbl, c in zip(all_labels, caches)
        ]

    def evaluate(self, clips: list[Clip], cfg: Config) -> Evaluation:
        """Render cues and score them against ground truth."""
        captions_cfg = cfg.pipeline.get("captions") or {}
        min_cue_sec = float(captions_cfg.get("min_cue_sec", 1.0))
        max_cue_sec = float(captions_cfg.get("max_cue_sec", 6.0))
        if min_cue_sec > max_cue_sec:
            raise ValueError(
                f"captions.min_cue_sec ({min_cue_sec}) cannot exceed captions.max_cue_sec ({max_cue_sec})"
            )

        text_emb = ev_comp.text_emb_for(cfg, [c.cache for c in clips])

        matches: list[CueMatch] = []
        per_clip: dict[str, dict[str, Any]] = {}
        tol = float(cfg.pipeline.get("eval", {}).get("onset_tolerance_sec", 0.5))

        for clip in clips:
            cache = clip.cache
            lbl = clip.labels
            if hasattr(cache, "duration") and cache.duration != lbl.duration:
                from dataclasses import replace
                cache = replace(cache, duration=lbl.duration)

            cond_segs = ev_comp.labeled_env_segments(lbl, cfg, confidence=0.7)
            events = ev_comp.replay_events(cache, cond_segs, cfg, text_emb)

            sorted_events = sorted(events, key=lambda e: e.start)
            cues = events_to_cues(sorted_events, min_cue=min_cue_sec, max_cue=max_cue_sec)
            pred_cues = [
                CaptionCue(start=c.start, end=c.end, class_id=ev.class_id)
                for c, ev in zip(cues, sorted_events)
            ]

            m = match_cues(pred_cues, lbl, cfg, tol)
            matches.append(m)

            per_clip[lbl.video_id] = {
                "stats": [float(m.tp_fine), float(m.tp_coarse), float(m.fp), float(m.fn)],
                "tp_fine": m.tp_fine,
                "tp_coarse": m.tp_coarse,
                "fp": m.fp,
                "fn": m.fn,
                "minutes": m.minutes,
                "safety_tp": m.safety_tp,
                "safety_total": m.safety_total,
            }

        total_stats = np.sum(
            [per_clip[lbl.video_id]["stats"] for lbl in [c.labels for c in clips]],
            axis=0,
        ) if clips else np.zeros(4, dtype=np.float64)
        obj = float(self.stat_objective(total_stats))

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

        all_durations = [dur for m in matches for dur in m.cue_durations]
        mean_cue_sec = float(np.mean(all_durations)) if all_durations else 0.0

        all_offset_errors = [err for m in matches for err in m.offset_errors]
        if all_offset_errors:
            counts, bin_edges = np.histogram(all_offset_errors, bins=5)
            offset_hist = {
                "counts": [int(x) for x in counts],
                "bin_edges": [float(x) for x in bin_edges],
            }
        else:
            offset_hist = {
                "counts": [0, 0, 0, 0, 0],
                "bin_edges": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }

        metrics = {
            "objective": obj,
            "safety_recall": safety_recall,
            "mean_cue_sec": mean_cue_sec,
            "fp_per_min": fp_per_min,
        }

        diagnostics = {
            "offset_error_histogram": offset_hist,
            "offset_error": offset_hist,
        }

        return Evaluation(
            objective=obj,
            metrics=metrics,
            per_clip=per_clip,
            diagnostics=diagnostics,
        )

    def limits(self, metrics: dict[str, float], best: dict[str, Any] | None) -> dict[str, bool]:
        """Check safety_recall vs best."""
        if best is None:
            return {"safety_recall": True}
        best_safety = float(
            best.get("safety_recall", best.get("metrics", {}).get("safety_recall", 0.0))
        )
        return {
            "safety_recall": bool(metrics["safety_recall"] >= best_safety - 1e-9),
        }

    def space(self, cfg: Config, labels_dir: Path | None = None) -> list[dict[str, Any]]:
        """Search space for captions stage."""
        captions_cfg = cfg.pipeline.get("captions") or {}
        min_cue = float(captions_cfg.get("min_cue_sec", 1.0))
        max_cue = float(captions_cfg.get("max_cue_sec", 6.0))
        return [
            {
                "name": "captions.min_cue_sec",
                "type": "float",
                "bounds": [0.2, 3.0],
                "current": min_cue,
                "shared": False,
            },
            {
                "name": "captions.max_cue_sec",
                "type": "float",
                "bounds": [2.0, 12.0],
                "current": max_cue,
                "shared": False,
            },
        ]

    def stat_objective(self, totals: np.ndarray) -> np.ndarray | float:
        """totals: (B, 4) or (4,) [tp_fine, tp_coarse, fp, fn] -> micro F0.5."""
        arr = np.asarray(totals, dtype=np.float64)
        if arr.ndim == 1:
            f = _vectorized_f_beta(arr.reshape(1, -1), beta=0.5)
            return float(f[0])
        return _vectorized_f_beta(arr, beta=0.5)


COMPONENT = CaptionsComponent()
