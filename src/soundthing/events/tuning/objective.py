"""Fixed tuning objective and evaluation metrics for the events stage."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from soundthing.config import Config
from soundthing.contracts import CaptionEvent, ClipLabels, LabeledEvent

OBJECTIVE = "f0.5_conditioned_micro"


@dataclass
class ClipMatch:
    video_id: str
    minutes: float
    tp_fine: int
    tp_coarse: int
    fp: int
    fn: int
    pairs: list[tuple[CaptionEvent, LabeledEvent]] = field(default_factory=list)
    false_pos: list[CaptionEvent] = field(default_factory=list)
    missed: list[LabeledEvent] = field(default_factory=list)
    safety_tp: int = 0
    safety_total: int = 0


def _coarse_key(cid: str, cfg: Config) -> str:
    """Return the coarse group id or cid if no coarse group exists."""
    if cid in cfg.coarse_groups:
        return cid
    try:
        return cfg.coarse_of(cid) or cid
    except KeyError:
        return cid


def match_clip(pred: list[CaptionEvent], labels: ClipLabels, cfg: Config, tol: float) -> ClipMatch:
    """Greedy onset matching matching soundthing.evaluate semantics with coarse credit."""
    truth = labels.events
    used_pred = [False] * len(pred)
    used_truth = [False] * len(truth)

    sorted_pred_indices = sorted(range(len(pred)), key=lambda i: (pred[i].start, pred[i].class_id))

    fine_pairs: list[tuple[CaptionEvent, LabeledEvent]] = []
    for i in sorted_pred_indices:
        p_ev = pred[i]
        best_j: int | None = None
        best_d = tol
        for j, t_ev in enumerate(truth):
            if not used_truth[j] and t_ev.class_id == p_ev.class_id:
                d = abs(p_ev.start - t_ev.start)
                if d <= best_d:
                    best_j = j
                    best_d = d
        if best_j is not None:
            used_truth[best_j] = True
            used_pred[i] = True
            fine_pairs.append((p_ev, truth[best_j]))

    coarse_pairs: list[tuple[CaptionEvent, LabeledEvent]] = []
    leftover_p_indices = [i for i in range(len(pred)) if not used_pred[i]]
    sorted_coarse_pred_indices = sorted(
        leftover_p_indices,
        key=lambda i: (pred[i].start, _coarse_key(pred[i].class_id, cfg)),
    )
    for i in sorted_coarse_pred_indices:
        p_ev = pred[i]
        p_key = _coarse_key(p_ev.class_id, cfg)
        best_j = None
        best_d = tol
        for j, t_ev in enumerate(truth):
            if not used_truth[j] and _coarse_key(t_ev.class_id, cfg) == p_key:
                d = abs(p_ev.start - t_ev.start)
                if d <= best_d:
                    best_j = j
                    best_d = d
        if best_j is not None:
            used_truth[best_j] = True
            used_pred[i] = True
            coarse_pairs.append((p_ev, truth[best_j]))

    tp_fine = len(fine_pairs)
    tp_coarse = len(coarse_pairs)
    fp = len(pred) - tp_fine - tp_coarse
    fn = len(truth) - tp_fine - tp_coarse
    pairs = fine_pairs + coarse_pairs
    false_pos = [pred[i] for i in range(len(pred)) if not used_pred[i]]
    missed = [truth[j] for j in range(len(truth)) if not used_truth[j]]
    minutes = labels.duration / 60.0

    safety_ids = cfg.safety_ids()
    safety_total = sum(1 for t in truth if t.class_id in safety_ids)
    safety_tp = sum(1 for _, t in pairs if t.class_id in safety_ids)

    return ClipMatch(
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
    )


def f_beta(clips: list[ClipMatch], beta: float = 0.5) -> float:
    """Pooled micro F-beta across clips with coarse-only matches at 0.5 credit."""
    tp_fine = sum(c.tp_fine for c in clips)
    tp_coarse = sum(c.tp_coarse for c in clips)
    fp = sum(c.fp for c in clips)
    fn = sum(c.fn for c in clips)

    credit = tp_fine + 0.5 * tp_coarse
    if credit == 0.0:
        return 0.0

    fp_prime = fp + 0.5 * tp_coarse
    fn_prime = fn + 0.5 * tp_coarse

    p = credit / (credit + fp_prime)
    r = credit / (credit + fn_prime)

    b2 = beta * beta
    denom = b2 * p + r
    if denom == 0.0:
        return 0.0
    return float((1.0 + b2) * p * r / denom)


def grid_f1(
    preds_by_clip: dict[str, list[CaptionEvent]],
    labels_by_clip: dict[str, ClipLabels],
    slice_sec: float = 1.0,
) -> float:
    """Time-sliced grid F1 (fine class ids) pooled across all clips."""
    if slice_sec <= 0:
        raise ValueError(f"slice_sec must be > 0, got {slice_sec}")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for vid, labels in labels_by_clip.items():
        preds = preds_by_clip.get(vid, [])
        num_slices = max(0, math.ceil(labels.duration / slice_sec))
        if num_slices == 0:
            continue

        pred_cells: set[tuple[int, str]] = set()
        for p in preds:
            k_min = max(0, int(p.start // slice_sec))
            k_max = min(num_slices - 1, int(p.end // slice_sec))
            for k in range(k_min, k_max + 1):
                if p.start < (k + 1) * slice_sec and p.end > k * slice_sec:
                    pred_cells.add((k, p.class_id))

        truth_cells: set[tuple[int, str]] = set()
        for t in labels.events:
            k_min = max(0, int(t.start // slice_sec))
            k_max = min(num_slices - 1, int(t.end // slice_sec))
            for k in range(k_min, k_max + 1):
                if t.start < (k + 1) * slice_sec and t.end > k * slice_sec:
                    truth_cells.add((k, t.class_id))

        tp = len(pred_cells & truth_cells)
        fp = len(pred_cells - truth_cells)
        fn = len(truth_cells - pred_cells)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    if total_tp == 0:
        return 0.0

    p = total_tp / (total_tp + total_fp)
    r = total_tp / (total_tp + total_fn)
    if p + r == 0.0:
        return 0.0
    return float(2.0 * p * r / (p + r))


def diagnostics(clips: list[ClipMatch], cfg: Config) -> dict[str, Any]:
    """Return diagnostic breakdowns: per_class, onset_error, and top_false_positives."""
    per_class: dict[str, dict[str, int]] = {}

    def _entry(cid: str) -> dict[str, int]:
        if cid not in per_class:
            per_class[cid] = {"tp": 0, "tp_coarse": 0, "fp": 0, "fn": 0}
        return per_class[cid]

    for c in clips:
        if len(c.pairs) == c.tp_fine + c.tp_coarse:
            for p, t in c.pairs[: c.tp_fine]:
                _entry(t.class_id)["tp"] += 1
            for p, t in c.pairs[c.tp_fine : c.tp_fine + c.tp_coarse]:
                _entry(t.class_id)["tp_coarse"] += 1
        else:
            for p, t in c.pairs:
                if p.class_id == t.class_id:
                    _entry(t.class_id)["tp"] += 1
                else:
                    _entry(t.class_id)["tp_coarse"] += 1
        for p in c.false_pos:
            _entry(p.class_id)["fp"] += 1
        for t in c.missed:
            _entry(t.class_id)["fn"] += 1

    sorted_per_class = {cid: per_class[cid] for cid in sorted(per_class)}

    errs: list[float] = [float(p.start - t.start) for c in clips for p, t in c.pairs]
    if not errs:
        onset_error: dict[str, Any] = {"mean": None, "median": None, "p90": None, "n": 0}
    else:
        abs_errs = [abs(e) for e in errs]
        onset_error = {
            "mean": float(np.mean(errs)),
            "median": float(np.median(errs)),
            "p90": float(np.percentile(abs_errs, 90)),
            "n": int(len(errs)),
        }

    raw_fps = []
    for c in clips:
        for p in c.false_pos:
            raw_fps.append({
                "video_id": c.video_id,
                "start": float(p.start),
                "class_id": p.class_id,
                "confidence": float(p.confidence),
            })

    top_fps = sorted(
        raw_fps,
        key=lambda x: (-x["confidence"], x["video_id"], x["start"]),
    )[:20]

    return {
        "per_class": sorted_per_class,
        "onset_error": onset_error,
        "top_false_positives": top_fps,
    }
