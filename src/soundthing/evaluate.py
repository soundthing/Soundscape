"""Evaluation harness — OWNER: Person 3.

Event metrics: per-clip greedy matching of predicted to labeled events of the same
class with |onset difference| <= tolerance. Reported at two levels:
  fine   — class_id must match exactly (predicted coarse labels count as wrong)
  coarse — both sides mapped to their coarse group (classes without one stay as-is)
False positives per minute = unmatched predictions / labeled minutes.
Environment accuracy = time-weighted agreement of predicted vs labeled env.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import Config
from .contracts import CaptionEvent, ClipLabels, EnvSegment


@dataclass
class EventCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    minutes: float = 0.0

    def __iadd__(self, o: "EventCounts") -> "EventCounts":
        self.tp += o.tp
        self.fp += o.fp
        self.fn += o.fn
        self.minutes += o.minutes
        return self

    def summary(self) -> dict:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        fpm = self.fp / self.minutes if self.minutes else 0.0
        return {**asdict(self), "precision": p, "recall": r, "f1": f1, "fp_per_min": fpm}


def match_events(
    pred: list[tuple[float, str]], truth: list[tuple[float, str]], tolerance: float
) -> tuple[int, int, int]:
    """pred/truth: (onset, label). Greedy closest-onset matching per label."""
    used = [False] * len(truth)
    tp = 0
    for onset, label in sorted(pred):
        best, best_d = None, tolerance
        for j, (t_on, t_label) in enumerate(truth):
            d = abs(onset - t_on)
            if not used[j] and t_label == label and d <= best_d:
                best, best_d = j, d
        if best is not None:
            used[best] = True
            tp += 1
    return tp, len(pred) - tp, len(truth) - tp


def score_events(
    pred: list[CaptionEvent], labels: ClipLabels, cfg: Config, tolerance: float, coarse: bool = False
) -> EventCounts:
    def key(cid: str) -> str:
        if not coarse or cid in cfg.coarse_groups:
            return cid
        try:
            return cfg.coarse_of(cid) or cid
        except KeyError:
            return cid

    p = [(e.start, key(e.class_id)) for e in pred]
    t = [(e.start, key(e.class_id)) for e in labels.events]
    tp, fp, fn = match_events(p, t, tolerance)
    return EventCounts(tp, fp, fn, labels.duration / 60.0)


def env_accuracy(pred: list[EnvSegment], labels: ClipLabels) -> float:
    total = sum(s.end - s.start for s in labels.segments)
    if not total:
        return 0.0
    agree = 0.0
    for ls in labels.segments:
        for ps in pred:
            if ps.env == ls.env:
                agree += max(0.0, min(ls.end, ps.end) - max(ls.start, ps.start))
    return agree / total


def check_labels(labels: ClipLabels, cfg: Config) -> list[str]:
    problems = []
    for e in labels.events:
        if e.class_id not in cfg.class_ids:
            problems.append(f"{labels.video_id}: unknown class_id {e.class_id!r} at {e.start}")
    for s in labels.segments:
        if s.env not in cfg.env_ids:
            problems.append(f"{labels.video_id}: unknown env {s.env!r} at {s.start}")
    return problems
