"""Post-processing: FrameScores -> display-ready CaptionEvents.

Offline, so smoothing may use future context (two-sided filters).
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..contracts import CaptionEvent
from .types import FrameScores


def smooth(scores: FrameScores, width_sec: float) -> FrameScores:
    """Two-sided median (or mean) filter along time, per class."""
    raise NotImplementedError("TODO(Person 1)")


def threshold_to_events(scores: FrameScores, cfg: Config) -> list[CaptionEvent]:
    """Per-class thresholds (cfg.threshold), drop events < min_event_sec, merge
    same-class events separated by < merge_gap_sec. confidence = peak score."""
    raise NotImplementedError("TODO(Person 1)")


def collapse_confusable(events: list[CaptionEvent], scores: FrameScores, cfg: Config) -> list[CaptionEvent]:
    """If the runner-up class within an event is in the same coarse group and
    within coarse_margin, relabel to the coarse group (class_id = group id)."""
    raise NotImplementedError("TODO(Person 1)")


def priority_filter(events: list[CaptionEvent], cfg: Config) -> list[CaptionEvent]:
    """Caption sustained classes once per sustained_repeat_sec; cap concurrent
    events at max_on_screen, keeping safety classes first, then by confidence."""
    raise NotImplementedError("TODO(Person 1)")


def add_modifiers(events: list[CaptionEvent], stereo: np.ndarray | None, sr: int) -> list[CaptionEvent]:
    """Optional: left/right from stereo balance; approaching/receding from loudness trend."""
    return events
