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


from dataclasses import replace


def _compute_clip_median_db(mono: np.ndarray, frame_len: int, eps: float = 1e-12) -> float | None:
    clip_n_frames = len(mono) // frame_len
    if clip_n_frames == 0:
        return None
    frames = mono[: clip_n_frames * frame_len].reshape(clip_n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    valid = frame_rms >= 1e-6
    if not np.any(valid):
        return None
    frame_dbs = 20.0 * np.log10(frame_rms[valid] + eps)
    return float(np.median(frame_dbs))


def add_modifiers(
    events: list[CaptionEvent],
    stereo: np.ndarray | None,
    sr: int,
    cfg: Config,
) -> list[CaptionEvent]:
    """Add left/right, approaching/receding, loud/quiet modifiers to events."""
    mod_cfg = (cfg.pipeline.get("events") or {}).get("modifiers") or {}
    if stereo is None or not mod_cfg.get("enabled", True):
        return events

    side_balance = float(mod_cfg.get("side_balance", 0.35))
    trend_db = float(mod_cfg.get("trend_db", 6.0))
    min_trend_sec = float(mod_cfg.get("min_trend_sec", 1.5))
    loud_db = float(mod_cfg.get("loud_db", 12.0))
    quiet_db = float(mod_cfg.get("quiet_db", -12.0))
    frame_sec = float(mod_cfg.get("frame_sec", 0.1))

    stereo_arr = np.asarray(stereo, dtype=np.float64)
    n_samples = len(stereo_arr)
    if n_samples == 0:
        return [replace(e, modifiers=list(dict.fromkeys(e.modifiers))) for e in events]

    if stereo_arr.ndim == 1:
        mono = stereo_arr
        has_stereo = False
    elif stereo_arr.ndim == 2:
        has_stereo = stereo_arr.shape[1] >= 2
        mono = stereo_arr.mean(axis=1)
    else:
        return events

    frame_len = max(1, int(round(frame_sec * sr)))
    clip_median_db = _compute_clip_median_db(mono, frame_len)

    out_events: list[CaptionEvent] = []
    for ev in events:
        current_mods: list[str] = list(dict.fromkeys(ev.modifiers))

        start_sample = int(round(ev.start * sr))
        end_sample = int(round(ev.end * sr))
        start_idx = max(0, min(n_samples, start_sample))
        end_idx = max(0, min(n_samples, end_sample))
        event_len = end_idx - start_idx

        # Events with < 1 frame of audio get no modifiers
        if event_len < frame_len:
            out_events.append(replace(ev, modifiers=current_mods))
            continue

        detected: list[str] = []

        # 1. left / right (only when >= 2 channels)
        if has_stereo:
            L_seg = stereo_arr[start_idx:end_idx, 0]
            R_seg = stereo_arr[start_idx:end_idx, 1]
            rms_L = float(np.sqrt(np.mean(L_seg**2)))
            rms_R = float(np.sqrt(np.mean(R_seg**2)))
            balance = (rms_R - rms_L) / (rms_R + rms_L + 1e-12)
            if balance >= side_balance:
                detected.append("right")
            elif balance <= -side_balance:
                detected.append("left")

        # 2. approaching / receding (only if end - start >= min_trend_sec)
        duration = ev.end - ev.start
        if duration >= min_trend_sec:
            ev_mono = mono[start_idx:end_idx]
            n_ev_frames = len(ev_mono) // frame_len
            if n_ev_frames >= 2:
                ev_frames = ev_mono[: n_ev_frames * frame_len].reshape(n_ev_frames, frame_len)
                ev_frame_rms = np.sqrt(np.mean(ev_frames**2, axis=1))
                ev_frame_dbs = 20.0 * np.log10(ev_frame_rms + 1e-12)
                times = np.arange(n_ev_frames) * frame_sec
                slope = float(np.polyfit(times, ev_frame_dbs, 1)[0])
                delta_db = slope * duration
                if delta_db >= trend_db:
                    detected.append("approaching")
                elif delta_db <= -trend_db:
                    detected.append("receding")

        # 3. loud / quiet
        if clip_median_db is not None:
            ev_mono = mono[start_idx:end_idx]
            ev_rms = float(np.sqrt(np.mean(ev_mono**2)))
            ev_db = 20.0 * np.log10(ev_rms + 1e-12)
            diff_db = ev_db - clip_median_db
            if diff_db >= loud_db:
                detected.append("loud")
            elif diff_db <= quiet_db:
                detected.append("quiet")

        # Merge detected into current_mods without duplicates
        merged_mods = list(current_mods)
        for m in detected:
            if m not in merged_mods:
                merged_mods.append(m)

        out_events.append(replace(ev, modifiers=merged_mods))

    return out_events

