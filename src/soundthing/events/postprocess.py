"""Post-processing: FrameScores -> display-ready CaptionEvents.

Offline, so smoothing may use future context (two-sided filters).
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..contracts import CaptionEvent, MODIFIERS
from .types import FrameScores


def _safe_coarse_of(class_id: str, cfg: Config) -> str | None:
    try:
        return cfg.coarse_of(class_id)
    except KeyError:
        return None


def smooth(scores: FrameScores, width_sec: float) -> FrameScores:
    """Two-sided running median filter along time, per class.

    Window length in frames k = round(width_sec / scores.hop), forced odd (k+1 if even).
    k <= 1 or width_sec <= 0 returns input unchanged. Edge-padded so output has the same shape.
    """
    if width_sec <= 0 or scores.scores.shape[0] <= 1:
        return scores

    k = int(round(width_sec / scores.hop))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return scores

    r = (k - 1) // 2
    padded = np.pad(scores.scores, ((r, r), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=k, axis=0)
    smoothed = np.median(windows, axis=-1)
    return scores.copy_with(smoothed.astype(scores.scores.dtype, copy=False))


def threshold_to_events(scores: FrameScores, cfg: Config) -> list[CaptionEvent]:
    """Per-class thresholds (cfg.threshold), merge same-class events separated by
    < merge_gap_sec, then drop events < min_event_sec (merge before drop, so
    fragments of one sound survive). confidence = peak score inside final span."""
    pcfg = cfg.pipeline.get("events", {}).get("postprocess", {})
    min_event_sec = float(pcfg.get("min_event_sec", 0.3))
    merge_gap_sec = float(pcfg.get("merge_gap_sec", 0.5))

    events: list[CaptionEvent] = []
    if scores.scores.shape[0] == 0:
        return events

    for c, class_id in enumerate(scores.class_ids):
        col = scores.scores[:, c]
        thr = cfg.threshold(class_id)
        active = col >= thr
        if not np.any(active):
            continue

        padded = np.pad(active, (1, 1), mode="constant", constant_values=False)
        diff = np.diff(padded.astype(np.int8))
        starts_idx = np.where(diff == 1)[0]
        ends_idx = np.where(diff == -1)[0] - 1

        runs: list[tuple[float, float]] = []
        for i, j in zip(starts_idx, ends_idx):
            s = max(0.0, float(scores.times[i] - scores.hop / 2.0))
            e = float(scores.times[j] + scores.hop / 2.0)
            runs.append((s, e))

        # First merge runs separated by < merge_gap_sec
        merged_spans: list[tuple[float, float]] = []
        for s, e in runs:
            if not merged_spans:
                merged_spans.append((s, e))
            else:
                prev_s, prev_e = merged_spans[-1]
                if (s - prev_e) < merge_gap_sec:
                    merged_spans[-1] = (prev_s, max(prev_e, e))
                else:
                    merged_spans.append((s, e))

        # Then drop events shorter than min_event_sec (merge before drop)
        kept_spans = [(s, e) for s, e in merged_spans if (e - s) >= min_event_sec]

        for s, e in kept_spans:
            in_span = (scores.times >= s - 1e-6) & (scores.times <= e + 1e-6)
            if np.any(in_span):
                peak = float(np.max(col[in_span]))
            else:
                nearest = int(np.argmin(np.abs(scores.times - (s + e) / 2.0)))
                peak = float(col[nearest])
            conf = round(float(np.clip(peak, 0.0, 1.0)), 3)
            start_r = round(s, 2)
            end_r = round(e, 2)
            if end_r <= start_r:
                end_r = start_r + 0.01

            events.append(
                CaptionEvent(
                    start=start_r,
                    end=end_r,
                    label=cfg.caption_for(class_id),
                    class_id=class_id,
                    confidence=conf,
                )
            )

    events.sort(key=lambda ev: (ev.start, ev.class_id))
    return events


def collapse_confusable(
    events: list[CaptionEvent], scores: FrameScores, cfg: Config
) -> list[CaptionEvent]:
    """If the runner-up class within an event is in the same coarse group and
    within coarse_margin, relabel to the coarse group (class_id = group id).
    Merge overlapping or touching events sharing the same class_id."""
    if not events:
        return []

    pcfg = cfg.pipeline.get("events", {}).get("postprocess", {})
    coarse_margin = float(pcfg.get("coarse_margin", 0.1))

    relabeled: list[CaptionEvent] = []
    for ev in events:
        group_id = _safe_coarse_of(ev.class_id, cfg)

        if group_id is not None and ev.class_id in scores.class_ids and scores.scores.shape[0] > 0:
            in_span = (scores.times >= ev.start - 1e-6) & (scores.times <= ev.end + 1e-6)
            if np.any(in_span):
                frame_scores = scores.scores[in_span, :]
            else:
                nearest = int(np.argmin(np.abs(scores.times - (ev.start + ev.end) / 2.0)))
                frame_scores = scores.scores[nearest : nearest + 1, :]

            peaks = np.max(frame_scores, axis=0)
            own_idx = scores.class_ids.index(ev.class_id)
            own_peak = float(peaks[own_idx])

            other_indices = [idx for idx in range(len(scores.class_ids)) if idx != own_idx]
            if other_indices:
                runner_up_idx = max(
                    other_indices,
                    key=lambda idx: (
                        peaks[idx],
                        _safe_coarse_of(scores.class_ids[idx], cfg) == group_id,
                    ),
                )
                runner_up_cid = scores.class_ids[runner_up_idx]
                runner_up_peak = float(peaks[runner_up_idx])
                runner_up_group = _safe_coarse_of(runner_up_cid, cfg)

                if runner_up_group == group_id and (own_peak - runner_up_peak) <= coarse_margin:
                    relabeled.append(
                        CaptionEvent(
                            start=ev.start,
                            end=ev.end,
                            label=cfg.caption_for(group_id),
                            class_id=group_id,
                            confidence=ev.confidence,
                            modifiers=list(ev.modifiers),
                        )
                    )
                    continue

        relabeled.append(ev)

    # After relabelling, merge overlapping (or touching) events that now share the same class_id
    # into one (min start, max end, max confidence).
    class_order = list(dict.fromkeys(e.class_id for e in relabeled))
    merged_events: list[CaptionEvent] = []

    for cid in class_order:
        cid_events = [e for e in relabeled if e.class_id == cid]
        cid_events.sort(key=lambda e: (e.start, e.end))

        cur_merged: list[CaptionEvent] = []
        for e in cid_events:
            if not cur_merged:
                cur_merged.append(e)
            else:
                prev = cur_merged[-1]
                if e.start <= prev.end + 1e-6:
                    new_start = min(prev.start, e.start)
                    new_end = max(prev.end, e.end)
                    new_conf = max(prev.confidence, e.confidence)
                    new_mods = [
                        m for m in MODIFIERS if m in prev.modifiers or m in e.modifiers
                    ]
                    cur_merged[-1] = CaptionEvent(
                        start=round(new_start, 2),
                        end=round(new_end, 2),
                        label=prev.label,
                        class_id=prev.class_id,
                        confidence=round(new_conf, 3),
                        modifiers=new_mods,
                    )
                else:
                    cur_merged.append(e)
        merged_events.extend(cur_merged)

    merged_events.sort(key=lambda e: (e.start, e.end, e.class_id))
    return merged_events


def priority_filter(events: list[CaptionEvent], cfg: Config) -> list[CaptionEvent]:
    """Caption sustained classes once per sustained_repeat_sec; cap concurrent
    events at max_on_screen, keeping safety classes first, then by confidence."""
    if not events:
        return []

    pcfg = cfg.pipeline.get("events", {}).get("postprocess", {})
    sustained_repeat_sec = float(pcfg.get("sustained_repeat_sec", 60.0))
    max_on_screen = int(pcfg.get("max_on_screen", 2))
    min_cue_sec = float(cfg.pipeline.get("captions", {}).get("min_cue_sec", 1.0))

    # a. Sustained filtering
    events_sorted = sorted(events, key=lambda e: (e.start, e.end))
    kept_sustained: list[CaptionEvent] = []
    last_kept_start_by_class: dict[str, float] = {}

    for ev in events_sorted:
        is_sustained = False
        try:
            sc = cfg.sound_class(ev.class_id)
            is_sustained = sc.sustained
        except KeyError:
            is_sustained = False

        if is_sustained:
            if ev.class_id in last_kept_start_by_class:
                last_start = last_kept_start_by_class[ev.class_id]
                if (ev.start - last_start) < sustained_repeat_sec:
                    continue
            last_kept_start_by_class[ev.class_id] = ev.start

        kept_sustained.append(ev)

    # b. On-screen cap
    safety_ids = cfg.safety_ids()

    def priority_key(ev: CaptionEvent) -> tuple:
        is_safety = ev.class_id in safety_ids
        return (
            0 if is_safety else 1,
            -ev.confidence,
            ev.start,
            ev.end,
            ev.class_id,
        )

    priority_ordered = sorted(kept_sustained, key=priority_key)
    accepted: list[CaptionEvent] = []
    accepted_spans: list[tuple[float, float]] = []

    for ev in priority_ordered:
        disp_start = ev.start
        disp_end = max(ev.end, ev.start + min_cue_sec)

        overlapping_spans = [
            (s, e)
            for s, e in accepted_spans
            if max(disp_start, s) < min(disp_end, e)
        ]

        can_accept = True
        if len(overlapping_spans) >= max_on_screen:
            test_instants = {disp_start}
            for s, e in overlapping_spans:
                if disp_start <= s < disp_end:
                    test_instants.add(s)

            for t in test_instants:
                count = sum(1 for s, e in overlapping_spans if s <= t < e)
                if count >= max_on_screen:
                    can_accept = False
                    break

        if can_accept:
            accepted.append(ev)
            accepted_spans.append((disp_start, disp_end))

    accepted.sort(key=lambda e: (e.start, e.end, e.class_id))
    return accepted


def add_modifiers(events: list[CaptionEvent], stereo: np.ndarray | None, sr: int) -> list[CaptionEvent]:
    """Optional: left/right from stereo balance; approaching/receding from loudness trend."""
    return events
