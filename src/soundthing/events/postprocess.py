"""Threshold, merge and label events without score smoothing or display limits."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..config import Config
from ..contracts import CaptionEvent, MODIFIERS
from .types import FrameScores

SAFETY_HYSTERESIS_RATIO: float = 0.33


def _safe_coarse_of(class_id: str, cfg: Config) -> str | None:
    try:
        return cfg.coarse_of(class_id)
    except KeyError:
        return None


def threshold_to_events(
    scores: FrameScores, cfg: Config, evidence: FrameScores | None = None
) -> list[CaptionEvent]:
    """Per-class thresholds (cfg.threshold), merge same-class events separated by
    < merge_gap_sec, then drop events < min_event_sec (merge before drop, so
    fragments of one sound survive). confidence = peak score inside final span.

    Classes in postprocess.evidence_thresholds must also reach that score in
    `evidence` (the pre-prior scores), so the stricter bar judges acoustic
    evidence and environment priors don't move it."""
    pcfg = cfg.pipeline.get("events", {}).get("postprocess", {})
    min_event_sec = float(pcfg.get("min_event_sec", 0.3))
    merge_gap_sec = float(pcfg.get("merge_gap_sec", 0.5))
    evidence_thr = pcfg.get("evidence_thresholds") or {}
    ev_index = {cid: i for i, cid in enumerate(evidence.class_ids)} if evidence is not None else {}

    events: list[CaptionEvent] = []
    if scores.scores.shape[0] == 0:
        return events

    safety = cfg.safety_ids()
    for c, class_id in enumerate(scores.class_ids):
        col = scores.scores[:, c]
        thr = cfg.threshold(class_id)
        active = col >= thr
        if class_id in evidence_thr and class_id not in safety:
            ev_col = evidence.scores[:, ev_index[class_id]] if class_id in ev_index else col
            active &= ev_col >= float(evidence_thr[class_id])
        if not np.any(active):
            continue
        if class_id in safety:
            # Hysteresis: a safety run is the whole region above a lower threshold
            # around frames that reach thr, so onsets aren't clipped by the ramp-up.
            low = col >= SAFETY_HYSTERESIS_RATIO * thr
            lp = np.pad(low, (1, 1), mode="constant", constant_values=False)
            ld = np.diff(lp.astype(np.int8))
            for i, j in zip(np.where(ld == 1)[0], np.where(ld == -1)[0]):
                if np.any(active[i:j]):
                    active[i:j] = True

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

    hierarchy = cfg.hierarchy()
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

            other_indices = [idx for idx, cid in enumerate(scores.class_ids)
                             if idx != own_idx and cid not in cfg.coarse_groups
                             and cid not in hierarchy.get(ev.class_id, ())
                             and ev.class_id not in hierarchy.get(cid, ())]
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

    return _merge_same_class(relabeled)


def _merge_same_class(relabeled: list[CaptionEvent]) -> list[CaptionEvent]:
    """Merge overlapping (or touching) events that share the same class_id
    into one (min start, max end, max confidence)."""
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


def relabel_children(
    events: list[CaptionEvent], scores: FrameScores, cfg: Config
) -> list[CaptionEvent]:
    """Relabel each parent event in postprocess.relabel_children (parent -> children)
    to the child with the highest mean score over the event's span. The parent still
    decides whether and where an event fires; only its label changes. Ground truth
    names the specific child (man/woman speaking, not speech), and onset matching
    gives a parent label no credit. Relabeled events merge with overlapping events
    of the same child. A parent whose best child mean is below relabel_min_score
    keeps its own label (no evidence for any child)."""
    pcfg = cfg.pipeline.get("events", {}).get("postprocess", {})
    mapping = pcfg.get("relabel_children") or {}
    min_score = float(pcfg.get("relabel_min_score", 0.02))
    if not events or not mapping or scores.scores.shape[0] == 0:
        return events

    index = {cid: i for i, cid in enumerate(scores.class_ids)}
    out: list[CaptionEvent] = []
    changed = False
    for ev in events:
        children = [c for c in (mapping.get(ev.class_id) or []) if c in index]
        if not children:
            out.append(ev)
            continue
        in_span = (scores.times >= ev.start - 1e-6) & (scores.times <= ev.end + 1e-6)
        if np.any(in_span):
            frame_scores = scores.scores[in_span, :]
        else:
            nearest = int(np.argmin(np.abs(scores.times - (ev.start + ev.end) / 2.0)))
            frame_scores = scores.scores[nearest : nearest + 1, :]
        means = [float(np.mean(frame_scores[:, index[c]])) for c in children]
        best = int(np.argmax(means))
        if means[best] < min_score:
            out.append(ev)
            continue
        child = children[best]
        out.append(replace(ev, class_id=child, label=cfg.caption_for(child)))
        changed = True
    return _merge_same_class(out) if changed else events


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
