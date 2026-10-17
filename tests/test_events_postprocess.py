import time
import numpy as np
import pytest

from soundthing import contracts as C
from soundthing.events import postprocess as pp
from soundthing.events.types import FrameScores


def _find_coarse_pair(cfg):
    """Find two classes that share a coarse group, plus a third class with a different/no group."""
    from collections import defaultdict

    by_group = defaultdict(list)
    for c in cfg.classes:
        if c.coarse:
            by_group[c.coarse].append(c.id)

    target_group = None
    c1, c2 = None, None
    for group, cids in by_group.items():
        if len(cids) >= 2:
            target_group = group
            c1, c2 = cids[0], cids[1]
            break
    assert c1 is not None and c2 is not None

    diff_class = None
    for c in cfg.classes:
        if c.coarse != target_group and not c.safety:
            diff_class = c.id
            break
    assert diff_class is not None

    return target_group, c1, c2, diff_class


def test_smooth_removes_spike_keeps_plateau_no_mutation(cfg):
    """Median smoothing removes a 1-frame spike, keeps a 3+-frame plateau, preserves shape and does not mutate input."""
    T, C_dim = 15, len(cfg.class_ids)
    hop = 0.25
    times = np.arange(T) * hop + 0.125
    scores = np.zeros((T, C_dim), dtype=np.float32)

    # 1-frame spike at index 2 (t=0.625)
    scores[2, 0] = 0.9
    # 3-frame plateau at indices 6, 7, 8
    scores[6:9, 0] = 0.85

    fs = FrameScores(times=times, class_ids=cfg.class_ids, scores=scores, hop=hop)
    scores_copy = scores.copy()

    # width_sec = 0.5 -> k = round(0.5 / 0.25) = 2 -> forced odd: 3
    smoothed = pp.smooth(fs, width_sec=0.5)

    # Must not mutate input
    assert np.array_equal(fs.scores, scores_copy)

    # Same shape
    assert smoothed.scores.shape == fs.scores.shape

    # 1-frame spike removed
    assert smoothed.scores[2, 0] == 0.0

    # 3-frame plateau kept
    assert np.allclose(smoothed.scores[6:9, 0], 0.85)

    # Performance check on large array (T=2400, C=61)
    big_scores = np.random.rand(2400, C_dim).astype(np.float32)
    big_times = np.arange(2400) * hop + 0.125
    big_fs = FrameScores(times=big_times, class_ids=cfg.class_ids, scores=big_scores, hop=hop)

    t0 = time.perf_counter()
    big_smoothed = pp.smooth(big_fs, width_sec=0.5)
    elapsed = time.perf_counter() - t0

    assert big_smoothed.scores.shape == (2400, C_dim)
    assert elapsed < 0.5, f"smooth took {elapsed:.3f}s, expected << 1s"


def test_smooth_width_zero_or_negative_is_identity(cfg):
    """Smooth with width <= 0 or k <= 1 returns input unchanged."""
    times = np.array([0.1, 0.35, 0.6])
    scores = np.array([[0.1], [0.8], [0.2]])
    fs = FrameScores(times=times, class_ids=[cfg.class_ids[0]], scores=scores, hop=0.25)

    out0 = pp.smooth(fs, width_sec=0.0)
    assert np.array_equal(out0.scores, fs.scores)

    out_neg = pp.smooth(fs, width_sec=-0.5)
    assert np.array_equal(out_neg.scores, fs.scores)


def test_threshold_uses_per_class_thresholds(cfg):
    """A score between default and a class's higher threshold does not fire."""
    # Find a class with threshold > default
    high_thr_id = None
    for cid, thr in cfg.thresholds.classes.items():
        if thr > cfg.thresholds.default:
            high_thr_id = cid
            break
    assert high_thr_id is not None

    # Find a non-safety class with default threshold
    default_id = None
    for cid in cfg.class_ids:
        if cid not in cfg.thresholds.classes and cid not in cfg.safety_ids():
            default_id = cid
            break
    assert default_id is not None

    high_thr = cfg.threshold(high_thr_id)
    def_thr = cfg.threshold(default_id)
    assert high_thr > def_thr

    mid_score = (high_thr + def_thr) / 2.0

    times = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
    cids = [high_thr_id, default_id]
    scores = np.zeros((len(times), 2), dtype=np.float32)
    scores[:, 0] = mid_score  # high_thr_id: below its threshold
    scores[:, 1] = mid_score  # default_id: above its threshold

    fs = FrameScores(times=times, class_ids=cids, scores=scores, hop=0.25)
    events = pp.threshold_to_events(fs, cfg)

    # default_id should produce an event; high_thr_id should not
    event_cids = [e.class_id for e in events]
    assert default_id in event_cids
    assert high_thr_id not in event_cids

    for e in events:
        e.validate()


def test_start_end_hop_math(cfg):
    """Run from frame i to j becomes start = max(0, times[i] - hop/2), end = times[j] + hop/2."""
    cid = cfg.class_ids[0]
    hop = 0.2
    # times[0] = 0.05 -> times[0] - hop/2 = 0.05 - 0.1 = -0.05 -> clamped to 0
    # Run over frames 0, 1: times[1] = 0.25 -> times[1] + hop/2 = 0.25 + 0.1 = 0.35
    times = np.array([0.05, 0.25, 0.45, 0.65])
    scores = np.zeros((4, 1), dtype=np.float32)
    scores[0:2, 0] = 0.9  # frames 0 and 1 active

    # Configure min_event_sec = 0.1 so 0.35s is kept
    cfg.pipeline["events"]["postprocess"]["min_event_sec"] = 0.1

    fs = FrameScores(times=times, class_ids=[cid], scores=scores, hop=hop)
    events = pp.threshold_to_events(fs, cfg)

    assert len(events) == 1
    ev = events[0]
    assert ev.start == 0.0  # max(0, -0.05)
    assert ev.end == 0.35   # 0.25 + 0.1
    ev.validate()


def test_merge_before_drop(cfg):
    """Two short fragments with small gap merge into one kept event; isolated blip drops."""
    cid = cfg.class_ids[0]
    hop = 0.25
    times = np.arange(20) * hop + 0.125
    scores = np.zeros((20, 1), dtype=np.float32)

    # Frame 1 active (duration hop=0.25 < min_event_sec 0.3)
    scores[1, 0] = 0.8
    # Frame 2 is zero (gap = hop = 0.25 < merge_gap_sec 0.5)
    # Frame 3 active (duration hop=0.25 < min_event_sec 0.3)
    scores[3, 0] = 0.8
    # Together after merge: span from frame 1 start (0.25) to frame 3 end (1.0) = 0.75 >= 0.3

    # Isolated short blip at frame 15: duration 0.25 < 0.3, surrounded by zeros
    scores[15, 0] = 0.8

    cfg.pipeline["events"]["postprocess"]["min_event_sec"] = 0.3
    cfg.pipeline["events"]["postprocess"]["merge_gap_sec"] = 0.5

    fs = FrameScores(times=times, class_ids=[cid], scores=scores, hop=hop)
    events = pp.threshold_to_events(fs, cfg)

    assert len(events) == 1
    ev = events[0]
    assert ev.start == round(times[1] - hop / 2, 2)
    assert ev.end == round(times[3] + hop / 2, 2)
    ev.validate()


def test_confidence_is_peak(cfg):
    """Event confidence is peak score inside the final event span, clipped to [0,1] and rounded to 3 decimals."""
    cid = cfg.class_ids[0]
    hop = 0.25
    times = np.array([0.5, 0.75, 1.0, 1.25])
    scores = np.zeros((4, 1), dtype=np.float32)
    scores[:, 0] = [0.65, 0.8764, 0.72, 0.60]

    fs = FrameScores(times=times, class_ids=[cid], scores=scores, hop=hop)
    events = pp.threshold_to_events(fs, cfg)

    assert len(events) == 1
    assert events[0].confidence == 0.876
    events[0].validate()


def test_collapse_to_coarse_small_margin(cfg):
    """Collapse to coarse when runner-up is same group and (own - runner-up) <= coarse_margin."""
    group, c1, c2, _ = _find_coarse_pair(cfg)
    hop = 0.25
    times = np.array([1.0, 1.25, 1.5])
    cids = [c1, c2]

    scores = np.zeros((3, 2), dtype=np.float32)
    scores[:, 0] = 0.80  # c1 own peak
    scores[:, 1] = 0.75  # c2 runner-up peak (diff = 0.05 <= coarse_margin 0.1)

    ev = C.CaptionEvent(start=1.0, end=1.5, label=cfg.caption_for(c1), class_id=c1, confidence=0.80)
    fs = FrameScores(times=times, class_ids=cids, scores=scores, hop=hop)

    collapsed = pp.collapse_confusable([ev], fs, cfg)

    assert len(collapsed) == 1
    assert collapsed[0].class_id == group
    assert collapsed[0].label == cfg.caption_for(group)
    assert collapsed[0].confidence == 0.80
    assert collapsed[0].start == 1.0
    assert collapsed[0].end == 1.5
    collapsed[0].validate()


def test_no_collapse_large_margin_or_different_group(cfg):
    """No collapse if margin is large or runner-up belongs to a different coarse group."""
    group, c1, c2, diff_class = _find_coarse_pair(cfg)
    hop = 0.25
    times = np.array([1.0, 1.25, 1.5])
    cids = [c1, c2, diff_class]

    # Case 1: same group, but margin large (0.80 - 0.60 = 0.20 > 0.1)
    scores1 = np.zeros((3, 3), dtype=np.float32)
    scores1[:, 0] = 0.80
    scores1[:, 1] = 0.60
    scores1[:, 2] = 0.10
    fs1 = FrameScores(times=times, class_ids=cids, scores=scores1, hop=hop)

    ev1 = C.CaptionEvent(start=1.0, end=1.5, label=cfg.caption_for(c1), class_id=c1, confidence=0.80)
    out1 = pp.collapse_confusable([ev1], fs1, cfg)
    assert out1[0].class_id == c1

    # Case 2: small margin, but runner-up belongs to a different group
    scores2 = np.zeros((3, 3), dtype=np.float32)
    scores2[:, 0] = 0.80
    scores2[:, 1] = 0.30  # same group c2 low
    scores2[:, 2] = 0.78  # diff_class runner-up (diff = 0.02, but different group)
    fs2 = FrameScores(times=times, class_ids=cids, scores=scores2, hop=hop)

    ev2 = C.CaptionEvent(start=1.0, end=1.5, label=cfg.caption_for(c1), class_id=c1, confidence=0.80)
    out2 = pp.collapse_confusable([ev2], fs2, cfg)
    assert out2[0].class_id == c1


def test_overlapping_collapsed_events_merge(cfg):
    """Overlapping or touching events that now share the same class_id merge into one."""
    group, c1, c2, _ = _find_coarse_pair(cfg)
    hop = 0.25
    times = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    cids = [c1, c2]

    scores = np.zeros((5, 2), dtype=np.float32)
    # At t in [1.0, 2.0], c1 own 0.80, c2 0.75 -> collapses to group
    scores[0:3, 0] = 0.80
    scores[0:3, 1] = 0.75
    # At t in [2.0, 3.0], c2 own 0.85, c1 0.80 -> collapses to group
    scores[2:5, 0] = 0.80
    scores[2:5, 1] = 0.85

    fs = FrameScores(times=times, class_ids=cids, scores=scores, hop=hop)

    ev1 = C.CaptionEvent(start=1.0, end=2.0, label=cfg.caption_for(c1), class_id=c1, confidence=0.80)
    ev2 = C.CaptionEvent(start=2.0, end=3.0, label=cfg.caption_for(c2), class_id=c2, confidence=0.85)

    merged = pp.collapse_confusable([ev1, ev2], fs, cfg)

    # They touch at 2.0 and both collapse to group -> must merge into one
    assert len(merged) == 1
    assert merged[0].class_id == group
    assert merged[0].start == 1.0
    assert merged[0].end == 3.0
    assert merged[0].confidence == 0.85
    merged[0].validate()


def test_sustained_repeat_suppression(cfg):
    """Sustained classes are kept only if no kept event of same class started < sustained_repeat_sec earlier."""
    # Find a sustained class
    sustained_cid = None
    for c in cfg.classes:
        if c.sustained:
            sustained_cid = c.id
            break
    assert sustained_cid is not None

    # Find a non-sustained class
    non_sustained_cid = None
    for c in cfg.classes:
        if not c.sustained:
            non_sustained_cid = c.id
            break
    assert non_sustained_cid is not None

    cfg.pipeline["events"]["postprocess"]["sustained_repeat_sec"] = 60.0
    cfg.pipeline["events"]["postprocess"]["max_on_screen"] = 10  # don't cap on screen here

    label_sust = cfg.caption_for(sustained_cid)
    label_non = cfg.caption_for(non_sustained_cid)

    events = [
        C.CaptionEvent(start=5.0, end=6.0, label=label_sust, class_id=sustained_cid, confidence=0.8),
        C.CaptionEvent(start=15.0, end=16.0, label=label_sust, class_id=sustained_cid, confidence=0.8),   # 10s later -> drop
        C.CaptionEvent(start=20.0, end=21.0, label=label_non, class_id=non_sustained_cid, confidence=0.8), # non-sustained -> keep
        C.CaptionEvent(start=25.0, end=26.0, label=label_non, class_id=non_sustained_cid, confidence=0.8), # non-sustained -> keep
        C.CaptionEvent(start=64.0, end=65.0, label=label_sust, class_id=sustained_cid, confidence=0.8),   # 59s after 5.0 -> drop
        C.CaptionEvent(start=66.0, end=67.0, label=label_sust, class_id=sustained_cid, confidence=0.8),   # 61s after 5.0 -> keep
    ]

    filtered = pp.priority_filter(events, cfg)

    sust_events = [e for e in filtered if e.class_id == sustained_cid]
    non_sust_events = [e for e in filtered if e.class_id == non_sustained_cid]

    assert len(non_sust_events) == 2
    assert len(sust_events) == 2
    assert sust_events[0].start == 5.0
    assert sust_events[1].start == 66.0

    for e in filtered:
        e.validate()


def test_on_screen_cap_safety_over_confidence(cfg):
    """On-screen cap keeps a lower-confidence safety event over a higher-confidence non-safety one."""
    safety_cid = sorted(cfg.safety_ids())[0]

    non_safety_cid = None
    for c in cfg.classes:
        if not c.safety:
            non_safety_cid = c.id
            break
    assert non_safety_cid is not None

    cfg.pipeline["events"]["postprocess"]["max_on_screen"] = 1

    # Both overlap in time: non-safety conf=0.95, safety conf=0.55
    ev_non_safety = C.CaptionEvent(start=1.0, end=3.0, label=cfg.caption_for(non_safety_cid), class_id=non_safety_cid, confidence=0.95)
    ev_safety = C.CaptionEvent(start=1.5, end=3.5, label=cfg.caption_for(safety_cid), class_id=safety_cid, confidence=0.55)

    filtered = pp.priority_filter([ev_non_safety, ev_safety], cfg)

    assert len(filtered) == 1
    assert filtered[0].class_id == safety_cid
    filtered[0].validate()


def test_on_screen_cap_respects_min_cue_sec_extension(cfg):
    """On-screen cap respects display span extended to min_cue_sec."""
    cids = [c.id for c in cfg.classes if not c.safety][:2]
    cid1, cid2 = cids[0], cids[1]

    cfg.pipeline["events"]["postprocess"]["max_on_screen"] = 1
    cfg.pipeline["captions"]["min_cue_sec"] = 1.0

    # ev1 has duration 0.2s (1.0 to 1.2), but display span extended to [1.0, 2.0]
    # Higher confidence so it gets accepted first
    ev1 = C.CaptionEvent(start=1.0, end=1.2, label=cfg.caption_for(cid1), class_id=cid1, confidence=0.9)

    # ev2 starts at 1.5, ending at 2.5. Overlaps with extended display span of ev1 [1.0, 2.0]
    ev2 = C.CaptionEvent(start=1.5, end=2.5, label=cfg.caption_for(cid2), class_id=cid2, confidence=0.8)

    filtered = pp.priority_filter([ev1, ev2], cfg)

    # ev2 must be rejected because ev1 is on screen until 2.0
    assert len(filtered) == 1
    assert filtered[0] == ev1

    # If ev3 starts at 2.0, it should be accepted because ev1 ended at 2.0
    ev3 = C.CaptionEvent(start=2.0, end=3.0, label=cfg.caption_for(cid2), class_id=cid2, confidence=0.8)
    filtered2 = pp.priority_filter([ev1, ev3], cfg)
    assert len(filtered2) == 2
    for e in filtered2:
        e.validate()
