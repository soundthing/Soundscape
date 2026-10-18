"""Tests for fixed tuning objective, metrics, and diagnostics."""

from __future__ import annotations

import json
import random
from typing import Any

import numpy as np
import pytest

from soundthing.config import Config
from soundthing.contracts import CaptionEvent, ClipLabels, LabeledEvent
from soundthing.evaluate import score_events
from soundthing.harness.objective import (
    OBJECTIVE,
    ClipMatch,
    diagnostics,
    f_beta,
    grid_f1,
    match_clip,
)


def _make_pred(start: float, end: float, class_id: str, conf: float = 0.9) -> CaptionEvent:
    return CaptionEvent(start=start, end=end, label="caption", class_id=class_id, confidence=conf)


def _make_truth(start: float, end: float, class_id: str) -> LabeledEvent:
    return LabeledEvent(start=start, end=end, class_id=class_id)


def test_randomized_agreement_with_evaluate(cfg: Config) -> None:
    """Randomized agreement between match_clip and evaluate.score_events."""
    rng = random.Random(1337)
    fine_with_cg = [c.id for c in cfg.classes if c.coarse is not None]
    fine_without_cg = [c.id for c in cfg.classes if c.coarse is None]
    coarse_group_ids = list(cfg.coarse_groups.keys())

    all_pred_ids = fine_with_cg + fine_without_cg + coarse_group_ids
    all_truth_ids = fine_with_cg + fine_without_cg

    # Run 250 randomized trials
    for trial in range(250):
        # Mix in trials with no coarse groups to test the zero coarse TP case
        only_no_coarse = (trial % 5 == 0) and bool(fine_without_cg)
        pred_pool = fine_without_cg if only_no_coarse else all_pred_ids
        truth_pool = fine_without_cg if only_no_coarse else all_truth_ids

        n_preds = rng.randint(0, 8)
        n_truths = rng.randint(0, 8)

        preds: list[CaptionEvent] = []
        for _ in range(n_preds):
            cid = rng.choice(pred_pool)
            # Onsets on coarse grid (multiples of 0.2s) so ties and overlaps occur
            start = round(rng.uniform(0.0, 5.0) * 5) / 5
            end = start + rng.choice([0.2, 0.4, 0.6, 1.0])
            conf = round(rng.uniform(0.5, 0.99), 2)
            preds.append(_make_pred(start, end, cid, conf))

        truths: list[LabeledEvent] = []
        for _ in range(n_truths):
            cid = rng.choice(truth_pool)
            start = round(rng.uniform(0.0, 5.0) * 5) / 5
            end = start + rng.choice([0.2, 0.4, 0.6, 1.0])
            truths.append(_make_truth(start, end, cid))

        labels = ClipLabels(video_id=f"clip_{trial}", duration=10.0, labeler="test", segments=[], events=truths)
        tol = rng.choice([0.2, 0.5, 1.0])

        m = match_clip(preds, labels, cfg, tol)
        ev_fine = score_events(preds, labels, cfg, tol, coarse=False)

        assert m.tp_fine == ev_fine.tp
        assert m.tp_fine + m.tp_coarse + m.fp == len(preds)
        assert m.tp_fine + m.tp_coarse + m.fn == len(truths)

        if only_no_coarse:
            assert m.tp_coarse == 0
            assert m.fp == ev_fine.fp
            assert m.fn == ev_fine.fn


def test_tie_breaking_later_truth_wins(cfg: Config) -> None:
    """When two truths have identical onset distance, the later truth wins."""
    cid = cfg.class_ids[0]
    pred = _make_pred(1.0, 1.5, cid)
    t0 = _make_truth(1.2, 1.6, cid)
    t1 = _make_truth(0.8, 1.4, cid)

    labels = ClipLabels(video_id="clip_tie", duration=10.0, labeler="test", segments=[], events=[t0, t1])
    m = match_clip([pred], labels, cfg, tol=0.5)

    assert len(m.pairs) == 1
    assert m.pairs[0][1] is t1  # Later truth in labels.events wins on tie

    # Identical onset tie
    t_exact0 = _make_truth(1.0, 1.5, cid)
    t_exact1 = _make_truth(1.0, 1.5, cid)
    labels2 = ClipLabels(video_id="clip_tie2", duration=10.0, labeler="test", segments=[], events=[t_exact0, t_exact1])
    m2 = match_clip([pred], labels2, cfg, tol=0.5)
    assert len(m2.pairs) == 1
    assert m2.pairs[0][1] is t_exact1


def test_hand_computed_f_beta() -> None:
    """Hand-computed F_beta: arithmetic verification and pooling vs average."""
    # Hand case:
    # tp_fine = 3, tp_coarse = 1, fp = 2, fn = 1
    # credit   = tp_fine + 0.5 * tp_coarse = 3 + 0.5 * 1 = 3.5
    # fp'      = fp + 0.5 * tp_coarse      = 2 + 0.5 * 1 = 2.5
    # fn'      = fn + 0.5 * tp_coarse      = 1 + 0.5 * 1 = 1.5
    # P        = credit / (credit + fp')   = 3.5 / (3.5 + 2.5) = 3.5 / 6.0 = 7/12
    # R        = credit / (credit + fn')   = 3.5 / (3.5 + 1.5) = 3.5 / 5.0 = 7/10
    #
    # For beta = 1.0 (harmonic mean):
    # F1 = 2 * P * R / (P + R)
    #    = 2 * (7/12) * (7/10) / (7/12 + 7/10)
    #    = (49/60) / (77/60) = 49 / 77 = 7 / 11 (~0.6363636363636364)
    #
    # For beta = 0.5 (precision-weighted):
    # beta^2   = 0.25
    # (1 + beta^2) * P * R = 1.25 * (7/12) * (7/10) = 245 / 480
    # beta^2 * P + R       = 0.25 * (7/12) + (7/10) = 7/48 + 7/10 = 406 / 480
    # F0.5 = 245 / 406 = 35 / 58 (~0.603448275862069)
    clip = ClipMatch(
        video_id="v1",
        minutes=1.0,
        tp_fine=3,
        tp_coarse=1,
        fp=2,
        fn=1,
    )

    f05 = f_beta([clip], beta=0.5)
    assert f05 == pytest.approx(35.0 / 58.0)

    f1 = f_beta([clip], beta=1.0)
    assert f1 == pytest.approx(7.0 / 11.0)

    # All-zero edge cases
    assert f_beta([], beta=0.5) == 0.0
    assert f_beta([ClipMatch("v", 1.0, 0, 0, 0, 0)], beta=0.5) == 0.0
    assert f_beta([ClipMatch("v", 1.0, 0, 0, 5, 5)], beta=0.5) == 0.0

    # Micro-pooling across clips differs from macro-averaging per-clip F
    clip_dense = ClipMatch("c1", 1.0, tp_fine=10, tp_coarse=0, fp=0, fn=0)  # F = 1.0
    clip_sparse = ClipMatch("c2", 1.0, tp_fine=1, tp_coarse=0, fp=10, fn=10)  # P=1/11, R=1/11, F=1/11
    pooled_f = f_beta([clip_dense, clip_sparse], beta=0.5)
    mean_f = (f_beta([clip_dense], beta=0.5) + f_beta([clip_sparse], beta=0.5)) / 2.0
    assert pooled_f != pytest.approx(mean_f)


def test_coarse_pass_matching(cfg: Config) -> None:
    """Fine-class sibling and coarse group ID matching at coarse level."""
    fine1 = next(c for c in cfg.classes if c.coarse is not None)
    cg_id = fine1.coarse
    fine2 = next(c for c in cfg.classes if c.coarse == cg_id and c.id != fine1.id)

    # 1. Sibling prediction matches only at coarse level
    pred_sibling = _make_pred(1.0, 1.5, fine2.id)
    truth1 = _make_truth(1.0, 1.5, fine1.id)
    labels = ClipLabels(video_id="c1", duration=10.0, labeler="test", segments=[], events=[truth1])
    m_sibling = match_clip([pred_sibling], labels, cfg, tol=0.5)
    assert m_sibling.tp_fine == 0
    assert m_sibling.tp_coarse == 1
    assert m_sibling.fp == 0
    assert m_sibling.fn == 0

    # 2. Coarse-group-id prediction matches fine truth at coarse level
    assert cg_id is not None
    pred_cg = _make_pred(1.0, 1.5, cg_id)
    m_cg = match_clip([pred_cg], labels, cfg, tol=0.5)
    assert m_cg.tp_fine == 0
    assert m_cg.tp_coarse == 1
    assert m_cg.fp == 0
    assert m_cg.fn == 0

    # 3. Outside tolerance gives fp and fn
    pred_far = _make_pred(1.0, 1.5, fine2.id)
    truth_far = _make_truth(3.0, 3.5, fine1.id)
    labels_far = ClipLabels(video_id="c2", duration=10.0, labeler="test", segments=[], events=[truth_far])
    m_far = match_clip([pred_far], labels_far, cfg, tol=0.5)
    assert m_far.tp_fine == 0
    assert m_far.tp_coarse == 0
    assert m_far.fp == 1
    assert m_far.fn == 1


def test_safety_tp_and_safety_total(cfg: Config) -> None:
    """Safety counts credit fine and coarse matches, ignoring non-safety."""
    s_fine = next(c for c in cfg.classes if c.safety and c.coarse is not None)
    cg_id = s_fine.coarse
    s_sibling = next(c for c in cfg.classes if c.coarse == cg_id and c.id != s_fine.id)
    s_missed = next(c for c in cfg.classes if c.safety and c.id not in (s_fine.id, s_sibling.id))
    non_safety = next(c for c in cfg.classes if not c.safety)

    truth_s_fine = _make_truth(1.0, 1.5, s_fine.id)       # Matched fine
    truth_s_coarse = _make_truth(3.0, 3.5, s_fine.id)     # Matched coarse
    truth_s_missed = _make_truth(5.0, 5.5, s_missed.id)   # Missed
    truth_non_safety = _make_truth(7.0, 7.5, non_safety.id) # Non-safety fine match

    truths = [truth_s_fine, truth_s_coarse, truth_s_missed, truth_non_safety]
    preds = [
        _make_pred(1.0, 1.5, s_fine.id),    # Fine match for truth_s_fine
        _make_pred(3.0, 3.5, s_sibling.id), # Coarse match for truth_s_coarse
        _make_pred(7.0, 7.5, non_safety.id),# Fine match for non-safety
    ]

    labels = ClipLabels(video_id="c_safety", duration=10.0, labeler="test", segments=[], events=truths)
    m = match_clip(preds, labels, cfg, tol=0.5)

    assert m.safety_total == 3  # s_fine, s_fine, s_missed
    assert m.safety_tp == 2     # s_fine (fine), s_fine (coarse)
    assert m.tp_fine == 2
    assert m.tp_coarse == 1
    assert m.fp == 0
    assert m.fn == 1


def test_grid_f1_boundaries_and_missing_clip(cfg: Config) -> None:
    """grid_f1 boundary semantics and clips missing from predictions."""
    cid = cfg.class_ids[0]

    # Event ending exactly at slice boundary does not activate the next slice
    truth_evt = _make_truth(0.2, 1.0, cid)  # Overlaps [0, 1), not [1, 2)
    pred_evt = _make_pred(0.4, 1.0, cid)    # Overlaps [0, 1), not [1, 2)
    labels = ClipLabels(video_id="clip1", duration=3.0, labeler="test", segments=[], events=[truth_evt])

    score = grid_f1({"clip1": [pred_evt]}, {"clip1": labels}, slice_sec=1.0)
    assert score == 1.0

    # Overlapping boundary into second slice activates second slice
    pred_overlap = _make_pred(0.4, 1.2, cid)  # Overlaps [0, 1) and [1, 2)
    score_overlap = grid_f1({"clip1": [pred_overlap]}, {"clip1": labels}, slice_sec=1.0)
    # TP = 1 (cell 0), FP = 1 (cell 1), FN = 0 -> P = 0.5, R = 1.0 -> F1 = 2/3
    assert score_overlap == pytest.approx(2.0 / 3.0)

    # Missing clip: predictions dictionary does not include clip2
    labels2 = ClipLabels(video_id="clip2", duration=2.0, labeler="test", segments=[], events=[_make_truth(0.1, 0.9, cid)])
    score_missing = grid_f1(
        {"clip1": [pred_evt]},
        {"clip1": labels, "clip2": labels2},
        slice_sec=1.0,
    )
    # Clip 1 has 1 TP, 0 FP, 0 FN. Clip 2 has 1 FN.
    # Total TP = 1, total FP = 0, total FN = 1 -> P = 1.0, R = 0.5 -> F1 = 2/3
    assert score_missing == pytest.approx(2.0 / 3.0)

    # Zero TP returns 0.0
    assert grid_f1({}, {"clip1": labels}, slice_sec=1.0) == 0.0


def test_diagnostics_structure_and_json(cfg: Config) -> None:
    """diagnostics returns sorted per_class, onset statistics, top FPs, and serializes to JSON."""
    c_fine = next(c for c in cfg.classes if c.coarse is not None)
    cg_id = c_fine.coarse
    c_sibling = next(c for c in cfg.classes if c.coarse == cg_id and c.id != c_fine.id)

    # Construct pairs with known onset errors: -0.2, -0.1, 0.0, 0.1, 0.4
    offsets = [-0.2, -0.1, 0.0, 0.1, 0.4]
    pairs: list[tuple[CaptionEvent, LabeledEvent]] = []
    for off in offsets:
        t = _make_truth(1.0, 1.5, c_fine.id)
        p = _make_pred(1.0 + off, 1.5 + off, c_fine.id)
        pairs.append((p, t))

    # Add 25 false positives with varying confidences
    fps: list[CaptionEvent] = []
    for idx in range(25):
        fps.append(_make_pred(float(idx), float(idx) + 0.5, c_sibling.id, conf=0.5 + (idx % 10) * 0.05))

    clip = ClipMatch(
        video_id="diag_clip",
        minutes=2.0,
        tp_fine=len(pairs),
        tp_coarse=0,
        fp=len(fps),
        fn=0,
        pairs=pairs,
        false_pos=fps,
        missed=[],
    )

    diag = diagnostics([clip], cfg)

    # 1. per_class keys sorted and present
    assert "per_class" in diag
    assert list(diag["per_class"].keys()) == sorted(diag["per_class"].keys())
    assert c_fine.id in diag["per_class"]
    assert diag["per_class"][c_fine.id]["tp"] == len(pairs)
    assert c_sibling.id in diag["per_class"]
    assert diag["per_class"][c_sibling.id]["fp"] == len(fps)

    # 2. onset_error statistics
    err_stat = diag["onset_error"]
    assert err_stat["n"] == 5
    assert err_stat["mean"] == pytest.approx(float(np.mean(offsets)))
    assert err_stat["median"] == pytest.approx(0.0)
    assert err_stat["p90"] == pytest.approx(float(np.percentile(np.abs(offsets), 90)))

    # Empty pairs onset error
    empty_diag = diagnostics([ClipMatch("empty", 1.0, 0, 0, 0, 0, pairs=[])], cfg)
    assert empty_diag["onset_error"] == {"mean": None, "median": None, "p90": None, "n": 0}

    # 3. top_false_positives cap of 20 and ordering: conf desc, video_id asc, start asc
    top_fps = diag["top_false_positives"]
    assert len(top_fps) == 20
    for i in range(len(top_fps) - 1):
        a = top_fps[i]
        b = top_fps[i + 1]
        key_a = (-a["confidence"], a["video_id"], a["start"])
        key_b = (-b["confidence"], b["video_id"], b["start"])
        assert key_a <= key_b

    # 4. JSON serializability
    dumped = json.dumps(diag)
    loaded = json.loads(dumped)
    assert loaded["onset_error"]["n"] == 5
    assert len(loaded["top_false_positives"]) == 20
    assert OBJECTIVE == "f0.5_conditioned_micro"
