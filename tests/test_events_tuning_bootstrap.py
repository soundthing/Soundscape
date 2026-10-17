"""Tests for paired bootstrap acceptance test and pooled F_beta objective."""

from __future__ import annotations

import pytest

from soundthing.events.tuning.bootstrap import accepts, paired_bootstrap, pooled_f_beta


def test_pooled_f_beta_hand_computed_cases():
    """Verify pooled_f_beta hand-computed boundary and reference cases."""
    # 1. All zero -> 0.0
    assert pooled_f_beta(0.0, 0.0, 0.0, 0.0) == 0.0

    # 2. tp_fine=8, tp_coarse=0, fp=2, fn=2, beta=0.5 -> 0.8
    # P = 8/10 = 0.8, R = 8/10 = 0.8 -> F0.5 = 0.8
    assert pooled_f_beta(8.0, 0.0, 2.0, 2.0, beta=0.5) == pytest.approx(0.8)

    # 3. Coarse-only: tp_fine=0, tp_coarse=2, fp=0, fn=0 -> P=R=0.5 -> 0.5
    # credit = 1.0, fp' = 1.0, fn' = 1.0 -> P = 0.5, R = 0.5 -> F0.5 = 0.5
    assert pooled_f_beta(0.0, 2.0, 0.0, 0.0, beta=0.5) == pytest.approx(0.5)

    # 4. beta=1 case: P=0.75 (tp=6, fp=2), R=0.6 (tp=6, fn=4) -> F1 = 2*0.75*0.6 / 1.35 = 2/3
    assert pooled_f_beta(6.0, 0.0, 2.0, 4.0, beta=1.0) == pytest.approx(2.0 / 3.0)


def test_paired_bootstrap_identical_candidates():
    """Identical candidate and best yield gain == 0 and lo == hi == 0."""
    per_clip = {
        f"vid_{i}": {
            "conditioned": {
                "tp_fine": 5 + i % 3,
                "tp_coarse": i % 2,
                "fp": 2,
                "fn": 1,
            }
        }
        for i in range(15)
    }

    ci = paired_bootstrap(per_clip, per_clip, n=500, seed=42)
    assert ci["gain"] == pytest.approx(0.0)
    assert ci["lo"] == pytest.approx(0.0)
    assert ci["hi"] == pytest.approx(0.0)
    assert accepts(ci) is False


def test_paired_bootstrap_strictly_better_and_worse():
    """Candidate strictly better on all 30 clips gives lo > 0 and accepts; worse gives hi < 0."""
    cand_per_clip = {}
    best_per_clip = {}

    for i in range(30):
        vid = f"clip_{i:02d}"
        cand_per_clip[vid] = {
            "conditioned": {"tp_fine": 10, "tp_coarse": 1, "fp": 0, "fn": 1}
        }
        best_per_clip[vid] = {
            "conditioned": {"tp_fine": 10, "tp_coarse": 1, "fp": 8, "fn": 1}
        }

    better_ci = paired_bootstrap(cand_per_clip, best_per_clip, n=1000, seed=123)
    assert better_ci["gain"] > 0
    assert better_ci["lo"] > 0
    assert accepts(better_ci) is True

    worse_ci = paired_bootstrap(best_per_clip, cand_per_clip, n=1000, seed=123)
    assert worse_ci["gain"] < 0
    assert worse_ci["hi"] < 0
    assert accepts(worse_ci) is False


def test_paired_bootstrap_determinism_and_seed():
    """Same seed produces identical results; different seed produces different bounds on noisy data."""
    cand = {}
    best = {}
    for i in range(25):
        vid = f"clip_{i}"
        cand[vid] = {"conditioned": {"tp_fine": 5 + (i * 3) % 7, "tp_coarse": i % 2, "fp": i % 3, "fn": (i + 1) % 4}}
        best[vid] = {"conditioned": {"tp_fine": 5 + (i * 2) % 6, "tp_coarse": (i + 1) % 2, "fp": (i + 1) % 4, "fn": i % 3}}

    run1 = paired_bootstrap(cand, best, n=1000, seed=42)
    run2 = paired_bootstrap(cand, best, n=1000, seed=42)
    assert run1 == run2

    run3 = paired_bootstrap(cand, best, n=1000, seed=999)
    assert run1["gain"] == pytest.approx(run3["gain"])
    assert run1["lo"] != run3["lo"] or run1["hi"] != run3["hi"]


def test_paired_bootstrap_clip_intersection():
    """Only intersection of clips is used; disjoint sets raise ValueError."""
    cand = {
        "clip1": {"conditioned": {"tp_fine": 5, "tp_coarse": 0, "fp": 1, "fn": 1}},
        "clip2": {"conditioned": {"tp_fine": 6, "tp_coarse": 1, "fp": 2, "fn": 1}},
        "clip3": {"conditioned": {"tp_fine": 4, "tp_coarse": 0, "fp": 0, "fn": 2}},
    }
    best = {
        "clip2": {"conditioned": {"tp_fine": 5, "tp_coarse": 1, "fp": 2, "fn": 2}},
        "clip3": {"conditioned": {"tp_fine": 3, "tp_coarse": 0, "fp": 1, "fn": 2}},
        "clip4": {"conditioned": {"tp_fine": 8, "tp_coarse": 2, "fp": 1, "fn": 0}},
    }

    ci = paired_bootstrap(cand, best, n=500, seed=0)
    assert isinstance(ci, dict)
    assert "gain" in ci and "lo" in ci and "hi" in ci

    # Disjoint sets raise ValueError
    disjoint_cand = {"clipA": {"conditioned": {"tp_fine": 1, "tp_coarse": 0, "fp": 0, "fn": 0}}}
    disjoint_best = {"clipB": {"conditioned": {"tp_fine": 1, "tp_coarse": 0, "fp": 0, "fn": 0}}}
    with pytest.raises(ValueError, match="No common clips"):
        paired_bootstrap(disjoint_cand, disjoint_best)
