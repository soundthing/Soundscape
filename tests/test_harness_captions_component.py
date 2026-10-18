"""Tests for the harness captions component (Track C)."""

from __future__ import annotations

import copy
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from soundthing.config import Config
from soundthing.contracts import (
    CaptionEvent,
    ClipLabels,
    EnvSegment,
    LabeledEvent,
    LabeledSegment,
    save_labels,
)
from soundthing.harness.components import captions as captions_mod
from soundthing.harness.components.base import Clip, Component
from soundthing.harness.components.captions import (
    COMPONENT,
    CaptionCue,
    CaptionsComponent,
    match_cues,
)


@pytest.fixture(autouse=True)
def isolate_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path / "tuning"))


def _make_truth(start: float, end: float, class_id: str) -> LabeledEvent:
    return LabeledEvent(start=start, end=end, class_id=class_id)


def _make_labels(
    video_id: str,
    duration: float,
    events: list[LabeledEvent],
    segments: list[LabeledSegment] | None = None,
) -> ClipLabels:
    return ClipLabels(
        video_id=video_id,
        duration=duration,
        events=events,
        segments=segments or [],
        labeler="test",
    )


def _with_captions_cfg(cfg: Config, min_cue: float, max_cue: float) -> Config:
    new_cfg = copy.deepcopy(cfg)
    new_cfg.pipeline.setdefault("captions", {})
    new_cfg.pipeline["captions"]["min_cue_sec"] = min_cue
    new_cfg.pipeline["captions"]["max_cue_sec"] = max_cue
    return new_cfg


def test_captions_component_protocol(cfg: Config) -> None:
    """COMPONENT implements the Component protocol with correct metadata."""
    assert isinstance(COMPONENT, Component)
    assert COMPONENT.name == "captions"
    assert COMPONENT.sections == ("captions.",)
    assert COMPONENT.code_paths == ("src/soundthing/captions.py",)
    assert COMPONENT.tuning_log == "src/soundthing/harness/references/captions-tuning-log.md"
    assert COMPONENT.n_stats == 4
    assert COMPONENT.build_cache("vid", Path("audio.wav"), cfg) is None
    key = COMPONENT.cache_key(cfg)
    assert isinstance(key, str) and len(key) > 0


def test_cue_matching_fine(cfg: Config) -> None:
    """Exact class matches with onset and offset within tolerance yield fine TP."""
    labels = _make_labels(
        "clip1",
        10.0,
        [_make_truth(start=2.0, end=4.0, class_id="dog_bark")],
    )
    # Cue: start 2.1 (diff 0.1 <= 0.5 tol), end 4.2 (diff 0.2 <= max(0.5, 0.5*2)=1.0)
    cues = [CaptionCue(start=2.1, end=4.2, class_id="dog_bark")]
    match = match_cues(cues, labels, cfg, tol=0.5)

    assert match.tp_fine == 1
    assert match.tp_coarse == 0
    assert match.fp == 0
    assert match.fn == 0
    assert len(match.pairs) == 1
    assert match.pairs[0][1].class_id == "dog_bark"


def test_cue_matching_coarse_half_credit(cfg: Config) -> None:
    """Different class in same coarse group yields coarse TP."""
    cg_name = next(iter(cfg.coarse_groups.keys()))
    members = [c.id for c in cfg.classes if cfg.coarse_of(c.id) == cg_name]
    assert len(members) >= 2, "Need at least 2 members in coarse group for test"
    class_a, class_b = members[0], members[1]

    labels = _make_labels(
        "clip1",
        10.0,
        [_make_truth(start=1.0, end=3.0, class_id=class_a)],
    )
    cues = [CaptionCue(start=1.1, end=3.1, class_id=class_b)]
    match = match_cues(cues, labels, cfg, tol=0.5)

    assert match.tp_fine == 0
    assert match.tp_coarse == 1
    assert match.fp == 0
    assert match.fn == 0


def test_cue_matching_offset_too_far(cfg: Config) -> None:
    """Cue with onset within tol but offset too far produces FP and FN."""
    # Truth duration = 1.0s, offset tolerance = max(0.5, 0.5*1.0) = 0.5s
    labels = _make_labels(
        "clip1",
        10.0,
        [_make_truth(start=2.0, end=3.0, class_id="dog_bark")],
    )
    # Cue end is 4.0s (offset diff = 1.0s > 0.5s tol)
    cues = [CaptionCue(start=2.0, end=4.0, class_id="dog_bark")]
    match = match_cues(cues, labels, cfg, tol=0.5)

    assert match.tp_fine == 0
    assert match.tp_coarse == 0
    assert match.fp == 1
    assert match.fn == 1


def test_cue_matching_onset_tolerance(cfg: Config) -> None:
    """Onsets within tolerance match; onsets exceeding tolerance do not."""
    labels = _make_labels(
        "clip1",
        10.0,
        [_make_truth(start=2.0, end=4.0, class_id="dog_bark")],
    )
    # Within onset tol 0.5 (diff = 0.4)
    cues_within = [CaptionCue(start=2.4, end=4.2, class_id="dog_bark")]
    m1 = match_cues(cues_within, labels, cfg, tol=0.5)
    assert m1.tp_fine == 1

    # Outside onset tol 0.5 (diff = 0.6)
    cues_outside = [CaptionCue(start=2.6, end=4.2, class_id="dog_bark")]
    m2 = match_cues(cues_outside, labels, cfg, tol=0.5)
    assert m2.tp_fine == 0
    assert m2.fp == 1
    assert m2.fn == 1


def test_stat_objective_consistency(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """stat_objective(sum stats) exactly reproduces evaluation.objective."""
    fake_events_mod = types.ModuleType("soundthing.harness.components.events")

    def fake_replay(cache: Any, segs: Any, c: Config, text_emb: Any) -> list[CaptionEvent]:
        return [
            CaptionEvent(start=1.0, end=3.0, label="bark", class_id="dog_bark", confidence=0.9),
            CaptionEvent(start=5.0, end=6.0, label="car", class_id="car_horn", confidence=0.8),
        ]

    def fake_labeled_env(labels: ClipLabels, c: Config, confidence: float = 0.7) -> list[EnvSegment]:
        return [
            EnvSegment(
                start=0.0,
                end=labels.duration,
                env="street",
                confidence=confidence,
                probs={"street": confidence, "general": 1.0 - confidence},
            )
        ]

    fake_events_mod.replay_events = fake_replay  # type: ignore[attr-defined]
    fake_events_mod.text_emb_for = lambda c, caches: None  # type: ignore[attr-defined]
    fake_events_mod.labeled_env_segments = fake_labeled_env  # type: ignore[attr-defined]

    monkeypatch.setattr(captions_mod, "ev_comp", fake_events_mod)

    labels = _make_labels(
        "clip_test",
        10.0,
        [
            _make_truth(start=1.0, end=3.0, class_id="dog_bark"),
            _make_truth(start=5.0, end=6.0, class_id="car_horn"),
        ],
    )
    fake_cache = types.SimpleNamespace(duration=10.0)
    clip = Clip(video_id="clip_test", labels=labels, cache=fake_cache)

    eval_res = COMPONENT.evaluate([clip], cfg)
    sum_stats = np.sum([clip_dict["stats"] for clip_dict in eval_res.per_clip.values()], axis=0)
    reproduced_obj = COMPONENT.stat_objective(sum_stats)

    assert reproduced_obj == eval_res.objective
    assert eval_res.objective > 0.0

    # Also test 2D array input (B, 4)
    totals_2d = np.array([sum_stats, sum_stats])
    vec_res = COMPONENT.stat_objective(totals_2d)
    assert isinstance(vec_res, np.ndarray)
    assert vec_res.shape == (2,)
    assert np.allclose(vec_res, eval_res.objective)


def test_raising_max_cue_sec_changes_score_for_long_events(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raising max_cue_sec allows long events to match, changing the objective score."""
    fake_events_mod = types.ModuleType("soundthing.harness.components.events")

    # Replay produces a 12s long event
    def fake_replay(cache: Any, segs: Any, c: Config, text_emb: Any) -> list[CaptionEvent]:
        return [CaptionEvent(start=0.0, end=12.0, label="alarm", class_id="siren", confidence=0.9)]

    def fake_labeled_env(labels: ClipLabels, c: Config, confidence: float = 0.7) -> list[EnvSegment]:
        return [
            EnvSegment(
                start=0.0,
                end=labels.duration,
                env="street",
                confidence=confidence,
                probs={"street": confidence, "general": 1.0 - confidence},
            )
        ]

    fake_events_mod.replay_events = fake_replay  # type: ignore[attr-defined]
    fake_events_mod.text_emb_for = lambda c, caches: None  # type: ignore[attr-defined]
    fake_events_mod.labeled_env_segments = fake_labeled_env  # type: ignore[attr-defined]

    monkeypatch.setattr(captions_mod, "ev_comp", fake_events_mod)

    labels = _make_labels(
        "clip_long",
        15.0,
        [_make_truth(start=0.0, end=12.0, class_id="siren")],
    )
    fake_cache = types.SimpleNamespace(duration=15.0)
    clip = Clip(video_id="clip_long", labels=labels, cache=fake_cache)

    # 1. Short max_cue_sec = 2.0 -> clamped cue is [0.0, 2.0].
    # Offset diff = 12.0 - 2.0 = 10.0s > max(0.5, 0.5*12) = 6.0s -> does not match!
    cfg_short = _with_captions_cfg(cfg, min_cue=1.0, max_cue=2.0)
    eval_short = COMPONENT.evaluate([clip], cfg_short)
    assert eval_short.objective == 0.0

    # 2. Long max_cue_sec = 12.0 -> cue is [0.0, 12.0].
    # Offset diff = 0.0s <= 6.0s -> matches!
    cfg_long = _with_captions_cfg(cfg, min_cue=1.0, max_cue=12.0)
    eval_long = COMPONENT.evaluate([clip], cfg_long)
    assert eval_long.objective > 0.0
    assert eval_long.objective != eval_short.objective


def test_min_cue_greater_than_max_raises(cfg: Config) -> None:
    """Candidate with min_cue_sec > max_cue_sec is invalid and raises ValueError in evaluate."""
    invalid_cfg = _with_captions_cfg(cfg, min_cue=5.0, max_cue=2.0)
    labels = _make_labels("clip1", 5.0, [_make_truth(1.0, 2.0, "dog_bark")])
    clip = Clip(video_id="clip1", labels=labels, cache=types.SimpleNamespace(duration=5.0))

    with pytest.raises(ValueError, match="captions.min_cue_sec"):
        COMPONENT.evaluate([clip], invalid_cfg)


def test_captions_limits() -> None:
    """Captions limits checks safety_recall vs best ledger entry."""
    metrics_pass = {"safety_recall": 0.85, "objective": 0.7}
    metrics_fail = {"safety_recall": 0.65, "objective": 0.7}

    # No best -> True
    limits_no_best = COMPONENT.limits(metrics_fail, None)
    assert limits_no_best == {"safety_recall": True}

    # Best with safety_recall = 0.8
    best = {"safety_recall": 0.8, "objective": 0.6}
    assert COMPONENT.limits(metrics_pass, best) == {"safety_recall": True}
    assert COMPONENT.limits(metrics_fail, best) == {"safety_recall": False}


def test_captions_space(cfg: Config) -> None:
    """Captions space returns min_cue_sec and max_cue_sec specifications."""
    space_entries = COMPONENT.space(cfg)
    assert len(space_entries) == 2
    names = {entry["name"]: entry for entry in space_entries}
    assert "captions.min_cue_sec" in names
    assert names["captions.min_cue_sec"]["bounds"] == [0.2, 3.0]
    assert "captions.max_cue_sec" in names
    assert names["captions.max_cue_sec"]["bounds"] == [2.0, 12.0]


def test_captions_load_clips_missing_cache_raises(cfg: Config, tmp_path: Path) -> None:
    """Missing cache raises FileNotFoundError naming the clip and cache key."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True)
    save_labels(
        _make_labels("missing_clip", 5.0, [_make_truth(1.0, 2.0, "dog_bark")]),
        labels_dir / "missing_clip.json",
    )
    with pytest.raises(FileNotFoundError, match="missing_clip"):
        COMPONENT.load_clips(cfg, labels_dir)


def test_captions_diagnostics_histogram(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Diagnostics include 5-bin offset error histogram."""
    fake_events_mod = types.ModuleType("soundthing.harness.components.events")

    def fake_replay(cache: Any, segs: Any, c: Config, text_emb: Any) -> list[CaptionEvent]:
        return [
            CaptionEvent(start=1.0, end=3.2, label="bark", class_id="dog_bark", confidence=0.9),
            CaptionEvent(start=5.0, end=7.5, label="horn", class_id="car_horn", confidence=0.8),
        ]

    def fake_labeled_env(labels: ClipLabels, c: Config, confidence: float = 0.7) -> list[EnvSegment]:
        return [
            EnvSegment(
                start=0.0,
                end=labels.duration,
                env="street",
                confidence=confidence,
                probs={"street": confidence, "general": 1.0 - confidence},
            )
        ]

    fake_events_mod.replay_events = fake_replay  # type: ignore[attr-defined]
    fake_events_mod.text_emb_for = lambda c, caches: None  # type: ignore[attr-defined]
    fake_events_mod.labeled_env_segments = fake_labeled_env  # type: ignore[attr-defined]

    monkeypatch.setattr(captions_mod, "ev_comp", fake_events_mod)

    labels = _make_labels(
        "clip_diag",
        10.0,
        [
            _make_truth(start=1.0, end=3.0, class_id="dog_bark"),
            _make_truth(start=5.0, end=7.0, class_id="car_horn"),
        ],
    )
    clip = Clip(video_id="clip_diag", labels=labels, cache=types.SimpleNamespace(duration=10.0))
    res = COMPONENT.evaluate([clip], cfg)

    assert "offset_error_histogram" in res.diagnostics
    hist = res.diagnostics["offset_error_histogram"]
    assert len(hist["counts"]) == 5
    assert len(hist["bin_edges"]) == 6
    assert sum(hist["counts"]) == 2
