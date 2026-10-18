"""Tests for the harness pipeline component (Track C)."""

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
from soundthing.harness.components import pipeline as pipeline_mod
from soundthing.harness.components.base import Clip, Component
from soundthing.harness.components.pipeline import (
    COMPONENT,
    PipelineClipCache,
    PipelineComponent,
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


def test_pipeline_component_protocol(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """COMPONENT implements the Component protocol with correct metadata."""
    assert isinstance(COMPONENT, Component)
    assert COMPONENT.name == "pipeline"
    assert COMPONENT.sections == ("events.", "thresholds.", "environment.", "captions.")
    assert COMPONENT.code_paths == (
        "src/soundthing/events",
        "src/soundthing/environment",
        "src/soundthing/captions.py",
    )
    assert COMPONENT.tuning_log == "src/soundthing/harness/references/pipeline-tuning-log.md"
    assert COMPONENT.n_stats == 4
    assert COMPONENT.build_cache("vid", Path("audio.wav"), cfg) is None

    fake_env_mod = types.ModuleType("soundthing.harness.components.environment")
    fake_env_mod.COMPONENT = types.SimpleNamespace(cache_key=lambda c: "envkey123")  # type: ignore[attr-defined]
    monkeypatch.setattr(pipeline_mod, "env_comp", fake_env_mod)

    key = COMPONENT.cache_key(cfg)
    assert isinstance(key, str)
    assert "+" in key
    assert key.endswith("envkey123")


def test_predicted_env_passed_to_events_replay(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Predicted env (not labeled) is what events replay receives (asserted via recording fake)."""
    recorded_segments: list[list[EnvSegment]] = []

    # Fake environment component
    fake_env_mod = types.ModuleType("soundthing.harness.components.environment")
    predicted_segs = [
        EnvSegment(
            start=0.0,
            end=10.0,
            env="kitchen",
            confidence=0.85,
            probs={"kitchen": 0.85, "general": 0.15},
        )
    ]
    fake_env_mod.predict_segments = lambda cache, c: predicted_segs  # type: ignore[attr-defined]

    # Fake events component
    fake_events_mod = types.ModuleType("soundthing.harness.components.events")

    def fake_replay(cache: Any, segs: Any, c: Config, text_emb: Any) -> list[CaptionEvent]:
        if segs is not None:
            recorded_segments.append(segs)
        return [CaptionEvent(start=1.0, end=3.0, label="clatter", class_id="dishes", confidence=0.9)]

    fake_events_mod.replay_events = fake_replay  # type: ignore[attr-defined]
    fake_events_mod.text_emb_for = lambda c, caches: None  # type: ignore[attr-defined]

    monkeypatch.setattr(pipeline_mod, "env_comp", fake_env_mod)
    monkeypatch.setattr(pipeline_mod, "ev_comp", fake_events_mod)

    # Labeled segments say "street", not "kitchen"
    labeled_segs = [LabeledSegment(start=0.0, end=10.0, env="street")]
    labels = _make_labels(
        "clip_p1",
        10.0,
        [_make_truth(1.0, 3.0, "dishes")],
        segments=labeled_segs,
    )

    clip_cache = PipelineClipCache(
        events_cache=types.SimpleNamespace(duration=10.0),
        env_cache=types.SimpleNamespace(duration=10.0),
    )
    clip = Clip(video_id="clip_p1", labels=labels, cache=clip_cache)

    eval_res = COMPONENT.evaluate([clip], cfg)

    assert len(recorded_segments) == 1
    passed_env = recorded_segments[0][0].env
    assert passed_env == "kitchen", f"Expected predicted env 'kitchen', got {passed_env!r}"
    assert passed_env != "street", "Events replay must receive predicted env, NOT labeled env"
    assert eval_res.objective > 0.0


def test_missing_env_cache_raises_filenotfound(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing env cache raises FileNotFoundError naming the missing cache."""
    fake_events_mod = types.ModuleType("soundthing.harness.components.events")
    fake_events_mod.load_event_caches = lambda c, lbls: [types.SimpleNamespace(duration=5.0)]  # type: ignore[attr-defined]

    fake_env_mod = types.ModuleType("soundthing.harness.components.environment")

    def fake_load_env(vid: str, c: Config) -> Any:
        raise FileNotFoundError(f"Environment cache missing for clip {vid!r} with key 'env_abc'")

    fake_env_mod.load_env_cache = fake_load_env  # type: ignore[attr-defined]

    monkeypatch.setattr(pipeline_mod, "ev_comp", fake_events_mod)
    monkeypatch.setattr(pipeline_mod, "env_comp", fake_env_mod)

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True)
    save_labels(
        _make_labels("clip_missing_env", 5.0, [_make_truth(1.0, 2.0, "dog_bark")]),
        labels_dir / "clip_missing_env.json",
    )

    with pytest.raises(FileNotFoundError, match="Environment cache missing for clip 'clip_missing_env'"):
        COMPONENT.load_clips(cfg, labels_dir)


def test_pipeline_space_deduplicated_union(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline space is the de-duplicated union of events, environment, and captions spaces."""
    fake_events = types.SimpleNamespace(
        space=lambda c, l: [
            {"name": "events.param1", "bounds": [0.0, 1.0]},
            {"name": "shared.threshold", "bounds": [0.1, 0.9]},
        ]
    )
    fake_env = types.SimpleNamespace(
        space=lambda c, l: [
            {"name": "environment.temp", "bounds": [0.01, 1.0]},
            {"name": "shared.threshold", "bounds": [0.1, 0.9]},
        ]
    )
    fake_captions = types.SimpleNamespace(
        space=lambda c, l: [
            {"name": "captions.min_cue_sec", "bounds": [0.2, 3.0]},
            {"name": "captions.max_cue_sec", "bounds": [2.0, 12.0]},
        ]
    )

    registry = {
        "events": fake_events,
        "environment": fake_env,
        "captions": fake_captions,
    }

    monkeypatch.setattr(pipeline_mod, "get", lambda name: registry[name])

    union_space = COMPONENT.space(cfg)
    names = [item["name"] for item in union_space]

    assert names == [
        "events.param1",
        "shared.threshold",
        "environment.temp",
        "captions.min_cue_sec",
        "captions.max_cue_sec",
    ]
    assert len(names) == len(set(names)), "Space contains duplicates"


def test_pipeline_limits() -> None:
    """Pipeline limits: no best -> above_baseline only; else safety_recall, fp_per_min, above_baseline."""
    metrics_good = {
        "objective": 0.8,
        "baseline": 0.7,
        "safety_recall": 0.9,
        "fp_per_min": 1.5,
    }
    metrics_below_baseline = {
        "objective": 0.65,
        "baseline": 0.7,
        "safety_recall": 0.9,
        "fp_per_min": 1.5,
    }

    # 1. No best -> above_baseline only
    limits_no_best_pass = COMPONENT.limits(metrics_good, None)
    assert limits_no_best_pass == {"above_baseline": True}

    limits_no_best_fail = COMPONENT.limits(metrics_below_baseline, None)
    assert limits_no_best_fail == {"above_baseline": False}

    # 2. With best
    best = {
        "safety_recall": 0.85,
        "fp_per_min": 2.0,
        "objective": 0.75,
    }

    limits_best_pass = COMPONENT.limits(metrics_good, best)
    assert limits_best_pass == {
        "safety_recall": True,
        "fp_per_min": True,
        "above_baseline": True,
    }

    # Fails safety recall (0.80 < 0.85)
    metrics_low_safety = copy.deepcopy(metrics_good)
    metrics_low_safety["safety_recall"] = 0.80
    assert COMPONENT.limits(metrics_low_safety, best)["safety_recall"] is False

    # Fails fp_per_min (2.5 > 2.0)
    metrics_high_fp = copy.deepcopy(metrics_good)
    metrics_high_fp["fp_per_min"] = 2.5
    assert COMPONENT.limits(metrics_high_fp, best)["fp_per_min"] is False

    # Fails above_baseline
    assert COMPONENT.limits(metrics_below_baseline, best)["above_baseline"] is False


def test_pipeline_env_time_acc_diagnostic(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """env_time_acc correctly computes time-weighted accuracy of predicted vs labeled env."""
    fake_env_mod = types.ModuleType("soundthing.harness.components.environment")
    # Predicted: 0-8s street, 8-10s park
    pred_segs = [
        EnvSegment(0.0, 8.0, "street", 0.9, probs={"street": 0.9, "general": 0.1}),
        EnvSegment(8.0, 10.0, "park", 0.9, probs={"park": 0.9, "general": 0.1}),
    ]
    fake_env_mod.predict_segments = lambda cache, c: pred_segs  # type: ignore[attr-defined]

    fake_events_mod = types.ModuleType("soundthing.harness.components.events")
    fake_events_mod.replay_events = lambda cache, segs, c, text_emb: [
        CaptionEvent(start=1.0, end=3.0, label="bark", class_id="dog_bark", confidence=0.9)
    ]
    fake_events_mod.text_emb_for = lambda c, caches: None  # type: ignore[attr-defined]

    monkeypatch.setattr(pipeline_mod, "env_comp", fake_env_mod)
    monkeypatch.setattr(pipeline_mod, "ev_comp", fake_events_mod)

    # Labeled: 0-6s street (agree 6s), 6-10s park (agree 2s) -> total 8s / 10s = 0.8
    labeled_segs = [
        LabeledSegment(start=0.0, end=6.0, env="street"),
        LabeledSegment(start=6.0, end=10.0, env="park"),
    ]
    labels = _make_labels(
        "clip_acc",
        10.0,
        [_make_truth(1.0, 3.0, "dog_bark")],
        segments=labeled_segs,
    )
    clip_cache = PipelineClipCache(
        events_cache=types.SimpleNamespace(duration=10.0),
        env_cache=types.SimpleNamespace(duration=10.0),
    )
    clip = Clip(video_id="clip_acc", labels=labels, cache=clip_cache)

    eval_res = COMPONENT.evaluate([clip], cfg)
    assert np.isclose(eval_res.metrics["env_time_acc"], 0.8)
    assert np.isclose(eval_res.diagnostics["env_time_acc"], 0.8)


def test_pipeline_clips_without_env_labels_scored(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Clips without env labels (audio only) are still scored."""
    fake_env_mod = types.ModuleType("soundthing.harness.components.environment")
    pred_segs = [
        EnvSegment(0.0, 5.0, "street", 0.9, probs={"street": 0.9, "general": 0.1}),
    ]
    fake_env_mod.predict_segments = lambda cache, c: pred_segs  # type: ignore[attr-defined]

    fake_events_mod = types.ModuleType("soundthing.harness.components.events")
    fake_events_mod.replay_events = lambda cache, segs, c, text_emb: [
        CaptionEvent(start=1.0, end=2.5, label="bark", class_id="dog_bark", confidence=0.9)
    ]
    fake_events_mod.text_emb_for = lambda c, caches: None  # type: ignore[attr-defined]

    monkeypatch.setattr(pipeline_mod, "env_comp", fake_env_mod)
    monkeypatch.setattr(pipeline_mod, "ev_comp", fake_events_mod)

    # Empty segments list
    labels = _make_labels(
        "clip_no_env",
        5.0,
        [_make_truth(1.0, 2.5, "dog_bark")],
        segments=[],
    )
    clip_cache = PipelineClipCache(
        events_cache=types.SimpleNamespace(duration=5.0),
        env_cache=types.SimpleNamespace(duration=5.0),
    )
    clip = Clip(video_id="clip_no_env", labels=labels, cache=clip_cache)

    eval_res = COMPONENT.evaluate([clip], cfg)
    assert eval_res.objective > 0.0
    assert eval_res.metrics["env_time_acc"] == 0.0

    sum_stats = np.sum([c["stats"] for c in eval_res.per_clip.values()], axis=0)
    assert COMPONENT.stat_objective(sum_stats) == eval_res.objective
    vec_res = COMPONENT.stat_objective(np.array([sum_stats, sum_stats]))
    assert isinstance(vec_res, np.ndarray) and vec_res.shape == (2,)

