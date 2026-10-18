"""Tests for the harness environment component (Track B)."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from soundthing.config import load_config
from soundthing.contracts import (
    ClipLabels,
    EnvSegment,
    ExtractedInputs,
    LabeledSegment,
    save_labels,
)
from soundthing.environment import classifier
from soundthing.harness.components.base import Clip, Component
from soundthing.harness.components.environment import (
    COMPONENT,
    EnvCache,
    build_cache,
    build_env_caches,
    cache_key,
    cache_path,
    evaluate,
    limits,
    load_cache_file,
    load_clips,
    load_env_cache,
    predict_segments,
    save_env_cache,
    space,
    stat_objective,
)


def test_component_protocol(cfg):
    assert isinstance(COMPONENT, Component)
    assert COMPONENT.name == "environment"
    assert COMPONENT.owner == "person2"
    assert COMPONENT.sections == ("environment.",)
    assert COMPONENT.code_paths == ("src/soundthing/environment",)
    assert COMPONENT.tuning_log == "src/soundthing/environment/tuning-log.md"
    assert COMPONENT.n_stats == 2 * len(cfg.environments)


def test_classify_equals_segments_from_logits(cfg, fake_inputs, monkeypatch):
    inputs = ExtractedInputs(
        video_id="fake_01",
        video_path=str(fake_inputs.parent / "fake.mp4"),
        audio_path=str(fake_inputs.parent / "audio.wav"),
        frames_dir=str(fake_inputs.parent / "frames"),
        fps=1.0,
        duration=30.0,
        sample_rate=16000,
    )
    env_ids = [e.id for e in cfg.environments if e.id != "general"]
    n_envs = len(env_ids)
    spans = [(0.0, 10.0), (10.0, 30.0)]
    audio_logits = np.zeros((2, n_envs), dtype=np.float32)
    audio_logits[0, 0] = 2.0
    audio_logits[1, 1] = 2.0
    visual_logits = np.zeros((2, n_envs), dtype=np.float32)
    visual_logits[0, 0] = 1.0
    visual_logits[1, 1] = 1.0

    monkeypatch.setattr(classifier, "detect_spans", lambda *args: spans)
    monkeypatch.setattr(classifier, "audio_env_logits", lambda *args: audio_logits)
    monkeypatch.setattr(classifier, "visual_env_logits", lambda *args: visual_logits)

    frames = Path(inputs.frames_dir)
    frames.mkdir(parents=True, exist_ok=True)
    (frames / "frame_000000.jpg").write_bytes(b"placeholder")

    classify_res = classifier.classify(inputs, cfg)
    direct_res = classifier.segments_from_logits(
        spans, audio_logits, visual_logits, env_ids, cfg, inputs.duration
    )

    assert len(classify_res) == len(direct_res)
    for s1, s2 in zip(classify_res, direct_res):
        assert s1.start == s2.start
        assert s1.end == s2.end
        assert s1.env == s2.env
        assert s1.confidence == pytest.approx(s2.confidence)
        assert s1.probs == pytest.approx(s2.probs)

    with pytest.raises(ValueError, match="at least one"):
        classifier.segments_from_logits(spans, None, None, env_ids, cfg, 30.0)


def test_cache_save_load_round_trip(tmp_path):
    spans = [(0.0, 5.0), (5.0, 15.0)]
    env_ids = ["kitchen", "street", "car"]
    audio_logits = np.array([[1.0, 0.5, -0.2], [-0.1, 2.0, 0.3]], dtype=np.float32)
    visual_logits = np.array([[0.8, 0.1, 0.0], [0.2, 1.5, -0.4]], dtype=np.float32)

    cache = EnvCache(
        video_id="clip_rt",
        duration=15.0,
        key="deadbeef1234",
        spans=spans,
        env_ids=env_ids,
        audio_logits=audio_logits,
        visual_logits=visual_logits,
    )

    save_path = tmp_path / "cache.npz"
    save_env_cache(cache, save_path)
    loaded = load_cache_file(save_path)

    assert loaded.video_id == cache.video_id
    assert loaded.duration == cache.duration
    assert loaded.key == cache.key
    assert loaded.spans == cache.spans
    assert loaded.env_ids == cache.env_ids
    np.testing.assert_allclose(loaded.audio_logits, cache.audio_logits, rtol=1e-5)
    np.testing.assert_allclose(loaded.visual_logits, cache.visual_logits, rtol=1e-5)

    # Audio-only round-trip (visual_logits is None)
    cache_audio_only = EnvCache(
        video_id="clip_audio",
        duration=10.0,
        key="deadbeef1234",
        spans=[(0.0, 10.0)],
        env_ids=env_ids,
        audio_logits=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        visual_logits=None,
    )
    save_path_ao = tmp_path / "cache_ao.npz"
    save_env_cache(cache_audio_only, save_path_ao)
    loaded_ao = load_cache_file(save_path_ao)
    assert loaded_ao.visual_logits is None
    np.testing.assert_allclose(loaded_ao.audio_logits, cache_audio_only.audio_logits, rtol=1e-5)

    # Empty spans round-trip
    cache_empty_spans = EnvCache(
        video_id="clip_empty",
        duration=0.0,
        key="deadbeef1234",
        spans=[],
        env_ids=env_ids,
        audio_logits=np.zeros((0, len(env_ids)), dtype=np.float32),
        visual_logits=None,
    )
    save_path_empty = tmp_path / "cache_empty.npz"
    save_env_cache(cache_empty_spans, save_path_empty)
    loaded_empty = load_cache_file(save_path_empty)
    assert loaded_empty.spans == []


def test_cache_key_dependencies(cfg):
    k0 = cache_key(cfg)
    assert isinstance(k0, str)
    assert len(k0) == 12

    # Segmentation changes invalidate cache_key
    cfg_seg1 = copy.deepcopy(cfg)
    cfg_seg1.pipeline["environment"]["segmentation"]["threshold"] = 42.0
    assert cache_key(cfg_seg1) != k0

    cfg_seg2 = copy.deepcopy(cfg)
    cfg_seg2.pipeline["environment"]["segmentation"]["max_segment_sec"] = 45.0
    assert cache_key(cfg_seg2) != k0

    # Prompts invalidate cache_key
    cfg_prompt = copy.deepcopy(cfg)
    cfg_prompt.environments[1].audio_prompts.append("new audio prompt")
    assert cache_key(cfg_prompt) != k0

    cfg_vprompt = copy.deepcopy(cfg)
    cfg_vprompt.environments[1].visual_prompts.append("new visual prompt")
    assert cache_key(cfg_vprompt) != k0

    # Model changes invalidate cache_key
    cfg_model = copy.deepcopy(cfg)
    cfg_model.pipeline["environment"]["audio"]["model"] = "other/clap-model"
    assert cache_key(cfg_model) != k0

    # Fusion weights and threshold do NOT invalidate cache_key
    cfg_fusion = copy.deepcopy(cfg)
    cfg_fusion.pipeline["environment"]["fusion"]["audio_weight"] = 0.9
    cfg_fusion.pipeline["environment"]["fusion"]["visual_weight"] = 0.1
    cfg_fusion.pipeline["environment"]["fusion"]["temperature"] = 0.5
    cfg_fusion.pipeline["environment"]["fusion"]["visual_similarity_floor"] = 0.5
    cfg_fusion.pipeline["environment"]["confidence_threshold"] = 0.8
    assert cache_key(cfg_fusion) == k0


def test_predict_replay_changes_with_temp_and_threshold_without_recache(cfg):
    env_ids = [e.id for e in cfg.environments if e.id != "general"]
    logits = np.zeros((1, len(env_ids)), dtype=np.float32)
    logits[0, 0] = 0.01

    cache = EnvCache(
        video_id="clip_replay",
        duration=10.0,
        key=cache_key(cfg),
        spans=[(0.0, 10.0)],
        env_ids=env_ids,
        audio_logits=logits,
        visual_logits=None,
    )

    # Under default cfg (temp 0.01, confidence_threshold 0.45), max prob is ~0.185 < 0.45 -> general
    preds_base = predict_segments(cache, cfg)
    assert preds_base[0].env == "general"

    # Lower confidence threshold to 0.10 -> accepts the predicted env instead of general
    cfg_thresh = copy.deepcopy(cfg)
    cfg_thresh.pipeline["environment"]["confidence_threshold"] = 0.10
    assert cache_key(cfg_thresh) == cache.key
    preds_thresh = predict_segments(cache, cfg_thresh)
    assert preds_thresh[0].env == env_ids[0]

    # Lower temperature to 0.001 -> sharpens distribution so max prob > 0.45 without recache
    cfg_temp = copy.deepcopy(cfg)
    cfg_temp.pipeline["environment"]["fusion"]["temperature"] = 0.001
    assert cache_key(cfg_temp) == cache.key
    preds_temp = predict_segments(cache, cfg_temp)
    assert preds_temp[0].env == env_ids[0]


def test_objective_and_stat_objective_agree(cfg):
    all_envs = list(cfg.environments)
    env_a = all_envs[1].id
    env_b = all_envs[2].id

    # Clip 1: 10s labeled as env_a; predicted as env_a for [0, 6], env_b for [6, 10]
    # Clip 2: 20s labeled as env_b; predicted as env_b for [0, 20]
    labels1 = ClipLabels(
        video_id="c1",
        duration=10.0,
        labeler="test",
        segments=[LabeledSegment(0.0, 10.0, env_a)],
        events=[],
    )
    labels2 = ClipLabels(
        video_id="c2",
        duration=20.0,
        labeler="test",
        segments=[LabeledSegment(0.0, 20.0, env_b)],
        events=[],
    )

    env_ids = [e.id for e in cfg.environments if e.id != "general"]
    idx_a = env_ids.index(env_a)
    idx_b = env_ids.index(env_b)

    logits1 = np.zeros((2, len(env_ids)), dtype=np.float32)
    logits1[0, idx_a] = 10.0
    logits1[1, idx_b] = 10.0

    logits2 = np.zeros((1, len(env_ids)), dtype=np.float32)
    logits2[0, idx_b] = 10.0

    cache1 = EnvCache("c1", 10.0, cache_key(cfg), [(0.0, 6.0), (6.0, 10.0)], env_ids, logits1, None)
    cache2 = EnvCache("c2", 20.0, cache_key(cfg), [(0.0, 20.0)], env_ids, logits2, None)

    clips = [Clip("c1", labels1, cache1), Clip("c2", labels2, cache2)]
    eval_res = COMPONENT.evaluate(clips, cfg)

    # In c1: env_a correct = 6.0, total = 10.0
    # In c2: env_b correct = 20.0, total = 20.0
    # Total correct = 26.0, total seconds = 30.0 -> time_acc = 26 / 30 = 0.866666...
    # env_a acc = 6 / 10 = 0.6, env_b acc = 20 / 20 = 1.0 -> macro_acc = (0.6 + 1.0) / 2 = 0.8
    # objective = 0.5 * (26 / 30) + 0.5 * 0.8 = 0.8333333333333334
    expected_time_acc = 26.0 / 30.0
    expected_macro_acc = 0.8
    expected_obj = 0.5 * expected_time_acc + 0.5 * expected_macro_acc

    assert eval_res.metrics["time_acc"] == pytest.approx(expected_time_acc)
    assert eval_res.metrics["macro_acc"] == pytest.approx(expected_macro_acc)
    assert eval_res.objective == pytest.approx(expected_obj)

    summed_stats = np.sum([c["stats"] for c in eval_res.per_clip.values()], axis=0)
    stat_obj = COMPONENT.stat_objective(summed_stats)

    assert stat_obj == eval_res.objective
    assert stat_obj == pytest.approx(expected_obj)

    # Vectorized 2D test
    totals_2d = np.array([summed_stats, summed_stats], dtype=np.float64)
    stat_objs = COMPONENT.stat_objective(totals_2d)
    assert isinstance(stat_objs, np.ndarray)
    assert stat_objs.shape == (2,)
    np.testing.assert_allclose(stat_objs, [expected_obj, expected_obj])

    # Confusion matrix in diagnostics
    assert eval_res.diagnostics["confusion"][env_a][env_a] == pytest.approx(6.0)
    assert eval_res.diagnostics["confusion"][env_a][env_b] == pytest.approx(4.0)
    assert eval_res.diagnostics["confusion"][env_b][env_b] == pytest.approx(20.0)
    assert eval_res.diagnostics["per_env_accuracy"][env_a] == pytest.approx(0.6)
    assert eval_res.diagnostics["per_env_accuracy"][env_b] == pytest.approx(1.0)


def test_labeled_general_matches_predicted_general(cfg):
    # Labeled as general, predicted as general counts correct
    labels = ClipLabels(
        video_id="c_gen",
        duration=10.0,
        labeler="test",
        segments=[LabeledSegment(0.0, 10.0, "general")],
        events=[],
    )
    env_ids = [e.id for e in cfg.environments if e.id != "general"]
    # All zero logits with high confidence_threshold produces general
    logits = np.zeros((1, len(env_ids)), dtype=np.float32)
    cache = EnvCache("c_gen", 10.0, cache_key(cfg), [(0.0, 10.0)], env_ids, logits, None)

    cfg_gen = copy.deepcopy(cfg)
    cfg_gen.pipeline["environment"]["confidence_threshold"] = 0.999

    eval_res = COMPONENT.evaluate([Clip("c_gen", labels, cache)], cfg_gen)
    assert eval_res.metrics["time_acc"] == pytest.approx(1.0)
    assert eval_res.metrics["macro_acc"] == pytest.approx(1.0)
    assert eval_res.objective == pytest.approx(1.0)
    assert eval_res.metrics["general_rate"] == pytest.approx(1.0)


def test_clip_with_no_env_segments_is_skipped(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path))
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Clip with env segments
    l1 = ClipLabels(
        video_id="c_has_env",
        duration=10.0,
        labeler="test",
        segments=[LabeledSegment(0.0, 10.0, cfg.environments[1].id)],
        events=[],
    )
    save_labels(l1, labels_dir / "c1.json")

    # Clip with NO env segments
    l2 = ClipLabels(
        video_id="c_no_env",
        duration=10.0,
        labeler="test",
        segments=[],
        events=[],
    )
    save_labels(l2, labels_dir / "c2.json")

    # Cache only c_has_env
    env_ids = [e.id for e in cfg.environments if e.id != "general"]
    cache1 = EnvCache(
        video_id="c_has_env",
        duration=10.0,
        key=cache_key(cfg),
        spans=[(0.0, 10.0)],
        env_ids=env_ids,
        audio_logits=np.zeros((1, len(env_ids)), dtype=np.float32),
        visual_logits=None,
    )
    save_env_cache(cache1, cache_path("c_has_env", cache_key(cfg)))

    clips = load_clips(cfg, labels_dir)
    assert len(clips) == 1
    assert clips[0].video_id == "c_has_env"


def test_missing_cache_raises_file_not_found(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path))
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    l1 = ClipLabels(
        video_id="c_missing_cache",
        duration=10.0,
        labeler="test",
        segments=[LabeledSegment(0.0, 10.0, cfg.environments[1].id)],
        events=[],
    )
    save_labels(l1, labels_dir / "c_missing.json")

    with pytest.raises(FileNotFoundError) as exc_info:
        load_clips(cfg, labels_dir)

    err = str(exc_info.value)
    assert "c_missing_cache" in err
    assert cache_key(cfg) in err

    with pytest.raises(FileNotFoundError) as exc_info2:
        load_env_cache("c_missing_cache", cfg)
    assert "c_missing_cache" in str(exc_info2.value)
    assert cache_key(cfg) in str(exc_info2.value)


def test_limits_behaviour():
    metrics = {"time_acc": 0.85, "macro_acc": 0.80}

    # No best: all True
    assert limits(metrics, None) == {"time_acc": True, "macro_acc": True}

    # Equal to best: True
    best_exact = {"time_acc": 0.85, "macro_acc": 0.80}
    assert limits(metrics, best_exact) == {"time_acc": True, "macro_acc": True}

    # Within 1e-9 tolerance: True
    best_near = {"time_acc": 0.85 + 5e-10, "macro_acc": 0.80}
    assert limits(metrics, best_near) == {"time_acc": True, "macro_acc": True}

    # Drop in time_acc: False
    best_higher_time = {"time_acc": 0.86, "macro_acc": 0.80}
    assert limits(metrics, best_higher_time) == {"time_acc": False, "macro_acc": True}

    # Drop in macro_acc: False
    best_higher_macro = {"time_acc": 0.85, "macro_acc": 0.82}
    assert limits(metrics, best_higher_macro) == {"time_acc": True, "macro_acc": False}

    # Object with attributes
    best_obj = SimpleNamespace(time_acc=0.84, macro_acc=0.79)
    assert limits(metrics, best_obj) == {"time_acc": True, "macro_acc": True}


def test_space_structure_and_bounds(cfg):
    entries = COMPONENT.space(cfg)
    idx = {e["name"]: e for e in entries}

    expected_params = [
        "environment.fusion.audio_weight",
        "environment.fusion.visual_weight",
        "environment.fusion.temperature",
        "environment.fusion.visual_similarity_floor",
        "environment.confidence_threshold",
        "environment.segmentation.threshold",
        "environment.segmentation.min_segment_sec",
        "environment.segmentation.max_segment_sec",
    ]

    for p in expected_params:
        assert p in idx, f"Missing parameter {p}"
        entry = idx[p]
        assert entry["type"] == "float"
        assert len(entry["bounds"]) == 2
        assert entry["bounds"][0] <= entry["bounds"][1]
        assert entry["shared"] is False

    assert idx["environment.fusion.audio_weight"]["bounds"] == [0.0, 1.0]
    assert idx["environment.fusion.visual_weight"]["bounds"] == [0.0, 1.0]
    assert idx["environment.fusion.temperature"]["bounds"] == [0.001, 1.0]
    assert idx["environment.fusion.temperature"].get("scale") == "log"
    assert idx["environment.fusion.visual_similarity_floor"]["bounds"] == [-1.0, 1.0]
    assert idx["environment.confidence_threshold"]["bounds"] == [0.0, 1.0]

    for seg_p in [
        "environment.segmentation.threshold",
        "environment.segmentation.min_segment_sec",
        "environment.segmentation.max_segment_sec",
    ]:
        assert idx[seg_p].get("needs_recache") is True


def test_build_cache_and_build_env_caches(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path))

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / "clip_build.wav"
    sf.write(str(wav_path), np.zeros(16000 * 2, dtype=np.float32), 16000)

    env_ids = [e.id for e in cfg.environments if e.id != "general"]
    dummy_logits = np.ones((1, len(env_ids)), dtype=np.float32)

    monkeypatch.setattr(
        "soundthing.harness.components.environment.audio_env_logits",
        lambda *args: dummy_logits,
    )

    out_path = build_cache("clip_build", wav_path, cfg)
    assert out_path.is_file()
    assert out_path == cache_path("clip_build", cache_key(cfg))

    loaded = load_env_cache("clip_build", cfg)
    assert loaded.video_id == "clip_build"
    assert loaded.duration == pytest.approx(2.0)
    assert loaded.visual_logits is None
    np.testing.assert_allclose(loaded.audio_logits, dummy_logits)

    # Test build_env_caches
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Clip 1 with env segments
    l1 = ClipLabels("clip_build", 2.0, "test", [LabeledSegment(0.0, 2.0, cfg.environments[1].id)], [])
    save_labels(l1, labels_dir / "clip_build.json")

    # Clip 2 with no env segments
    l2 = ClipLabels("clip_no_env", 2.0, "test", [], [])
    save_labels(l2, labels_dir / "clip_no_env.json")

    count = build_env_caches(labels_dir, audio_dir, cfg)
    assert count == 1
