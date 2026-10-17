"""Tests for events tuning search space description and overrides validation."""

from __future__ import annotations


import pytest

from soundthing import contracts as C
from soundthing.events.tuning import paths
from soundthing.events.tuning.space import (
    BOUNDS,
    MIN_CLASS_EVENTS,
    THRESHOLD_BOUNDS,
    describe_space,
    dev_event_counts,
    is_shared,
    space_index,
    validate_overrides,
)


def test_describe_space_defaults_and_repo_config(cfg):
    """describe_space(cfg, None) matches BOUNDS, yaml values, and has no per-class thresholds."""
    space = describe_space(cfg, None)
    idx = space_index(space)

    # Contains every BOUNDS key
    for key, spec in BOUNDS.items():
        assert key in idx, f"Missing BOUNDS key: {key}"
        entry = idx[key]
        assert entry["type"] == spec.type
        assert entry["bounds"] == [spec.lo, spec.hi]
        assert entry["shared"] is False

    # onset_shift_sec has default 0.0
    assert idx["events.postprocess.onset_shift_sec"]["current"] == 0.0

    # Currents equal yaml values
    events_cfg = cfg.pipeline["events"]
    assert idx["events.clap.calibration.center"]["current"] == events_cfg["clap"]["calibration"]["center"]
    assert idx["events.clap.calibration.scale"]["current"] == events_cfg["clap"]["calibration"]["scale"]
    assert idx["events.vad.suppress_under_speech"]["current"] == events_cfg["vad"]["suppress_under_speech"]
    assert idx["events.vad.threshold"]["current"] == events_cfg["vad"]["threshold"]
    assert idx["events.vad.min_speech_sec"]["current"] == events_cfg["vad"]["min_speech_sec"]
    assert idx["events.vad.pad_sec"]["current"] == events_cfg["vad"]["pad_sec"]
    assert idx["events.vad.merge_gap_sec"]["current"] == events_cfg["vad"]["merge_gap_sec"]
    assert idx["events.postprocess.smoothing_sec"]["current"] == events_cfg["postprocess"]["smoothing_sec"]
    assert idx["events.postprocess.min_event_sec"]["current"] == events_cfg["postprocess"]["min_event_sec"]
    assert idx["events.postprocess.merge_gap_sec"]["current"] == events_cfg["postprocess"]["merge_gap_sec"]
    assert idx["events.postprocess.coarse_margin"]["current"] == events_cfg["postprocess"]["coarse_margin"]
    assert idx["events.postprocess.sustained_repeat_sec"]["current"] == events_cfg["postprocess"]["sustained_repeat_sec"]
    assert idx["events.postprocess.max_on_screen"]["current"] == events_cfg["postprocess"]["max_on_screen"]
    assert idx["thresholds.default"]["current"] == cfg.thresholds.default
    assert idx["thresholds.safety_default"]["current"] == cfg.thresholds.safety_default

    # No per-class thresholds when dev_event_counts is None
    for entry in space:
        assert not entry["name"].startswith("thresholds.classes.")


def test_describe_space_per_class_threshold_min_events(cfg):
    """Only classes with >= MIN_CLASS_EVENTS get a per-class threshold entry."""
    cid0 = cfg.class_ids[0]
    cid1 = cfg.class_ids[1]
    counts = {cid0: MIN_CLASS_EVENTS, cid1: MIN_CLASS_EVENTS - 1}

    space = describe_space(cfg, counts)
    idx = space_index(space)

    assert f"thresholds.classes.{cid0}" in idx
    entry0 = idx[f"thresholds.classes.{cid0}"]
    assert entry0["type"] == "float"
    assert entry0["bounds"] == [THRESHOLD_BOUNDS[0], THRESHOLD_BOUNDS[1]]
    assert entry0["current"] == cfg.threshold(cid0)
    assert entry0["shared"] is False

    assert f"thresholds.classes.{cid1}" not in idx

    for other_cid in cfg.class_ids[2:]:
        assert f"thresholds.classes.{other_cid}" not in idx


def test_describe_space_shared_entries(cfg):
    """Shared entries exist with shared=True for vocabulary and priors; non-shared for events.*."""
    space = describe_space(cfg, None)
    idx = space_index(space)

    for cid in cfg.class_ids:
        clap_key = f"vocabulary.{cid}.clap"
        audioset_key = f"vocabulary.{cid}.audioset"
        assert clap_key in idx
        assert idx[clap_key]["shared"] is True
        assert idx[clap_key]["type"] == "list[str]"
        assert idx[clap_key]["bounds"] is None
        assert idx[clap_key]["current"] == cfg.sound_class(cid).clap

        assert audioset_key in idx
        assert idx[audioset_key]["shared"] is True
        assert idx[audioset_key]["type"] == "list[str]"
        assert idx[audioset_key]["bounds"] is None
        assert idx[audioset_key]["current"] == cfg.sound_class(cid).audioset

    assert "priors.floor" in idx
    assert idx["priors.floor"]["shared"] is True
    assert idx["priors.floor"]["bounds"] == [0.0, 1.0]

    for env in cfg.priors.environments:
        env_def_key = f"priors.{env}.default"
        assert env_def_key in idx
        assert idx[env_def_key]["shared"] is True

    # events.* must be shared=False
    for name, entry in idx.items():
        if name.startswith("events."):
            assert entry["shared"] is False


def test_validate_overrides(cfg):
    """validate_overrides accepts valid values and rejects malformed/out-of-bounds inputs."""
    cid0 = cfg.class_ids[0]
    cid1 = cfg.class_ids[1]
    counts = {cid0: MIN_CLASS_EVENTS}
    space = describe_space(cfg, counts)

    # Valid in-bounds values
    validate_overrides({"events.vad.threshold": 0.5}, space)
    validate_overrides({"events.postprocess.max_on_screen": 3}, space)
    validate_overrides({f"thresholds.classes.{cid0}": 0.7}, space)
    validate_overrides({f"vocabulary.{cid0}.clap": ["a prompt", "another prompt"]}, space)

    # Unknown key
    with pytest.raises(ValueError, match="Unknown parameter 'events.unknown_param'"):
        validate_overrides({"events.unknown_param": 0.5}, space)

    # Out of bounds float
    with pytest.raises(ValueError, match="outside bounds"):
        validate_overrides({"events.vad.threshold": 1.5}, space)

    # Out of bounds int
    with pytest.raises(ValueError, match="outside bounds"):
        validate_overrides({"events.postprocess.max_on_screen": 10}, space)

    # Non-int for max_on_screen
    with pytest.raises(ValueError, match="non-integral"):
        validate_overrides({"events.postprocess.max_on_screen": 2.5}, space)

    # Bool rejected
    with pytest.raises(ValueError, match="boolean"):
        validate_overrides({"events.vad.threshold": True}, space)

    with pytest.raises(ValueError, match="boolean"):
        validate_overrides({"events.postprocess.max_on_screen": False}, space)

    # Per-class threshold below MIN_CLASS_EVENTS: error message mentions MIN_CLASS_EVENTS
    with pytest.raises(ValueError, match="MIN_CLASS_EVENTS"):
        validate_overrides({f"thresholds.classes.{cid1}": 0.6}, space)

    # Non-list or non-string list for list[str]
    with pytest.raises(ValueError, match="list\\[str\\]"):
        validate_overrides({f"vocabulary.{cid0}.clap": "not a list"}, space)

    with pytest.raises(ValueError, match="list\\[str\\]"):
        validate_overrides({f"vocabulary.{cid0}.clap": [123]}, space)


def test_is_shared_and_dev_event_counts(cfg, tmp_path, monkeypatch):
    """Test is_shared helper and dev_event_counts counting logic."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "tuning"))

    cid0 = cfg.class_ids[0]
    cid1 = cfg.class_ids[1]
    env0 = cfg.env_ids[0]

    assert is_shared(f"vocabulary.{cid0}.clap") is True
    assert is_shared("priors.floor") is True
    assert is_shared(f"priors.{env0}.default") is True
    assert is_shared("events.vad.threshold") is False
    assert is_shared(f"thresholds.classes.{cid0}") is False

    # Missing directory returns {}
    missing_dir = tmp_path / "tuning" / "nonexistent"
    assert dev_event_counts(missing_dir) == {}
    assert dev_event_counts() == {}

    # Directory with label files
    labels_dir = tmp_path / "tuning" / "dev" / "labels"
    labels_dir.mkdir(parents=True)

    clip1 = C.ClipLabels(
        video_id="clip1",
        duration=10.0,
        labeler="tester",
        segments=[],
        events=[
            C.LabeledEvent(start=1.0, end=2.0, class_id=cid0),
            C.LabeledEvent(start=3.0, end=4.0, class_id=cid0),
            C.LabeledEvent(start=5.0, end=6.0, class_id=cid1),
        ],
    )
    clip2 = C.ClipLabels(
        video_id="clip2",
        duration=10.0,
        labeler="tester",
        segments=[],
        events=[
            C.LabeledEvent(start=2.0, end=3.0, class_id=cid0),
        ],
    )
    C.save_labels(clip1, labels_dir / "clip1.json")
    C.save_labels(clip2, labels_dir / "clip2.json")

    counts = dev_event_counts(labels_dir)
    assert counts == {cid0: 3, cid1: 1}

    # Default labels dir via paths.dev_labels_dir()
    default_counts = dev_event_counts()
    assert default_counts == {cid0: 3, cid1: 1}
