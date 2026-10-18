from __future__ import annotations

import json
from pathlib import Path
import shutil
import numpy as np
import pytest

from soundthing.config import Config
from soundthing.contracts import (
    GENERAL_ENV,
    ClipLabels,
    EnvSegment,
    LabeledEvent,
    LabeledSegment,
    save_labels,
    validate_segments,
)
from soundthing.events.core import EventModels
from soundthing.events.tagger import build_tagger_map
from soundthing.events.cache import (
    build_cache,
    cache_key,
    cache_path,
    save_cache,
)
from soundthing.harness import paths
from soundthing.events.replay import replay
from soundthing.harness.overrides import apply_overrides
from soundthing.harness.score import labeled_env_segments, score
from soundthing.events import vad


class FakeTagger:
    def __init__(
        self,
        names: list[str],
        reg_label: str | None = None,
        safety_label: str | None = None,
    ) -> None:
        self.names = names
        self.reg_label = reg_label
        self.safety_label = safety_label

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        b = audio.shape[0] if audio.ndim > 1 else 1
        probs = np.full((b, len(self.names)), 0.05, dtype=np.float32)
        peak = float(np.max(np.abs(audio)))
        if 0.3 <= peak < 0.7 and self.reg_label and self.reg_label in self.names:
            probs[:, self.names.index(self.reg_label)] = 0.95
        elif peak >= 0.7 and self.safety_label and self.safety_label in self.names:
            probs[:, self.names.index(self.safety_label)] = 0.95
        return probs


@pytest.fixture(autouse=True)
def set_tuning_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path / "tuning"))


def _setup_clip_and_cache(
    cfg: Config,
    labels_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    video_id: str = "clip_synth",
    duration: float = 5.0,
    include_safety: bool = True,
) -> tuple[ClipLabels, str, str]:
    labels_dir.mkdir(parents=True, exist_ok=True)
    env_id = next(e for e in cfg.env_ids if e != GENERAL_ENV)

    # Pick a regular class with audioset labels and high prior in env_id
    sc = next(
        c
        for c in cfg.classes
        if c.audioset
        and not c.safety
        and cfg.priors.environments[env_id].weight(c.id) >= 0.7
    )
    # Pick a safety class with audioset labels
    safety_sc = next(c for c in cfg.classes if c.audioset and c.safety)

    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    tagger_map = build_tagger_map(cfg, audioset_names)

    reg_label = sc.audioset[0]
    safety_label = safety_sc.audioset[0] if include_safety else None

    tagger = FakeTagger(
        audioset_names,
        reg_label=reg_label,
        safety_label=safety_label,
    )
    models = EventModels(
        tagger=tagger,
        class_ids=list(cfg.class_ids),
        tagger_map=tagger_map,
        audioset_names=audioset_names,
    )

    monkeypatch.setattr(
        vad, "speech_probs", lambda a, s, c: np.zeros(10, dtype=np.float32)
    )

    sr = 16000
    audio = np.zeros(int(duration * sr), dtype=np.float32)
    # Pulse 1: regular class
    audio[int(0.5 * sr) : int(1.5 * sr)] = 0.5
    # Pulse 2: safety class (separated in time to avoid priority_filter collapse)
    if include_safety:
        audio[int(3.0 * sr) : int(4.0 * sr)] = 0.9

    c = build_cache(audio, sr, video_id, cfg, models=models)
    key = cache_key(cfg)
    save_cache(c, cache_path(video_id, key))

    # Replay under baseline to obtain the exact predicted event spans for ground truth
    synth_events = replay(c, None, cfg)
    labeled_events = [
        LabeledEvent(start=e.start, end=e.end, class_id=e.class_id)
        for e in synth_events
    ]

    lbl = ClipLabels(
        video_id=video_id,
        duration=duration,
        labeler="test",
        segments=[LabeledSegment(start=0.0, end=duration, env=env_id)],
        events=labeled_events,
    )
    save_labels(lbl, labels_dir / f"{video_id}.json")
    return lbl, sc.id, safety_sc.id


# ===========================================================================
# labeled_env_segments tests
# ===========================================================================


def test_labeled_env_segments_gap_filling_and_probs(cfg: Config):
    env_id = next(e for e in cfg.env_ids if e != GENERAL_ENV)
    duration = 10.0

    # Labeled segments leave gaps: [0, 2], [5, 7], and [9, 10]
    segments = [
        LabeledSegment(start=2.0, end=5.0, env=env_id),
        LabeledSegment(start=7.0, end=9.0, env=env_id),
    ]
    labels = ClipLabels(
        video_id="clip_gaps",
        duration=duration,
        labeler="test",
        segments=segments,
        events=[],
    )

    result = labeled_env_segments(labels, cfg, confidence=0.7)
    validate_segments(result, duration)

    assert len(result) == 5

    # Gap 1: [0.0, 2.0] GENERAL_ENV
    assert result[0].start == 0.0
    assert result[0].end == 2.0
    assert result[0].env == GENERAL_ENV
    assert result[0].confidence == 1.0
    assert result[0].probs == {GENERAL_ENV: 1.0}

    # Segment 1: [2.0, 5.0] env_id
    assert result[1].start == 2.0
    assert result[1].end == 5.0
    assert result[1].env == env_id
    assert result[1].confidence == 0.7
    assert result[1].probs == {env_id: 0.7, GENERAL_ENV: pytest.approx(0.3)}

    # Gap 2: [5.0, 7.0] GENERAL_ENV
    assert result[2].start == 5.0
    assert result[2].end == 7.0
    assert result[2].env == GENERAL_ENV
    assert result[2].confidence == 1.0
    assert result[2].probs == {GENERAL_ENV: 1.0}

    # Segment 2: [7.0, 9.0] env_id
    assert result[3].start == 7.0
    assert result[3].end == 9.0
    assert result[3].env == env_id
    assert result[3].confidence == 0.7
    assert result[3].probs == {env_id: 0.7, GENERAL_ENV: pytest.approx(0.3)}

    # Gap 3: [9.0, 10.0] GENERAL_ENV
    assert result[4].start == 9.0
    assert result[4].end == 10.0
    assert result[4].env == GENERAL_ENV
    assert result[4].confidence == 1.0
    assert result[4].probs == {GENERAL_ENV: 1.0}

    # Verify probs sum to 1 for every segment
    for seg in result:
        assert sum(seg.probs.values()) == pytest.approx(1.0)


def test_labeled_env_segments_empty(cfg: Config):
    duration = 6.0
    labels = ClipLabels(
        video_id="clip_empty",
        duration=duration,
        labeler="test",
        segments=[],
        events=[],
    )

    result = labeled_env_segments(labels, cfg)
    validate_segments(result, duration)

    assert len(result) == 1
    assert result[0].start == 0.0
    assert result[0].end == duration
    assert result[0].env == GENERAL_ENV
    assert result[0].confidence == 1.0
    assert result[0].probs == {GENERAL_ENV: 1.0}


def test_labeled_env_segments_general_env(cfg: Config):
    duration = 4.0
    labels = ClipLabels(
        video_id="clip_gen",
        duration=duration,
        labeler="test",
        segments=[LabeledSegment(start=0.0, end=duration, env=GENERAL_ENV)],
        events=[],
    )

    result = labeled_env_segments(labels, cfg, confidence=0.7)
    validate_segments(result, duration)

    assert len(result) == 1
    assert result[0].env == GENERAL_ENV
    assert result[0].confidence == 1.0
    assert result[0].probs == {GENERAL_ENV: 1.0}


def test_labeled_env_segments_unknown_env_raises(cfg: Config):
    labels = ClipLabels(
        video_id="clip_bad_env",
        duration=5.0,
        labeler="test",
        segments=[
            LabeledSegment(start=0.0, end=5.0, env="nonexistent_environment_xyz")
        ],
        events=[],
    )
    with pytest.raises(KeyError, match="nonexistent_environment_xyz"):
        labeled_env_segments(labels, cfg)


# ===========================================================================
# score() tests
# ===========================================================================


def test_score_matching_clip_modes_and_counts(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    labels_dir = tmp_path / "dev_labels"
    lbl, sc_id, safety_id = _setup_clip_and_cache(cfg, labels_dir, monkeypatch)

    res = score(cfg=cfg, labels_dir=labels_dir)

    assert res.objective > 0.0
    assert res.baseline > 0.0
    assert res.wrong_env > 0.0
    assert res.safety_recall == 1.0
    assert res.fp_per_min == pytest.approx(0.0)
    assert res.split == "dev"
    assert res.passed is True

    vid = lbl.video_id
    assert vid in res.per_clip
    modes = res.per_clip[vid]
    assert set(modes.keys()) == {"conditioned", "baseline", "wrong_env", "stats"}
    assert len(modes["stats"]) == 4

    count_keys = {
        "tp_fine",
        "tp_coarse",
        "fp",
        "fn",
        "minutes",
        "safety_tp",
        "safety_total",
    }
    for mode in ("conditioned", "baseline", "wrong_env"):
        assert set(modes[mode].keys()) == count_keys
        for k in ("tp_fine", "tp_coarse", "fp", "fn", "safety_tp", "safety_total"):
            assert isinstance(modes[mode][k], int)
        assert isinstance(modes[mode]["minutes"], float)

    assert modes["conditioned"]["tp_fine"] > 0
    assert modes["conditioned"]["safety_tp"] > 0
    assert modes["conditioned"]["safety_total"] > 0


def test_score_missing_cache_raises_filenotfound(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    env_id = next(e for e in cfg.env_ids if e != GENERAL_ENV)
    sc = next(c for c in cfg.classes if c.audioset and not c.safety)

    vid = "missing_cache_clip"
    lbl = ClipLabels(
        video_id=vid,
        duration=2.0,
        labeler="test",
        segments=[LabeledSegment(start=0.0, end=2.0, env=env_id)],
        events=[LabeledEvent(start=0.5, end=1.5, class_id=sc.id)],
    )
    save_labels(lbl, labels_dir / f"{vid}.json")

    key = cache_key(cfg)
    with pytest.raises(FileNotFoundError) as exc_info:
        score(cfg=cfg, labels_dir=labels_dir)

    msg = str(exc_info.value)
    assert vid in msg
    assert key in msg
    assert "caches must be built/recached" in msg


def test_score_cfg_none_reads_caller_working_tree(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Set up synthetic data
    labels_dir = tmp_path / "dev_labels"
    lbl, sc_id, _ = _setup_clip_and_cache(cfg, labels_dir, monkeypatch)

    # Baseline score with repo config
    res_repo = score(cfg=cfg, labels_dir=labels_dir)
    assert res_repo.objective > 0.0

    # Copy repo config to tmp_path / "config"
    config_dir = tmp_path / "config"
    shutil.copytree(cfg.config_dir, config_dir)

    # Edit thresholds.yaml in tmp_path/"config" to raise threshold to 0.99
    thr_file = config_dir / "thresholds.yaml"
    text_modified = (
        "default: 0.99\nsafety_default: 0.99\nclasses:\n"
        + f"  {sc_id}: 0.99\n"
    )
    thr_file.write_text(text_modified, encoding="utf-8")

    # Change working directory to tmp_path
    monkeypatch.chdir(tmp_path)

    # Score with cfg=None (must read Path.cwd() / "config")
    res_modified = score(cfg=None, labels_dir=labels_dir)

    # With 0.99 threshold, the 0.95 prob tagger detection no longer fires
    assert res_modified.objective == 0.0
    assert res_modified.objective != res_repo.objective


def test_score_overrides_change_result(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    labels_dir = tmp_path / "dev_labels"
    lbl, sc_id, _ = _setup_clip_and_cache(
        cfg, labels_dir, monkeypatch, include_safety=False
    )

    res_base = score(cfg=cfg, labels_dir=labels_dir)
    assert res_base.per_clip[lbl.video_id]["conditioned"]["tp_fine"] > 0
    assert res_base.overrides == {}

    # Override threshold for sc_id to 0.99
    override_key = f"thresholds.classes.{sc_id}"
    overrides = {override_key: 0.99}
    res_over = score(cfg=cfg, labels_dir=labels_dir, overrides=overrides)

    assert res_over.overrides == overrides
    assert res_over.per_clip[lbl.video_id]["conditioned"]["tp_fine"] == 0
    assert res_over.objective < res_base.objective


def test_score_limits_behavior(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    labels_dir = tmp_path / "dev_labels"
    lbl, sc_id, safety_id = _setup_clip_and_cache(cfg, labels_dir, monkeypatch)

    base_res = score(cfg=cfg, labels_dir=labels_dir)

    # 1. With best=None: all limits pass when objective >= baseline
    assert base_res.limits == {
        "safety_recall": True,
        "fp_per_min": True,
        "above_baseline": True,
        "wrong_env": True,
    }
    assert base_res.passed is True

    # Craft an unfavorable prior override so conditioned drops below baseline.
    # Odds priors keep a 0.95 score fairly high (the labeled env is mixed with neutral
    # general), so also raise the class minimum to just under the raw score: baseline
    # still detects it, the downweighted conditioned score does not.
    env_id = lbl.segments[0].env
    unfavorable_overrides = {
        "priors.floor": 0.01,  # odds mode needs a positive floor
        f"priors.{env_id}.default": 0.01,
        f"priors.{env_id}.classes.{sc_id}": 0.01,
        f"thresholds.classes.{sc_id}": 0.9,
    }
    unfavorable_cfg = apply_overrides(cfg, unfavorable_overrides)
    res_below_base = score(
        cfg=unfavorable_cfg, labels_dir=labels_dir
    )
    # With best=None, only above_baseline should fail
    assert res_below_base.objective < res_below_base.baseline
    assert res_below_base.limits["above_baseline"] is False
    assert res_below_base.limits["safety_recall"] is True
    assert res_below_base.limits["fp_per_min"] is True
    assert res_below_base.limits["wrong_env"] is True
    assert res_below_base.passed is False

    # 2. With weak best: all limits pass
    weak_best = {
        "safety_recall": 0.0,
        "fp_per_min": 999.0,
        "wrong_env": 0.0,
    }
    res_weak = score(cfg=cfg, labels_dir=labels_dir, best=weak_best)
    assert all(res_weak.limits.values())
    assert res_weak.passed is True

    # 3. With best having higher safety_recall -> safety_recall limit fails
    best_high_sr = {
        "safety_recall": base_res.safety_recall + 0.1,
        "fp_per_min": base_res.fp_per_min + 10.0,
        "wrong_env": 0.0,
    }
    res_high_sr = score(cfg=cfg, labels_dir=labels_dir, best=best_high_sr)
    assert res_high_sr.limits["safety_recall"] is False
    assert res_high_sr.passed is False

    # 4. With best having lower fp_per_min -> fp_per_min limit fails
    best_low_fp = {
        "safety_recall": 0.0,
        "fp_per_min": base_res.fp_per_min - 0.5,
        "wrong_env": 0.0,
    }
    res_low_fp = score(cfg=cfg, labels_dir=labels_dir, best=best_low_fp)
    assert res_low_fp.limits["fp_per_min"] is False
    assert res_low_fp.passed is False

    # 5. With best having higher wrong_env -> wrong_env limit fails
    best_high_we = {
        "safety_recall": 0.0,
        "fp_per_min": 999.0,
        "wrong_env": base_res.wrong_env + 0.5,
    }
    res_high_we = score(cfg=cfg, labels_dir=labels_dir, best=best_high_we)
    assert res_high_we.limits["wrong_env"] is False
    assert res_high_we.passed is False


def test_score_determinism(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    labels_dir = tmp_path / "dev_labels"
    _setup_clip_and_cache(cfg, labels_dir, monkeypatch)

    r1 = score(cfg=cfg, labels_dir=labels_dir)
    r2 = score(cfg=cfg, labels_dir=labels_dir)

    d1 = r1.to_dict()
    d2 = r2.to_dict()

    assert d1["run_id"] != d2["run_id"]
    d1["run_id"] = "fixed"
    d2["run_id"] = "fixed"
    assert d1 == d2


def test_score_to_dict_json_serializable(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    labels_dir = tmp_path / "dev_labels"
    _setup_clip_and_cache(cfg, labels_dir, monkeypatch)

    r = score(cfg=cfg, labels_dir=labels_dir)
    d = r.to_dict()

    serialized = json.dumps(d)
    deserialized = json.loads(serialized)
    assert deserialized == d


def test_score_does_not_create_ledger(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    assert not paths.ledger_path().exists()

    labels_dir = tmp_path / "dev_labels"
    _setup_clip_and_cache(cfg, labels_dir, monkeypatch)

    score(cfg=cfg, labels_dir=labels_dir)

    assert not paths.ledger_path().exists()


def test_score_split_handling(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    with pytest.raises(ValueError, match="Unknown split"):
        score(cfg=cfg, split="invalid_split_name")
