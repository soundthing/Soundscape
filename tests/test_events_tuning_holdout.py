from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from soundthing.config import Config
from soundthing.contracts import ClipLabels, LabeledEvent, LabeledSegment, save_labels
from soundthing.events.tuning import paths
from soundthing.events.tuning.holdout import (
    assert_disjoint,
    dev_ids,
    holdout_audio_dir,
    holdout_ids,
    holdout_labels_dir,
    import_holdout,
)


@pytest.fixture(autouse=True)
def set_tuning_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))


def _make_labels(video_id: str, cfg: Config) -> ClipLabels:
    return ClipLabels(
        video_id=video_id,
        duration=3.0,
        labeler="test_annotator",
        segments=[LabeledSegment(start=0.0, end=3.0, env=cfg.env_ids[0])],
        events=[LabeledEvent(start=0.5, end=1.5, class_id=cfg.class_ids[0])],
    )


def _write_fake_audio(path: Path, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(sr, dtype=np.float32)
    sf.write(str(path), data, sr)


def test_holdout_ids_fresh_home() -> None:
    assert holdout_ids() == []
    assert dev_ids() == []


def test_import_copies_labels_and_fake_wav(tmp_path: Path, cfg: Config) -> None:
    src_labels = tmp_path / "src_labels"
    src_audio = tmp_path / "src_audio"
    labels = _make_labels("clip01", cfg)
    save_labels(labels, src_labels / "clip01.json")
    _write_fake_audio(src_audio / "clip01.wav")

    imported = import_holdout(src_labels, src_audio, cfg=cfg)

    assert imported == ["clip01"]
    assert holdout_ids() == ["clip01"]
    dest_json = holdout_labels_dir() / "clip01.json"
    dest_wav = holdout_audio_dir() / "clip01.wav"
    assert dest_json.exists()
    assert dest_wav.exists()
    assert dest_json.resolve() == (tmp_path / "holdout" / "labels" / "clip01.json").resolve()
    assert dest_wav.resolve() == (tmp_path / "holdout" / "audio" / "clip01.wav").resolve()


def test_missing_audio_warns_but_imports_labels(tmp_path: Path, cfg: Config) -> None:
    src_labels = tmp_path / "src_labels_missing"
    labels = _make_labels("clip_no_audio", cfg)
    save_labels(labels, src_labels / "clip_no_audio.json")
    empty_audio = tmp_path / "empty_audio"
    empty_audio.mkdir(parents=True, exist_ok=True)

    with pytest.warns(UserWarning, match="clip_no_audio"):
        imported = import_holdout(src_labels, empty_audio, cfg=cfg)

    assert imported == ["clip_no_audio"]
    assert holdout_ids() == ["clip_no_audio"]
    assert (holdout_labels_dir() / "clip_no_audio.json").exists()
    assert not (holdout_audio_dir() / "clip_no_audio.wav").exists()


def test_src_audio_none_warns_and_imports_labels(tmp_path: Path, cfg: Config) -> None:
    src_labels = tmp_path / "src_labels_none"
    labels = _make_labels("clip_none", cfg)
    save_labels(labels, src_labels / "clip_none.json")

    with pytest.warns(UserWarning, match="clip_none"):
        imported = import_holdout(src_labels, None, cfg=cfg)

    assert imported == ["clip_none"]
    assert holdout_ids() == ["clip_none"]
    assert (holdout_labels_dir() / "clip_none.json").exists()


def test_audio_extension_support(tmp_path: Path, cfg: Config) -> None:
    src_labels = tmp_path / "src_labels_ext"
    src_audio = tmp_path / "src_audio_ext"
    labels = _make_labels("clip_flac", cfg)
    save_labels(labels, src_labels / "clip_flac.json")
    _write_fake_audio(src_audio / "clip_flac.flac")

    imported = import_holdout(src_labels, src_audio, cfg=cfg)

    assert imported == ["clip_flac"]
    assert (holdout_audio_dir() / "clip_flac.flac").exists()


def test_unknown_class_id_raises_and_nothing_imported(tmp_path: Path, cfg: Config) -> None:
    src_labels = tmp_path / "src_bad_class"
    good_labels = _make_labels("clip_good", cfg)
    save_labels(good_good := good_labels, src_labels / "clip_good.json")

    bad_labels = ClipLabels(
        video_id="clip_bad",
        duration=3.0,
        labeler="test",
        segments=[LabeledSegment(start=0.0, end=3.0, env=cfg.env_ids[0])],
        events=[LabeledEvent(start=0.5, end=1.5, class_id="nonexistent_class_xyz")],
    )
    save_labels(bad_labels, src_labels / "clip_bad.json")

    with pytest.raises(ValueError, match="nonexistent_class_xyz"):
        import_holdout(src_labels, cfg=cfg)

    assert holdout_ids() == []
    assert not (holdout_labels_dir() / "clip_good.json").exists()
    assert not (holdout_labels_dir() / "clip_bad.json").exists()


def test_unknown_env_id_raises_and_nothing_imported(tmp_path: Path, cfg: Config) -> None:
    src_labels = tmp_path / "src_bad_env"
    bad_labels = ClipLabels(
        video_id="clip_bad_env",
        duration=3.0,
        labeler="test",
        segments=[LabeledSegment(start=0.0, end=3.0, env="nonexistent_env_xyz")],
        events=[LabeledEvent(start=0.5, end=1.5, class_id=cfg.class_ids[0])],
    )
    save_labels(bad_labels, src_labels / "clip_bad_env.json")

    with pytest.raises(ValueError, match="nonexistent_env_xyz"):
        import_holdout(src_labels, cfg=cfg)

    assert holdout_ids() == []
    assert not (holdout_labels_dir() / "clip_bad_env.json").exists()


def test_stem_mismatch_raises_and_nothing_imported(tmp_path: Path, cfg: Config) -> None:
    src_labels = tmp_path / "src_stem_mismatch"
    labels = _make_labels("correct_id", cfg)
    save_labels(labels, src_labels / "wrong_stem.json")

    with pytest.raises(ValueError, match="wrong_stem"):
        import_holdout(src_labels, cfg=cfg)

    assert holdout_ids() == []


def test_dev_overlap_refused_and_assert_disjoint(tmp_path: Path, cfg: Config) -> None:
    dev_dir = paths.dev_labels_dir()
    save_labels(_make_labels("clip_dev", cfg), dev_dir / "clip_dev.json")
    assert dev_ids() == ["clip_dev"]

    src_labels = tmp_path / "src_dev_overlap"
    save_labels(_make_labels("clip_dev", cfg), src_labels / "clip_dev.json")

    with pytest.raises(ValueError, match="clip_dev"):
        import_holdout(src_labels, cfg=cfg)

    assert holdout_ids() == []

    src_disjoint = tmp_path / "src_disjoint"
    save_labels(_make_labels("clip_holdout", cfg), src_disjoint / "clip_holdout.json")
    imported = import_holdout(src_disjoint, None, cfg=cfg)
    assert imported == ["clip_holdout"]
    assert holdout_ids() == ["clip_holdout"]

    assert_disjoint(["other_clip", "clip_dev"])

    with pytest.raises(ValueError, match="clip_holdout"):
        assert_disjoint(["clip_holdout", "clip_dev"])


def test_repo_labels_import_cleanly(cfg: Config) -> None:
    repo_labels_dir = Path("labels")
    repo_files = sorted(repo_labels_dir.glob("*.json"))
    if not repo_files:
        pytest.skip("no labels/*.json present in repo worktree")

    imported = import_holdout(repo_labels_dir, cfg=cfg)
    assert imported == [p.stem for p in repo_files]


def test_import_with_cfg_default_loads_config(tmp_path: Path, cfg: Config) -> None:
    src_labels = tmp_path / "src_default_cfg"
    save_labels(_make_labels("clip_default", cfg), src_labels / "clip_default.json")

    imported = import_holdout(src_labels)
    assert imported == ["clip_default"]
    assert holdout_ids() == ["clip_default"]


def test_import_overwrite_allowed(tmp_path: Path, cfg: Config) -> None:
    src_labels = tmp_path / "src_overwrite"
    save_labels(_make_labels("clip_reimport", cfg), src_labels / "clip_reimport.json")

    first = import_holdout(src_labels, cfg=cfg)
    second = import_holdout(src_labels, cfg=cfg)

    assert first == ["clip_reimport"]
    assert second == ["clip_reimport"]
    assert holdout_ids() == ["clip_reimport"]
