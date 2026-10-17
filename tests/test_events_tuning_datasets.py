"""Tests for synthetic dev mixes from ESC-50 and dataset YAML."""

from __future__ import annotations

import ast
import csv
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from soundthing.config import load_config
from soundthing.contracts import ClipLabels, LabeledEvent, load_labels, save_labels
import soundthing.events.tuning as tuning_pkg
from soundthing.events.tuning import datasets, holdout
from soundthing.events.tuning.datasets import (
    _load_mono_16k,
    _trim_silence,
    audioset_strong_labels,
    build_dev_caches,
    esc50_mixes,
    fsd50k_mixes,
    load_datasets_yaml,
)

ALL_50_ESC50_CATEGORIES = {
    "airplane",
    "breathing",
    "brushing_teeth",
    "can_opening",
    "car_horn",
    "cat",
    "chainsaw",
    "chirping_birds",
    "church_bells",
    "clapping",
    "clock_alarm",
    "clock_tick",
    "coughing",
    "cow",
    "crackling_fire",
    "crickets",
    "crow",
    "crying_baby",
    "dog",
    "door_wood_creaks",
    "door_wood_knock",
    "drinking_sipping",
    "engine",
    "fireworks",
    "footsteps",
    "frog",
    "glass_breaking",
    "hand_saw",
    "helicopter",
    "hen",
    "insects",
    "keyboard_typing",
    "laughing",
    "mouse_click",
    "pig",
    "pouring_water",
    "rain",
    "rooster",
    "sea_waves",
    "sheep",
    "siren",
    "sneezing",
    "snoring",
    "thunderstorm",
    "toilet_flush",
    "train",
    "vacuum_cleaner",
    "washing_machine",
    "water_drops",
    "wind",
}


def test_datasets_yaml_structure_and_vocabulary():
    """Verify datasets.yaml covers all 50 categories and maps to valid classes."""
    cfg = load_config()
    data = load_datasets_yaml()

    assert "esc50" in data
    esc_map = data["esc50"]["map"]
    esc_unmapped = set(data["esc50"]["unmapped"])
    esc_bg = set(data["esc50"]["background"])

    valid_class_ids = set(cfg.class_ids)
    for category, class_id in esc_map.items():
        assert class_id in valid_class_ids, f"{category} -> {class_id} not in vocabulary"

    map_cats = set(esc_map.keys())
    assert map_cats.isdisjoint(esc_unmapped), "map and unmapped categories must be disjoint"
    assert (map_cats | esc_unmapped) == ALL_50_ESC50_CATEGORIES
    assert len(map_cats | esc_unmapped) == 50
    assert esc_bg.issubset(esc_unmapped), "background must be a subset of unmapped categories"


def test_trim_silence_finds_burst_within_one_frame():
    """_trim_silence finds a tone burst placed at a known sample within one frame."""
    sr = 16000
    mix_cfg = {
        "trim_frame_sec": 0.02,
        "trim_rel_db": -35.0,
        "trim_floor_dbfs": -60.0,
        "max_event_sec": 8.0,
    }
    frame_len = int(round(sr * mix_cfg["trim_frame_sec"]))

    total_samples = 32000
    audio = np.zeros(total_samples, dtype=np.float32)

    known_start = 16000
    known_end = 24000
    t = np.linspace(0, (known_end - known_start) / sr, known_end - known_start, endpoint=False)
    audio[known_start:known_end] = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    start_sample, end_sample = _trim_silence(audio, sr, mix_cfg)
    assert abs(start_sample - known_start) <= frame_len
    assert abs(end_sample - known_end) <= frame_len

    silent = np.zeros(16000, dtype=np.float32)
    assert _trim_silence(silent, sr, mix_cfg) == (0, 0)


@pytest.fixture
def fake_esc50(tmp_path: Path) -> Path:
    """Create a minimal fake ESC-50 dataset with 44.1 kHz audio and silence padding."""
    esc50_dir = tmp_path / "fake_esc50"
    audio_dir = esc50_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    data = load_datasets_yaml()
    mapped_categories = sorted(data["esc50"]["map"].keys())[:3]
    bg_category = data["esc50"]["background"][0]

    sr_raw = 44100
    rows = []
    # Create files for folds 1 and 2
    for fold in (1, 2):
        # Mapped categories: tone bursts with silence padding
        for idx, cat in enumerate(mapped_categories):
            filename = f"{fold}-1000{idx}-A-{fold}.wav"
            tone_len = int(0.4 * sr_raw)
            pad_before = int(0.2 * sr_raw)
            pad_after = int(0.2 * sr_raw)
            t = np.linspace(0, 0.4, tone_len, endpoint=False)
            tone = 0.6 * np.sin(2 * np.pi * (300 + idx * 200) * t).astype(np.float32)
            audio = np.concatenate([
                np.zeros(pad_before, dtype=np.float32),
                tone,
                np.zeros(pad_after, dtype=np.float32),
            ])
            sf.write(str(audio_dir / filename), audio, sr_raw)
            rows.append({
                "filename": filename,
                "fold": str(fold),
                "target": str(idx),
                "category": cat,
                "esc10": "False",
                "src_file": filename,
                "take": "A",
            })

        # Background category: steady quiet noise
        bg_filename = f"{fold}-99990-A-{fold}.wav"
        rng = np.random.default_rng(fold)
        bg_audio = rng.normal(0.0, 0.02, size=int(1.0 * sr_raw)).astype(np.float32)
        sf.write(str(audio_dir / bg_filename), bg_audio, sr_raw)
        rows.append({
            "filename": bg_filename,
            "fold": str(fold),
            "target": "99",
            "category": bg_category,
            "esc10": "False",
            "src_file": bg_filename,
            "take": "A",
        })

    csv_path = esc50_dir / "esc50.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return esc50_dir


def test_esc50_mixes_generation(fake_esc50: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """End-to-end synthetic dev mix creation, determinism, audio quality, and labels."""
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    cfg = load_config()

    orig_yaml = load_datasets_yaml()
    fast_yaml = json.loads(json.dumps(orig_yaml))
    fast_yaml["mix"]["clip_sec"] = 4.0
    fast_yaml["mix"]["min_gap_sec"] = 0.2
    fast_yaml["mix"]["max_event_sec"] = 0.8
    monkeypatch.setattr(datasets, "load_datasets_yaml", lambda: fast_yaml)

    out1 = tmp_path / "mixes_1"
    labels1 = esc50_mixes(fake_esc50, cfg, n_per_env=2, seed=1, out_dir=out1)

    expected_clips_count = 2 * len(cfg.env_ids)
    assert len(labels1) == expected_clips_count

    clip_sec = float(fast_yaml["mix"]["clip_sec"])
    expected_samples = int(round(clip_sec * 16000))
    valid_class_ids = set(cfg.class_ids)

    for p in labels1:
        assert p.is_file()
        labels = load_labels(p)
        labels.validate()

        assert labels.duration == clip_sec
        assert len(labels.segments) == 1
        seg = labels.segments[0]
        assert seg.start == 0.0
        assert seg.end == clip_sec
        assert seg.env in cfg.env_ids

        wav_path = out1 / "audio" / f"{labels.video_id}.wav"
        assert wav_path.is_file()
        audio, sr = sf.read(str(wav_path))
        assert sr == 16000
        assert audio.ndim == 1
        assert len(audio) == expected_samples

        for ev in labels.events:
            assert ev.class_id in valid_class_ids

        # Sorted and non-overlapping events
        for a, b in zip(labels.events, labels.events[1:]):
            assert a.start < a.end <= b.start

        # At each labeled onset, the event is audible (RMS 50ms after >> RMS 50ms before)
        window = int(0.05 * 16000)
        for ev in labels.events:
            onset_sample = int(round(ev.start * 16000))
            if onset_sample >= window:
                rms_before = float(np.sqrt(np.mean(audio[onset_sample - window : onset_sample] ** 2)))
                rms_after = float(np.sqrt(np.mean(audio[onset_sample : onset_sample + window] ** 2)))
                assert rms_after > 1.5 * rms_before, f"Onset at {ev.start}s not audible: {rms_after} vs {rms_before}"

    # Verify manifest
    manifest_file = out1 / "manifest.json"
    assert manifest_file.is_file()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "esc50"
    assert manifest["seed"] == 1
    assert manifest["n_per_env"] == 2
    assert len(manifest["clips"]) == expected_clips_count

    for video_id, clip_info in manifest["clips"].items():
        assert "env" in clip_info
        fold = clip_info["fold"]
        assert fold in (1, 2)
        for src in clip_info["sources"]:
            assert src["fold"] == fold

    # Determinism: same seed gives byte-identical label json and audio
    out2 = tmp_path / "mixes_2"
    labels2 = esc50_mixes(fake_esc50, cfg, n_per_env=2, seed=1, out_dir=out2)
    assert len(labels1) == len(labels2)

    for p1, p2 in zip(labels1, labels2):
        assert p1.read_bytes() == p2.read_bytes()
        vid = p1.stem
        wav1 = out1 / "audio" / f"{vid}.wav"
        wav2 = out2 / "audio" / f"{vid}.wav"
        assert wav1.read_bytes() == wav2.read_bytes()

    # Different seed produces different audio / labels
    out3 = tmp_path / "mixes_3"
    labels3 = esc50_mixes(fake_esc50, cfg, n_per_env=2, seed=2, out_dir=out3)
    assert len(labels3) == expected_clips_count
    assert [p.name for p in labels1] != [p.name for p in labels3]

    labels1_obj = load_labels(labels1[0])
    labels3_obj = load_labels(labels3[0])
    wav1_data, _ = sf.read(str(out1 / "audio" / f"{labels1_obj.video_id}.wav"))
    wav3_data, _ = sf.read(str(out3 / "audio" / f"{labels3_obj.video_id}.wav"))
    assert not np.array_equal(wav1_data, wav3_data)


def test_invalid_mapped_id_raises_value_error(fake_esc50: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A mapped class id not in vocabulary raises ValueError naming the id."""
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    cfg = load_config()

    orig_yaml = load_datasets_yaml()
    bad_yaml = json.loads(json.dumps(orig_yaml))
    first_cat = list(bad_yaml["esc50"]["map"].keys())[0]
    bad_yaml["esc50"]["map"][first_cat] = "nonexistent_class_id_xyz"

    monkeypatch.setattr(datasets, "load_datasets_yaml", lambda: bad_yaml)

    with pytest.raises(ValueError, match="nonexistent_class_id_xyz"):
        esc50_mixes(fake_esc50, cfg, n_per_env=1, seed=1, out_dir=tmp_path / "out")


def test_prior_weighted_sampling(fake_esc50: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Sampling uses prior weights: weight given only to one candidate forces all events to that class."""
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    cfg = load_config()

    orig_yaml = load_datasets_yaml()
    fast_yaml = json.loads(json.dumps(orig_yaml))
    fast_yaml["mix"]["clip_sec"] = 4.0
    fast_yaml["mix"]["min_gap_sec"] = 0.2
    fast_yaml["mix"]["max_event_sec"] = 0.8
    monkeypatch.setattr(datasets, "load_datasets_yaml", lambda: fast_yaml)

    target_class = fast_yaml["esc50"]["map"][list(fast_yaml["esc50"]["map"].keys())[0]]

    def fake_prior_weights(config: Any, env_probs: dict[str, float], candidates: list[str]) -> np.ndarray:
        w = np.zeros(len(candidates), dtype=np.float64)
        if target_class in candidates:
            w[candidates.index(target_class)] = 1.0
        return w

    monkeypatch.setattr(datasets, "prior_weights", fake_prior_weights)

    out = tmp_path / "priors_out"
    labels_list = esc50_mixes(fake_esc50, cfg, n_per_env=2, seed=1, out_dir=out)

    events_checked = 0
    for p in labels_list:
        labels = load_labels(p)
        for ev in labels.events:
            assert ev.class_id == target_class
            events_checked += 1

    assert events_checked > 0, "Expected at least one event across all mixes to verify class"


def test_fsd50k_yaml_structure_and_vocabulary():
    """Verify datasets.yaml covers FSD50K with valid vocabulary class ids and disjoint background."""
    cfg = load_config()
    data = load_datasets_yaml()

    assert "fsd50k" in data
    fsd_map = data["fsd50k"]["map"]
    fsd_bg = set(data["fsd50k"].get("background", []))

    valid_class_ids = set(cfg.class_ids)
    assert len(fsd_map) > 0, "fsd50k map must not be empty"
    for cat, cid in fsd_map.items():
        assert cid in valid_class_ids, f"{cat} -> {cid} not in vocabulary"

    assert fsd_bg, "fsd50k background must not be empty"
    assert set(fsd_map.keys()).isdisjoint(fsd_bg), "fsd50k map and background must be disjoint"

    assert "audioset" in data
    assert data["audioset"]["clip_sec"] == 10.0


@pytest.fixture
def fake_fsd50k(tmp_path: Path) -> Path:
    """Create a minimal fake FSD50K dataset."""
    fsd_dir = tmp_path / "fake_fsd50k"
    gt_dir = fsd_dir / "FSD50K.ground_truth"
    dev_audio_dir = fsd_dir / "FSD50K.dev_audio"
    gt_dir.mkdir(parents=True, exist_ok=True)
    dev_audio_dir.mkdir(parents=True, exist_ok=True)

    data = load_datasets_yaml()
    fsd_map = data["fsd50k"]["map"]
    fsd_bg = data["fsd50k"]["background"]

    # Pick two categories that map to different class ids
    mapped_items = list(fsd_map.items())
    cat1, cid1 = mapped_items[0]
    cat2, cid2 = next((c, cid) for c, cid in mapped_items[1:] if cid != cid1)
    # Pick another category that also maps to cid1 (if any, else just use cat1)
    cat1_alias = next((c for c, cid in mapped_items if cid == cid1 and c != cat1), cat1)

    bg_cat1 = fsd_bg[0]
    bg_cat2 = fsd_bg[1] if len(fsd_bg) > 1 else fsd_bg[0]

    sr_raw = 44100

    def make_tone(path: Path, freq: float = 440.0) -> None:
        t = np.linspace(0, 0.5, int(0.5 * sr_raw), endpoint=False)
        tone = 0.6 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        pad = np.zeros(int(0.1 * sr_raw), dtype=np.float32)
        audio = np.concatenate([pad, tone, pad])
        sf.write(str(path), audio, sr_raw)

    def make_noise(path: Path) -> None:
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 0.02, size=int(1.0 * sr_raw)).astype(np.float32)
        sf.write(str(path), noise, sr_raw)

    # 1. Foreground clip mapping to cid1 (with two labels mapping to same class), train split
    make_tone(dev_audio_dir / "101.wav", 300.0)
    # 2. Foreground clip mapping to cid2, val split
    make_tone(dev_audio_dir / "102.wav", 500.0)
    # 3. Multi-label clip mapping to two distinct classes (cat1, cat2), train split (MUST BE EXCLUDED)
    make_tone(dev_audio_dir / "103.wav", 700.0)
    # 4. Background clip, train split
    make_noise(dev_audio_dir / "104.wav")
    # 5. Background clip, val split
    make_noise(dev_audio_dir / "105.wav")

    rows = [
        {"fname": "101", "labels": f"{cat1},{cat1_alias}", "mids": "/m/01,/m/02", "split": "train"},
        {"fname": "102", "labels": cat2, "mids": "/m/03", "split": "val"},
        {"fname": "103", "labels": f"{cat1},{cat2}", "mids": "/m/01,/m/03", "split": "train"},
        {"fname": "104", "labels": bg_cat1, "mids": "/m/04", "split": "train"},
        {"fname": "105", "labels": bg_cat2, "mids": "/m/05", "split": "val"},
        {"fname": "999", "labels": cat1, "mids": "/m/01", "split": "train"},
    ]

    csv_path = gt_dir / "dev.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fname", "labels", "mids", "split"])
        writer.writeheader()
        writer.writerows(rows)

    return fsd_dir


def test_fsd50k_mixes_generation(fake_fsd50k: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """FSD50K mixes: single fold per mix, multi-label exclusion, validation, determinism."""
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    cfg = load_config()

    orig_yaml = load_datasets_yaml()
    fast_yaml = json.loads(json.dumps(orig_yaml))
    fast_yaml["mix"]["clip_sec"] = 4.0
    fast_yaml["mix"]["min_gap_sec"] = 0.2
    fast_yaml["mix"]["max_event_sec"] = 0.8
    monkeypatch.setattr(datasets, "load_datasets_yaml", lambda: fast_yaml)

    out1 = tmp_path / "fsd_mixes_1"
    labels1 = fsd50k_mixes(fake_fsd50k, cfg, n_per_env=2, seed=42, out_dir=out1, split="dev")

    expected_clips_count = 2 * len(cfg.env_ids)
    assert len(labels1) == expected_clips_count

    manifest_file = out1 / "manifest.json"
    assert manifest_file.is_file()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "fsd50k"
    assert manifest["seed"] == 42
    assert manifest["n_per_env"] == 2

    for p in labels1:
        assert p.is_file()
        labels = load_labels(p)
        labels.validate()
        assert labels.labeler == "fsd50k-synthetic"
        assert labels.duration == 4.0
        assert len(labels.segments) == 1
        assert labels.segments[0].env in cfg.env_ids

    # Check manifest clips: single fold per mix and multi-label clip 103 never appears
    for video_id, clip_info in manifest["clips"].items():
        fold = clip_info["fold"]
        assert fold in ("train", "val")
        for src in clip_info["sources"]:
            assert src["fold"] == fold
            assert "103" not in src["file"], "Multi-label clip 103 mapping to two classes must be excluded"
            assert "999" not in src["file"], "Nonexistent wav 999 must not appear"

    # Determinism: same seed gives byte-identical label json and audio
    out2 = tmp_path / "fsd_mixes_2"
    labels2 = fsd50k_mixes(fake_fsd50k, cfg, n_per_env=2, seed=42, out_dir=out2, split="dev")
    assert len(labels1) == len(labels2)
    for p1, p2 in zip(labels1, labels2):
        assert p1.read_bytes() == p2.read_bytes()
        vid = p1.stem
        wav1 = out1 / "audio" / f"{vid}.wav"
        wav2 = out2 / "audio" / f"{vid}.wav"
        assert wav1.read_bytes() == wav2.read_bytes()


def test_audioset_strong_labels_conversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """AudioSet strong labels: event mapping, clipping, merging, and negative handling."""
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    cfg = load_config()

    classes_with_audioset = [c for c in cfg.classes if c.audioset]
    c_a = classes_with_audioset[0]
    c_b = classes_with_audioset[1]
    name_a = c_a.audioset[0]
    name_b = c_b.audioset[0]

    csv_path = tmp_path / "class_labels_indices.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "mid", "display_name"])
        writer.writeheader()
        writer.writerow({"index": "0", "mid": "/m/mid_a", "display_name": name_a})
        writer.writerow({"index": "1", "mid": "/m/mid_b", "display_name": name_b})
        writer.writerow({"index": "2", "mid": "/m/mid_unmapped", "display_name": "Unmapped Sound"})

    tsv_path = tmp_path / "audioset_strong.tsv"
    tsv_content = (
        "segment_id\tstart_time_seconds\tend_time_seconds\tlabel\n"
        "seg_1\t1.0\t3.0\t/m/mid_a\n"
        "seg_1\t2.0\t4.5\t/m/mid_a\n"
        "seg_1\t5.0\t6.0\t/m/mid_unmapped\n"
        "seg_1\t8.0\t12.0\t/m/mid_b\n"
        "seg_2\t2.0\t4.0\t/m/mid_unmapped\n"
    )
    tsv_path.write_text(tsv_content, encoding="utf-8")

    with pytest.raises(ValueError, match="class_labels_csv is required"):
        audioset_strong_labels(tsv_path, cfg, out_dir=tmp_path / "out", class_labels_csv=None)

    out_dir = tmp_path / "as_out"
    label_paths = audioset_strong_labels(tsv_path, cfg, out_dir=out_dir, class_labels_csv=csv_path)
    assert len(label_paths) == 2

    # Verify seg_1
    lbl_1 = load_labels(out_dir / "labels" / "seg_1.json")
    lbl_1.validate()
    assert lbl_1.video_id == "seg_1"
    assert lbl_1.duration == 10.0
    assert lbl_1.labeler == "audioset-strong"
    assert lbl_1.segments == []
    assert len(lbl_1.events) == 2
    assert lbl_1.events[0].class_id == c_a.id
    assert lbl_1.events[0].start == 1.0
    assert lbl_1.events[0].end == 4.5
    assert lbl_1.events[1].class_id == c_b.id
    assert lbl_1.events[1].start == 8.0
    assert lbl_1.events[1].end == 10.0

    # Verify seg_2 (negative clip)
    lbl_2 = load_labels(out_dir / "labels" / "seg_2.json")
    lbl_2.validate()
    assert lbl_2.video_id == "seg_2"
    assert lbl_2.duration == 10.0
    assert lbl_2.labeler == "audioset-strong"
    assert lbl_2.segments == []
    assert lbl_2.events == []


def test_build_dev_caches_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Dev cache builds: fake cache module, skips without audio, respects overwrite and holdout."""
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path))
    cfg = load_config()

    built_calls: list[tuple[str, int, Any]] = []
    saved_calls: list[tuple[str, Path]] = []

    class FakeClipCache:
        def __init__(self, video_id: str):
            self.video_id = video_id

    fake_cache = types.ModuleType("soundthing.events.tuning.cache")

    def fake_cache_key(config: Any) -> str:
        return "cachekey_abc"

    def fake_cache_path(video_id: str, key: str, root: Path | None = None) -> Path:
        base = Path(root) if root else (tmp_path / "cache")
        return base / key / f"{video_id}.npz"

    def fake_build_cache(
        audio: np.ndarray, sr: int, video_id: str, config: Any, models: Any = None
    ) -> FakeClipCache:
        built_calls.append((video_id, sr, models))
        return FakeClipCache(video_id)

    def fake_save_cache(cache: FakeClipCache, path: Path) -> None:
        saved_calls.append((cache.video_id, Path(path)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("dummy", encoding="utf-8")

    fake_cache.cache_key = fake_cache_key
    fake_cache.cache_path = fake_cache_path
    fake_cache.build_cache = fake_build_cache
    fake_cache.save_cache = fake_save_cache

    monkeypatch.setitem(sys.modules, "soundthing.events.tuning.cache", fake_cache)
    monkeypatch.setattr(tuning_pkg, "cache", fake_cache, raising=False)

    labels_dir = tmp_path / "labels"
    audio_dir = tmp_path / "audio"
    labels_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    for vid in ("clip1", "clip2", "clip_no_audio"):
        save_labels(
            ClipLabels(video_id=vid, duration=5.0, labeler="test", segments=[], events=[]),
            labels_dir / f"{vid}.json",
        )

    sr = 16000
    silent = np.zeros(sr, dtype=np.float32)
    sf.write(str(audio_dir / "clip1.wav"), silent, sr)
    sf.write(str(audio_dir / "clip2.flac"), silent, sr)

    cache_root = tmp_path / "custom_cache"
    sentinel_models = object()

    with pytest.warns(UserWarning, match="1 labels have no matching audio"):
        paths1 = build_dev_caches(
            audio_dir=audio_dir,
            labels_dir=labels_dir,
            cfg=cfg,
            models=sentinel_models,
            root=cache_root,
        )

    assert len(paths1) == 2
    assert [p.stem for p in paths1] == ["clip1", "clip2"]
    assert len(built_calls) == 2
    assert built_calls[0][0] == "clip1"
    assert built_calls[0][2] is sentinel_models
    assert built_calls[1][0] == "clip2"

    # Rebuilding with overwrite=False skips existing
    built_calls.clear()
    paths2 = build_dev_caches(
        audio_dir=audio_dir,
        labels_dir=labels_dir,
        cfg=cfg,
        models=sentinel_models,
        overwrite=False,
        root=cache_root,
    )
    assert paths2 == paths1
    assert len(built_calls) == 0

    # Rebuilding with overwrite=True rebuilds both
    built_calls.clear()
    paths3 = build_dev_caches(
        audio_dir=audio_dir,
        labels_dir=labels_dir,
        cfg=cfg,
        models=sentinel_models,
        overwrite=True,
        root=cache_root,
    )
    assert paths3 == paths1
    assert len(built_calls) == 2

    # Holdout guard: if any id overlaps with holdout, raise ValueError and build nothing

    holdout_dir = holdout.holdout_labels_dir()
    holdout_dir.mkdir(parents=True, exist_ok=True)
    save_labels(
        ClipLabels(video_id="clip1", duration=5.0, labeler="test", segments=[], events=[]),
        holdout_dir / "clip1.json",
    )

    built_calls.clear()
    with pytest.raises(ValueError, match="IDs overlap with holdout split"):
        build_dev_caches(
            audio_dir=audio_dir,
            labels_dir=labels_dir,
            cfg=cfg,
            models=sentinel_models,
            overwrite=True,
            root=cache_root,
        )
    assert len(built_calls) == 0


def test_no_hardcoded_ids_in_datasets_py():
    """Ensure no vocabulary class id or environment id string literals appear in datasets.py."""
    cfg = load_config()
    class_ids = set(cfg.class_ids)
    env_ids = set(cfg.env_ids)

    src_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "soundthing"
        / "events"
        / "tuning"
        / "datasets.py"
    )
    tree = ast.parse(src_path.read_text(encoding="utf-8"))

    found_ids: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            if val in class_ids:
                found_ids.append(f"class_id: {val} at line {node.lineno}")
            if val in env_ids:
                found_ids.append(f"env_id: {val} at line {node.lineno}")

    assert not found_ids, f"Hardcoded vocabulary or environment ids found in datasets.py: {found_ids}"


def test_fsd50k_map_has_no_parent_of_another_mapped_class():
    """Parent labels co-occur with child labels on every child clip (FSD50K propagates)."""
    fsd_map = datasets.load_datasets_yaml()["fsd50k"]["map"]
    for parent in ("Alarm", "Door", "Glass"):
        assert parent not in fsd_map
