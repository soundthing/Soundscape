"""Tests for synthetic dev mixes from ESC-50 and dataset YAML."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from soundthing.config import load_config
from soundthing.contracts import load_labels
from soundthing.events.tuning import datasets
from soundthing.events.tuning.datasets import (
    _load_mono_16k,
    _trim_silence,
    esc50_mixes,
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
