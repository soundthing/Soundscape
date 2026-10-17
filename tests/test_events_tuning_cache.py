from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from soundthing.config import Config
from soundthing.events import vad
from soundthing.events.core import EventModels
from soundthing.events.tagger import build_tagger_map
from soundthing.events.tuning.cache import (
    CACHE_VERSION,
    ClipCache,
    build_cache,
    cache_key,
    cache_path,
    load_cache,
    save_cache,
)
from soundthing.events.tuning.paths import cache_root


class FakeTagger:
    def __init__(self, names: list[str], default_prob: float = 0.1) -> None:
        self.names = names
        self.default_prob = default_prob

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        b = audio.shape[0] if audio.ndim > 1 else 1
        return np.full((b, len(self.names)), self.default_prob, dtype=np.float32)


class FakeClap:
    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def embed_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        vec = np.ones(self.dim, dtype=np.float32)
        return vec / np.linalg.norm(vec)


@pytest.fixture(autouse=True)
def set_tuning_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SOUNDTHING_TUNING_HOME", str(tmp_path / "tuning"))


def test_cache_key_dependencies(cfg: Config):
    base_key = cache_key(cfg)
    assert isinstance(base_key, str)
    assert len(base_key) == 12

    # 1. Must change when cache-affecting keys change:
    # window_sec
    p = copy.deepcopy(cfg.pipeline)
    p["events"]["window_sec"] = float(p["events"]["window_sec"]) + 0.5
    assert cache_key(replace(cfg, pipeline=p)) != base_key

    # hop_sec
    p = copy.deepcopy(cfg.pipeline)
    p["events"]["hop_sec"] = float(p["events"]["hop_sec"]) + 0.1
    assert cache_key(replace(cfg, pipeline=p)) != base_key

    # tagger model
    p = copy.deepcopy(cfg.pipeline)
    p["events"]["tagger"]["model"] = "other_tagger"
    assert cache_key(replace(cfg, pipeline=p)) != base_key

    # clap model
    p = copy.deepcopy(cfg.pipeline)
    p["events"]["clap"]["model"] = "other_clap"
    assert cache_key(replace(cfg, pipeline=p)) != base_key

    # clap enabled
    p = copy.deepcopy(cfg.pipeline)
    p["events"]["clap"]["enabled"] = not p["events"]["clap"].get("enabled", True)
    assert cache_key(replace(cfg, pipeline=p)) != base_key

    # vad model
    p = copy.deepcopy(cfg.pipeline)
    p["events"]["vad"]["model"] = "none"
    assert cache_key(replace(cfg, pipeline=p)) != base_key

    # 2. Must NOT change with postprocess, thresholds, calibration, priors, etc.:
    # thresholds
    thr = copy.deepcopy(cfg.thresholds)
    thr.default = 0.99
    assert cache_key(replace(cfg, thresholds=thr)) == base_key

    # calibration
    p = copy.deepcopy(cfg.pipeline)
    p["events"]["clap"]["calibration"]["center"] = 0.99
    p["events"]["clap"]["calibration"]["scale"] = 99.0
    assert cache_key(replace(cfg, pipeline=p)) == base_key

    # vad threshold, pad, merge, suppress
    p = copy.deepcopy(cfg.pipeline)
    p["events"]["vad"]["threshold"] = 0.99
    p["events"]["vad"]["pad_sec"] = 0.99
    p["events"]["vad"]["merge_gap_sec"] = 0.99
    p["events"]["vad"]["suppress_under_speech"] = 0.99
    assert cache_key(replace(cfg, pipeline=p)) == base_key

    # postprocess (smoothing_sec, onset_shift_sec, coarse_margin, min_event_sec)
    p = copy.deepcopy(cfg.pipeline)
    p["events"]["postprocess"]["smoothing_sec"] = 2.0
    p["events"]["postprocess"]["onset_shift_sec"] = 0.5
    assert cache_key(replace(cfg, pipeline=p)) == base_key

    # priors
    pri = copy.deepcopy(cfg.priors)
    pri.floor = 0.99
    assert cache_key(replace(cfg, priors=pri)) == base_key


def test_cache_path_layout(tmp_path: Path):
    key = "abcdef123456"
    vid = "clip01"

    expected_default = cache_root() / key / f"{vid}.npz"
    assert cache_path(vid, key) == expected_default

    custom_root = tmp_path / "custom_cache"
    assert cache_path(vid, key, root=custom_root) == custom_root / key / f"{vid}.npz"


def test_build_cache_structure_and_dtypes(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    tagger_map = build_tagger_map(cfg, audioset_names)
    models = EventModels(
        tagger=FakeTagger(audioset_names, default_prob=0.2),
        clap=FakeClap(dim=16),
        class_ids=list(cfg.class_ids),
        tagger_map=tagger_map,
        clap_text_emb=None,
        audioset_names=audioset_names,
    )

    fake_vad_probs = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    monkeypatch.setattr(vad, "speech_probs", lambda audio, sr, c: fake_vad_probs)

    sr = 16000
    duration = 2.0
    audio = np.zeros(int(duration * sr), dtype=np.float32)

    cache = build_cache(audio, sr, "test_clip", cfg, models=models)

    assert cache.video_id == "test_clip"
    assert cache.duration == pytest.approx(duration)
    assert cache.key == cache_key(cfg)
    assert cache.window_sec == cfg.pipeline["events"]["window_sec"]
    assert cache.hop == cfg.pipeline["events"]["hop_sec"]
    assert cache.audioset_names == audioset_names
    assert cache.vad_chunk_sec == vad.VAD_CHUNK_SEC

    assert cache.tagger_raw.dtype == np.float16
    assert cache.clap_emb.dtype == np.float16
    assert cache.vad_probs.dtype == np.float32
    assert np.array_equal(cache.vad_probs, fake_vad_probs)

    win = int(cache.window_sec * sr)
    hop = int(cache.hop * sr)
    expected_n_windows = len(range(0, max(1, len(audio) - win + 1), hop))
    assert len(cache.times) == expected_n_windows
    assert cache.tagger_raw.shape == (expected_n_windows, len(audioset_names))
    assert cache.clap_emb.shape == (expected_n_windows, 16)

    models_no_clap = replace(models, clap=None)
    cache_no_clap = build_cache(audio, sr, "test_clip", cfg, models=models_no_clap)
    assert cache_no_clap.clap_emb is None

    models_no_names = replace(models, audioset_names=None)
    with pytest.raises(ValueError, match="audioset_names"):
        build_cache(audio, sr, "test_clip", cfg, models=models_no_names)


def test_save_load_cache_round_trip(cfg: Config, tmp_path: Path):
    times = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    tagger_raw = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float16)
    clap_emb = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], dtype=np.float16)
    vad_probs = np.array([0.05, 0.85, 0.1], dtype=np.float32)

    cache = ClipCache(
        video_id="clip_rt",
        duration=3.0,
        key=cache_key(cfg),
        window_sec=1.0,
        hop=0.25,
        times=times,
        audioset_names=["Label_A", "Label_B"],
        tagger_raw=tagger_raw,
        clap_emb=clap_emb,
        vad_chunk_sec=vad.VAD_CHUNK_SEC,
        vad_probs=vad_probs,
    )

    path = tmp_path / "cache.npz"
    save_cache(cache, path)
    assert path.exists()

    loaded = load_cache(path)
    assert loaded.video_id == cache.video_id
    assert loaded.duration == cache.duration
    assert loaded.key == cache.key
    assert loaded.window_sec == cache.window_sec
    assert loaded.hop == cache.hop
    assert loaded.audioset_names == cache.audioset_names
    assert loaded.vad_chunk_sec == cache.vad_chunk_sec

    assert np.array_equal(loaded.times, cache.times)
    assert np.array_equal(loaded.tagger_raw, cache.tagger_raw)
    assert np.array_equal(loaded.clap_emb, cache.clap_emb)
    assert np.array_equal(loaded.vad_probs, cache.vad_probs)

    assert loaded.tagger_raw.dtype == np.float16
    assert loaded.clap_emb.dtype == np.float16
    assert loaded.vad_probs.dtype == np.float32

    cache_no_clap = replace(cache, clap_emb=None)
    path_no_clap = tmp_path / "cache_no_clap.npz"
    save_cache(cache_no_clap, path_no_clap)
    loaded_no_clap = load_cache(path_no_clap)
    assert loaded_no_clap.clap_emb is None

    bad_meta = {
        "video_id": "bad",
        "duration": 1.0,
        "key": "bad_key",
        "window_sec": 1.0,
        "hop": 0.25,
        "audioset_names": [],
        "vad_chunk_sec": 0.032,
        "has_clap": False,
        "cache_version": 999,
    }
    bad_path = tmp_path / "bad_version.npz"
    np.savez_compressed(
        bad_path,
        meta=np.array(json.dumps(bad_meta)),
        times=np.array([0.5]),
        tagger_raw=np.zeros((1, 0), dtype=np.float16),
        vad_probs=np.zeros((1,), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="Cache version mismatch"):
        load_cache(bad_path)
