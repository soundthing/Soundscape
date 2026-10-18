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
from soundthing.events.cache import (
    CACHE_VERSION,
    ClipCache,
    build_cache,
    cache_key,
    cache_path,
    load_cache,
    save_cache,
)
from soundthing.harness.paths import cache_root


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


class FakeSeparator:
    def __init__(self, multiplier: float = 0.0) -> None:
        self.multiplier = multiplier
        self.call_count = 0

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        self.call_count += 1
        return (audio * self.multiplier).astype(np.float32)


class AudioDependentFakeTagger:
    def __init__(
        self,
        names: list[str],
        gated_windows: set[int] | None = None,
        gate_label: str = "Speech",
        gate_prob: float = 0.9,
    ) -> None:
        self.names = names
        self.gated_windows = gated_windows or set()
        self.gate_idx = names.index(gate_label) if gate_label in names else 0
        self.gate_prob = gate_prob
        self.call_count = 0

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        idx = self.call_count
        self.call_count += 1
        b = audio.shape[0] if audio.ndim > 1 else 1
        mean_amp = float(np.mean(np.abs(audio)))
        probs = np.full((b, len(self.names)), max(0.01, mean_amp * 0.1), dtype=np.float32)
        if idx in self.gated_windows:
            probs[:, self.gate_idx] = self.gate_prob
        return probs


class AudioDependentFakeClap:
    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.call_count = 0

    def embed_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        self.call_count += 1
        mean_amp = float(np.mean(np.abs(audio)))
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[0] = mean_amp + 0.1
        vec[1] = 1.0
        return vec / np.linalg.norm(vec)


@pytest.fixture(autouse=True)
def set_tuning_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SOUNDTHING_HARNESS_HOME", str(tmp_path / "tuning"))


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


def test_save_load_cache_separation_fields(cfg: Config, tmp_path: Path):
    base = ClipCache(
        video_id="clip_sep",
        duration=2.0,
        key=cache_key(cfg),
        window_sec=1.0,
        hop=0.5,
        times=np.array([0.5, 1.0, 1.5]),
        audioset_names=["Label_A", "Label_B"],
        tagger_raw=np.zeros((3, 2), dtype=np.float16),
        clap_emb=None,
        vad_chunk_sec=vad.VAD_CHUNK_SEC,
        vad_probs=np.zeros(4, dtype=np.float32),
    )
    assert base.sep_mask is None and base.tagger_sep is None and base.clap_emb_sep is None

    path_plain = tmp_path / "plain.npz"
    save_cache(base, path_plain)
    plain = load_cache(path_plain)
    assert plain.sep_mask is None and plain.tagger_sep is None and plain.clap_emb_sep is None

    sep = replace(
        base,
        sep_mask=np.array([False, True, True]),
        tagger_sep=np.array([[0.0, 0.0], [0.7, 0.1], [0.6, 0.2]], dtype=np.float16),
        clap_emb_sep=np.array([[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]], dtype=np.float16),
    )
    path_sep = tmp_path / "sep.npz"
    save_cache(sep, path_sep)
    loaded = load_cache(path_sep)
    assert loaded.sep_mask.dtype == bool
    assert np.array_equal(loaded.sep_mask, sep.sep_mask)
    assert loaded.tagger_sep.dtype == np.float16
    assert np.array_equal(loaded.tagger_sep, sep.tagger_sep)
    assert np.array_equal(loaded.clap_emb_sep, sep.clap_emb_sep)


def test_cache_key_separation_dependencies(cfg: Config):
    """Test cache_key dependencies for source separation."""
    repo_key = cache_key(cfg)

    # 1. Unchanged when disabled (compare to cfg without separation block AND to repo cfg)
    p_no_sep = copy.deepcopy(cfg.pipeline)
    p_no_sep["events"].pop("separation", None)
    cfg_no_sep = replace(cfg, pipeline=p_no_sep)
    assert cache_key(cfg_no_sep) == repo_key

    p_disabled = copy.deepcopy(cfg.pipeline)
    p_disabled["events"]["separation"] = {"enabled": False, "model": "other_model"}
    assert cache_key(replace(cfg, pipeline=p_disabled)) == repo_key

    # 2. Changes when enabled
    p_enabled = copy.deepcopy(cfg.pipeline)
    p_enabled["events"]["separation"] = {
        "enabled": True,
        "model": "htdemucs",
        "gate_threshold": 0.3,
        "gate_vad": True,
        "pad_sec": 0.5,
        "crossfade_sec": 0.05,
        "chunk_sec": 10.0,
    }
    cfg_enabled = replace(cfg, pipeline=p_enabled)
    enabled_key = cache_key(cfg_enabled)
    assert enabled_key != repo_key

    # 3. Changes with gate_threshold / model
    p_thr = copy.deepcopy(p_enabled)
    p_thr["events"]["separation"]["gate_threshold"] = 0.5
    assert cache_key(replace(cfg, pipeline=p_thr)) != enabled_key

    p_model = copy.deepcopy(p_enabled)
    p_model["events"]["separation"]["model"] = "mdx_extra"
    assert cache_key(replace(cfg, pipeline=p_model)) != enabled_key

    # 4. Unchanged by replay-time keys (fusion, sep_weight, suppress_under_speech)
    p_fusion = copy.deepcopy(p_enabled)
    p_fusion["events"]["separation"]["fusion"] = "sep"
    assert cache_key(replace(cfg, pipeline=p_fusion)) == enabled_key

    p_weight = copy.deepcopy(p_enabled)
    p_weight["events"]["separation"]["sep_weight"] = 0.8
    assert cache_key(replace(cfg, pipeline=p_weight)) == enabled_key

    p_suppress = copy.deepcopy(p_enabled)
    p_suppress["events"]["separation"]["suppress_under_speech"] = 0.5
    assert cache_key(replace(cfg, pipeline=p_suppress)) == enabled_key


def test_build_cache_separation_disabled(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    """build_cache with separation disabled leaves sep fields as None."""
    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    tagger_map = build_tagger_map(cfg, audioset_names)
    separator = FakeSeparator()
    models = EventModels(
        tagger=FakeTagger(audioset_names),
        clap=FakeClap(dim=16),
        class_ids=list(cfg.class_ids),
        tagger_map=tagger_map,
        clap_text_emb=None,
        audioset_names=audioset_names,
        separator=separator,
    )
    monkeypatch.setattr(vad, "speech_probs", lambda audio, sr, c: np.array([], dtype=np.float32))

    sr = 16000
    audio = np.ones(int(2.0 * sr), dtype=np.float32) * 0.5
    cache = build_cache(audio, sr, "clip_disabled", cfg, models=models)

    assert cache.sep_mask is None
    assert cache.tagger_sep is None
    assert cache.clap_emb_sep is None
    assert separator.call_count == 0


def test_build_cache_separation_enabled(cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """build_cache rescores gated windows on separated audio."""
    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    gate_label = "Speech" if "Speech" in audioset_names else audioset_names[0]
    tagger_map = build_tagger_map(cfg, audioset_names)

    sr = 16000
    duration = 4.0
    audio = np.ones(int(duration * sr), dtype=np.float32) * 0.5

    p_sep = copy.deepcopy(cfg.pipeline)
    p_sep["events"]["window_sec"] = 1.0
    p_sep["events"]["hop_sec"] = 0.5
    p_sep["events"]["separation"] = {
        "enabled": True,
        "model": "htdemucs",
        "gate_labels": [gate_label],
        "gate_threshold": 0.3,
        "gate_vad": False,
        "pad_sec": 0.5,
        "crossfade_sec": 0.05,
        "chunk_sec": 10.0,
    }
    cfg_sep = replace(cfg, pipeline=p_sep)

    monkeypatch.setattr(vad, "speech_probs", lambda a, s, c: np.array([], dtype=np.float32))

    separator = FakeSeparator(multiplier=0.0)
    tagger = AudioDependentFakeTagger(audioset_names, gated_windows={3}, gate_label=gate_label, gate_prob=0.9)
    clap = AudioDependentFakeClap(dim=16)

    models = EventModels(
        tagger=tagger,
        clap=clap,
        class_ids=list(cfg_sep.class_ids),
        tagger_map=tagger_map,
        clap_text_emb=None,
        audioset_names=audioset_names,
        separator=separator,
    )

    cache = build_cache(audio, sr, "clip_sep_enabled", cfg_sep, models=models)

    # Separator called exactly once per clip
    assert separator.call_count == 1

    # sep_mask True only for windows fully inside padded span [1.0, 3.0]
    expected_mask = np.array([False, False, True, True, True, False, False], dtype=bool)
    assert cache.sep_mask is not None
    assert cache.sep_mask.dtype == bool
    assert np.array_equal(cache.sep_mask, expected_mask)

    # tagger_sep differs exactly on masked rows; unmasked rows are equal
    assert cache.tagger_sep is not None
    assert cache.tagger_sep.dtype == np.float16
    for i in range(len(cache.times)):
        if cache.sep_mask[i]:
            assert not np.array_equal(cache.tagger_sep[i], cache.tagger_raw[i])
        else:
            assert np.array_equal(cache.tagger_sep[i], cache.tagger_raw[i])

    # clap_emb_sep differs exactly on masked rows when CLAP is on
    assert cache.clap_emb is not None
    assert cache.clap_emb_sep is not None
    assert cache.clap_emb_sep.dtype == np.float16
    for i in range(len(cache.times)):
        if cache.sep_mask[i]:
            assert not np.array_equal(cache.clap_emb_sep[i], cache.clap_emb[i])
        else:
            assert np.array_equal(cache.clap_emb_sep[i], cache.clap_emb[i])

    # Save/load round trip preserves all separation fields
    cache_file = tmp_path / "round_trip.npz"
    save_cache(cache, cache_file)
    loaded = load_cache(cache_file)
    assert loaded.sep_mask.dtype == bool
    assert np.array_equal(loaded.sep_mask, cache.sep_mask)
    assert loaded.tagger_sep.dtype == np.float16
    assert np.array_equal(loaded.tagger_sep, cache.tagger_sep)
    assert loaded.clap_emb_sep.dtype == np.float16
    assert np.array_equal(loaded.clap_emb_sep, cache.clap_emb_sep)

    # When CLAP is off, clap_emb_sep is None
    separator_no_clap = FakeSeparator(multiplier=0.0)
    tagger_no_clap = AudioDependentFakeTagger(audioset_names, gated_windows={3}, gate_label=gate_label, gate_prob=0.9)
    models_no_clap = replace(models, clap=None, tagger=tagger_no_clap, separator=separator_no_clap)
    cache_no_clap = build_cache(audio, sr, "clip_no_clap", cfg_sep, models=models_no_clap)
    assert cache_no_clap.clap_emb is None
    assert cache_no_clap.clap_emb_sep is None
    assert cache_no_clap.sep_mask is not None
    assert np.array_equal(cache_no_clap.sep_mask, expected_mask)


def test_build_cache_separation_no_gated_windows(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    """When enabled but no windows are gated: sep_mask all False, separator not called, tagger_sep == tagger_raw."""
    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    tagger_map = build_tagger_map(cfg, audioset_names)

    p_sep = copy.deepcopy(cfg.pipeline)
    p_sep["events"]["separation"] = {
        "enabled": True,
        "model": "htdemucs",
        "gate_labels": ["Speech"],
        "gate_threshold": 0.5,
        "gate_vad": False,
    }
    cfg_sep = replace(cfg, pipeline=p_sep)

    monkeypatch.setattr(vad, "speech_probs", lambda a, s, c: np.array([], dtype=np.float32))

    separator = FakeSeparator()
    tagger = FakeTagger(audioset_names, default_prob=0.1)
    models = EventModels(
        tagger=tagger,
        clap=FakeClap(dim=16),
        class_ids=list(cfg_sep.class_ids),
        tagger_map=tagger_map,
        clap_text_emb=None,
        audioset_names=audioset_names,
        separator=separator,
    )

    sr = 16000
    audio = np.ones(int(2.0 * sr), dtype=np.float32)
    cache = build_cache(audio, sr, "clip_no_gates", cfg_sep, models=models)

    assert cache.sep_mask is not None
    assert not cache.sep_mask.any()
    assert separator.call_count == 0
    assert np.array_equal(cache.tagger_sep, cache.tagger_raw)
    assert np.array_equal(cache.clap_emb_sep, cache.clap_emb)


def test_build_cache_separation_missing_separator_raises(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    """When separation is enabled but models.separator is None, build_cache raises ValueError."""
    monkeypatch.setattr(vad, "speech_probs", lambda a, s, c: np.array([], dtype=np.float32))

    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    tagger_map = build_tagger_map(cfg, audioset_names)

    p_sep = copy.deepcopy(cfg.pipeline)
    p_sep["events"]["separation"] = {"enabled": True}
    cfg_sep = replace(cfg, pipeline=p_sep)

    models = EventModels(
        tagger=FakeTagger(audioset_names),
        clap=None,
        class_ids=list(cfg_sep.class_ids),
        tagger_map=tagger_map,
        clap_text_emb=None,
        audioset_names=audioset_names,
        separator=None,
    )

    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    with pytest.raises(ValueError, match="separator"):
        build_cache(audio, sr, "clip_err", cfg_sep, models=models)
