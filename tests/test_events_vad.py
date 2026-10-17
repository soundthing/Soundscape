from __future__ import annotations

import copy
import sys
import types
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest

from soundthing.events import vad


def test_empty_audio_returns_empty(cfg):
    assert vad.speech_regions(np.array([], dtype=np.float32), 16000, cfg) == []
    assert vad.speech_regions(np.zeros((0, 2), dtype=np.float32), 16000, cfg) == []


def test_model_none_never_calls_loader(cfg, monkeypatch):
    loader_called = False

    def fake_loader():
        nonlocal loader_called
        loader_called = True
        return None

    monkeypatch.setattr(vad, "_load_silero", fake_loader)

    pipeline = copy.deepcopy(cfg.pipeline)
    pipeline["events"]["vad"]["model"] = "none"
    cfg_none = replace(cfg, pipeline=pipeline)

    audio = np.zeros(16000, dtype=np.float32)
    regions = vad.speech_regions(audio, 16000, cfg_none)
    assert regions == []
    assert not loader_called


def test_unknown_model_raises_value_error(cfg):
    pipeline = copy.deepcopy(cfg.pipeline)
    pipeline["events"]["vad"]["model"] = "unsupported_model"
    cfg_unknown = replace(cfg, pipeline=pipeline)

    audio = np.zeros(16000, dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown VAD model"):
        vad.speech_regions(audio, 16000, cfg_unknown)


def test_stereo_input_averaged(cfg, monkeypatch):
    monkeypatch.setattr(vad, "_load_silero", lambda: "fake_model")
    recorded: list[np.ndarray] = []

    def fake_probs(model, audio16k):
        recorded.append(audio16k)
        return np.zeros(0, dtype=np.float32)

    monkeypatch.setattr(vad, "_silero_probs", fake_probs)

    # 1. Channels-last: shape (16000, 2)
    ch1 = np.ones(16000, dtype=np.float32) * 0.2
    ch2 = np.ones(16000, dtype=np.float32) * 0.8
    stereo_last = np.column_stack([ch1, ch2])
    vad.speech_regions(stereo_last, 16000, cfg)

    assert len(recorded) == 1
    assert recorded[0].ndim == 1
    assert recorded[0].shape == (16000,)
    assert np.allclose(recorded[0], 0.5)

    # 2. Channels-first: shape (2, 16000)
    stereo_first = np.stack([ch1, ch2], axis=0)
    vad.speech_regions(stereo_first, 16000, cfg)

    assert len(recorded) == 2
    assert recorded[1].ndim == 1
    assert recorded[1].shape == (16000,)
    assert np.allclose(recorded[1], 0.5)


def test_fake_timestamps_padding_and_clipping(cfg, monkeypatch):
    monkeypatch.setattr(vad, "_load_silero", lambda: "fake_model")
    # 10s audio duration
    audio = np.zeros(16000 * 10, dtype=np.float32)

    # 10s audio: 313 chunks of 512 samples.
    # Chunk 1..31 (0.032 to 0.992) -> padded by 0.1 -> [0.0, 1.092]
    # Chunk 297..312 (9.504 to 9.984) -> padded by 0.1 -> [9.404, 10.0]
    probs = np.zeros(313, dtype=np.float32)
    probs[1:31] = 0.9
    probs[297:312] = 0.9
    monkeypatch.setattr(vad, "_silero_probs", lambda model, audio16k: probs)

    regions = vad.speech_regions(audio, 16000, cfg)
    # 0.032 - 0.1 = -0.068 -> clipped to 0.0; 0.992 + 0.1 = 1.092
    # 9.504 - 0.1 = 9.404; 9.984 + 0.1 = 10.084 -> clipped to 10.0
    assert regions == [(0.0, 1.092), (9.404, 10.0)]


def test_fake_timestamps_merging_and_separation(cfg, monkeypatch):
    monkeypatch.setattr(vad, "_load_silero", lambda: "fake_model")
    audio = np.zeros(16000 * 10, dtype=np.float32)

    # pad_sec=0.1, merge_gap_sec=0.3
    # Span 1: chunks 31..62 (0.992 to 1.984) -> padded: [0.892, 2.084]
    # Span 2: chunks 68..93 (2.176 to 2.976) -> padded: [2.076, 3.076]
    # Gap between them = 2.076 - 2.084 < 0 -> merges into [0.892, 3.076]
    # Span 3: chunks 156..187 (4.992 to 5.984) -> padded: [4.892, 6.084]
    # Gap from 3.076 to 4.892 >= 0.3 -> stays separate
    probs = np.zeros(313, dtype=np.float32)
    probs[31:62] = 0.9
    probs[68:93] = 0.9
    probs[156:187] = 0.9
    monkeypatch.setattr(vad, "_silero_probs", lambda m, a: probs)

    regions = vad.speech_regions(audio, 16000, cfg)
    assert len(regions) == 2
    assert regions[0] == (0.892, 3.076)
    assert regions[1] == (4.892, 6.084)


def test_fake_timestamps_sorted_output(cfg, monkeypatch):
    monkeypatch.setattr(vad, "_load_silero", lambda: "fake_model")
    audio = np.zeros(16000 * 10, dtype=np.float32)

    probs = np.zeros(313, dtype=np.float32)
    probs[31:62] = 0.9
    probs[156:187] = 0.9
    monkeypatch.setattr(vad, "_silero_probs", lambda m, a: probs)

    regions = vad.speech_regions(audio, 16000, cfg)
    assert regions == [(0.892, 2.084), (4.892, 6.084)]


def test_loader_cached_across_calls(cfg, monkeypatch):
    vad._load_silero.cache_clear()
    load_mock = mock.Mock(return_value="fake_model")
    fake_module = types.ModuleType("silero_vad")
    fake_module.load_silero_vad = load_mock
    monkeypatch.setitem(sys.modules, "silero_vad", fake_module)
    monkeypatch.setattr(vad, "_silero_probs", lambda *args, **kwargs: np.zeros(0, dtype=np.float32))

    audio = np.zeros(16000, dtype=np.float32)
    res1 = vad.speech_regions(audio, 16000, cfg)
    res2 = vad.speech_regions(audio, 16000, cfg)

    assert res1 == []
    assert res2 == []
    assert load_mock.call_count == 1
    vad._load_silero.cache_clear()


def test_speech_mask_integration(cfg, monkeypatch):
    monkeypatch.setattr(vad, "_load_silero", lambda: "fake_model")
    audio = np.zeros(16000 * 6, dtype=np.float32)

    # 6s audio = 188 chunks
    # Speech span around 2.0 to 4.0: chunks 63 to 125
    probs = np.zeros(188, dtype=np.float32)
    probs[63:125] = 1.0
    monkeypatch.setattr(vad, "_silero_probs", lambda m, a: probs)
    regions = vad.speech_regions(audio, 16000, cfg)
    assert len(regions) == 1
    s, e = regions[0]
    assert abs(s - 1.916) < 1e-3
    assert abs(e - 4.1) < 1e-3

    times = np.arange(0.0, 6.0, 0.5)
    mask = vad.speech_mask(times, regions)

    expected = (times >= s) & (times < e)
    assert np.array_equal(mask, expected)
    assert mask.dtype == bool


def test_resample_on_non_16k_sr(cfg, monkeypatch):
    monkeypatch.setattr(vad, "_load_silero", lambda: "fake_model")
    resample_called = False

    def fake_resample(audio, orig_sr, target_sr=16000):
        nonlocal resample_called
        resample_called = True
        assert orig_sr == 48000
        assert target_sr == 16000
        return np.zeros(16000, dtype=np.float32)

    monkeypatch.setattr(vad, "_resample", fake_resample)
    monkeypatch.setattr(vad, "_silero_probs", lambda m, a: np.zeros(0, dtype=np.float32))

    audio48k = np.zeros(48000, dtype=np.float32)
    vad.speech_regions(audio48k, 48000, cfg)
    assert resample_called


def test_speech_regions_from_probs_hysteresis_and_min_speech(cfg):
    """Test hysteresis on/off and min_speech filtering in speech_regions_from_probs."""
    chunk_sec = 0.1
    duration = 5.0

    # 1. Short burst: 2 chunks (0.2 s) >= 0.5 -> shorter than 0.25 min_speech -> dropped!
    short_burst = np.array([0.1, 0.8, 0.8, 0.1, 0.1], dtype=np.float32)
    assert vad.speech_regions_from_probs(short_burst, chunk_sec, duration, cfg) == []

    # 2. Hysteresis on and off:
    # chunk 0: 0.1 (off)
    # chunk 1: 0.6 (turns ON: start = 1*0.1 = 0.1)
    # chunk 2: 0.4 (>= 0.35 -> stays ON)
    # chunk 3: 0.35 (>= 0.35 -> stays ON)
    # chunk 4: 0.30 (< 0.35 -> turns OFF: end = 4*0.1 = 0.4)
    # Raw span: [0.1, 0.4) length 0.3s >= 0.25s min_speech.
    # Padded by pad_sec=0.1: [0.0, 0.5]
    hyst_probs = np.array([0.1, 0.6, 0.4, 0.35, 0.30, 0.1], dtype=np.float32)
    regions = vad.speech_regions_from_probs(hyst_probs, chunk_sec, duration, cfg)
    assert regions == [(0.0, 0.5)]

    # 3. Model none and empty probs return empty
    p_none = copy.deepcopy(cfg.pipeline)
    p_none["events"]["vad"]["model"] = "none"
    cfg_none = replace(cfg, pipeline=p_none)
    assert vad.speech_regions_from_probs(hyst_probs, chunk_sec, duration, cfg_none) == []
    assert vad.speech_regions_from_probs(np.array([], dtype=np.float32), chunk_sec, duration, cfg) == []


def test_real_model_guarded(cfg):
    pytest.importorskip("silero_vad")
    pytest.importorskip("torch")

    audio = np.zeros(16000, dtype=np.float32)
    regions = vad.speech_regions(audio, 16000, cfg)
    assert isinstance(regions, list)
