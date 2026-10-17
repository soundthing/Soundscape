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

    def fake_timestamps(model, audio16k, threshold, min_speech_sec):
        recorded.append(audio16k)
        return []

    monkeypatch.setattr(vad, "_silero_timestamps", fake_timestamps)

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

    # Raw span near 0 and near 10.0; default pad_sec is 0.1
    raw_spans = [(0.05, 1.0), (9.5, 9.95)]
    monkeypatch.setattr(
        vad,
        "_silero_timestamps",
        lambda model, audio16k, threshold, min_speech_sec: raw_spans,
    )

    regions = vad.speech_regions(audio, 16000, cfg)
    # 0.05 - 0.1 = -0.05 -> clipped to 0.0; 1.0 + 0.1 = 1.1
    # 9.5 - 0.1 = 9.4; 9.95 + 0.1 = 10.05 -> clipped to 10.0
    assert regions == [(0.0, 1.1), (9.4, 10.0)]


def test_fake_timestamps_merging_and_separation(cfg, monkeypatch):
    monkeypatch.setattr(vad, "_load_silero", lambda: "fake_model")
    audio = np.zeros(16000 * 10, dtype=np.float32)

    # pad_sec=0.1, merge_gap_sec=0.3
    # Span 1 (1.0, 2.0) -> padded: [0.9, 2.1]
    # Span 2 (2.2, 3.0) -> padded: [2.1, 3.1]
    # Gap between them = 2.1 - 2.1 = 0.0 < 0.3 -> merges into [0.9, 3.1]
    # Span 3 (5.0, 6.0) -> padded: [4.9, 6.1]
    # Gap from 3.1 to 4.9 = 1.8 >= 0.3 -> stays separate
    # Span 4 (7.0, 8.0) -> padded: [6.9, 8.1]
    # Span 5 (7.5, 8.5) -> padded: [7.4, 8.6]
    # Overlapping after padding -> merges into [6.9, 8.6]
    spans = [(1.0, 2.0), (2.2, 3.0), (5.0, 6.0), (7.0, 8.0), (7.5, 8.5)]
    monkeypatch.setattr(vad, "_silero_timestamps", lambda m, a, t, min_s: spans)

    regions = vad.speech_regions(audio, 16000, cfg)
    assert regions == [(0.9, 3.1), (4.9, 6.1), (6.9, 8.6)]


def test_fake_timestamps_sorted_output(cfg, monkeypatch):
    monkeypatch.setattr(vad, "_load_silero", lambda: "fake_model")
    audio = np.zeros(16000 * 10, dtype=np.float32)

    # Raw spans deliberately in reverse order
    spans = [(5.0, 6.0), (1.0, 2.0)]
    monkeypatch.setattr(vad, "_silero_timestamps", lambda m, a, t, min_s: spans)

    regions = vad.speech_regions(audio, 16000, cfg)
    assert regions == [(0.9, 2.1), (4.9, 6.1)]


def test_loader_cached_across_calls(cfg, monkeypatch):
    vad._load_silero.cache_clear()
    load_mock = mock.Mock(return_value="fake_model")
    fake_module = types.ModuleType("silero_vad")
    fake_module.load_silero_vad = load_mock
    monkeypatch.setitem(sys.modules, "silero_vad", fake_module)
    monkeypatch.setattr(vad, "_silero_timestamps", lambda *args, **kwargs: [])

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

    # Raw span: (2.0, 4.0) -> padded: (1.9, 4.1)
    monkeypatch.setattr(vad, "_silero_timestamps", lambda m, a, t, min_s: [(2.0, 4.0)])
    regions = vad.speech_regions(audio, 16000, cfg)
    assert regions == [(1.9, 4.1)]

    times = np.arange(0.0, 6.0, 0.5)
    mask = vad.speech_mask(times, regions)

    expected = (times >= 1.9) & (times < 4.1)
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
    monkeypatch.setattr(vad, "_silero_timestamps", lambda m, a, t, min_s: [])

    audio48k = np.zeros(48000, dtype=np.float32)
    vad.speech_regions(audio48k, 48000, cfg)
    assert resample_called


def test_real_model_guarded(cfg):
    pytest.importorskip("silero_vad")
    pytest.importorskip("torch")

    audio = np.zeros(16000, dtype=np.float32)
    regions = vad.speech_regions(audio, 16000, cfg)
    assert isinstance(regions, list)
