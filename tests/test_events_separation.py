"""Tests for soundthing.events.separation."""

from __future__ import annotations

import copy
from dataclasses import replace
import sys
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from soundthing.config import Config
from soundthing.events import vad
from soundthing.events.separation import (
    DEFAULTS,
    DemucsSeparator,
    gate_spans,
    is_enabled,
    load_separator,
    sep_config,
    separate_spans,
    window_mask,
)


def test_sep_config_defaults_and_overlay(cfg: Config):
    """Verify DEFAULTS, overlaying with cfg, and immutability of DEFAULTS."""
    # Check default dictionary values and types
    assert DEFAULTS["enabled"] is False
    assert DEFAULTS["model"] == "htdemucs"
    assert DEFAULTS["gate_labels"] == [
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Conversation",
        "Narration, monologue",
        "Music",
        "Singing",
        "Song",
    ]
    assert DEFAULTS["gate_threshold"] == 0.3
    assert DEFAULTS["gate_vad"] is True
    assert DEFAULTS["pad_sec"] == 0.5
    assert DEFAULTS["crossfade_sec"] == 0.05
    assert DEFAULTS["chunk_sec"] == 10.0
    assert DEFAULTS["fusion"] == "max"
    assert DEFAULTS["sep_weight"] == 0.5
    assert DEFAULTS["suppress_under_speech"] == 1.0

    # Default extraction
    conf = sep_config(cfg)
    assert conf == DEFAULTS
    assert is_enabled(cfg) is False

    # Overlay
    p = copy.deepcopy(cfg.pipeline)
    p["events"]["separation"] = {
        "enabled": True,
        "model": "custom_demucs",
        "gate_labels": ["Speech", "Music"],
        "chunk_sec": 4.0,
    }
    cfg_over = replace(cfg, pipeline=p)
    conf_over = sep_config(cfg_over)
    assert conf_over["enabled"] is True
    assert conf_over["model"] == "custom_demucs"
    assert conf_over["gate_labels"] == ["Speech", "Music"]
    assert conf_over["chunk_sec"] == 4.0
    assert conf_over["gate_threshold"] == 0.3  # inherited default
    assert is_enabled(cfg_over) is True

    # Mutating returned config must not mutate DEFAULTS
    conf_over["gate_labels"].append("New_Label")
    assert "New_Label" not in DEFAULTS["gate_labels"]
    assert DEFAULTS["enabled"] is False


def test_gate_spans_threshold_and_unknown_labels(cfg: Config):
    """Test label threshold gating and unknown labels ignored."""
    audioset_names = ["Speech", "Music", "Bark", "Whistle"]
    times = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    window_sec = 1.0
    duration = 5.0

    p = copy.deepcopy(cfg.pipeline)
    p["events"]["separation"] = {
        "gate_labels": ["Speech", "Music", "NonExistentLabel"],
        "gate_threshold": 0.3,
        "gate_vad": False,
        "pad_sec": 0.0,
    }
    test_cfg = replace(cfg, pipeline=p)

    # Window 0: max in gate labels is 0.2 < 0.3 -> not gated
    # Window 1: Speech prob is 0.4 >= 0.3 -> gated [1.5, 2.5]
    # Window 2: Bark prob is 0.9 (high), but Bark is not in gate_labels -> not gated
    tagger_raw = np.array(
        [
            [0.2, 0.1, 0.0, 0.0],
            [0.4, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.0],
        ],
        dtype=np.float16,
    )
    vad_probs = np.zeros(10, dtype=np.float32)

    spans = gate_spans(
        times,
        tagger_raw,
        audioset_names,
        vad_probs,
        vad_chunk_sec=0.032,
        duration=duration,
        window_sec=window_sec,
        cfg=test_cfg,
    )
    assert len(spans) == 1
    assert spans[0] == (1.5, 2.5)


def test_gate_spans_padding_clipping_and_merging(cfg: Config):
    """Test padding, clipping to [0, duration], and merging touching/overlapping spans."""
    audioset_names = ["Speech"]
    times = np.array([0.2, 1.2, 9.8], dtype=np.float64)
    window_sec = 1.0
    duration = 10.0

    p = copy.deepcopy(cfg.pipeline)
    p["events"]["separation"] = {
        "gate_labels": ["Speech"],
        "gate_threshold": 0.3,
        "gate_vad": False,
        "pad_sec": 0.5,
    }
    test_cfg = replace(cfg, pipeline=p)

    # All 3 windows trigger
    # Win 0: time 0.2 -> span [-0.3, 0.7], padded by 0.5 -> [-0.8, 1.2], clipped -> [0.0, 1.2]
    # Win 1: time 1.2 -> span [0.7, 1.7], padded by 0.5 -> [0.2, 2.2], clipped -> [0.2, 2.2]
    # Win 0 and Win 1 overlap -> merged [0.0, 2.2]
    # Win 2: time 9.8 -> span [9.3, 10.3], padded by 0.5 -> [8.8, 10.8], clipped -> [8.8, 10.0]
    tagger_raw = np.full((3, 1), 0.5, dtype=np.float16)
    vad_probs = np.zeros(10, dtype=np.float32)

    spans = gate_spans(
        times,
        tagger_raw,
        audioset_names,
        vad_probs,
        vad_chunk_sec=0.032,
        duration=duration,
        window_sec=window_sec,
        cfg=test_cfg,
    )
    assert len(spans) == 2
    assert spans[0] == (0.0, 2.2)
    assert spans[1] == (8.8, 10.0)


def test_gate_spans_gate_vad_on_vs_off(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    """Test gate_vad toggle and inclusion of VAD speech regions."""
    audioset_names = ["Speech"]
    times = np.array([2.0], dtype=np.float64)
    # Tagger below threshold
    tagger_raw = np.array([[0.1]], dtype=np.float16)
    duration = 5.0

    # Mock vad.speech_regions_from_probs to return a clear speech region
    fake_speech_regions = [(1.0, 2.0)]
    monkeypatch.setattr(
        vad, "speech_regions_from_probs", lambda probs, chunk_sec, dur, c: fake_speech_regions
    )
    fake_vad_probs = np.ones(20, dtype=np.float32)

    # When gate_vad is False:
    p_off = copy.deepcopy(cfg.pipeline)
    p_off["events"]["separation"] = {
        "gate_labels": ["Speech"],
        "gate_threshold": 0.3,
        "gate_vad": False,
        "pad_sec": 0.1,
    }
    cfg_off = replace(cfg, pipeline=p_off)
    spans_off = gate_spans(
        times,
        tagger_raw,
        audioset_names,
        fake_vad_probs,
        vad_chunk_sec=0.032,
        duration=duration,
        window_sec=1.0,
        cfg=cfg_off,
    )
    assert spans_off == []

    # When gate_vad is True:
    p_on = copy.deepcopy(cfg.pipeline)
    p_on["events"]["separation"] = {
        "gate_labels": ["Speech"],
        "gate_threshold": 0.3,
        "gate_vad": True,
        "pad_sec": 0.1,
    }
    cfg_on = replace(cfg, pipeline=p_on)
    spans_on = gate_spans(
        times,
        tagger_raw,
        audioset_names,
        fake_vad_probs,
        vad_chunk_sec=0.032,
        duration=duration,
        window_sec=1.0,
        cfg=cfg_on,
    )
    # fake region [1.0, 2.0] padded by 0.1 -> [0.9, 2.1]
    assert len(spans_on) == 1
    assert spans_on[0] == (pytest.approx(0.9), pytest.approx(2.1))


def test_gate_spans_empty_inputs(cfg: Config):
    """Test empty times, zero duration, or no triggered windows return empty list."""
    audioset_names = ["Speech"]
    times_empty = np.zeros(0, dtype=np.float64)
    tagger_raw_empty = np.zeros((0, 1), dtype=np.float16)
    vad_probs_empty = np.zeros(0, dtype=np.float32)

    # Empty times and vad
    res = gate_spans(
        times_empty,
        tagger_raw_empty,
        audioset_names,
        vad_probs_empty,
        vad_chunk_sec=0.032,
        duration=10.0,
        window_sec=1.0,
        cfg=cfg,
    )
    assert res == []

    # Duration <= 0
    times = np.array([1.0], dtype=np.float64)
    tagger_raw = np.array([[0.9]], dtype=np.float16)
    res_dur0 = gate_spans(
        times,
        tagger_raw,
        audioset_names,
        vad_probs_empty,
        vad_chunk_sec=0.032,
        duration=0.0,
        window_sec=1.0,
        cfg=cfg,
    )
    assert res_dur0 == []


def test_window_mask_cases():
    """Test window_mask fully inside span, partial, tolerance, and empty cases."""
    times = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
    window_sec = 1.0

    # Spans: [0.5, 1.5]
    # Win 0: [0.0, 1.0] -> partial (left outside) -> False
    # Win 1: [0.5, 1.5] -> exact match -> True
    # Win 2: [1.0, 2.0] -> partial (right outside) -> False
    # Win 3: [1.5, 2.5] -> outside -> False
    # Win 4: [2.0, 3.0] -> outside -> False
    spans = [(0.5, 1.5)]
    mask = window_mask(times, window_sec, spans)
    assert mask.dtype == bool
    assert np.array_equal(mask, [False, True, False, False, False])

    # Tolerance test (1e-6)
    # Win 1 starts at 0.5. Span starts at 0.5 + 5e-7 (within 1e-6)
    spans_tol = [(0.5000005, 1.5)]
    mask_tol = window_mask(times, window_sec, spans_tol)
    assert mask_tol[1]

    # Beyond tolerance
    spans_beyond = [(0.501, 1.5)]
    mask_beyond = window_mask(times, window_sec, spans_beyond)
    assert not mask_beyond[1]

    # Empty spans
    mask_empty_spans = window_mask(times, window_sec, [])
    assert len(mask_empty_spans) == len(times)
    assert not np.any(mask_empty_spans)

    # Empty times
    mask_empty_times = window_mask(np.zeros(0), window_sec, spans)
    assert len(mask_empty_times) == 0
    assert mask_empty_times.dtype == bool


def test_separate_spans_fake_separator(cfg: Config):
    """Test separate_spans with a fake separator, verifying length, dtype, crossfades, and immutability."""
    sr = 1000
    n_samples = 3000
    audio = np.ones(n_samples, dtype=np.float32)

    # Fake separator returns audio * 0.5
    fake_separator = MagicMock(side_effect=lambda x, sr_arg: x * 0.5)

    p = copy.deepcopy(cfg.pipeline)
    p["events"]["separation"] = {
        "chunk_sec": 1.0,
        "crossfade_sec": 0.05,
    }
    test_cfg = replace(cfg, pipeline=p)

    # Span from 1.0s to 2.0s (samples 1000 to 2000)
    spans = [(1.0, 2.0)]
    out = separate_spans(audio, sr, spans, fake_separator, test_cfg)

    # Output length and dtype
    assert len(out) == len(audio)
    assert out.dtype == np.float32

    # Input not mutated
    assert np.all(audio == 1.0)

    # Untouched outside spans
    assert np.all(out[:1000] == 1.0)
    assert np.all(out[2000:] == 1.0)

    # Interior away from edges has exact separated values (0.5)
    # crossfade_len is 0.05 * 1000 = 50 samples
    # Left edge: 1000 to 1050; Right edge: 1950 to 2000
    # Interior: 1050 to 1950
    assert np.allclose(out[1050:1950], 0.5)

    # Crossfade blends at span edges
    # At sample 1000: weight of audio is 1.0, out is 1.0
    # Between 1000 and 1050, values smoothly decrease from 1.0 to 0.5
    assert out[1000] == pytest.approx(1.0)
    assert np.all(np.diff(out[1000:1050]) < 0)
    # Between 1950 and 2000, values smoothly increase from 0.5 to 1.0
    assert np.all(np.diff(out[1950:2000]) > 0)


def test_separate_spans_chunk_joins(cfg: Config):
    """Test that a span longer than chunk_sec blends smoothly at chunk joins."""
    sr = 1000
    audio = np.ones(4000, dtype=np.float32)

    # Span from 1.0s to 3.0s (2000 samples)
    # chunk_sec = 1.0 (1000 samples)
    # Pieces: [1000, 2000) and [2000, 3000)
    # Join is at sample 2000
    p = copy.deepcopy(cfg.pipeline)
    p["events"]["separation"] = {
        "chunk_sec": 1.0,
        "crossfade_sec": 0.05,  # 50 samples
    }
    test_cfg = replace(cfg, pipeline=p)

    # Separator returns 0.2 for piece 0 (starts before 1500) and 0.8 for piece 1
    def tagging_separator(x: np.ndarray, sr_arg: int) -> np.ndarray:
        # Check if x is piece 0 or piece 1
        # Piece 0 ext_start is 1000 - 50 = 950
        # Piece 1 ext_start is 2000 - 50 = 1950
        # In our test audio, audio is all 1.0, so distinguish by length or call count
        if not hasattr(tagging_separator, "count"):
            tagging_separator.count = 0  # type: ignore
        val = 0.2 if tagging_separator.count == 0 else 0.8  # type: ignore
        tagging_separator.count += 1  # type: ignore
        return np.full_like(x, val)

    out = separate_spans(audio, sr, [(1.0, 3.0)], tagging_separator, test_cfg)

    # Join is at 2000. Blend is centered at 2000 over 50 samples: [1975, 2025]
    # Interior of piece 0 (away from span edge 1050 and join 1975) is exact 0.2
    assert np.allclose(out[1100:1950], 0.2)
    # Interior of piece 1 (away from join 2025 and span edge 2950) is exact 0.8
    assert np.allclose(out[2050:2900], 0.8)

    # Monotonic blend across join
    assert np.all(np.diff(out[1975:2025]) > 0)


def test_separate_spans_crossfade_zero_hard_cuts(cfg: Config):
    """Test crossfade_sec=0 produces hard cuts at edges and joins."""
    sr = 1000
    audio = np.ones(3000, dtype=np.float32)

    p = copy.deepcopy(cfg.pipeline)
    p["events"]["separation"] = {
        "chunk_sec": 0.5,
        "crossfade_sec": 0.0,
    }
    test_cfg = replace(cfg, pipeline=p)

    fake_separator = lambda x, sr_arg: x * 0.4
    out = separate_spans(audio, sr, [(1.0, 2.0)], fake_separator, test_cfg)

    # Outside untouched
    assert out[999] == pytest.approx(1.0)
    # Hard step at span start
    assert out[1000] == pytest.approx(0.4)
    # Hard step at span end
    assert out[1999] == pytest.approx(0.4)
    assert out[2000] == pytest.approx(1.0)


def test_separate_spans_wrong_length_raises(cfg: Config):
    """Test ValueError is raised when separator returns unexpected length."""
    audio = np.ones(1000, dtype=np.float32)
    bad_separator = lambda x, sr: x[:-1]  # wrong length

    with pytest.raises(ValueError, match="output length mismatch"):
        separate_spans(audio, 1000, [(0.1, 0.5)], bad_separator, cfg)


def test_separate_spans_empty_spans_not_calling_separator(cfg: Config):
    """Test empty spans returns copy without calling separator."""
    audio = np.ones(500, dtype=np.float32)
    separator = MagicMock()

    out = separate_spans(audio, 1000, [], separator, cfg)
    assert np.array_equal(out, audio)
    separator.assert_not_called()


def test_load_separator_missing_demucs_raises(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    """Test load_separator raises ImportError with install hint when demucs is unavailable."""
    monkeypatch.setitem(sys.modules, "demucs", None)
    monkeypatch.setitem(sys.modules, "demucs.pretrained", None)

    with pytest.raises(ImportError, match=r'pip install -e "\.\[separation\]"'):
        load_separator(cfg)


def test_demucs_separator_mock_pipeline():
    """Test DemucsSeparator operations (resampling, apply_model, channel average, pad/trim)."""
    import torch

    class FakeModel:
        samplerate = 44100
        sources = ["drums", "bass", "other", "vocals"]

        def eval(self):
            return self

    mock_model = FakeModel()
    sep = DemucsSeparator(mock_model, device="cpu")

    # Mock demucs.apply.apply_model
    # When called with x shape (1, 2, N_resampled)
    def fake_apply(model, x, device=None):
        # Return shape (1, 4, 2, N_resampled)
        # drums = 1, bass = 2, other = 3, vocals = 100
        # sum of non-vocals = 1 + 2 + 3 = 6
        n_samples = x.shape[-1]
        out = torch.zeros((1, 4, 2, n_samples), dtype=torch.float32)
        out[0, 0] = 1.0
        out[0, 1] = 2.0
        out[0, 2] = 3.0
        out[0, 3] = 100.0
        return out

    # We monkeypatch demucs and demucs.apply
    mock_demucs = MagicMock()
    mock_apply_module = MagicMock()
    mock_apply_module.apply_model = fake_apply
    mock_demucs.apply = mock_apply_module
    sys.modules["demucs"] = mock_demucs
    sys.modules["demucs.apply"] = mock_apply_module

    try:
        # At model sample rate (no resampling)
        sr = 44100
        audio = np.ones(4410, dtype=np.float32)  # 0.1s
        res = sep(audio, sr)
        assert len(res) == len(audio)
        assert res.dtype == np.float32
        assert np.allclose(res, 6.0)

        # With resampling (sr != 44100)
        sr_16k = 16000
        audio_16k = np.ones(1600, dtype=np.float32)
        res_16k = sep(audio_16k, sr_16k)
        assert len(res_16k) == len(audio_16k)
        assert res_16k.dtype == np.float32
        # Interior away from sinc filter edges is 6.0
        assert np.allclose(res_16k[50:-50], 6.0, atol=1e-2)
    finally:
        sys.modules.pop("demucs.apply", None)
        sys.modules.pop("demucs", None)


def test_load_separator_real_demucs_skipped(cfg: Config):
    """Test that skips if demucs is not installed."""
    pytest.importorskip("demucs")
    sep = load_separator(cfg)
    assert isinstance(sep, DemucsSeparator)


def test_module_imports_without_heavy_deps(monkeypatch: pytest.MonkeyPatch):
    """Verify module can be imported when torch, torchaudio, and demucs are not installed."""
    import importlib
    import soundthing.events.separation

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torchaudio", None)
    monkeypatch.setitem(sys.modules, "demucs", None)

    importlib.reload(soundthing.events.separation)

