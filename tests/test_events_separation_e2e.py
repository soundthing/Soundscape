"""End-to-end tests for gated source separation in event detection."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from soundthing.config import Config
from soundthing.contracts import CaptionEvent, ExtractedInputs
from soundthing.events.core import EventModels
import soundthing.events.detector as detector
import soundthing.events.separation as separation
from soundthing.events.tagger import build_tagger_map
from soundthing.events.tuning.cache import cache_key
from soundthing.events import vad


class FakeAudioDependentTagger:
    """Fake tagger returning probabilities based on frequency content of the audio window."""

    def __init__(self, audioset_names: list[str], target_label: str) -> None:
        self.audioset_names = audioset_names
        self.target_label = target_label
        self.speech_idx = audioset_names.index("Speech")
        self.event_idx = audioset_names.index(target_label)
        self.call_count = 0

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        self.call_count += 1
        b = audio.shape[0] if audio.ndim > 1 else 1
        probs = np.zeros((b, len(self.audioset_names)), dtype=np.float32)
        audio_2d = audio if audio.ndim > 1 else audio[None, :]

        for i in range(b):
            x = audio_2d[i]
            if len(x) == 0:
                continue
            fft = np.fft.rfft(x)
            freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
            power = np.abs(fft) ** 2

            speech_band = (freqs >= 950) & (freqs <= 1050)
            event_band = (freqs >= 400) & (freqs <= 480)

            speech_power = float(np.mean(power[speech_band])) if np.any(speech_band) else 0.0
            event_power = float(np.mean(power[event_band])) if np.any(event_band) else 0.0

            has_speech = speech_power > 1e4
            has_event = event_power > 1e4

            if has_speech:
                # Loud speech masks the target event
                probs[i, self.speech_idx] = 0.95
                probs[i, self.event_idx] = 0.10
            elif has_event:
                # Separated/unmasked target event is clearly detected
                probs[i, self.speech_idx] = 0.0
                probs[i, self.event_idx] = 0.95
            else:
                probs[i, self.speech_idx] = 0.0
                probs[i, self.event_idx] = 0.0

        return probs


class FakeSeparator:
    """Fake separator callable removing the speech component (>700 Hz)."""

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        self.call_count += 1
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
        fft[freqs > 700] = 0.0
        return np.fft.irfft(fft, n=len(audio)).astype(np.float32)


@pytest.fixture(autouse=True)
def clean_detector_caches():
    detector._clear_caches()
    yield
    detector._clear_caches()


@pytest.fixture
def fake_env(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    # Pick a non-safety class with an AudioSet mapping programmatically
    target_class = None
    target_label = None
    for sc in cfg.classes:
        if not sc.safety and sc.audioset:
            target_class = sc.id
            target_label = sc.audioset[0]
            break
    assert target_class is not None and target_label is not None

    audioset_names = sorted({n for c in cfg.classes for n in c.audioset} | {"Speech"})
    tagger = FakeAudioDependentTagger(audioset_names, target_label)
    separator = FakeSeparator()

    sep_load_calls = 0

    def fake_load_separator(c: Config):
        nonlocal sep_load_calls
        sep_load_calls += 1
        return separator

    monkeypatch.setattr(separation, "load_separator", fake_load_separator)
    monkeypatch.setattr(detector.separation, "load_separator", fake_load_separator)
    monkeypatch.setattr(
        vad,
        "speech_probs",
        lambda audio, sr, c: np.zeros(max(1, int(len(audio) / sr / vad.VAD_CHUNK_SEC)), dtype=np.float32),
    )
    monkeypatch.setattr(detector, "load_tagger", lambda c: (tagger, audioset_names))
    monkeypatch.setattr(detector, "load_clap", lambda c: None)
    monkeypatch.setattr(detector, "text_embeddings", lambda clap, c: None)

    class EnvHelper:
        @property
        def target_class(self) -> str:
            return target_class

        @property
        def target_label(self) -> str:
            return target_label

        @property
        def tagger(self) -> FakeAudioDependentTagger:
            return tagger

        @property
        def separator(self) -> FakeSeparator:
            return separator

        @property
        def sep_load_calls(self) -> int:
            return sep_load_calls

    return EnvHelper()


def make_synthetic_inputs(
    tmp_path: Path,
    duration: float = 6.0,
    sr: int = 16000,
    video_id: str = "clip_sep_e2e",
) -> ExtractedInputs:
    """Create a clip where middle seconds contain event + loud speech component."""
    clip_dir = tmp_path / video_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    audio_path = clip_dir / "audio.wav"

    n_samples = int(duration * sr)
    t = np.arange(n_samples, dtype=np.float32) / float(sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    # Middle seconds [2.0, 4.0]: event (440 Hz) + loud speech (1000 Hz)
    mid = (t >= 2.0) & (t < 4.0)
    event_tone = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    speech_tone = 0.8 * np.sin(2.0 * np.pi * 1000.0 * t)
    audio[mid] = (event_tone[mid] + speech_tone[mid]).astype(np.float32)

    sf.write(str(audio_path), audio, sr)
    return ExtractedInputs(
        video_id=video_id,
        video_path=str(clip_dir / f"{video_id}.mp4"),
        audio_path=str(audio_path),
        frames_dir=str(clip_dir / "frames"),
        fps=1.0,
        duration=duration,
        sample_rate=sr,
    )


def test_separation_end_to_end_scenarios(cfg: Config, fake_env, tmp_path: Path):
    """End-to-end test verifying detection with separation disabled vs enabled (fusion max and sep)."""
    inputs = make_synthetic_inputs(tmp_path, duration=6.0)

    # Config variations
    p_dis = copy.deepcopy(cfg.pipeline)
    p_dis["events"]["separation"]["enabled"] = False
    cfg_dis = replace(cfg, pipeline=p_dis)

    p_max = copy.deepcopy(cfg.pipeline)
    p_max["events"]["separation"]["enabled"] = True
    p_max["events"]["separation"]["fusion"] = "max"
    cfg_max = replace(cfg, pipeline=p_max)

    p_sep = copy.deepcopy(cfg.pipeline)
    p_sep["events"]["separation"]["enabled"] = True
    p_sep["events"]["separation"]["fusion"] = "sep"
    cfg_sep = replace(cfg, pipeline=p_sep)

    # d) Cache keys: disabled vs enabled differ; enabled with sep vs max share cache_key
    key_dis = cache_key(cfg_dis)
    key_max = cache_key(cfg_max)
    key_sep = cache_key(cfg_sep)
    assert key_dis != key_max, "disabled and enabled configs must produce different cache keys"
    assert key_max == key_sep, "fusion max and sep share cache_key because fusion is replay-time only"

    # a) Separation disabled -> target event is NOT detected in the masked region
    detector._clear_caches()
    events_dis = detector.detect(inputs, None, cfg_dis)
    target_dis = [e for e in events_dis if e.class_id == fake_env.target_class]
    assert len(target_dis) == 0, "target event must not be detected when speech masks it and separation is disabled"

    # b) Separation enabled, fusion max -> target event IS detected there,
    # and built cache has sep_mask True only around the gated region
    detector._clear_caches()
    events_max = detector.detect(inputs, None, cfg_max)
    target_max = [e for e in events_max if e.class_id == fake_env.target_class]
    assert len(target_max) > 0, "target event must be detected when separation unmasks it"
    for ev in target_max:
        ev.validate()
        # Event is within the middle region
        assert ev.start >= 1.5 and ev.end <= 4.5

    # Check built cache sep_mask
    assert detector._clip_cache_memo is not None
    cache = detector._clip_cache_memo[1]
    assert cache.sep_mask is not None
    assert cache.sep_mask.dtype == bool
    assert cache.sep_mask.any(), "sep_mask must be True in the gated region"
    assert (~cache.sep_mask).any(), "sep_mask must be False outside the gated region"

    # Gated region is around [2.0, 4.0] (with padding ~[1.5, 4.5])
    # Windows well outside (e.g. times < 1.0 or times > 5.0) must have sep_mask False
    outside_mask = (cache.times < 1.0) | (cache.times > 5.0)
    assert not np.any(cache.sep_mask[outside_mask]), "sep_mask must be False outside the gated region"
    inside_mask = (cache.times >= 2.0) & (cache.times <= 4.0)
    assert np.all(cache.sep_mask[inside_mask]), "sep_mask must be True in the core gated region"

    # c) Enabled -> separator was called (once per clip) and models.separator was set via separation.load_separator
    assert fake_env.sep_load_calls == 1, "separation.load_separator must be called when loading models with separation enabled"
    models = detector.get_models(cfg_max)
    assert models.separator is fake_env.separator, "models.separator must be set via monkeypatched load_separator"
    assert fake_env.separator.call_count == 1, "separator must be called exactly once for the clip's gated span"


def test_real_demucs_smoke(cfg: Config):
    """Smoke test running the real DemucsSeparator on 2 s of synthetic audio at 16 kHz."""
    pytest.importorskip("demucs")
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")

    sep = separation.load_separator(cfg)
    assert isinstance(sep, separation.DemucsSeparator)

    sr = 16000
    duration = 2.0
    n_samples = int(duration * sr)
    t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float32)
    audio = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    out = sep(audio, sr)

    assert isinstance(out, np.ndarray)
    assert out.shape == (n_samples,)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
