from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from soundthing.config import Config
from soundthing.contracts import CaptionEvent, EnvSegment, ExtractedInputs
from soundthing.events.core import EventModels
import soundthing.events.detector as detector
from soundthing.events.tagger import build_tagger_map


class FakeTagger:
    """Fake tagger callable returning fixed/controllable probabilities."""

    def __init__(self, names: list[str], default_prob: float = 0.0) -> None:
        self.names = names
        self.default_prob = default_prob
        self.scores_by_name: dict[str, float] = {}
        self.call_count = 0

    def set_score(self, name: str, prob: float) -> None:
        self.scores_by_name[name] = prob

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        self.call_count += 1
        b = audio.shape[0] if audio.ndim > 1 else 1
        probs = np.full((b, len(self.names)), self.default_prob, dtype=np.float32)
        for name, prob in self.scores_by_name.items():
            if name in self.names:
                probs[:, self.names.index(name)] = prob
        return probs


@pytest.fixture(autouse=True)
def clean_detector_caches():
    detector._clear_caches()
    yield
    detector._clear_caches()


@pytest.fixture
def fake_env(cfg: Config, monkeypatch):
    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    tagger_map = build_tagger_map(cfg, audioset_names)
    tagger = FakeTagger(audioset_names, default_prob=0.0)

    load_calls = 0

    from soundthing.events import vad

    def fake_load_models(c: Config) -> EventModels:
        nonlocal load_calls
        load_calls += 1
        return EventModels(
            tagger=tagger,
            clap=None,
            class_ids=list(c.class_ids),
            tagger_map=tagger_map,
            clap_text_emb=None,
            audioset_names=audioset_names,
        )

    monkeypatch.setattr(detector, "load_models", fake_load_models)
    monkeypatch.setattr(vad, "speech_probs", lambda audio, sr, c: np.array([], dtype=np.float32))

    class Helper:
        @property
        def tagger(self) -> FakeTagger:
            return tagger

        @property
        def calls(self) -> int:
            return load_calls

    return Helper()


def make_inputs(
    tmp_path: Path,
    video_id: str = "clip01",
    duration: float = 3.0,
    channels: int = 1,
    sr: int = 16000,
) -> ExtractedInputs:
    clip_dir = tmp_path / video_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    audio_path = clip_dir / "audio.wav"
    n_samples = int(duration * sr)
    if channels == 1:
        data = np.zeros(n_samples, dtype=np.float32)
    else:
        data = np.zeros((n_samples, channels), dtype=np.float32)
    sf.write(str(audio_path), data, sr)
    return ExtractedInputs(
        video_id=video_id,
        video_path=str(clip_dir / f"{video_id}.mp4"),
        audio_path=str(audio_path),
        frames_dir=str(clip_dir / "frames"),
        fps=1.0,
        duration=duration,
        sample_rate=sr,
    )


def test_detect_returns_valid_sorted_events_in_both_modes(cfg: Config, fake_env, tmp_path: Path):
    sc = next(c for c in cfg.classes if c.audioset)
    fake_env.tagger.set_score(sc.audioset[0], 0.95)

    inputs = make_inputs(tmp_path, duration=3.0)
    segments = [
        EnvSegment(
            start=0.0,
            end=3.0,
            env=cfg.env_ids[0],
            confidence=1.0,
            probs={cfg.env_ids[0]: 1.0},
        )
    ]

    baseline_evs = detector.detect(inputs, None, cfg)
    conditioned_evs = detector.detect(inputs, segments, cfg)

    assert len(baseline_evs) > 0
    assert len(conditioned_evs) > 0

    for ev in baseline_evs:
        ev.validate()
    for ev in conditioned_evs:
        ev.validate()

    assert baseline_evs == sorted(baseline_evs, key=lambda e: (e.start, e.end))
    assert conditioned_evs == sorted(conditioned_evs, key=lambda e: (e.start, e.end))


def test_models_and_raw_scores_cached_across_modes(cfg: Config, fake_env, tmp_path: Path):
    sc = next(c for c in cfg.classes if c.audioset)
    fake_env.tagger.set_score(sc.audioset[0], 0.95)

    inputs = make_inputs(tmp_path, duration=3.0)
    segments = [
        EnvSegment(
            start=0.0,
            end=3.0,
            env=cfg.env_ids[0],
            confidence=1.0,
            probs={cfg.env_ids[0]: 1.0},
        )
    ]

    assert fake_env.calls == 0
    detector.detect(inputs, None, cfg)
    assert fake_env.calls == 1
    tagger_calls_after_first = fake_env.tagger.call_count
    assert tagger_calls_after_first > 0

    # Second call for the same file in conditioned mode
    detector.detect(inputs, segments, cfg)
    assert fake_env.calls == 1, "load_models must be called exactly once across baseline+conditioned"
    assert (
        fake_env.tagger.call_count == tagger_calls_after_first
    ), "the tagger must not be called again on the second (conditioned) call for the same file"


def test_conditioning_changes_output_for_downweighted_class(cfg: Config, fake_env, tmp_path: Path):
    # Dynamically select an environment and a non-safety class downweighted in priors
    chosen_env = None
    chosen_class = None
    for env_id, ep in cfg.priors.environments.items():
        for cid, weight in ep.classes.items():
            if cid not in cfg.safety_ids() and weight <= 0.4:
                sc = cfg.sound_class(cid)
                if sc.audioset:
                    chosen_env = env_id
                    chosen_class = cid
                    break
        if chosen_env is not None:
            break

    if chosen_env is None:
        for env_id, ep in cfg.priors.environments.items():
            if ep.default <= 0.3:
                for sc in cfg.classes:
                    if not sc.safety and sc.id not in ep.classes and sc.audioset:
                        chosen_env = env_id
                        chosen_class = sc.id
                        break
            if chosen_env is not None:
                break

    assert chosen_env is not None and chosen_class is not None
    sc = cfg.sound_class(chosen_class)
    thr = cfg.threshold(chosen_class)

    # Set score just above threshold so baseline fires but conditioned drops below threshold
    firing_prob = min(0.95, thr + 0.1)
    fake_env.tagger.set_score(sc.audioset[0], firing_prob)

    inputs = make_inputs(tmp_path, duration=3.0)
    segments = [
        EnvSegment(
            start=0.0,
            end=3.0,
            env=chosen_env,
            confidence=1.0,
            probs={chosen_env: 1.0},
        )
    ]

    baseline_evs = detector.detect(inputs, None, cfg)
    conditioned_evs = detector.detect(inputs, segments, cfg)

    baseline_cids = {e.class_id for e in baseline_evs}
    conditioned_cids = {e.class_id for e in conditioned_evs}

    assert chosen_class in baseline_cids
    assert chosen_class not in conditioned_cids or baseline_evs != conditioned_evs


def test_audio_path_2d_averaged_to_mono(cfg: Config, fake_env, tmp_path: Path):
    sc = next(c for c in cfg.classes if c.audioset)
    fake_env.tagger.set_score(sc.audioset[0], 0.95)

    inputs = make_inputs(tmp_path, duration=2.0, channels=2)
    evs = detector.detect(inputs, None, cfg)
    assert len(evs) > 0
    for e in evs:
        e.validate()


def test_stereo_path_triggers_add_modifiers_with_2d_array(
    cfg: Config, fake_env, tmp_path: Path, monkeypatch
):
    inputs = make_inputs(tmp_path, duration=2.0, channels=1)
    stereo_path = tmp_path / "stereo.wav"
    sf.write(str(stereo_path), np.zeros((16000 * 2, 2), dtype=np.float32), 16000)
    inputs.audio_stereo_path = str(stereo_path)

    recorded: dict = {}
    orig_add_modifiers = detector.pp.add_modifiers

    def spy_add_modifiers(events, stereo, sr, c):
        recorded["stereo"] = stereo
        recorded["sr"] = sr
        return orig_add_modifiers(events, stereo, sr, c)

    monkeypatch.setattr(detector.pp, "add_modifiers", spy_add_modifiers)

    detector.detect(inputs, None, cfg)
    assert "stereo" in recorded
    assert recorded["stereo"] is not None
    assert isinstance(recorded["stereo"], np.ndarray)
    assert recorded["stereo"].ndim == 2
    assert recorded["stereo"].shape[1] == 2


def test_stereo_path_absent_calls_add_modifiers_with_none(
    cfg: Config, fake_env, tmp_path: Path, monkeypatch
):
    inputs = make_inputs(tmp_path, duration=2.0, channels=1)
    inputs.audio_stereo_path = None

    recorded: dict = {}
    orig_add_modifiers = detector.pp.add_modifiers

    def spy_add_modifiers(events, stereo, sr, c):
        recorded["stereo"] = stereo
        recorded["sr"] = sr
        return orig_add_modifiers(events, stereo, sr, c)

    monkeypatch.setattr(detector.pp, "add_modifiers", spy_add_modifiers)

    detector.detect(inputs, None, cfg)
    assert "stereo" in recorded
    assert recorded["stereo"] is None


def test_events_clipped_to_duration(cfg: Config, fake_env, tmp_path: Path):
    sc = next(c for c in cfg.classes if c.audioset)
    fake_env.tagger.set_score(sc.audioset[0], 0.95)

    # Audio is 3.0 s, but inputs.duration is 2.0 s
    inputs = make_inputs(tmp_path, duration=3.0)
    inputs.duration = 2.0

    evs = detector.detect(inputs, None, cfg)
    assert len(evs) > 0
    for e in evs:
        assert e.start < 2.0
        assert e.end <= 2.0
        assert e.end > e.start
        e.validate()


def test_clear_caches_reloads_models_and_rescores(cfg: Config, fake_env, tmp_path: Path):
    sc = next(c for c in cfg.classes if c.audioset)
    fake_env.tagger.set_score(sc.audioset[0], 0.95)

    inputs = make_inputs(tmp_path, duration=2.0)
    detector.detect(inputs, None, cfg)
    assert fake_env.calls == 1
    calls_before = fake_env.tagger.call_count

    detector._clear_caches()
    detector.detect(inputs, None, cfg)
    assert fake_env.calls == 2
    assert fake_env.tagger.call_count > calls_before


def test_load_models_and_cache_key_with_separation(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    """Test load_models and _models_cache_key with separation enabled and disabled."""
    key_disabled = detector._models_cache_key(cfg)

    sep_load_calls = 0
    fake_separator = object()

    def fake_load_separator(c: Config):
        nonlocal sep_load_calls
        sep_load_calls += 1
        return fake_separator

    audioset_names = sorted({n for c in cfg.classes for n in c.audioset})
    monkeypatch.setattr(detector, "load_tagger", lambda c: (FakeTagger(audioset_names), audioset_names))
    monkeypatch.setattr(detector, "load_clap", lambda c: None)
    monkeypatch.setattr(detector, "text_embeddings", lambda clap, c: None)
    monkeypatch.setattr(detector.separation, "load_separator", fake_load_separator)

    # When disabled: separator is None and load_separator is not called
    models_disabled = detector.load_models(cfg)
    assert models_disabled.separator is None
    assert sep_load_calls == 0

    # When enabled: load_separator is called and attached to EventModels.separator
    p_sep = copy.deepcopy(cfg.pipeline)
    p_sep["events"]["separation"]["enabled"] = True
    cfg_sep = replace(cfg, pipeline=p_sep)

    key_enabled = detector._models_cache_key(cfg_sep)
    assert key_enabled != key_disabled

    models_enabled = detector.load_models(cfg_sep)
    assert models_enabled.separator is fake_separator
    assert sep_load_calls == 1

    # get_models caches loaded models
    detector._clear_caches()
    m1 = detector.get_models(cfg_sep)
    m2 = detector.get_models(cfg_sep)
    assert m1 is m2
    assert m1.separator is fake_separator
    assert sep_load_calls == 2
