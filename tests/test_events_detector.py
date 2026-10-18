from __future__ import annotations

from copy import deepcopy
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
from soundthing.events.types import FrameScores


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

    def fake_load_models(c: Config) -> EventModels:
        nonlocal load_calls
        load_calls += 1
        return EventModels(
            tagger=tagger,
            class_ids=list(c.class_ids),
            tagger_map=tagger_map,
        )

    monkeypatch.setattr(detector, "load_models", fake_load_models)
    monkeypatch.setattr(detector, "speech_regions", lambda audio, sr, c: [])

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


@pytest.mark.parametrize('conditioned', [False, True])
@pytest.mark.parametrize('legacy_settings', [False, True])
def test_detect_preserves_brief_repeated_and_simultaneous_events(
    cfg, fake_env, tmp_path, monkeypatch, conditioned, legacy_settings,
):
    from soundthing.captions import events_to_cues

    local_cfg = deepcopy(cfg)
    local_cfg.pipeline['events']['modifiers']['enabled'] = False
    if legacy_settings:
        # Old local configs remain loadable; these retired settings have no effect.
        local_cfg.pipeline['events']['postprocess'].update(
            smoothing_sec=20, sustained_repeat_sec=60, max_on_screen=1,
        )
    local_cfg.validate()
    assert local_cfg.sound_class('typing').sustained

    class_ids = ['typing', 'cough', 'audioset_m_04rlf']  # unrelated Music label
    times = np.arange(17) * .25 + .5
    scores = np.zeros((len(times), len(class_ids)))
    scores[:6, 0] = .95
    scores[10:16, 0] = .95  # typing stops and restarts well inside 60 seconds
    scores[[2, 4], 1] = .9  # nearby isolated peaks survive via ordinary gap merging
    scores[:, 2] = .85
    original_scores = scores.copy()
    frame_scores = FrameScores(times, class_ids, scores, .25)
    monkeypatch.setattr(detector, 'score_audio', lambda *args: frame_scores)

    inputs = make_inputs(tmp_path, duration=5)
    segments = [EnvSegment(0, 5, 'general', 1, {'general': 1})] if conditioned else None
    events = detector.detect(inputs, segments, local_cfg)

    assert {event.class_id for event in events} == set(class_ids)
    assert len(events) == 4
    typing = [event for event in events if event.class_id == 'typing']
    assert len(typing) == 2 and typing[1].start - typing[0].start < 60
    cough, = [event for event in events if event.class_id == 'cough']
    assert cough.confidence == .9 and cough.end - cough.start > .7
    for event in events:
        event.validate()
    cues = events_to_cues(events, hierarchy=local_cfg.hierarchy())
    assert sum(cue.start <= 1 < cue.end for cue in cues) == 3
    np.testing.assert_array_equal(frame_scores.scores, original_scores)


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


def test_full_vocabulary_speech_is_not_suppressed_by_its_own_vad(cfg, fake_env, monkeypatch, tmp_path):
    speech = next(c for c in cfg.classes if c.audioset == ['Speech'])
    assert speech.id in cfg.pipeline['events']['vad']['exempt_classes']
    assert speech.id not in cfg.safety_ids()
    fake_env.tagger.set_score('Speech', .95)
    monkeypatch.setattr(detector, 'speech_regions', lambda *args: [(0., 3.)])
    result = detector.detect(make_inputs(tmp_path, duration=3.), None, cfg)
    assert speech.id in {event.class_id for event in result}
