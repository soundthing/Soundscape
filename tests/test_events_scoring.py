from __future__ import annotations

import copy
import sys

import numpy as np
import pytest

from soundthing.config import Config
from soundthing.events.core import EventModels, score_window
import soundthing.events.detector as detector
from soundthing.events.tagger import build_tagger_map, load_tagger


class FakeTagger:
    """Fake tagger callable returning fixed/controllable probabilities."""

    def __init__(self, names: list[str], default_prob: float = 0.1) -> None:
        self.names = names
        self.default_prob = default_prob
        self.scores_by_name: dict[str, float] = {}

    def set_score(self, name: str, prob: float) -> None:
        self.scores_by_name[name] = prob

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        B = audio.shape[0] if audio.ndim > 1 else 1
        probs = np.full((B, len(self.names)), self.default_prob, dtype=np.float32)
        for name, prob in self.scores_by_name.items():
            if name in self.names:
                probs[:, self.names.index(name)] = prob
        return probs


def test_no_torch_imported_at_top_level():
    """Importing events modules must not import torch or heavy ML libs at top level."""
    torch_previously_loaded = "torch" in sys.modules

    import soundthing.events.core
    import soundthing.events.detector
    import soundthing.events.tagger

    assert soundthing.events.tagger is not None
    assert soundthing.events.core is not None
    assert soundthing.events.detector is not None

    if not torch_previously_loaded:
        assert "torch" not in sys.modules, "torch was imported at module top level"
        assert "torchaudio" not in sys.modules, "torchaudio was imported at module top level"
        assert "panns_inference" not in sys.modules, "panns_inference was imported at module top level"
        assert "transformers" not in sys.modules, "transformers was imported at module top level"


def test_unsupported_tagger_model_raises(cfg: Config):
    """Unsupported tagger model raises NotImplementedError with a clear message."""
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg.pipeline["events"]["tagger"]["model"] = "beats"
    with pytest.raises(NotImplementedError, match="Unsupported tagger model: 'beats'"):
        load_tagger(bad_cfg)


def test_score_window_tagger_only(cfg: Config):
    """Every mapped class keeps its tagger score; unmapped classes remain silent."""
    audioset_names = sorted({name for c in cfg.classes for name in c.audioset})
    tagger_map = build_tagger_map(cfg, audioset_names)
    fake_tagger = FakeTagger(audioset_names, default_prob=0.35)

    models = EventModels(
        tagger=fake_tagger,
        class_ids=list(cfg.class_ids),
        tagger_map=tagger_map,
    )

    audio = np.zeros(16000, dtype=np.float32)
    scores = score_window(audio, 16000, models)

    assert scores.shape == (len(cfg.classes),)
    assert scores.dtype == np.float32
    assert np.all((scores >= 0.0) & (scores <= 1.0))

    for i, c in enumerate(cfg.classes):
        if c.audioset:
            assert abs(scores[i] - 0.35) < 1e-5
        else:
            assert scores[i] == 0.0


def test_load_models_needs_only_the_tagger(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    """Event loading works when Transformers is unavailable to this process."""
    audioset_names = sorted({name for c in cfg.classes for name in c.audioset})
    fake_tagger = FakeTagger(audioset_names)
    monkeypatch.setattr(detector, "load_tagger", lambda c: (fake_tagger, audioset_names))
    monkeypatch.setitem(sys.modules, "transformers", None)

    models = detector.load_models(cfg)
    assert isinstance(models, EventModels)
    assert models.tagger is fake_tagger
    assert models.class_ids == cfg.class_ids
    assert models.tagger_map.shape == (len(cfg.classes), len(audioset_names))
    scores = score_window(np.zeros(16000, dtype=np.float32), 16000, models)
    np.testing.assert_allclose(scores, np.where(models.tagger_map.any(axis=1), .1, 0))


def test_obsolete_zero_shot_settings_cannot_change_event_scores(cfg, monkeypatch):
    """Old local configs cannot reactivate the removed sound-event scorer."""
    legacy = copy.deepcopy(cfg)
    legacy.pipeline["events"]["clap"] = {
        "enabled": True,
        "model": "invalid-removed-event-model",
        "calibration": {"center": -1, "scale": 1000},
    }
    audioset_names = sorted({name for c in cfg.classes for name in c.audioset})
    fake_tagger = FakeTagger(audioset_names, default_prob=.2)
    monkeypatch.setattr(detector, "load_tagger", lambda c: (fake_tagger, audioset_names))
    detector._clear_caches()
    try:
        current = detector.get_models(cfg)
        assert detector.get_models(legacy) is current
        scores = score_window(np.zeros(16000, dtype=np.float32), 16000, current)
        np.testing.assert_allclose(scores, np.where(current.tagger_map.any(axis=1), .2, 0))
    finally:
        detector._clear_caches()


def test_load_models_tagger_map_keyerror(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    """Missing AudioSet label in tagger names propagates KeyError from build_tagger_map."""
    incomplete_names = ["Irrelevant Label Not In Vocabulary"]
    fake_tagger = FakeTagger(incomplete_names)

    monkeypatch.setattr(detector, "load_tagger", lambda c: (fake_tagger, incomplete_names))

    with pytest.raises(KeyError, match="not in tagger labels"):
        detector.load_models(cfg)


def test_real_model_smoke(cfg: Config):
    """Real model smoke test: runs only when panns_inference is installed, otherwise skips."""
    pytest.importorskip("panns_inference")
    tagger, names = load_tagger(cfg)
    audio = np.zeros((1, 16000), dtype=np.float32)
    scores = tagger(audio, 16000)
    assert scores.shape == (1, len(names))
