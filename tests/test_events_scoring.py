from __future__ import annotations

import copy
import sys

import numpy as np
import pytest

from soundthing.config import Config
from soundthing.events.clap import clap_scores, text_embeddings
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


class FakeClap:
    """Fake CLAP wrapper with deterministic embeddings."""

    def __init__(self, dim: int = 16, center: float = 0.2, scale: float = 20.0) -> None:
        self.dim = dim
        self.center = center
        self.scale = scale

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        if not prompts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for p in prompts:
            # Deterministic pseudo-random embedding from string hash
            rng = np.random.RandomState(abs(hash(p)) % (2**31))
            vec = rng.randn(self.dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec /= norm
            out.append(vec)
        return np.array(out, dtype=np.float32)

    def embed_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        seed = int(np.sum(np.abs(audio)) * 1000) % (2**31) + 42
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec /= norm
        return vec


def test_no_torch_imported_at_top_level():
    """Importing events modules must not import torch or heavy ML libs at top level."""
    torch_previously_loaded = "torch" in sys.modules

    import soundthing.events.clap
    import soundthing.events.core
    import soundthing.events.detector
    import soundthing.events.tagger

    assert soundthing.events.tagger is not None
    assert soundthing.events.clap is not None
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


def test_text_embeddings(cfg: Config):
    """text_embeddings returns (C, D) array: unit-normalised for classes with prompts, zero for others."""
    dim = 16
    fake_clap = FakeClap(dim=dim)
    text_emb = text_embeddings(fake_clap, cfg)

    assert text_emb.shape == (len(cfg.classes), dim)
    assert text_emb.dtype == np.float32

    has_prompts_count = 0
    no_prompts_count = 0
    for i, c in enumerate(cfg.classes):
        if c.clap:
            has_prompts_count += 1
            norm = np.linalg.norm(text_emb[i])
            assert abs(norm - 1.0) < 1e-5, f"Class {c.id} text embedding is not L2-normalised"
        else:
            no_prompts_count += 1
            assert np.all(text_emb[i] == 0.0), f"Class {c.id} without prompts must be zero row"

    assert has_prompts_count > 0
    assert no_prompts_count > 0


def test_clap_scores_range_and_zeros(cfg: Config):
    """clap_scores returns values in [0, 1] and exactly 0 for prompt-less classes."""
    fake_clap = FakeClap(dim=16, center=0.2, scale=20.0)
    text_emb = text_embeddings(fake_clap, cfg)

    audio = np.ones(16000, dtype=np.float32) * 0.1
    scores = clap_scores(fake_clap, audio, 16000, text_emb)

    assert scores.shape == (len(cfg.classes),)
    assert scores.dtype == np.float32
    assert np.all((scores >= 0.0) & (scores <= 1.0))

    for i, c in enumerate(cfg.classes):
        if not c.clap:
            assert scores[i] == 0.0, f"Class {c.id} without prompts must have score 0.0"
        else:
            assert scores[i] > 0.0


def test_clap_scores_calibration_monotonic():
    """clap_scores calibration must be strictly monotonic in cosine similarity."""
    dim = 4

    class ControllableClap:
        def __init__(self, vec: np.ndarray, center: float = 0.2, scale: float = 20.0) -> None:
            self.vec = vec
            self.center = center
            self.scale = scale

        def embed_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
            return self.vec

    # 1 class with unit text embedding along axis 0
    text_emb = np.zeros((1, dim), dtype=np.float32)
    text_emb[0, 0] = 1.0

    sim_values = [-0.8, -0.4, 0.0, 0.2, 0.6, 0.95]
    computed = []
    dummy_audio = np.zeros(1600, dtype=np.float32)

    for sim in sim_values:
        audio_emb = np.zeros(dim, dtype=np.float32)
        audio_emb[0] = sim
        audio_emb[1] = np.sqrt(max(0.0, 1.0 - sim**2))
        clap = ControllableClap(audio_emb, center=0.2, scale=20.0)
        s = clap_scores(clap, dummy_audio, 16000, text_emb)
        computed.append(float(s[0]))

    # Must be strictly monotonic increasing
    for i in range(len(computed) - 1):
        assert computed[i] < computed[i + 1], f"Expected {computed[i]} < {computed[i + 1]}"

    # At sim == center (0.2), sigmoid((0.2 - 0.2) * scale) = sigmoid(0) = 0.5
    idx_center = sim_values.index(0.2)
    assert abs(computed[idx_center] - 0.5) < 1e-5


def test_score_window_tagger_only(cfg: Config):
    """score_window works when CLAP is None and outputs valid probabilities in [0, 1]."""
    audioset_names = sorted({name for c in cfg.classes for name in c.audioset})
    tagger_map = build_tagger_map(cfg, audioset_names)
    fake_tagger = FakeTagger(audioset_names, default_prob=0.35)

    models = EventModels(
        tagger=fake_tagger,
        clap=None,
        class_ids=list(cfg.class_ids),
        tagger_map=tagger_map,
        clap_text_emb=None,
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


def test_score_window_max_combination(cfg: Config):
    """score_window combines tagger and CLAP scores via element-wise max."""
    audioset_names = sorted({name for c in cfg.classes for name in c.audioset})
    tagger_map = build_tagger_map(cfg, audioset_names)
    fake_tagger = FakeTagger(audioset_names, default_prob=0.2)

    dim = 4

    class MockClap:
        def __init__(self) -> None:
            self.center = 0.2
            self.scale = 20.0

        def embed_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
            vec = np.zeros(dim, dtype=np.float32)
            vec[0] = 1.0
            return vec

    text_emb = np.zeros((len(cfg.classes), dim), dtype=np.float32)
    # Class 0: aligned with audio -> CLAP score ~ 1.0
    text_emb[0, 0] = 1.0
    # Class 1: opposite -> CLAP score ~ 0.0
    text_emb[1, 0] = -1.0

    models = EventModels(
        tagger=fake_tagger,
        clap=MockClap(),
        class_ids=list(cfg.class_ids),
        tagger_map=tagger_map,
        clap_text_emb=text_emb,
    )

    audio = np.zeros(16000, dtype=np.float32)
    scores = score_window(audio, 16000, models)

    assert scores.shape == (len(cfg.classes),)
    # Class 0: tagger <= 0.2, CLAP ~ 1.0 -> max is ~ 1.0
    assert scores[0] > 0.95
    # Class 1: tagger = 0.2, CLAP ~ 0.0 -> max is 0.2 (if class 1 has audioset)
    if cfg.classes[1].audioset:
        assert abs(scores[1] - 0.2) < 1e-4


def test_load_models_with_clap_enabled(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    """load_models loads both tagger and CLAP when clap.enabled is true."""
    audioset_names = sorted({name for c in cfg.classes for name in c.audioset})
    fake_tagger = FakeTagger(audioset_names)
    fake_clap = FakeClap(dim=8)

    monkeypatch.setattr(detector, "load_tagger", lambda c: (fake_tagger, audioset_names))
    monkeypatch.setattr(detector, "load_clap", lambda c: fake_clap)

    models = detector.load_models(cfg)
    assert isinstance(models, EventModels)
    assert models.tagger is fake_tagger
    assert models.clap is fake_clap
    assert models.clap_text_emb is not None
    assert models.clap_text_emb.shape == (len(cfg.classes), 8)
    assert models.class_ids == cfg.class_ids
    assert models.tagger_map.shape == (len(cfg.classes), len(audioset_names))
    assert models.audioset_names == audioset_names


def test_load_models_with_clap_disabled(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    """load_models skips CLAP loading when events.clap.enabled is false."""
    disabled_cfg = copy.deepcopy(cfg)
    disabled_cfg.pipeline["events"]["clap"]["enabled"] = False

    audioset_names = sorted({name for c in cfg.classes for name in c.audioset})
    fake_tagger = FakeTagger(audioset_names)

    monkeypatch.setattr(detector, "load_tagger", lambda c: (fake_tagger, audioset_names))

    def fail_load_clap(c):
        pytest.fail("load_clap should not be called when clap.enabled is false")

    monkeypatch.setattr(detector, "load_clap", fail_load_clap)

    models = detector.load_models(disabled_cfg)
    assert isinstance(models, EventModels)
    assert models.tagger is fake_tagger
    assert models.clap is None
    assert models.clap_text_emb is None
    assert models.class_ids == cfg.class_ids
    assert models.audioset_names == audioset_names


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
