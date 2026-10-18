import numpy as np

from soundthing.contracts import GENERAL_ENV
from soundthing.environment.fusion import fuse, to_segments
from soundthing.events.conditioning import prior_weights


def test_config_valid(cfg):
    cfg.validate()
    assert GENERAL_ENV in cfg.env_ids
    assert cfg.safety_ids()
    assert cfg.threshold("gunshot") == 0.75
    assert cfg.threshold("siren") == cfg.thresholds.safety_default
    assert cfg.threshold("typing") == cfg.thresholds.default


def test_general_means_no_conditioning(cfg):
    w = prior_weights(cfg, {GENERAL_ENV: 1.0}, cfg.class_ids)
    assert np.allclose(w, 1.0)


def test_safety_bypasses_priors(cfg):
    w = prior_weights(cfg, {"kitchen": 1.0}, ["siren", "car_horn", "microwave_beep", "train_horn"])
    assert list(w[:2]) == [1.0, 1.0]
    assert w[2] == cfg.priors.environments["kitchen"].weight("microwave_beep")
    assert w[3] == cfg.priors.environments["kitchen"].weight("train_horn")


def test_fusion_and_fallback():
    env_ids = ["kitchen", "street"]
    a = np.array([[0.9, 0.1], [0.5, 0.5]])
    v = np.array([[0.7, 0.3], [0.4, 0.6]])
    p = fuse(a, v, 0.5, 0.5)
    segs = to_segments([(0, 5), (5, 10)], p, env_ids, confidence_threshold=0.6)
    assert segs[0].env == "kitchen"
    assert segs[1].env == GENERAL_ENV
    assert abs(segs[1].probs[GENERAL_ENV] - (1 - segs[1].confidence)) < 1e-6
    for s in segs:
        s.validate()
