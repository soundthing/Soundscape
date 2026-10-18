"""Tests for tuning config overrides and touches_shared."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from soundthing.config import Config, ConfigError
from soundthing.harness.overrides import apply_overrides, touches_shared


def _assert_cfgs_equal(a: Config, b: Config) -> None:
    """Assert two Config objects are deeply equal in data."""
    assert a.pipeline == b.pipeline
    assert a.thresholds == b.thresholds
    assert a.priors == b.priors
    assert a.classes == b.classes
    assert a.environments == b.environments
    assert a.coarse_groups == b.coarse_groups


def test_overrides_preserve_input_cfg(cfg: Config) -> None:
    original = copy.deepcopy(cfg)
    cid = cfg.class_ids[0]
    env = cfg.env_ids[0]
    cg = next(iter(cfg.coarse_groups))

    overrides = {
        "events.window_sec": 2.0,
        "thresholds.default": 0.45,
        f"thresholds.classes.{cid}": 0.75,
        f"thresholds.classes.{cg}": 0.65,
        f"vocabulary.{cid}.clap": ["custom clap prompt"],
        "priors.floor": 0.3,
        f"priors.{env}.default": 0.4,
        f"priors.{env}.classes.{cid}": 0.8,
    }

    new_cfg = apply_overrides(cfg, overrides)
    # Input config must be untouched
    _assert_cfgs_equal(cfg, original)
    # Output config must reflect all changes
    assert new_cfg.pipeline["events"]["window_sec"] == 2.0
    assert new_cfg.thresholds.default == 0.45
    assert new_cfg.thresholds.classes[cid] == 0.75
    assert new_cfg.thresholds.classes[cg] == 0.65
    assert new_cfg.sound_class(cid).clap == ["custom clap prompt"]
    assert new_cfg.priors.floor == 0.3
    assert new_cfg.priors.environments[env].default == 0.4
    assert new_cfg.priors.environments[env].classes[cid] == 0.8


def test_events_nested_override(cfg: Config) -> None:
    original = copy.deepcopy(cfg)
    new_cfg = apply_overrides(cfg, {"events.tagger.model": "beats"})
    _assert_cfgs_equal(cfg, original)
    assert new_cfg.pipeline["events"]["tagger"]["model"] == "beats"


def test_thresholds_safety_default_override(cfg: Config) -> None:
    original = copy.deepcopy(cfg)
    new_cfg = apply_overrides(cfg, {"thresholds.safety_default": 0.72})
    _assert_cfgs_equal(cfg, original)
    assert new_cfg.thresholds.safety_default == 0.72


def test_vocabulary_audioset_override(cfg: Config) -> None:
    original = copy.deepcopy(cfg)
    cid = cfg.class_ids[0]
    new_cfg = apply_overrides(cfg, {f"vocabulary.{cid}.audioset": ["New AudioSet Name"]})
    _assert_cfgs_equal(cfg, original)
    assert new_cfg.sound_class(cid).audioset == ["New AudioSet Name"]


@pytest.mark.parametrize(
    "bad_key",
    [
        "events",
        "events.nonexistent_leaf_key",
        "events.tagger.nonexistent_nested_key",
        "events.tagger.model.bad_depth",
        "thresholds",
        "thresholds.invalid_sub",
        "thresholds.classes.nonexistent_class_xyz",
        "thresholds.classes.a.b",
        "vocabulary",
        "vocabulary.nonexistent_class_xyz.clap",
        "vocabulary.nonexistent_class_xyz.audioset",
        "priors",
        "priors.invalid_sub",
        "priors.nonexistent_env_xyz.default",
        "priors.nonexistent_env_xyz.classes.some_class",
        "unknown_section.some_key",
        "unknown_section",
    ],
)
def test_unknown_or_malformed_keys_raise_keyerror(cfg: Config, bad_key: str) -> None:
    with pytest.raises(KeyError) as exc_info:
        apply_overrides(cfg, {bad_key: 0.5})
    assert bad_key in str(exc_info.value) or bad_key.split(".")[0] in str(exc_info.value)


def test_vocabulary_invalid_field_raises_keyerror(cfg: Config) -> None:
    cid = cfg.class_ids[0]
    bad_key = f"vocabulary.{cid}.caption"
    with pytest.raises(KeyError) as exc_info:
        apply_overrides(cfg, {bad_key: ["x"]})
    assert bad_key in str(exc_info.value)


def test_priors_unknown_class_raises_keyerror(cfg: Config) -> None:
    env = cfg.env_ids[0]
    bad_key = f"priors.{env}.classes.nonexistent_class_xyz"
    with pytest.raises(KeyError) as exc_info:
        apply_overrides(cfg, {bad_key: 0.5})
    assert bad_key in str(exc_info.value)


@pytest.mark.parametrize(
    "override_dict",
    [
        {"thresholds.classes.__CID__": 1.5},
        {"thresholds.classes.__CID__": 0.0},
        {"priors.floor": 1.5},
        {"priors.floor": -0.1},
        {"priors.__ENV__.default": 2.0},
        {"priors.__ENV__.classes.__CID__": 1.5},
    ],
)
def test_invalid_values_raise_config_error(cfg: Config, override_dict: dict[str, Any]) -> None:
    cid = cfg.class_ids[0]
    env = cfg.env_ids[0]
    concrete: dict[str, Any] = {}
    for k, v in override_dict.items():
        concrete_key = k.replace("__CID__", cid).replace("__ENV__", env)
        concrete[concrete_key] = v

    with pytest.raises(ConfigError):
        apply_overrides(cfg, concrete)


def test_touches_shared(cfg: Config) -> None:
    cid = cfg.class_ids[0]
    env = cfg.env_ids[0]

    assert not touches_shared({})
    assert not touches_shared({"events.window_sec": 1.0})
    assert not touches_shared({"thresholds.default": 0.5})
    assert not touches_shared({f"thresholds.classes.{cid}": 0.6})
    assert touches_shared({f"vocabulary.{cid}.clap": ["prompt"]})
    assert touches_shared({f"vocabulary.{cid}.audioset": ["label"]})
    assert touches_shared({"priors.floor": 0.3})
    assert touches_shared({f"priors.{env}.default": 0.5})
    assert touches_shared({f"priors.{env}.classes.{cid}": 0.5})
    assert touches_shared({"events.window_sec": 1.0, f"vocabulary.{cid}.clap": ["prompt"]})
