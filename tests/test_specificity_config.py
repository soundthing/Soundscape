from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from soundthing.config import ConfigError, Thresholds, load_config
from soundthing.contracts import load_inputs
from soundthing.events import postprocess
from soundthing.events.hierarchy import suppress_ancestors
from soundthing.events.stub import detect_stub
from soundthing.events.types import FrameScores


def test_every_scored_label_has_an_explicit_specificity_assignment(cfg):
    assert len(cfg.classes) == 527
    assert set(cfg.class_ids) <= cfg.thresholds.class_tiers.keys()
    assert set(cfg.thresholds.class_tiers) == set(cfg.class_ids) | set(cfg.coarse_groups)
    assert cfg.priors.floor == .25
    assert cfg.threshold('microwave_beep') == .8
    assert cfg.threshold('typing') == .65
    assert cfg.threshold('audioset_m_04rlf') == .5  # Music


def test_one_tier_setting_changes_all_its_labels_but_preserves_scores(cfg):
    copy = deepcopy(cfg)
    copy.thresholds.tiers['fine'] = .9
    labels = [c.id for c in cfg.classes
              if copy.thresholds.class_tiers[c.id] == 'fine' and not c.safety]
    assert len(labels) > 100
    assert all(copy.threshold(cid) == .9 for cid in labels)
    scores = FrameScores(np.array([.5, .75]), ['microwave_beep'], np.full((2, 1), .85), .25)
    assert postprocess.threshold_to_events(scores, copy) == []
    event, = postprocess.threshold_to_events(scores, cfg)
    assert event.confidence == .85
    assert np.all(scores.scores == .85)


def test_safety_thresholds_are_exact_and_independent_of_tiers(cfg):
    copy = deepcopy(cfg)
    expected = {cid: cfg.threshold(cid) for cid in cfg.safety_ids()}
    for value in (.1, .99):
        copy.thresholds.tiers = {tier: value for tier in cfg.thresholds.tiers}
        copy.validate()
        assert {cid: copy.threshold(cid) for cid in copy.safety_ids()} == expected


def test_overrides_are_minimums_for_non_safety_labels(cfg):
    copy = deepcopy(cfg)
    assert copy.threshold('thud') == .6
    assert copy.threshold('door_slam') == .65  # old 0.55 minimum, new tier 0.65
    copy.thresholds.tiers['specific'] = .4
    assert copy.threshold('door_slam') == .55
    copy.thresholds.classes['typing'] = .9
    assert copy.threshold('typing') == .9


def test_unassigned_labels_and_legacy_thresholds_keep_the_old_lookup(cfg):
    copy = deepcopy(cfg)
    copy.thresholds = Thresholds(default=.4, safety_default=.6, classes={'gunshot': .75, 'typing': .55})
    assert copy.thresholds.tiers == copy.thresholds.class_tiers == {}
    assert copy.threshold('typing') == .55
    assert copy.threshold('microwave_beep') == .4
    assert copy.threshold('siren') == .6
    assert copy.threshold('gunshot') == .75
    assert copy.threshold('impact') == .4
    copy.validate()


def test_parent_survives_when_a_precise_child_misses_its_own_tier(cfg):
    # PANNs' independently scored Animal/Dog/Bark outputs stay independent.
    animal, dog, bark = 'audioset_m_0jbk', 'audioset_m_0bt9lr', 'dog_bark'
    scores = FrameScores(np.array([.5, .75]), [animal, dog, bark],
                         np.array([[.7, .6, .7], [.7, .6, .7]]), .25)
    events = suppress_ancestors(postprocess.threshold_to_events(scores, cfg), cfg.hierarchy())
    assert [e.class_id for e in events] == [animal]
    assert events[0].confidence == .7
    stronger = scores.copy_with(np.full((2, 3), .9))
    events = suppress_ancestors(postprocess.threshold_to_events(stronger, cfg), cfg.hierarchy())
    assert [e.class_id for e in events] == [bark]


def test_specificity_example_loads_without_new_vocabulary_or_schema(cfg, tmp_path):
    for path in cfg.config_dir.glob('*.yaml'):
        shutil.copy2(path, tmp_path / path.name)
    example = Path(__file__).parents[1] / 'docs/examples/specificity.yaml'
    shutil.copy2(example, tmp_path / 'thresholds.yaml')
    loaded = load_config(tmp_path)
    assert loaded.threshold('typing') == .65
    assert loaded.threshold('microwave_beep') == .8
    assert loaded.threshold('gunshot') == .75
    assert loaded.threshold('footsteps') == .5  # unassigned, no override
    raw = yaml.safe_load(example.read_text())
    del raw['tiers'], raw['class_tiers']
    (tmp_path / 'thresholds.yaml').write_text(yaml.safe_dump(raw))
    legacy = load_config(tmp_path)
    assert legacy.thresholds.tiers == legacy.thresholds.class_tiers == {}
    assert legacy.threshold('typing') == .5


def test_stub_uses_the_same_tunable_requirement(cfg, fake_inputs):
    copy = deepcopy(cfg)
    copy.classes = [copy.sound_class('typing')]
    copy.thresholds.classes = {}
    copy.thresholds.class_tiers = {'typing': 'specific'}
    inputs = load_inputs(fake_inputs)
    copy.thresholds.tiers['specific'] = .99
    assert detect_stub(inputs, None, copy) == []
    copy.thresholds.tiers['specific'] = .4
    events = detect_stub(inputs, None, copy)
    assert events and all(e.class_id == 'typing' for e in events)
    assert all(e.confidence >= .5 for e in events)  # retain original stub scores


@pytest.mark.parametrize('value', [0, 1, -.1, 1.1, float('nan'), float('inf'), True, '.5'])
def test_invalid_tier_scores_are_rejected(cfg, value):
    copy = deepcopy(cfg)
    copy.thresholds.tiers['fine'] = value
    with pytest.raises(ConfigError, match='tier.*finite and in'):
        copy.validate()


@pytest.mark.parametrize('name', ['', ' ', 3])
def test_invalid_tier_names_are_rejected(cfg, name):
    copy = deepcopy(cfg)
    copy.thresholds.tiers[name] = .7
    with pytest.raises(ConfigError, match='tier names'):
        copy.validate()


@pytest.mark.parametrize('cid, tier', [('missing_class', 'fine'), ('typing', 'missing_tier'),
                                    ('typing', None), ('typing', ['fine'])])
def test_invalid_tier_assignments_are_rejected(cfg, cid, tier):
    copy = deepcopy(cfg)
    copy.thresholds.class_tiers[cid] = tier
    with pytest.raises(ConfigError, match='unknown (class|tier)'):
        copy.validate()


@pytest.mark.parametrize('name', ['tiers', 'class_tiers'])
def test_tier_configuration_must_be_a_mapping(cfg, tmp_path, name):
    for path in cfg.config_dir.glob('*.yaml'):
        shutil.copy2(path, tmp_path / path.name)
    (tmp_path / 'thresholds.yaml').write_text(yaml.safe_dump({name: ['fine']}))
    with pytest.raises(ConfigError, match=f'{name} must be a mapping'):
        load_config(tmp_path)
