from copy import deepcopy

import pytest
import yaml

from soundthing.config import ConfigError, SoundClass


def test_old_classes_default_to_no_explicit_parents(cfg):
    assert SoundClass('x', 'x', audioset=['x']).parents == []
    cfg.validate()


def test_hierarchy_example_supports_transitive_and_multiple_parents(cfg):
    from pathlib import Path
    example = yaml.safe_load((Path(__file__).parents[1] / 'docs/examples/hierarchy.yaml').read_text())
    copy = deepcopy(cfg)
    copy.classes.extend(SoundClass(**record) for record in example['classes'])
    copy.classes.append(SoundClass('example_voice', 'voice', audioset=['Speech']))
    copy.classes[-2].parents.append('example_voice')
    copy.validate()
    assert copy.hierarchy()['example_bark'] == {'example_dog', 'example_animal', 'example_voice'}
    assert copy.hierarchy()['example_animal'] == set()


@pytest.mark.parametrize('parents, expected', [(['missing'], 'unknown parent'), (['siren'], 'cycle'),
                                             (['car_horn', 'car_horn'], 'duplicate parent'), ('siren', 'list of ids')])
def test_invalid_hierarchy_rejected(cfg, parents, expected):
    copy = deepcopy(cfg)
    copy.sound_class('siren').parents = parents
    with pytest.raises(ConfigError, match=expected):
        copy.validate()


def test_multi_node_cycle_rejected(cfg):
    copy = deepcopy(cfg)
    copy.sound_class('siren').parents = ['car_horn']
    copy.sound_class('car_horn').parents = ['siren']
    with pytest.raises(ConfigError, match='cycle'):
        copy.validate()
