from collections import Counter

import pytest

from soundthing.config import ConfigError, SoundClass
from soundthing.events.vocabulary import expand_vocabulary


def test_all_527_model_labels_have_independent_classes(cfg):
    counts=Counter(name for sound in cfg.classes for name in sound.audioset)
    assert len(counts)==527 and set(counts.values())=={1}
    assert len(cfg.classes)==527
    assert all(c.audioset for c in cfg.classes)
    assert cfg.sound_class('dog_bark').audioset==['Bark']
    dog=next(c for c in cfg.classes if c.audioset==['Dog'])
    animal=next(c for c in cfg.classes if c.audioset==['Animal'])
    assert dog.id in cfg.hierarchy()['dog_bark']
    assert animal.id in cfg.hierarchy()['dog_bark']
    assert dog.id not in cfg.hierarchy()[animal.id]


def test_unmapped_sound_classes_are_rejected(cfg):
    from copy import deepcopy

    invalid = deepcopy(cfg)
    invalid.classes.append(SoundClass('unmapped', 'unmapped'))
    with pytest.raises(ConfigError, match='class unmapped: needs audioset labels'):
        invalid.validate()


def test_retired_event_prompts_are_not_a_supported_sound_class_field():
    with pytest.raises(TypeError, match='clap'):
        SoundClass('unmapped', 'unmapped', clap=['an unsupported sound'])


def test_generator_preserves_ids_splits_aliases_and_projects_abstract_nodes():
    vocabulary={'coarse_groups':{},'classes':[{'id':'barking','caption':'barking','audioset':['Bark','Dog'],'safety':True}]}
    ontology=[{'id':'/a','name':'Animal','child_ids':['/abstract']},
              {'id':'/abstract','name':'Domestic animals','child_ids':['/d']},
              {'id':'/d','name':'Dog','child_ids':['/b']}, {'id':'/b','name':'Bark','child_ids':[]}]
    labels=[{'mid':'/a','display_name':'Animal'},{'mid':'/d','display_name':'Dog'},{'mid':'/b','display_name':'Bark'}]
    result,report=expand_vocabulary(vocabulary,ontology,labels)
    classes={c['id']:c for c in result['classes']}
    assert classes['barking']['audioset']==['Bark']
    assert classes['barking']['parents']==['audioset_d']
    assert classes['audioset_d']['parents']==['audioset_a']
    assert classes['audioset_d']['safety']
    assert report['panns_outputs']==3
    assert vocabulary['classes'][0]['audioset']==['Bark','Dog']
