import re

import numpy as np
import pytest

from soundthing.events.conditioning import adjust_scores
from soundthing.events.prior_fit import estimate_coefficients, match_environment


def test_odds_preserve_boundaries_and_keep_unlikely_sounds_possible():
    assert np.allclose(adjust_scores([0,.5,1],[1,1,1],application='odds'),[0,.5,1])
    assert adjust_scores(.5,.25,application='odds') < .5
    assert adjust_scores(.5,4,application='odds') > .5
    assert adjust_scores(.999,.25,application='odds') > .99
    assert np.all(np.isfinite(adjust_scores([0,1],[.25,4],application='odds')))
    assert np.allclose(adjust_scores([.4,.8],[.2,1]),[.4*(.25+.75*.2),.8])


def test_data_enrichment_boosts_relevant_events_without_zeroing_others():
    weights,base,conditional=estimate_coefficients([200,400,1,200],1000,[150,5,0,150],200,safety=[3])
    assert weights[0]>1 and weights[1]<1
    assert weights[2]==weights[3]==1
    assert np.all((weights>=.25)&(weights<=4))
    assert conditional[0]>base[0] and conditional[1]<base[1]


def test_unsupported_environments_and_classes_are_neutral():
    weights,*_=estimate_coefficients([100,200,0],1000,[1,1,0],2)
    assert np.all(weights==1)
    weights,*_=estimate_coefficients([100,200,0],1000,[100,100,0],100)
    assert weights[-1]==1


def test_estimator_can_preserve_unfloored_evidence_with_support_and_safety_guards():
    raw,*_=estimate_coefficients([100,100,1],100000,[0,0,0],10000,safety=[1],apply_floor=False)
    bounded,*_=estimate_coefficients([100,100,1],100000,[0,0,0],10000,safety=[1])
    expected=(200/(10000+200))**(100/(100+20))
    assert raw[0]==pytest.approx(expected)
    assert 0<raw[0]<.25 and bounded[0]==.25
    assert raw[1]==raw[2]==1


def test_caption_environment_matching_rejects_ambiguity_and_uncertainty():
    patterns={'kitchen':[re.compile(r'\bkitchen\b',re.I)], 'office':[re.compile(r'\boffice\b',re.I)]}
    uncertainty=re.compile(r'\bpossibly\b',re.I)
    assert match_environment('A kitchen counter',patterns,uncertainty)=='kitchen'
    assert match_environment('Possibly in a kitchen',patterns,uncertainty) is None
    assert match_environment('An office with a kitchen',patterns,uncertainty) is None
    assert match_environment('A roadside scene',patterns,uncertainty) is None


def test_bad_estimation_inputs_fail():
    with pytest.raises(ValueError):estimate_coefficients([2],1,[1],1)
    with pytest.raises(ValueError):estimate_coefficients([1],1,[1],1,floor=0)
    with pytest.raises(ValueError):adjust_scores(.5,1,application='unknown')


def test_fit_joins_only_training_ids_excludes_evaluation_and_deduplicates(cfg, tmp_path):
    import csv
    from copy import deepcopy
    import hashlib
    import json
    from soundthing.config import SoundClass
    from soundthing.events.prior_fit import fit
    copy=deepcopy(cfg)
    copy.priors.floor=.05  # A stale settings floor must not override the single runtime knob.
    copy.classes=[SoundClass('dog','dog',audioset=['Dog']),SoundClass('bark','bark',audioset=['Bark'],parents=['dog']),
                  SoundClass('speech','speech',audioset=['Speech'])]
    labels=tmp_path/'labels.csv'
    labels.write_text('index,mid,display_name\n0,/dog,Dog\n1,/bark,Bark\n2,/speech,Speech\n')
    (tmp_path/'ontology.json').write_text(json.dumps([
        {'id':'/dog','name':'Dog','child_ids':['/bark']}, {'id':'/bark','name':'Bark','child_ids':[]},
        {'id':'/speech','name':'Speech','child_ids':[]}]))
    ids=[f'v{i:010}' for i in range(40)]
    (tmp_path/'balanced_train_segments.csv').write_text('# training\n')
    (tmp_path/'unbalanced_train_segments.csv').write_text(''.join(f'{vid}, 0, 10, "/bark"\n' for vid in ids))
    (tmp_path/'eval_segments.csv').write_text(f'{ids[0]}, 0, 10, "/bark"\n')
    (tmp_path/'audiocaps_val.csv').write_text(f'youtube_id\n{ids[1]}\n')
    (tmp_path/'audiocaps_test.csv').write_text(f'youtube_id\n{ids[2]}\n')
    with (tmp_path/'Sound-VECaps_full.csv').open('w') as source:
        writer=csv.writer(source);writer.writerow(['id','caption'])
        writer.writerows([('Y'+vid,'In a kitchen') for vid in ids]+[('Y'+ids[3],'In a kitchen')])
    settings={'alpha':10.,'min_class':1,'min_environment':1,'floor':.25,'max_gain':4.,
              'uncertainty_pattern':r'possibly','environment_patterns':{'kitchen':[r'\bkitchen\b']}}
    priors,report=fit(copy,tmp_path,settings,labels,progress=lambda *args,**kw:None)
    train=sum(int.from_bytes(hashlib.sha256(v.encode()).digest()[:8],'big')%5!=0 for v in ids[3:])
    assert report['joined_clips']==37
    assert report['training_clips']==train
    assert report['heldout_clips']==37-train
    assert report['duplicate_captions']==1
    assert report['global_positive_counts']==[train,train,0]
    assert report['environment_stats']['kitchen']['training_clips']==train
    assert priors['environments']['office']['classes']=={'dog':1.,'bark':1.,'speech':1.}
    assert priors['environments']['kitchen']['classes']['speech']==1.
    assert report['sources']['Sound-VECaps_full.csv']['sha256']
    assert priors['coefficient_format']=='unfloored'
    assert priors['floor']==copy.priors.floor
    assert report['coefficient_format']=='unfloored'


def test_runtime_minimum_does_not_change_unfloored_estimates():
    # The public estimator must not change recovered evidence when policy changes.
    first,*_=estimate_coefficients([100],100000,[0],10000,floor=.25,apply_floor=False)
    second,*_=estimate_coefficients([100],100000,[0],10000,floor=.05,apply_floor=False)
    np.testing.assert_array_equal(first,second)
