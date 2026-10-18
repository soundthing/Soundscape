from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from soundthing.config import ConfigError, load_config
from soundthing.contracts import EnvSegment
from soundthing.events.conditioning import apply_priors, prior_weights
from soundthing.events.types import FrameScores


def test_fitted_priors_cover_every_class_without_vetoes(cfg):
    assert cfg.priors.application=='odds'
    for env in cfg.env_ids:
        weights=prior_weights(cfg,{env:1.},cfg.class_ids)
        assert np.all((weights>=cfg.priors.floor)&(weights<=cfg.priors.max_gain))
        for i,cid in enumerate(cfg.class_ids):
            if cid in cfg.safety_ids():assert weights[i]==1
    assert cfg.priors.environments['kitchen'].weight('sizzling')>1
    assert cfg.priors.environments['office'].weight('typing')>1
    assert all(set(ep.classes) <= set(cfg.class_ids) for ep in cfg.priors.environments.values())


def test_every_class_can_cross_its_threshold_in_every_environment(cfg):
    scores=FrameScores(np.array([.5]),cfg.class_ids,np.full((1,len(cfg.classes)),.9999),.5)
    thresholds=np.array([cfg.threshold(cid) for cid in cfg.class_ids])
    for env in cfg.env_ids:
        result=apply_priors(scores,[EnvSegment(0,1,env,1.,{env:1.})],cfg)
        assert np.all(result.scores[0]>thresholds),env
    assert np.all(scores.scores==.9999)


def test_old_config_defaults_to_multiplication(cfg,tmp_path):
    for file in cfg.config_dir.glob('*.yaml'):shutil.copy2(file,tmp_path/file.name)
    raw={'floor':.25,'environments':{env:{'default':1.,'classes':{}} for env in cfg.env_ids}}
    (tmp_path/'priors.yaml').write_text(yaml.safe_dump(raw))
    legacy=load_config(tmp_path)
    assert legacy.priors.application=='multiplicative'
    assert legacy.priors.max_gain==4
    assert legacy.priors.coefficient_format=='bounded'
    legacy.priors.environments['kitchen'].default=.2
    scores=FrameScores(np.array([.5]),['typing'],np.array([[.8]]),.5)
    result=apply_priors(scores,[EnvSegment(0,1,'kitchen',1.,{'kitchen':1.})],legacy)
    assert np.allclose(result.scores,[[.8*(.25+.75*.2)]])


@pytest.mark.parametrize('field,value', [('application','unknown'),('floor',0),('max_gain',.5),('max_gain',float('nan'))])
def test_invalid_odds_parameters_rejected(cfg,field,value):
    copy=deepcopy(cfg);setattr(copy.priors,field,value)
    with pytest.raises(ConfigError):copy.validate()


@pytest.mark.parametrize('value',[0,-.1,4.1,float('nan'),float('inf'),True])
def test_invalid_odds_coefficients_rejected(cfg,value):
    copy=deepcopy(cfg);copy.priors.environments['kitchen'].classes['typing']=value
    with pytest.raises(ConfigError):copy.validate()


def test_legacy_coefficient_range_is_not_loosened(cfg):
    copy=deepcopy(cfg);copy.priors.application='multiplicative';copy.priors.coefficient_format='bounded'
    with pytest.raises(ConfigError):copy.validate()


def test_general_must_remain_neutral(cfg):
    copy=deepcopy(cfg);copy.priors.environments['general'].default=2
    with pytest.raises(ConfigError,match='general must be neutral'):copy.validate()


def test_published_coefficients_reproduce_from_recorded_counts(cfg):
    import json
    from soundthing.events.prior_fit import estimate_coefficients
    report=json.loads((Path(__file__).parents[1]/'docs/research/sound-priors-report.json').read_text())
    # Preserve the original fitting evidence, including its retired neutral columns.
    selected=[report['classes'].index(cid) for cid in cfg.class_ids]
    assert sum(n>0 for n in report['global_positive_counts'])==527
    assert report['training_clips']+report['heldout_clips']==report['joined_clips']
    safety=[report['classes'].index(c.id) for c in cfg.classes if c.safety]
    params={k:report['settings'][k] for k in ['alpha','min_class','min_environment','floor','max_gain']}
    for env,stats in report['environment_stats'].items():
        weights,*_=estimate_coefficients(report['global_positive_counts'],report['training_clips'],stats['positive_counts'],
                                         stats['training_clips'],safety=safety,apply_floor=False,**params)
        actual=np.array([cfg.priors.environments[env].weight(cid) for cid in cfg.class_ids])
        assert np.allclose(actual,weights[selected],rtol=0,atol=.00000051),env
        assert stats['heldout_annotation_logloss_adjusted']<stats['heldout_annotation_logloss_baseline']


def test_legacy_bounded_format_still_rejects_values_below_floor(cfg):
    legacy=deepcopy(cfg)
    legacy.priors.coefficient_format='bounded'
    for ep in legacy.priors.environments.values():
        ep.classes={cid:max(legacy.priors.floor,w) for cid,w in ep.classes.items()}
    legacy.validate()
    legacy.priors.environments['kitchen'].classes['train_horn']=.1
    with pytest.raises(ConfigError,match='train_horn'):
        legacy.validate()


@pytest.mark.parametrize('format,application', [('unknown','odds'),('unfloored','multiplicative')])
def test_invalid_coefficient_format_rejected(cfg,format,application):
    copy=deepcopy(cfg)
    copy.priors.coefficient_format=format
    copy.priors.application=application
    with pytest.raises(ConfigError):copy.validate()


def test_all_former_floor_entries_are_recovered_without_changing_effective_defaults(cfg):
    import json
    from soundthing.events.prior_fit import estimate_coefficients
    report=json.loads((Path(__file__).parents[1]/'docs/research/sound-priors-report.json').read_text())
    selected=[report['classes'].index(cid) for cid in cfg.class_ids]
    safety=[report['classes'].index(cid) for cid in cfg.safety_ids()]
    params={key:report['settings'][key] for key in ['alpha','min_class','min_environment','floor','max_gain']}
    recovered=0
    expected={}
    assert cfg.priors.coefficient_format=='unfloored'
    for env,stats in report['environment_stats'].items():
        old,*_=estimate_coefficients(report['global_positive_counts'],report['training_clips'],stats['positive_counts'],
                                     stats['training_clips'],safety=safety,**params)
        expected[env]=np.round(old[selected],6)
        np.testing.assert_allclose(prior_weights(cfg,{env:1},cfg.class_ids),expected[env],rtol=0,atol=1e-12)
        recovered+=sum(w<cfg.priors.floor for w in cfg.priors.environments[env].classes.values())
    assert recovered==3045
    probabilities={'kitchen':.35,'office':.25,'nature':.2,'general':.2}
    mixed=sum(p*(np.ones(len(cfg.classes)) if env=='general' else expected[env]) for env,p in probabilities.items())
    np.testing.assert_allclose(prior_weights(cfg,probabilities,cfg.class_ids),mixed,rtol=0,atol=1e-12)


def test_runtime_floor_is_adjustable_without_rewriting_estimates(cfg):
    copy=deepcopy(cfg)
    raw=copy.priors.environments['kitchen'].weight('train_horn')
    assert 0<raw<.05
    for minimum,expected in [(.25,.25),(.05,.05),(.01,raw),(.8,.8)]:
        copy.priors.floor=minimum
        copy.validate()
        assert copy.priors.environments['kitchen'].weight('train_horn')==raw
        assert prior_weights(copy,{'kitchen':1},['train_horn'])[0]==expected
        # Clamp each environment before mixing, preserving uncertain/general behavior.
        assert prior_weights(copy,{'kitchen':.5,'general':.5},['train_horn'])[0]==(expected+1)/2


def test_unfloored_default_coefficient_is_bounded_and_safety_stays_neutral(cfg):
    copy=deepcopy(cfg)
    copy.priors.floor=.05
    copy.priors.environments['kitchen'].default=.01
    copy.priors.environments['kitchen'].classes.pop('train_horn')
    copy.validate()
    assert prior_weights(copy,{'kitchen':1},['train_horn','siren']).tolist()==[.05,1]


def test_stub_and_real_conditioning_share_unfloored_minimum(cfg,fake_inputs):
    from soundthing.contracts import load_inputs
    from soundthing.events.stub import detect_stub
    copy=deepcopy(cfg)
    copy.priors.floor=.05
    copy.validate()
    bounded=deepcopy(copy)
    bounded.priors.coefficient_format='bounded'
    for ep in bounded.priors.environments.values():
        ep.classes={cid:max(bounded.priors.floor,w) for cid,w in ep.classes.items()}
    bounded.validate()
    inputs=load_inputs(fake_inputs)
    segments=[EnvSegment(0,inputs.duration,'kitchen',1,{'kitchen':1})]
    assert detect_stub(inputs,segments,copy)==detect_stub(inputs,segments,bounded)
