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
    legacy.priors.environments['kitchen'].default=.2
    scores=FrameScores(np.array([.5]),['typing'],np.array([[.8]]),.5)
    result=apply_priors(scores,[EnvSegment(0,1,'kitchen',1.,{'kitchen':1.})],legacy)
    assert np.allclose(result.scores,[[.8*(.25+.75*.2)]])


@pytest.mark.parametrize('field,value', [('application','unknown'),('floor',0),('max_gain',.5),('max_gain',float('nan'))])
def test_invalid_odds_parameters_rejected(cfg,field,value):
    copy=deepcopy(cfg);setattr(copy.priors,field,value)
    with pytest.raises(ConfigError):copy.validate()


@pytest.mark.parametrize('value',[0,.1,4.1,float('nan'),float('inf')])
def test_invalid_odds_coefficients_rejected(cfg,value):
    copy=deepcopy(cfg);copy.priors.environments['kitchen'].classes['typing']=value
    with pytest.raises(ConfigError):copy.validate()


def test_legacy_coefficient_range_is_not_loosened(cfg):
    copy=deepcopy(cfg);copy.priors.application='multiplicative'
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
                                         stats['training_clips'],safety=safety,**params)
        actual=np.array([cfg.priors.environments[env].weight(cid) for cid in cfg.class_ids])
        assert np.allclose(actual,weights[selected],rtol=0,atol=.00000051),env
        assert stats['heldout_annotation_logloss_adjusted']<stats['heldout_annotation_logloss_baseline']
