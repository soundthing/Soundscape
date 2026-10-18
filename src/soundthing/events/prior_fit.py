"""Estimate environment/sound enrichment from AudioSet + Sound-VECaps metadata.

These are weak dataset associations, not calibrated real-world probabilities.
No audio is downloaded or inferred. Writes a reviewable proposal, never live priors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import yaml

from ..config import Config, load_config
from ..contracts import GENERAL_ENV


def read_segments(path: Path):
    with path.open(encoding='utf-8') as source:
        for row in csv.reader((line for line in source if not line.startswith('#')), skipinitialspace=True):
            if row:
                if len(row) != 4:
                    raise ValueError(f'Malformed AudioSet segment row in {path.name}')
                yield row[0], float(row[1]), row[3].split(',')


def match_environment(caption: str, patterns: dict[str, list[re.Pattern]], uncertainty: re.Pattern) -> str | None:
    if uncertainty.search(caption):
        return None
    matches = [env for env, expressions in patterns.items() if any(p.search(caption) for p in expressions)]
    return matches[0] if len(matches) == 1 else None


def estimate_coefficients(global_counts, total, env_counts, env_total, *, alpha=200., min_class=20, min_environment=50,
                          floor=.25, max_gain=4., safety=(), apply_floor=True):
    """Positive relative enrichment, optionally retaining estimates below the runtime floor."""
    global_counts = np.asarray(global_counts, dtype=float)
    env_counts = np.asarray(env_counts, dtype=float)
    if total <= 0 or alpha <= 0 or not 0 < floor <= 1 <= max_gain:
        raise ValueError('Invalid prior estimation parameters')
    if env_total < 0 or np.any(global_counts < 0) or np.any(global_counts > total) or np.any(env_counts < 0) or np.any(env_counts > env_total):
        raise ValueError('Invalid annotation counts')
    base = (global_counts + .5) / (total + 1.)
    conditional = (env_counts + alpha * base) / (env_total + alpha)
    reliability = global_counts / (global_counts + min_class)
    weights = np.exp(np.log(np.maximum(conditional, 1e-15) / base) * reliability)
    weights = np.minimum(weights, max_gain)
    if apply_floor:
        weights = np.maximum(weights, floor)
    weights[global_counts < min_class] = 1.
    if env_total < min_environment:
        weights[:] = 1.
    weights[list(safety)] = 1.
    return weights, base, conditional


def fit(cfg: Config, data: Path, settings: dict, labels_path: Path, *, progress=print):
    settings = dict(settings)
    unfloored = cfg.priors.coefficient_format == "unfloored"
    # New-format fitting and inference share one minimum in priors.yaml.
    if unfloored:
        settings['floor'] = cfg.priors.floor
    else:
        settings.setdefault('floor', cfg.priors.floor)
    with labels_path.open() as source:
        labels = list(csv.DictReader(source))
    model_index = {row['mid']: i for i, row in enumerate(labels)}
    names = {row['display_name']: row['mid'] for row in labels}
    ontology = json.loads((data/'ontology.json').read_text())
    node_parents = {node['id']: set() for node in ontology}
    for node in ontology:
        for child in node['child_ids']:
            node_parents[child].add(node['id'])
    ancestry = {}
    def ancestors(mid, active=frozenset()):
        if mid in active:
            raise ValueError('Ontology cycle')
        if mid not in ancestry:
            ancestry[mid] = node_parents[mid] | {a for p in node_parents[mid] for a in ancestors(p, active | {mid})}
        return ancestry[mid]
    masks = {mid: sum(1 << model_index[m] for m in ({mid} | ancestors(mid)) if m in model_index) for mid in model_index}
    forbidden = {vid for vid, _, _ in read_segments(data/'eval_segments.csv')}
    for name in ['audiocaps_val.csv','audiocaps_test.csv']:
        with (data/name).open() as source:
            forbidden.update(row['youtube_id'] for row in csv.DictReader(source))
    metadata = {}
    starts = {}
    ambiguous = set()
    for name in ['balanced_train_segments.csv','unbalanced_train_segments.csv']:
        for vid, start, mids in read_segments(data/name):
            if vid in forbidden:
                continue
            if vid in starts and starts[vid] != start:
                ambiguous.add(vid)
                continue
            starts[vid] = start
            mask = 0
            for mid in mids:
                if mid not in masks:
                    raise ValueError(f'Unknown AudioSet label: {mid}')
                mask |= masks[mid]
            metadata[vid] = mask
    del starts
    for vid in ambiguous:
        metadata.pop(vid, None)
    expressions = settings['environment_patterns']
    if not set(expressions) <= set(cfg.env_ids) - {GENERAL_ENV}:
        raise ValueError('Unknown or general environment in fitting patterns')
    patterns = {env: [re.compile(p, re.I) for p in patterns] for env, patterns in expressions.items()}
    uncertainty = re.compile(settings['uncertainty_pattern'], re.I)
    class_masks = []
    # Each label contributes once per clip, including its known ontology ancestors.
    for sound in cfg.classes:
        class_masks.append(sum(1 << model_index[names[name]] for name in sound.audioset))
    by_model = {i: [] for i in range(len(labels))}
    for ci, mask in enumerate(class_masks):
        for i in by_model:
            if mask & (1 << i):
                by_model[i].append(ci)
    width = len(cfg.classes)
    global_counts = np.zeros(width, dtype=np.int64)
    counts = {env: np.zeros(width, dtype=np.int64) for env in expressions}
    test_counts = {env: np.zeros(width, dtype=np.int64) for env in expressions}
    sizes = dict.fromkeys(expressions, 0); test_sizes = dict.fromkeys(expressions, 0)
    seen = set(); joined = total = heldout = missing = duplicate = 0
    with (data/'Sound-VECaps_full.csv').open(encoding='utf-8', newline='') as source:
        for row in csv.DictReader(source):
            raw_id = row['id']
            vid = raw_id[1:] if len(raw_id) == 12 and raw_id.startswith('Y') else raw_id
            if vid in seen:
                duplicate += 1
                continue
            seen.add(vid)
            mask = metadata.get(vid)
            if mask is None:
                missing += 1
                continue
            joined += 1
            indices = set()
            while mask:
                bit = mask & -mask
                indices.update(by_model[bit.bit_length()-1]); mask -= bit
            indices = list(indices)
            env = match_environment(row['caption'], patterns, uncertainty)
            is_test = int.from_bytes(hashlib.sha256(vid.encode()).digest()[:8], 'big') % 5 == 0
            if is_test:
                heldout += 1
                if env:
                    test_sizes[env] += 1; test_counts[env][indices] += 1
            else:
                total += 1; global_counts[indices] += 1
                if env:
                    sizes[env] += 1; counts[env][indices] += 1
            if joined % 250000 == 0:
                progress(f'Joined {joined:,} clips; environment matches: {sum(sizes.values())+sum(test_sizes.values()):,}', flush=True)
    if not total:
        raise ValueError('No training records joined')
    safety = [i for i, sound in enumerate(cfg.classes) if sound.safety]
    params = {k: settings[k] for k in ['alpha','min_class','min_environment','floor','max_gain']}
    priors = {'application': 'odds', 'coefficient_format': cfg.priors.coefficient_format,
              'floor': settings['floor'], 'max_gain': settings['max_gain'],
              'environments': {GENERAL_ENV: {'default': 1., 'classes': {}}}}
    report = {'coefficient_representation': 'Safety classes use neutral default=1; all classes were considered in estimation.',
              'dataset': 'Sound-VECaps 1.0 + AudioSet training annotations',
              'source_url': 'https://zenodo.org/records/12606207',
              'limitations': ['Environment captions are generated, not human scene ground truth.',
                             'AudioSet labels are incomplete positives; missing labels are not verified negatives.',
                             'Frequencies describe this selected dataset, not real-world event probabilities.',
                             'Held-out scores measure annotation association, not detector accuracy.'],
              'coefficient_format': cfg.priors.coefficient_format,
              'settings': settings, 'training_clips': total, 'heldout_clips': heldout,
              'joined_clips': joined, 'unjoined_or_excluded': missing, 'duplicate_captions': duplicate,
              'excluded_video_ids': len(forbidden), 'ambiguous_segments_excluded': len(ambiguous),
              'classes': cfg.class_ids, 'global_positive_counts': global_counts.tolist(),
              'environment_stats': {}, 'sources': {}}
    for env in cfg.env_ids:
        if env == GENERAL_ENV:
            continue
        weights, base, conditional = estimate_coefficients(global_counts, total, counts.get(env,np.zeros(width)), sizes.get(env,0),
                                                          safety=safety, apply_floor=not unfloored, **params)
        def stored_weight(weight):
            return float(f'{weight:.12g}') if unfloored else round(float(weight),6)
        priors['environments'][env] = {'default': 1., 'classes': {cid: stored_weight(w) for cid,w in zip(cfg.class_ids,weights) if cid not in cfg.safety_ids()}}
        n = test_sizes.get(env,0); positives = test_counts.get(env,np.zeros(width))
        effective = np.clip(weights, settings['floor'], settings['max_gain']) if unfloored else weights
        adjusted = base * effective / (1-base+base*effective)
        def logloss(probs):
            p = np.clip(probs,1e-12,1-1e-12)
            return float(-np.sum(positives*np.log(p)+(n-positives)*np.log1p(-p))/(n*width)) if n else None
        report['environment_stats'][env] = {
            'training_clips': sizes.get(env,0), 'heldout_clips': n,
            'positive_counts': counts.get(env,np.zeros(width,dtype=int)).tolist(),
            'conditional_label_frequency': conditional.tolist(),
            'neutral_classes': int(np.sum(weights==1)), 'boosted_classes': int(np.sum(weights>1)),
            'downweighted_classes': int(np.sum(weights<1)),
            'heldout_annotation_logloss_baseline': logloss(base),
            'heldout_annotation_logloss_adjusted': logloss(adjusted)}
    for path in [*data.glob('*.csv'),data/'ontology.json',labels_path]:
        digest = hashlib.sha256()
        with path.open('rb') as source:
            while block := source.read(1024*1024): digest.update(block)
        report['sources'][path.name] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
    return priors,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--settings',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    cfg=load_config(args.config)
    if args.output.resolve() == (args.config/'priors.yaml').resolve():
        parser.error('Write a proposal to a separate path; shared priors require review')
    settings=yaml.safe_load(args.settings.read_text()) if args.settings else cfg.pipeline['events']['prior_estimation']
    priors, report=fit(cfg,args.data,settings,args.labels)
    args.output.parent.mkdir(parents=True,exist_ok=True); args.report.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(yaml.safe_dump(priors,sort_keys=False),encoding='utf-8')
    args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({env:{k:v for k,v in stats.items() if not isinstance(v,list)} for env,stats in report['environment_stats'].items()},indent=2))


if __name__ == '__main__':
    main()
