"""Reproducibly expand the existing vocabulary to every PANNs AudioSet label.

Writes proposals only; the CLI never overwrites shared configuration implicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def expand_vocabulary(vocabulary: dict, ontology: list[dict], labels: list[dict]) -> tuple[dict, dict]:
    nodes = {node['id']: node for node in ontology}
    names = {node['name']: node['id'] for node in ontology}
    model_ids = {row['mid'] for row in labels}
    if len(model_ids) != len(labels) or any(mid not in nodes for mid in model_ids):
        raise ValueError('AudioSet label ids must be unique ontology nodes')
    parents: dict[str, set[str]] = {mid: set() for mid in nodes}
    for node in ontology:
        for child in node['child_ids']:
            if child not in nodes:
                raise ValueError(f'Unknown ontology child: {child}')
            parents[child].add(node['id'])
    ancestors: dict[str, set[str]] = {}
    def visit(mid, active=frozenset()):
        if mid in active:
            raise ValueError('Cycle in AudioSet ontology')
        if mid not in ancestors:
            ancestors[mid] = parents[mid] | {a for p in parents[mid] for a in visit(p, active | {mid})}
        return ancestors[mid]
    for mid in nodes:
        visit(mid)

    old = vocabulary['classes']
    canonical: dict[str, str] = {}
    for record in old:
        if record.get('audioset'):
            mid = names[record['audioset'][0]]
            if mid in canonical or mid not in model_ids:
                raise ValueError('Existing primary AudioSet labels must be unique model outputs')
            canonical[mid] = record['id']
    for row in labels:
        canonical.setdefault(row['mid'], 'audioset_' + row['mid'].strip('/').replace('/', '_'))

    # Preserve explicitly designated safety labels, including former aliases.
    safety_labels = {name for c in old if c.get('safety') for name in c.get('audioset', [])}
    new = []
    split_aliases = {}
    for record in old:
        result = dict(record)
        if record.get('audioset'):
            primary = record['audioset'][0]
            result['audioset'] = [primary]
            split_aliases[record['id']] = record['audioset'][1:]
            mid = names[primary]
            result['parents'] = sorted(canonical[p] for p in ancestors[mid] & model_ids
                                       if not any(p in ancestors[a] for a in ancestors[mid] & model_ids))
        new.append(result)
    old_ids = {c['id'] for c in old}
    for row in labels:
        mid = row['mid']
        if canonical[mid] in old_ids:
            continue
        direct = ancestors[mid] & model_ids
        closest = {p for p in direct if not any(p in ancestors[a] for a in direct)}
        record = {'id': canonical[mid], 'caption': row['display_name'].lower(),
                  'audioset': [row['display_name']], 'parents': sorted(canonical[p] for p in closest)}
        if row['display_name'] in safety_labels:
            record['safety'] = True
        new.append(record)
    return {'coarse_groups': vocabulary.get('coarse_groups', {}), 'classes': new}, {
        'panns_outputs': len(labels), 'caption_classes': len(new),
        'retained_ids': sorted(old_ids), 'split_aliases': {k: v for k, v in split_aliases.items() if v},
        'label_to_class': {row['mid']: canonical[row['mid']] for row in labels},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vocabulary', type=Path, required=True)
    parser.add_argument('--ontology', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.vocabulary.resolve():
        parser.error('Write a proposal to a separate file; shared config requires review')
    with args.labels.open() as source:
        labels = list(csv.DictReader(source))
    result, report = expand_vocabulary(yaml.safe_load(args.vocabulary.read_text()), json.loads(args.ontology.read_text()), labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
