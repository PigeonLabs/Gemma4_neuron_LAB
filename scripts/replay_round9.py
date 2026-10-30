#!/usr/bin/env python3
"""CPU reconciliation of original R9 rows, inputs, counters and frozen analyses.

Reusing the registered analysis is a deterministic replay, not independent
scientific evidence. Independent exact-oracle checks live in test_round9.py.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from neuron_lab.round9_worlds import prepared_cases
from neuron_lab.round9_analysis import analyze
from neuron_lab.runtime import save_json


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    started = time.monotonic(); cfg = yaml.safe_load((ROOT / 'configs/round9_P0.yaml').read_text())
    reg = read(ROOT / 'logs/summary/round9_P0_registration.json')
    for rel, h in {**reg['operative_source_sha256'], **reg['protected_prior_sha256']}.items():
        assert sha(ROOT / rel) == h, rel
    manifest = read(ROOT / 'runs/round9_P0_manifest.json'); encs = read(ROOT / 'runs/round9_P0_encoded_inputs.json')
    assert sha(ROOT / 'runs/round9_P0_manifest.json') == reg['manifest_sha256']
    assert sha(ROOT / 'runs/round9_P0_encoded_inputs.json') == reg['encoded_inputs_sha256']
    tech = read(ROOT / 'logs/summary/round9_P0_technical.json')
    assert tech['status'] == 'PASS' and tech['actual_forward_calls'] == 34
    assert all(all(rec['checks'].values()) for rec in tech['records'])
    details = {}; total = tech['actual_forward_calls']; world_counts = {}; anchors = 0; main_rows = 0
    for s in cfg['execution_order']:
        summary = read(ROOT / f'logs/summary/round9_P0_{s}.json'); assert summary['status'] == 'PASS'
        run = ROOT / 'runs' / summary['run_id']; raw = run / 'rows.jsonl'; assert sha(raw) == summary['raw_rows_sha256']
        rows = [json.loads(x) for x in raw.read_text().splitlines()]
        assert len(rows) == summary['actual_forward_calls'] == summary['completed_forward_calls']
        expected = list(prepared_cases(manifest, s, anchors=True))
        if summary['outcome']['gate'] != 'STOPPED_AT_BASIC_ANCHORS': expected += list(prepared_cases(manifest, s))
        assert [r['id'] for r in rows] == [r['id'] for r in expected]
        ids = set()
        for row, case in zip(rows, expected):
            assert row['id'] not in ids; ids.add(row['id'])
            enc = encs[case['id']]
            assert row['authoritative_input_sha256'] == enc['authoritative_input_sha256']
            assert row['tokens'] == enc['tokens']
            assert row['expected'] == case['expected'] and row['oracle'] == case['oracle'] and row['factors'] == case['factors']
            top = row['full_top1_id']; allowed = row['answer_token_ids']
            valid = top in allowed; pred = allowed.index(top) if valid else None
            assert row['format_valid'] == valid and row['full_prediction'] == pred
            correct = valid and case['expected'] is not None and case['semantic_codes'][pred] == case['expected']
            assert row['full_correct'] == bool(correct)
            assert row['semantic_one'] == row['semantic_yes'] == int(pred == 1)
            assert abs(row['semantic_logit_difference'] - (row['answer_logits'][1] - row['answer_logits'][0])) <= 1e-5
            assert 0 <= row['valid_probability_mass'] <= 1.000001
            assert abs(sum(row['restricted_probabilities']) - 1) < 1e-12
            assert not row['probe_delivery'], 'Unregistered scientific activation intervention'
        replay = analyze(s, rows, cfg)
        assert replay == summary['outcome'], 'Frozen analysis replay changed'
        meta = read(run / 'metadata.json')
        assert meta['config'] == cfg and summary['frozen_parameters_verified'] and summary['use_cache'] is False
        for rel, h in reg['operative_source_sha256'].items():
            if rel in meta['source_sha256']:
                assert meta['source_sha256'][rel] == h, rel
        counts = Counter(r['role'] for r in rows); nmain = counts['main'] + counts['reference']
        anchors += counts['anchor']; main_rows += nmain; total += len(rows)
        world_counts[s] = len({r['episode_id'] for r in rows if r['role'] == 'main'})
        details[s] = {'run_id': summary['run_id'], 'actual_science_forwards': len(rows), 'role_counts': dict(counts),
                      'classification': replay['classification'], 'gate': replay['gate'],
                      'raw_rows_sha256': sha(raw), 'source_input_counter_replay': 'PASS',
                      'recorded_full_vocab_mass_recomputed': False,
                      'full_vocab_limitation': 'Top1 and full-vocabulary mass are original forward records; all262144 logits were not persisted, so CPU replay validates their use/consistency, not a second model evaluation.'}
    budget = read(ROOT / 'runs/round9_global_budget.json')
    assert total == budget['calls'] <= cfg['P0_round_actual_forward_cap']
    assert [x['study'] for x in budget['completed']] == cfg['execution_order']
    out = {'schema_version': 1, 'status': 'PASS', 'kind': 'CPU raw/input/counter/frozen-analysis reconciliation',
           'new_model_forwards': 0, 'original_actual_model_forwards_verified': total,
           'shared_technical_forwards': tech['actual_forward_calls'], 'science_anchors': anchors, 'science_main_or_reference': main_rows,
           'independent_main_families': world_counts, 'studies': details,
           'protected_prior_files_verified': len(reg['protected_prior_sha256']),
           'registration_sha256': sha(ROOT / 'logs/summary/round9_P0_registration.json'),
           'script_sha256': sha(Path(__file__)), 'CPU_seconds': time.monotonic() - started,
           'independent_oracle_evidence': 'logs/summary/round9_CPU_validation.json; separate synthetic tests, not additional12B observations'}
    save_json(ROOT / 'logs/summary/round9_P0_replay.json', out)
    print(json.dumps({'status': 'PASS', 'original_actual_forwards': total, 'new_forwards': 0, 'studies': 5}), flush=True)


if __name__ == '__main__': main()
