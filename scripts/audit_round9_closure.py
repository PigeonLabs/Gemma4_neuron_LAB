#!/usr/bin/env python3
"""Check R9's actual work and allowed stop without asserting discovery success."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    started = time.monotonic()
    config = yaml.safe_load((ROOT / 'configs/round9_closure_audit.yaml').read_text())
    registration = read('logs/summary/round9_P0_registration.json')
    freeze = read('logs/summary/round9_measurement_freeze.json')
    protected = {**registration['protected_prior_sha256'], **freeze['protected_measured_sha256']}
    for path, expected in protected.items():
        assert sha(ROOT / path) == expected, path
    assert len(registration['protected_prior_sha256']) == 139
    assert len(freeze['protected_measured_sha256']) == 20
    for path, key in [('runs/round9_P0_manifest.json', 'manifest_sha256'),
                      ('runs/round9_P0_encoded_inputs.json', 'encoded_inputs_sha256')]:
        assert sha(ROOT / path) == registration[key]
    assert registration['model_outputs_seen'] == 0

    counts = {}
    for key in 'ABCDE':
        summary = read(f'logs/summary/round9_P0_{key}.json')
        raw = ROOT / 'runs' / summary['run_id'] / 'rows.jsonl'
        assert sha(raw) == summary['raw_rows_sha256']
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        counts[key] = len(rows)
        assert counts[key] == summary['actual_forward_calls']
        assert summary['P1'] == summary['P2'] == 'NOT_EVALUATED'
    technical = read('logs/summary/round9_P0_technical.json')
    assert technical['status'] == 'PASS' and technical['actual_forward_calls'] == 34
    assert all(all(item['checks'].values()) for item in technical['records'])
    assert counts == dict(A=896, B=96, C=96, D=540, E=128)
    total = sum(counts.values()) + technical['actual_forward_calls']
    assert total == read('runs/round9_global_budget.json')['calls'] == 1790
    replay = read('logs/summary/round9_P0_replay.json')
    assert replay['status'] == 'PASS' and replay['original_actual_model_forwards_verified'] == total
    assert replay['new_model_forwards'] == 0
    assert read('logs/summary/round9_chat_mode_audit.json')['registered_inputs_checked'] == 11900
    guard = read('logs/summary/round9_interpretation_guard.json')
    assert guard['status'] == 'PASS'
    assert all(guard['studies'][key]['primary_estimate'] is None and
               not guard['studies'][key]['estimand_valid'] for key in 'ABCE')
    assert guard['studies']['D']['primary_estimate'] == dict(selective_families=0, families=8)

    raw_review_files = {}
    for name in ['round9_peer_review', 'round9_result_peer_review']:
        peer = read(f'logs/summary/{name}.json')
        def verify_refs(value):
            if isinstance(value, dict):
                if 'local_path' in value and 'sha256' in value:
                    path = ROOT / value['local_path']
                    assert path.exists() and sha(path) == value['sha256'], str(path)
                    raw_review_files[value['local_path']] = value['sha256']
                for item in value.values():
                    verify_refs(item)
            elif isinstance(value, list):
                for item in value:
                    verify_refs(item)
        verify_refs(peer)
    peer = read('logs/summary/round9_result_peer_review.json')
    assert peer['status'] == 'BOTH_INDEPENDENT_AND_ADVERSARIAL_CROSS_COMPLETE'
    assert peer['adversarial_cross']['status'] == 'ACTUALLY_COMPLETED_BOTH_AFTER_INDEPENDENT_COMPLETION'
    for name, evidence in [('GPT', 'Response complete'), ('Claude', 'Claude finished the response')]:
        path = ROOT / peer['adversarial_cross'][name]['raw']['local_path']
        assert evidence in path.read_text()

    prior_lit = read('logs/summary/round7_literature_audit.json')['entries']
    mandatory = ['Where Steering Signals Come From', 'Representation Without Control',
                 'Stress-Testing Random-Vector Controls', 'SPARK', 'Angle–Norm Decomposition']
    for fragment in mandatory:
        papers = [p for p in prior_lit if fragment in p['title']]
        assert papers and papers[0]['read_sections'], fragment
    decision = read('logs/summary/round9_final_decision.json')
    ranking = read('logs/summary/paper_candidate_ranking.json')
    assert not decision['scientific_discovery_achieved']
    assert decision['qualified_new_GPU_studies'] == decision['paper_Go_count'] == 0
    assert ranking['Go_count'] == ranking['Hold_count'] == ranking['active_next_experiment_count'] == 0
    assert len(decision['screened_followup_ideas']) == 5
    nine = ['question', 'novelty', 'connection', 'design', 'controls_metrics',
            'outcomes', 'failure_confounds', 'cost_difficulty', 'paper_contribution']
    assert all(all(candidate.get(key) for key in nine) for candidate in decision['screened_followup_ideas'])
    for entry in ranking['entries']:
        if 'source_sha256' in entry:
            assert sha(ROOT / entry['source']) == entry['source_sha256'], entry['id']

    docs = ['README.md', 'reports/ROUND9_REVIEW_AND_NEXT_KO.md',
            'reports/ROUND9_EXECUTION_RESULTS_KO.md', 'reports/PAPER_CANDIDATE_RANKING_KO.md']
    assert '11900' in (ROOT / 'reports/ROUND9_REVIEW_AND_NEXT_KO.md').read_text()
    assert 'BOTH_INDEPENDENT_AND_ADVERSARIAL_CROSS_COMPLETE' in (ROOT / 'reports/ROUND9_EXECUTION_RESULTS_KO.md').read_text()
    sources = docs + ['logs/summary/round9_final_decision.json', 'logs/summary/round9_result_peer_review.json',
                      'logs/summary/paper_candidate_ranking.json', 'scripts/build_round9_review.py',
                      'scripts/build_round9_results.py', 'scripts/build_readme.py']
    elapsed = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    assert elapsed <= config['maximum_CPU_seconds'] and rss <= config['maximum_RSS_bytes']
    out = dict(schema_version=1, status='PASS', kind='CPU evidence and permitted unsuccessful-search stop audit',
               workflow_outcome=decision['status'], scientific_discovery_achieved=False,
               original_actual_R9_forwards_verified=total, new_model_forwards=0,
               prior_protected_files=139, R9_frozen_files=20,
               both_pre_P0_and_post_result_actual_review_evidence_verified=True,
               private_review_files_verified=len(raw_review_files),
               mandatory_prior_papers_methods_results_recorded=mandatory,
               P1='NOT_EVALUATED', P2='NOT_EVALUATED', paper_Go_count=0,
               allowed_stop_scope='Current search pool only; no universal exhaustion, no scientific success claim.',
               publication_readback='Separate subsequent receipt; this PASS is not a remote publication claim.',
               artifact_sha256={p: sha(ROOT / p) for p in sources},
               CPU_seconds=elapsed, CPU_peak_RSS_bytes=rss,
               script_sha256=sha(Path(__file__)), config_sha256=sha(ROOT / 'configs/round9_closure_audit.yaml'))
    (ROOT / 'logs/summary/round9_closure_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ['status', 'scientific_discovery_achieved',
        'original_actual_R9_forwards_verified', 'new_model_forwards', 'private_review_files_verified']}))


if __name__ == '__main__':
    main()
