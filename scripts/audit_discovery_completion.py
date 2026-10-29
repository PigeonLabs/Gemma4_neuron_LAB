"""Audit the full discovery workflow and its explicitly permitted stop branch.

Requires the retained local raw evidence and fetched Git objects. A PASS means
that the reported unsuccessful-search outcome is supported; it does not mean
that a publishable scientific topic was found. Publication readback is a
separate, subsequent receipt to avoid a self-referential content hash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs/summary/autonomous_discovery_completion_audit.json'
OBJECTIVE = Path('/home/jeong/.codex/attachments/44e4e250-07d3-4bfd-800a-57074b8ec2e9/goal-objective.md')
GIT = ['git', '--git-dir=runs/git-metadata', '--work-tree=.']
REVISION = '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def command(args):
    return subprocess.check_output(args, cwd=ROOT, stderr=subprocess.STDOUT)


def private_receipts(value, results):
    if isinstance(value, dict):
        if 'local_path' in value and 'sha256' in value:
            path = ROOT / value['local_path']
            assert path.is_file() and sha(path) == value['sha256'], path
            results[value['local_path']] = value['sha256']
        for child in value.values():
            private_receipts(child, results)
    elif isinstance(value, list):
        for child in value:
            private_receipts(child, results)


def main():
    baseline = read('logs/summary/round7_baseline_audit.json')
    assert sha(OBJECTIVE) == baseline['goal_objective_sha256']
    objective = OBJECTIVE.read_text()
    selection = read('logs/summary/round8_selection_results.json')
    assert selection['stop_rule_quote'] in objective
    assert baseline['status'] == 'PASS' and baseline['new_model_forwards'] == 0
    assert baseline['R2_F']['damage_gate'] is False
    assert all(not x['eligible'] for x in baseline['R3_A'])
    assert baseline['initial_top1_concentration']['test']['unique_changed_prompts'] == 1

    literature = read('logs/summary/round7_literature_audit.json')
    mandatory = ['source_selection', 'representation_control', 'random_controls', 'spark', 'angle_norm']
    entries = {x['id']: x for x in literature['entries']}
    assert all(entries[k]['read_sections'] and entries[k].get('method') and entries[k].get('result') for k in mandatory)
    later_literature = read('logs/summary/round8_literature_audit.json')
    assert len(later_literature['entries']) == 10

    candidates = read('logs/summary/round7_candidates.json')['ranked_candidates']
    fields = ['question', 'novelty', 'prior_evidence', 'design', 'controls_metrics', 'outcomes', 'confounds', 'cost', 'contribution']
    assert len(candidates) == 5 and {x['id'] for x in candidates} == set('ABCDE')
    assert all(all(x.get(k) for k in fields) for x in candidates)
    new_fields = ['question', 'novelty', 'prior_evidence', 'design', 'controls_metrics', 'outcome_interpretation', 'confounds', 'cost', 'paper_contribution']
    assert len(selection['items']) == 5 and all(all(x.get(k) for k in new_fields) for x in selection['items'])
    assert all(x['decision'] == 'No-Go' and x['model_forwards'] == 0 for x in selection['items'])
    assert selection['scientific_topic_found'] is False
    assert selection['paper_Go_count'] == selection['active_candidate_Hold_count'] == selection['selected_viable_new_model_studies'] == selection['new_Gemma_model_forwards'] == 0

    prereg_names = ['round7_A_preregistration_publication', 'round7_BCDE_preregistration_publication']
    registrations = []
    protected = set()
    totals = [0, 0]
    for letter in 'ABCDE':
        r = read(f'logs/summary/round7_{letter}_P0_registration.json')
        s = read(f'logs/summary/round7_{letter}_P0.json')
        t = read(f'logs/summary/round7_{letter}_P0_technical.json')
        load = read(f'logs/summary/round7_{letter}_P0_model_load.json')
        receipt = read('logs/summary/' + prereg_names[letter != 'A'] + '.json')
        commit = receipt['remote_sha']
        path = f'logs/summary/round7_{letter}_P0_registration.json'
        assert command(GIT + ['show', f'{commit}:{path}']) == (ROOT / path).read_bytes()
        timestamp = command(GIT + ['show', '-s', '--format=%cI', commit]).decode().strip()
        run_start = datetime.strptime(s['run_id'][:15], '%Y%m%dT%H%M%S').replace(tzinfo=ZoneInfo('Asia/Seoul'))
        assert datetime.fromisoformat(timestamp) < run_start
        assert r['study_model_outputs_seen'] == 0 and r['model_revision'] == s['model_revision'] == load['revision'] == REVISION
        assert s['status'] == t['status'] == load['status'] == 'PASS'
        assert s['use_cache'] is False and s['frozen_parameters_verified'] is True
        assert s['P1'] == s['P2'] == 'NOT_EVALUATED'
        cfg = r['scientific_protocol']
        assert s['actual_forward_calls'] + t['actual_forward_calls'] <= cfg['phase_actual_forward_cap']
        assert s['elapsed_seconds'] < cfg['max_seconds']
        assert t['actual_forward_calls'] == 34 <= cfg['technical_call_cap']
        assert load['config']['device'] == 'cuda:0' and load['environment']['gpu']
        assert s['primary_dtype'] == 'FP32_upcast_of_same_stored_BF16_values'
        assert len({x['dtype'] for x in t['records']}) == 2
        assert all(all(x['checks'].values()) for x in t['records'])
        for p, digest in {**r['operative_source_sha256'], **r['protected_prior_sha256']}.items():
            assert sha(ROOT / p) == digest, p
        protected.update(r['protected_prior_sha256'])
        totals[0] += s['actual_forward_calls']
        totals[1] += t['actual_forward_calls']
        registrations.append({'study': letter, 'preregistration_commit': commit,
                              'commit_time': timestamp, 'run_id': s['run_id'],
                              'technical_calls': t['actual_forward_calls'], 'science_calls': s['actual_forward_calls'],
                              'classification': s['outcome']['classification'], 'gate': s['outcome']['gate']})
    assert totals == [4108, 170] and len(protected) == 91

    replay_log = ROOT / 'runs/autonomous_discovery_final_replay.log'
    replay = json.loads(replay_log.read_text().strip().splitlines()[-1])
    assert replay == read('logs/summary/round7_replay.json') and replay['status'] == 'PASS'
    assert replay['total_actual_forwards'] == 4278 and replay['A_protocol_defect_preserved'] == 48
    assert replay['studies'][1]['query_blind_cache_pairs'] == 48
    assert replay['studies'][4]['prospective_predictions_verified'] == 80
    r6 = read('logs/summary/round6_execution_results.json')
    r6_replay = read('logs/summary/round6_execution_replay.json')
    assert r6['total_actual_forwards'] == 744 and r6_replay['status'] == 'PASS'
    r6_log = json.loads((ROOT / 'runs/autonomous_discovery_final_R6_replay.log').read_text().strip())
    assert r6_log == {'status': 'PASS', 'stages': 4, 'rows': 608, 'total_actual_forwards': 744, 'new_model_forwards': 0}

    checked_private = {}
    for name in ['round7_post_results_peer_review', 'round8_peer_review']:
        peer = read('logs/summary/' + name + '.json')
        assert 'COMPLETED' in peer['status']
        private_receipts(peer, checked_private)
    design = read('logs/summary/round7_design_peer_review.json')
    assert design['status'] == 'ACTUAL_BOTH_INDEPENDENT_AND_CROSS_COMPLETED'
    design_paths = ['common_packet.txt', 'cross_packet.txt', 'gpt_cross_full_AX.txt', 'claude_cross_full_AX.txt',
                    'gpt_independent_full_AX.txt', 'claude_independent_full_AX.txt']
    design_hashes = {p: sha(ROOT / 'runs/round7_peer_review' / p) for p in design_paths}
    assert design_hashes['common_packet.txt'] == design['common_core_sha256']
    assert design_hashes['cross_packet.txt'] == design['cross_core_sha256']
    assert design_hashes['gpt_cross_full_AX.txt'] == design['GPT']['raw_sha256']
    assert design_hashes['claude_cross_full_AX.txt'] == design['Claude']['raw_sha256']
    checked_private.update({'runs/round7_peer_review/' + p: h for p, h in design_hashes.items()})
    assert sha(ROOT / selection['peer_review_path']) == selection['peer_review_sha256']

    for script, result in [('scripts/audit_round8_s2_oracle.py', 'round8_s2_oracle_audit'),
                           ('scripts/audit_round8_reuse_identifiability.py', 'round8_reuse_identifiability_audit')]:
        a = read('logs/summary/' + result + '.json')
        assert a['status'] == 'PASS' and a['new_model_forwards'] == 0
        assert sha(ROOT / script) == a['source_sha256']
        assert sha(ROOT / a['budget_config']) == a['budget_config_sha256']
    reuse = read('logs/summary/round8_reuse_identifiability_audit.json')
    assert reuse['total_observables_compared'] == 6144
    assert reuse['same_input_different_table_repair']['observables_compared'] == 20736
    assert reuse['readout_extension']['not_a_Gemma_readout'] is True
    split = read('logs/summary/round7_rendered_split_audit.json')['studies']
    assert split['B']['P0_worlds'] == 12 and split['B']['P0_unique_rendered_world_views'] == 11
    assert split['E']['reserved_prompt_overlap_with_P0']['calibration'] == 0

    tests = (ROOT / 'runs/autonomous_discovery_final_pytest.log').read_text()
    assert re.search(r'157 passed in [0-9.]+s', tests)
    renders = {}
    for script in ['build_readme.py', 'build_round7_review_report.py', 'build_discovery_outcome.py']:
        renders[script] = command([str(ROOT / '.venv/bin/python'), 'scripts/' + script, '--check']).decode().strip()
    rank = read('logs/summary/paper_candidate_ranking.json')
    assert rank['Go_count'] == rank['Hold_count'] == rank['active_next_experiment_count'] == 0
    for entry in rank['entries']:
        assert sha(ROOT / entry['source']) == entry['source_sha256']

    prior_publications = []
    for name in prereg_names + ['round7_execution_publication', 'round7_closure_publication']:
        pub = read('logs/summary/' + name + '.json')
        assert pub['status'] == 'PASS' and pub['remote_readback'] and pub['every_file_hash_verified']
        assert len(pub['allowlist']) == pub['files']
        for entry in pub['allowlist']:
            raw = command(GIT + ['show', pub['remote_sha'] + ':' + entry['path']])
            assert hashlib.sha256(raw).hexdigest() == entry['sha256']
        prior_publications.append({'receipt': 'logs/summary/' + name + '.json', 'sha': pub['remote_sha'], 'verified_files': pub['files']})

    docs = ['README.md', 'reports/AUTONOMOUS_DISCOVERY_OUTCOME_KO.md', 'reports/PAPER_CANDIDATE_RANKING_KO.md', 'reports/ROUND7_REVIEW_AND_NEXT_KO.md']
    link_targets = []
    for name in docs:
        source = ROOT / name
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', source.read_text()):
            target = target.split('#')[0]
            if not target or re.match(r'[a-z]+:', target):
                continue
            path = (source.parent / target).resolve()
            assert path.exists() or path == OUT, (name, target)
            link_targets.append(path)
    command(GIT + ['diff', '--check'])

    source_paths = ['logs/summary/round7_baseline_audit.json', 'logs/summary/round7_candidates.json',
                    'logs/summary/round7_literature_audit.json', 'logs/summary/round8_literature_audit.json',
                    'logs/summary/round7_post_results_peer_review.json', 'logs/summary/round8_peer_review.json',
                    'logs/summary/round7_replay.json', 'logs/summary/round6_execution_results.json',
                    'logs/summary/round6_execution_replay.json', 'logs/summary/round7_rendered_split_audit.json',
                    'logs/summary/round8_selection_results.json', 'logs/summary/paper_candidate_ranking.json',
                    'logs/summary/round8_s2_oracle_audit.json', 'logs/summary/round8_reuse_identifiability_audit.json',
                    'configs/round8_selection.yaml', 'logs/summary/round8_selection_progress.json',
                    'docs/RESEARCH_LOG.md', 'scripts/audit_discovery_completion.py'] + docs
    result = {
        'schema_version': 1, 'date_kst': '2026-10-30', 'status': 'PASS',
        'scope': 'Full-objective evidence and explicit unsuccessful-search termination audit; publication readback is a separate subsequent receipt.',
        'objective_sha256': sha(OBJECTIVE), 'scientific_topic_found': False,
        'scientific_goal_status': 'NOT_ACHIEVED: no novel, causally validated publishable topic found',
        'workflow_terminal_branch': 'PERMITTED_UNSUCCESSFUL_SEARCH_STOP under objective4E and8',
        'stop_rule_quote': selection['stop_rule_quote'],
        'stopping_basis': selection['reason'], 'not_stopping_because': selection['not_a_reason'],
        'requirements': {
            'baseline_raw_reanalysis': {'status': 'PASS', 'evidence': 'logs/summary/round7_baseline_audit.json', 'R2_E_string_gain_fraction': baseline['R2_E']['MSE_reduction_fraction_by_task']['string_rule'], 'old_positive_not_erased': True},
            'mandatory_five_primary_methods': {'status': 'PASS', 'sources': [{'id': k, 'url': entries[k]['url'], 'read_sections': entries[k]['read_sections']} for k in mandatory]},
            'five_distinct_questions_and_nine_fields': {'status': 'PASS_FOR_R7', 'ids': [x['id'] for x in candidates], 'distinct_units': ['logical relevance of evidence', 'prospective cross-intervention prediction', 'pre-query carrier in a modified graph', 'obsolete history at matched final facts', 'answer-independent prefix ordering']},
            'actual_model_execution': {'status': 'ALL_FIVE_R7_BOUNDED_P0_EXECUTED', 'R6_actual_forwards': 744, 'R7_science_forwards': totals[0], 'R7_technical_forwards': totals[1], 'registration_before_outputs': registrations, 'independent_GPU_repetitions': 0},
            'P1_P2': {'status': 'NOT_EVALUATED', 'reason': 'Registered P0 readiness or effect gates did not permit promotion; neither full validation nor phenomenon absence is claimed.'},
            'technical_model_and_resource_provenance': {'status': 'PASS', 'revision': REVISION, 'device': 'cuda:0', 'BF16_and_FP32_hook_checks': 'PASS for each load', 'upcast_is_truth': False, 'single_GPU_sequential_execution': 'Recorded in each run; no new GPU forwards during selection audit'},
            'critical_reviews': {'status': 'ACTUAL_INDEPENDENT_AND_CROSS_COMPLETED', 'private_receipts_verified': len(checked_private), 'mode_limit': 'Actual GPT Pro web review and Claude Opus5.5 Extra; no standalone Deep Research artifact and no reviewer GPU reproduction', 'decision_rule': 'Raw evidence, primary methods and explicit countermodels; not reviewer consensus'},
            'next_five_selection': {'status': 'SCREENED_FIVE_ALL_NO_GO; NO_VIABLE_FIVE_SELECTED', 'Go': 0, 'Hold': 0, 'new_model_forwards': 0, 'nine_fields_retained': True, 'terminal_rule_applied_instead_of_manufacturing_topics': True},
            'data_separation_and_statistics': {'status': 'AUDITED_WITH_DISCLOSED_LIMITATIONS', 'B': '12 latent draws/11 rendered main views; unique-view weighting is posthoc sensitivity', 'unused_reserved_prompt_overlap_P2': {k: split[k]['reserved_prompt_overlap_with_P0']['P2_confirmation'] for k in 'ABCDE'}, 'used_E_calibration_P0_prompt_overlap': 0, 'B_prequery_cache_pairs': 48, 'E_predictions_saved_before_targets': 80, 'P1_P2_not_read_or_silently_replaced': True, 'world_cluster_not_condition_row_inference': True},
            'synthetic_vs_empirical': {'status': 'SEPARATED', 'CPU_raw_input_comparisons': 6144, 'CPU_table_repair_comparisons': 20736, 'actual_R8_model_round': False, 'constructed_equal_logit_head_is_actual_Gemma': False, 'novel_theorem_claim': False},
            'preservation': {'status': 'PASS', 'protected_prior_files': len(protected), 'operative_R7_sources_and_registrations_verified': True, 'A_conflicting_anchors_preserved': 48, 'old_gates_retuned': False},
            'tests_and_renderers': {'status': 'PASS', 'CPU_tests': 157, 'science_time_tests': 153, 'later_infrastructure_tests': 4, 'renderers': renders, 'Markdown_links': len(link_targets)},
            'research_logs_reports_ranking_README': {'status': 'PASS', 'new_topic_claimed': False, 'rank_is_execution_recommendation': False},
        },
        'final_CPU_logs_sha256': {p: sha(ROOT / p) for p in ['runs/autonomous_discovery_final_pytest.log', 'runs/autonomous_discovery_final_replay.log', 'runs/autonomous_discovery_final_R6_replay.log']},
        'private_evidence_sha256': checked_private,
        'prior_publication_readbacks': prior_publications,
        'publication_verification': {'required': True, 'receipt_path': 'logs/summary/autonomous_discovery_publication.json', 'reason_separate': 'Readback occurs after committing this content; a later receipt records its exact remote SHA and per-file hashes.'},
        'source_sha256': {p: sha(ROOT / p) for p in source_paths},
        'reopen_condition': selection['reopening_requires'],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    assert all(p.exists() for p in link_targets)
    print(json.dumps({'status': 'PASS', 'scientific_topic_found': False, 'permitted_stop_branch': True,
                      'R6_forwards': 744, 'R7_forwards': 4278, 'new_R8_forwards': 0,
                      'CPU_tests': 157, 'private_receipts_verified': len(checked_private)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
