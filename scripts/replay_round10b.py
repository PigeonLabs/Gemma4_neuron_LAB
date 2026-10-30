#!/usr/bin/env python3
"""CPU replay of R10b's actual tokens, declared futility, counters and analyses."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import resource
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from neuron_lab.round10b_repair import prepared_cases, generation_boundary, parse_generated_body, calibration_futility
from neuron_lab.round10b_analysis import analyze, anchor_quality, accuracy
from neuron_lab.input_contracts import token_fingerprint


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    from transformers import AutoTokenizer
    started = time.monotonic()
    cfg = yaml.safe_load((ROOT/'configs/round10b_repair.yaml').read_text())
    regpath = ROOT/'logs/summary/round10b_registration.json'
    reg = read(regpath)
    for path, expected in {**reg['operative_source_sha256'], **reg['protected_prior_sha256']}.items():
        assert sha(ROOT/path) == expected, path
    assert sha(ROOT/'runs/round10b_manifest.json') == reg['manifest_sha256']
    assert sha(ROOT/'runs/round10b_encoded_inputs.json') == reg['encoded_inputs_sha256']
    manifest = read(ROOT/'runs/round10b_manifest.json')
    encoded = read(ROOT/'runs/round10b_encoded_inputs.json')
    tok = AutoTokenizer.from_pretrained(Path((ROOT/'runs/model_path.txt').read_text().strip()),
        local_files_only=True, trust_remote_code=False)
    effective = read(ROOT/'logs/summary/round10b_effective_generation_config.json')['actual_neutral_greedy']
    assert not effective['do_sample'] and effective['suppress_tokens'] is None
    eos = effective['eos_token_id']; eos = [eos] if isinstance(eos, int) else eos
    special = set(tok.all_special_ids)
    technical = read(ROOT/'logs/summary/round10b_technical.json')
    cache = read(ROOT/'logs/summary/round10b_cache_technical.json')
    assert technical['status'] == cache['status'] == 'PASS'
    assert all(all(r['checks'].values()) for r in technical['records'])
    total = technical['actual_forward_calls'] + cache['actual_forward_calls']
    phases = {}; studies = {s: {} for s in cfg['studies']}; admitted = {}
    responses = 0; parse = Counter(); lines = 0
    cal_calls = p0_calls = 0; elapsed = 0.; peak = 0
    for split in ['calibration', 'P0']:
        for study in cfg['execution_order']:
            path = ROOT/f'logs/summary/round10b_{split}_{study}.json'
            if split == 'P0' and not admitted[study]:
                assert not path.exists()
                studies[study]['P0'] = {'status': 'NOT_EVALUATED', 'reason': 'CALIBRATION_NOT_READY'}
                continue
            summary = read(path); assert summary['status'] == 'PASS'
            assert summary['registration_sha256'] == sha(regpath)
            run = ROOT/'runs'/summary['run_id']; raw = run/'rows.jsonl'
            assert sha(raw) == summary['raw_rows_sha256']
            rows = [json.loads(line) for line in raw.read_text().splitlines()]
            meta = read(run/'metadata.json'); assert meta['config'] == cfg
            anchor_plan = list(prepared_cases(manifest, study, split, 'anchor'))
            anchors = [r for r in rows if r['role'] == 'anchor']
            main_plan = list(prepared_cases(manifest, study, split, 'main'))
            mains = [r for r in rows if r['role'] == 'main']
            quality = anchor_quality(anchors, cfg)
            futility = sentinel_futility = None
            if split == 'calibration':
                # Replay each prefix to prove the first declared stop was obeyed.
                for i in range(1, len(anchors)+1):
                    futility = calibration_futility(anchors[:i], anchor_plan, cfg)
                    if i < len(anchors):
                        assert not futility['stop'], 'Skipped declared first futility stop'
                if futility['stop']:
                    assert not mains
                else:
                    assert len(anchors) == len(anchor_plan)
                    for i in range(1, len(mains)+1):
                        invalid = sum(r['semantic_choice'] is None for r in mains[:i])
                        upper = (len(main_plan)-invalid)/len(main_plan)
                        if upper < cfg['main_parse_min']:
                            assert i == len(mains), 'Continued beyond declared sentinel futility'
                            sentinel_futility = {'planned': len(main_plan), 'observed': i,
                                'best_possible_final_parse_fraction': upper}
                    assert sentinel_futility or len(mains) == len(main_plan)
                sentinel_parse = accuracy(mains)['parsed_fraction'] if mains else None
                admitted[study] = (not futility['stop'] and not sentinel_futility and quality['passed']
                    and sentinel_parse is not None and sentinel_parse >= cfg['main_parse_min'])
                expected_outcome = {'gate': 'CALIBRATION_READY' if admitted[study] else 'CALIBRATION_NOT_READY',
                    'readiness': quality, 'sentinel_parse_fraction': sentinel_parse,
                    'elementary_anchor_futility': futility, 'sentinel_parse_futility': sentinel_futility,
                    'anchor_rows_planned': len(anchor_plan), 'anchor_rows_completed': len(anchors),
                    'sentinel_rows_planned': len(main_plan),
                    'single_line_contract_compliant': sum(r['single_line_contract_compliant'] for r in rows),
                    'sentinel_task_accuracy_not_used_for_admission': True,
                    'scientific_hypothesis_result': 'NOT_EVALUATED'}
                assert expected_outcome == summary['outcome']
            else:
                assert len(anchors) == len(anchor_plan)
                assert len(mains) == (len(main_plan) if quality['passed'] else 0)
                assert analyze(study, rows, cfg) == summary['outcome']
            expected = anchor_plan[:len(anchors)] + main_plan[:len(mains)]
            assert [r['id'] for r in rows] == [c['id'] for c in expected]
            for row, case in zip(rows, expected):
                assert all(row[k] == v for k, v in case.items()), row['id']
                enc = encoded[row['id']]
                rendered = tok.apply_chat_template([{'role':'user','content':case['text']}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                assert tok.encode(rendered, add_special_tokens=False) == enc['input_ids']
                assert row['input_tokens'] == len(enc['input_ids']) == len(enc['attention_mask'])
                assert all(v == 1 for v in enc['attention_mask'])
                assert row['authoritative_input_sha256'] == enc['authoritative_input_sha256'] == token_fingerprint(enc['input_ids'])
                ids = row['generated_token_ids']
                complete, exhausted, body = generation_boundary(ids, eos, cfg['max_new_tokens'])
                controls = [t for t in body if t in special]
                text = tok.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                assert text == row['generated_text']
                assert tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == row['generated_text_with_terminal_special']
                parsed = parse_generated_body(text, exhausted=exhausted, internal_special=bool(controls))
                choice = case['semantic_codes'].index(parsed['code']) if parsed['code'] is not None else None
                assert parsed['status'] == row['parse_status'] and parsed['code'] == row['parsed_code']
                assert row['semantic_choice'] == choice
                assert row['correct'] == (choice == case['truth'] if choice is not None else None)
                assert row['actual_model_forwards'] == row['generated_tokens'] == len(ids)
                assert row['generation_complete_EOS'] == complete and row['token_cap_exhausted'] == exhausted
                assert row['internal_special_token_ids'] == controls
                assert row['single_line_contract_compliant'] == (choice is not None and bool(re.fullmatch(r'FINAL: *[AB]', text.strip())))
                assert ids[0] == row['full_vocab_first_token_id']
                assert row['full_vocab_first_token_text'] == tok.decode([ids[0]], clean_up_tokenization_spaces=False)
            calls = sum(r['actual_model_forwards'] for r in rows)
            assert calls == summary['actual_forward_calls'] == summary['generated_tokens']
            assert len(rows) == summary['generated_responses']
            phase = {'responses': len(rows), 'anchor_responses': len(anchors), 'main_or_sentinel_responses': len(mains),
                'actual_forwards': calls, 'raw_rows_sha256': sha(raw),
                'parse_status': dict(Counter(r['parse_status'] for r in rows)),
                'single_line_contract_compliant': sum(r['single_line_contract_compliant'] for r in rows),
                'anchor_accuracy': accuracy(anchors), 'main_or_sentinel_accuracy': accuracy(mains) if mains else None,
                'outcome': summary['outcome'], 'run_id': summary['run_id'],
                'elapsed_seconds': summary['elapsed_seconds']}
            phases[f'{split}_{study}'] = phase; studies[study][split] = phase
            total += calls; responses += len(rows); parse.update(phase['parse_status']); lines += phase['single_line_contract_compliant']
            if split == 'calibration': cal_calls += calls
            else: p0_calls += calls
            elapsed += summary['elapsed_seconds']; peak = max(peak, summary['peak_allocated_bytes'])
    budget = read(ROOT/'runs/round10b_global_budget.json')
    assert total == budget['calls'] and responses == budget['generated_responses']
    assert total <= cfg['max_actual_forward_calls']
    assert total+cfg['prior_actual_forward_entries'] <= cfg['cumulative_forward_cap']
    assert budget['elapsed_seconds'] <= cfg['max_seconds']
    assert budget['elapsed_seconds']+cfg['prior_measured_GPU_seconds'] <= cfg['cumulative_GPU_seconds_cap']
    out = {'schema_version': 1, 'status': 'PASS', 'new_model_forwards': 0,
        'kind': 'CPU replay of actual token IDs, decoded final declarations, exact inputs, counters, declared futility and frozen analyses; no model logits recomputed',
        'original_R10b_forwards_verified': total, 'generated_responses_verified': responses,
        'protected_prior_files_verified': len(reg['protected_prior_sha256']),
        'calibration_admitted': admitted, 'phases': phases, 'registration_sha256': sha(regpath),
        'CPU_seconds': time.monotonic()-started, 'CPU_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'script_sha256': sha(Path(__file__))}
    save(ROOT/'logs/summary/round10b_replay.json', out)
    aggregate = {'schema_version': 1, 'status': 'ACTUAL_REPAIR_EXECUTION_COMPLETE', 'model_revision': cfg['model_revision'],
        'primary_dtype': cfg['primary_dtype'], 'measurement_scope': cfg['measurement_scope'],
        'earlier_R10_forwards': cfg['prior_actual_forward_entries'], 'R10b_technical_forwards': total-cal_calls-p0_calls,
        'R10b_calibration_forwards': cal_calls, 'R10b_P0_forwards': p0_calls,
        'R10b_total_forwards': total, 'cumulative_R10_R10b_forwards': total+cfg['prior_actual_forward_entries'],
        'R10b_responses': responses, 'parse_status': dict(parse), 'single_line_contract_compliant': lines,
        'calibration_admitted': admitted, 'studies': studies, 'P1': 'NOT_EVALUATED', 'P2': 'NOT_EVALUATED',
        'prior_measured_GPU_seconds': cfg['prior_measured_GPU_seconds'], 'R10b_process_elapsed_seconds': budget['elapsed_seconds'],
        'cumulative_measured_GPU_seconds': cfg['prior_measured_GPU_seconds']+budget['elapsed_seconds'],
        'peak_allocated_bytes': peak, 'replay_sha256': sha(ROOT/'logs/summary/round10b_replay.json'),
        'limitation': 'Fresh worlds plus changed output wording do not identify a causal effect of wording. Incomplete calibration is qualification failure, never a scientific Negative. Small family bootstrap intervals remain exploratory.'}
    save(ROOT/'logs/summary/round10b_aggregate.json', aggregate)
    print(json.dumps({k: aggregate[k] for k in ['status','R10b_total_forwards','R10b_responses','calibration_admitted','cumulative_R10_R10b_forwards']}))


if __name__ == '__main__':
    main()
