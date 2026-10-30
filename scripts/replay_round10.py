#!/usr/bin/env python3
"""CPU audit of actual R10 generated outputs and original registered analyses."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from neuron_lab.round10_repair import prepared_cases, parse_generated_body, generation_boundary
from neuron_lab.round10_analysis import analyze, anchor_quality, accuracy
from neuron_lab.input_contracts import token_fingerprint


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from transformers import AutoTokenizer
    started = time.monotonic()
    cfg = yaml.safe_load((ROOT / 'configs/round10_repair.yaml').read_text())
    reg = read(ROOT / 'logs/summary/round10_registration.json')
    assert sha(ROOT / 'configs/round10_repair.yaml') == reg['config_sha256']
    for path, expected in {**reg['operative_source_sha256'], **reg['protected_prior_sha256']}.items():
        assert sha(ROOT / path) == expected, path
    assert sha(ROOT / 'runs/round10_manifest.json') == reg['manifest_sha256']
    assert sha(ROOT / 'runs/round10_encoded_inputs.json') == reg['encoded_inputs_sha256']
    manifest = read(ROOT / 'runs/round10_manifest.json')
    encodings = read(ROOT / 'runs/round10_encoded_inputs.json')
    tokenizer = AutoTokenizer.from_pretrained(Path((ROOT / 'runs/model_path.txt').read_text().strip()),
        local_files_only=True, trust_remote_code=False)
    effective = read(ROOT / 'logs/summary/round10_effective_generation_config.json')['actual_neutral_greedy']
    assert not effective['do_sample'] and effective['suppress_tokens'] is None
    eos = effective['eos_token_id']; eos = [eos] if isinstance(eos, int) else eos
    special = set(tokenizer.all_special_ids)
    technical = read(ROOT / 'logs/summary/round10_technical.json')
    cached = read(ROOT / 'logs/summary/round10_cache_technical.json')
    assert technical['status'] == cached['status'] == 'PASS'
    assert all(all(r['checks'].values()) for r in technical['records'])
    total = technical['actual_forward_calls'] + cached['actual_forward_calls']
    phases = {}; omitted = []; responses = 0
    admitted = {}
    for split in ['calibration', 'P0']:
        for study in cfg['execution_order']:
            path = ROOT / f'logs/summary/round10_{split}_{study}.json'
            if split == 'P0' and not admitted[study]:
                assert not path.exists()
                omitted.append({'study': study, 'P0': 'NOT_EVALUATED', 'reason': 'CALIBRATION_NOT_READY'})
                continue
            summary = read(path); assert summary['status'] == 'PASS'
            run = ROOT / 'runs' / summary['run_id']; raw = run / 'rows.jsonl'
            assert sha(raw) == summary['raw_rows_sha256']
            rows = [json.loads(line) for line in raw.read_text().splitlines()]
            meta = read(run / 'metadata.json'); assert meta['config'] == cfg
            anchors = list(prepared_cases(manifest, study, split, 'anchor'))
            quality = anchor_quality([r for r in rows if r['role'] == 'anchor'], cfg)
            expected = anchors + (list(prepared_cases(manifest, study, split, 'main'))
                                 if split == 'calibration' or quality['passed'] else [])
            assert [r['id'] for r in rows] == [r['id'] for r in expected]
            for row, case in zip(rows, expected):
                for key, value in case.items():
                    assert row[key] == value, (row['id'], key)
                encoded = encodings[row['id']]
                assert row['authoritative_input_sha256'] == encoded['authoritative_input_sha256'] == token_fingerprint(encoded['input_ids'])
                ids = row['generated_token_ids']
                complete, exhausted, body = generation_boundary(ids, eos, cfg['max_new_tokens'])
                controls = [i for i in body if i in special]
                text = tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                assert text == row['generated_text']
                assert tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == row['generated_text_with_terminal_special']
                parsed = parse_generated_body(text, exhausted=exhausted, internal_special=bool(controls))
                semantic = case['semantic_codes'].index(parsed['code']) if parsed['code'] is not None else None
                assert row['parse_status'] == parsed['status'] and row['semantic_choice'] == semantic
                assert row['correct'] == ((semantic == row['truth']) if semantic is not None else None)
                assert row['actual_model_forwards'] == row['generated_tokens'] == len(ids)
                assert row['generation_complete_EOS'] == complete and row['token_cap_exhausted'] == exhausted
                assert row['internal_special_token_ids'] == controls
                assert ids[0] == row['full_vocab_first_token_id']
                assert row['full_vocab_first_token_text'] == tokenizer.decode([ids[0]], clean_up_tokenization_spaces=False)
            calls = sum(r['actual_model_forwards'] for r in rows)
            assert calls == summary['actual_forward_calls'] == summary['generated_tokens']
            assert len(rows) == summary['generated_responses']
            if split == 'calibration':
                sentinel_parse = accuracy([r for r in rows if r['role'] == 'main'])['parsed_fraction']
                admitted[study] = quality['passed'] and sentinel_parse >= cfg['main_parse_min']
                assert summary['outcome']['sentinel_parse_fraction'] == sentinel_parse
                assert summary['outcome']['readiness'] == quality
            else:
                assert analyze(study, rows, cfg) == summary['outcome']
            phases[f'{split}_{study}'] = {'responses': len(rows), 'actual_forwards': calls,
                'gate': summary['outcome']['gate'], 'raw_rows_sha256': sha(raw),
                'parse_status': dict(Counter(r['parse_status'] for r in rows))}
            total += calls; responses += len(rows)
    budget = read(ROOT / 'runs/round10_global_budget.json')
    assert total == budget['calls'] and responses == budget['generated_responses']
    assert total <= cfg['max_actual_forward_calls']
    smoke = read(ROOT / 'logs/summary/round10_generation_smoke.json')
    assert smoke['scientific_response_count'] == 0 and smoke['actual_forward_calls'] == 58
    out = {'schema_version': 1, 'status': 'PASS', 'kind': 'CPU original token/output/input/counter/registered-analysis replay',
           'new_model_forwards': 0, 'original_execution_forwards_verified': total,
           'earlier_separate_technical_forwards': 58, 'all_R10_actual_forwards_verified': total+58,
           'generated_responses_verified': responses, 'calibration_admitted': admitted,
           'omitted_P0': omitted, 'phases': phases,
           'protected_prior_files_verified': len(reg['protected_prior_sha256']),
           'recomputed_model_logits': False,
           'limitation': 'Replays actual decoded token IDs, syntax, exact-input identities, counters and frozen statistics. It is not a second model evaluation or independent scientific evidence.',
           'registration_sha256': sha(ROOT / 'logs/summary/round10_registration.json'),
           'script_sha256': sha(Path(__file__)), 'CPU_seconds': time.monotonic()-started,
           'CPU_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    (ROOT / 'logs/summary/round10_replay.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: out[k] for k in ['status', 'new_model_forwards', 'all_R10_actual_forwards_verified', 'generated_responses_verified']}))


if __name__ == '__main__':
    main()
