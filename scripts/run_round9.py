#!/usr/bin/env python3
"""Register or execute the frozen five-question R9 P0 with one GPU owner."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import resource
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from neuron_lab.runtime import ModelRuntime, Run, save_json, source_hashes
from neuron_lab.round6_engine import Probe, encode
from neuron_lab.round9_worlds import build_manifest, prepared_cases, digest
from neuron_lab.round9_analysis import analyze, anchor_quality
from neuron_lab.input_contracts import token_fingerprint, validate_splits

REG = ROOT / 'logs/summary/round9_P0_registration.json'
MAN = ROOT / 'runs/round9_P0_manifest.json'
ENC = ROOT / 'runs/round9_P0_encoded_inputs.json'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())


def authoritative_encoding(tokenizer, case, maximum):
    enc = encode(tokenizer, case['text'], maximum)
    prefix = case.get('prefix', '')
    if prefix:
        suffix = tokenizer.encode(prefix, add_special_tokens=False)
        templ = tokenizer.apply_chat_template([{'role': 'user', 'content': case['text']}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        assert tokenizer.encode(templ + prefix, add_special_tokens=False) == enc['input_ids'] + suffix, 'Noncanonical prefix boundary'
        enc['input_ids'] += suffix; enc['attention_mask'] += [1] * len(suffix)
        enc['tokens'] += len(suffix)
        enc['regions']['query'] = list(range(enc['query_start'], enc['tokens']))
        enc['regions']['all'] = list(range(enc['tokens']))
    assert enc['tokens'] <= maximum
    enc['authoritative_input_sha256'] = token_fingerprint(enc['input_ids'])
    return enc


def operative_paths():
    return ['src/neuron_lab/round9_worlds.py', 'src/neuron_lab/round9_analysis.py', 'scripts/run_round9.py',
            'tests/test_round9.py', 'configs/round9_P0.yaml', 'src/neuron_lab/input_contracts.py',
            'src/neuron_lab/runtime.py', 'src/neuron_lab/round6_engine.py', 'src/neuron_lab/round5_execution.py',
            'src/neuron_lab/round6_explicit_discovery.py', 'src/neuron_lab/round6_explicit_corpus.py',
            'configs/model.yaml']


def prepare(cp):
    from transformers import AutoTokenizer
    started = time.monotonic(); cfg = yaml.safe_load(cp.read_text())
    if REG.exists(): raise RuntimeError('Registration exists; never overwrite measured definitions')
    local = Path((ROOT / 'runs/model_path.txt').read_text().strip()); assert local.name == cfg['model_revision']
    tok = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
    assert all(len(tok.encode(x, add_special_tokens=False)) == 1 for x in 'ABCD01')
    assert all(tok.encode(s, add_special_tokens=False) == sum([tok.encode(c, add_special_tokens=False) for c in s], [])
               for s in ['00', '01', '10', '11', '000', '001', '010', '011', '100', '101', '110', '111'])
    manifest = build_manifest(cfg)
    from neuron_lab.round6_explicit_corpus import make_episode
    manifest['technical'] = [make_episode(cfg['corpus_seed'] + 99000000 + i, f'R9-technical-{i}', 'A', 'technical') for i in range(2)]
    encodings = {}; contracts = {}; counts = {}; maximum = 0; seen_ids = set()
    for study, detail in manifest['studies'].items():
        contract = {}
        for split, worlds in detail['splits'].items():
            contract[split] = []
            for w in worlds:
                hashes = []
                for c in prepared_world_cases(w):
                    assert c['id'] not in seen_ids; seen_ids.add(c['id'])
                    enc = authoritative_encoding(tok, c, cfg['max_input_tokens'])
                    encodings[c['id']] = enc; maximum = max(maximum, enc['tokens'])
                    hashes.append(enc['authoritative_input_sha256'])
                contract[split].append({'id': w['id'], 'observable_signature': w['observable_signature'], 'principal_input_sha256': hashes})
                if time.monotonic() - started > cfg['CPU_prepare_seconds_cap']: raise TimeoutError('CPU preparation budget')
        contracts[study] = validate_splits(contract)
        for c in list(prepared_cases(manifest, study, anchors=True)) + ([c for c in prepared_cases(manifest, study) if c['role'] == 'reference'] if study == 'D' else []):
            assert c['id'] not in seen_ids; seen_ids.add(c['id'])
            enc = authoritative_encoding(tok, c, cfg['max_input_tokens']); encodings[c['id']] = enc
            maximum = max(maximum, enc['tokens'])
        counts[study] = {role: len(list(prepared_cases(manifest, study, anchors=role == 'anchors'))) for role in ['anchors', 'main']}
        assert counts[study]['anchors'] == cfg['studies'][study]['anchors']
        assert counts[study]['main'] == cfg['studies'][study]['main']
    protected = read(ROOT / 'logs/summary/round9_progress.json')['protected_prior_source_sha256']
    for p, h in protected.items(): assert sha(ROOT / p) == h, p
    save_json(MAN, manifest); save_json(ENC, encodings)
    assert MAN.stat().st_size + ENC.stat().st_size < cfg['CPU_prepare_disk_gb_cap'] * 1e9
    for p in [local / 'tokenizer.json', local / 'tokenizer_config.json', local / 'config.json']:
        assert p.exists()
    operative = {p: sha(ROOT / p) for p in operative_paths()}
    reg = {'schema_version': 1, 'status': 'REGISTERED_BEFORE_ANY_R9_MODEL_OUTPUTS',
           'recorded_at_UTC': datetime.now(timezone.utc).isoformat(), 'model_outputs_seen': 0,
           'model_id': cfg['model_id'], 'model_revision': cfg['model_revision'], 'config_path': str(cp.relative_to(ROOT)),
           'config_sha256': sha(cp), 'operative_source_sha256': operative, 'protected_prior_sha256': protected,
           'manifest_sha256': sha(MAN), 'manifest_canonical_sha256': digest(manifest), 'encoded_inputs_sha256': sha(ENC),
           'tokenizer_file_sha256': {p.name: sha(p) for p in [local / 'tokenizer.json', local / 'tokenizer_config.json', local / 'config.json']},
           'counts': counts, 'maximum_input_tokens': maximum, 'input_contracts': contracts,
           'input_contract_scope': 'Principal scientific inputs deduplicated across all fresh splits; generic elementary anchors and the ONE shared uniform reference are separately declared and never independent families.',
           'CPU_prepare_seconds': time.monotonic() - started, 'CPU_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
           'all_measured_stages': 'NOT_EVALUATED', 'scientific_protocol': cfg}
    save_json(REG, reg)
    print(json.dumps({'status': 'PREPARED', 'counts': counts, 'maximum_tokens': maximum,
                      'authoritative_inputs': len(encodings), 'seconds': reg['CPU_prepare_seconds'], 'model_forwards': 0}), flush=True)


def prepared_world_cases(w):
    from neuron_lab.round9_worlds import CASES
    return CASES[w['study']](w)


def check(cp):
    cfg = yaml.safe_load(cp.read_text()); reg = read(REG); manifest = read(MAN)
    assert sha(cp) == reg['config_sha256'] and sha(MAN) == reg['manifest_sha256'] and sha(ENC) == reg['encoded_inputs_sha256']
    for p, h in {**reg['operative_source_sha256'], **reg['protected_prior_sha256']}.items(): assert sha(ROOT / p) == h, p
    assert digest(manifest) == reg['manifest_canonical_sha256']
    return cfg, reg, manifest


class Journal:
    def __init__(self, probe, encodings):
        self.p = probe; self.encodings = encodings; self.rows = []; self.file = (probe.run.path / 'rows.jsonl').open('x')

    def call(self, case):
        enc = self.encodings[case['id']]
        old, hooks, logits = self.p.forward(enc, full=True)
        codes = case['semantic_codes']; ids = [self.p.runtime.tokenizer.encode(c, add_special_tokens=False)[0] for c in codes]
        allowed = logits[0, ids].float(); probs = torch.softmax(allowed.double(), 0)
        top = int(logits[0].argmax().item()); pred = ids.index(top) if top in ids else None
        fullmass = float(torch.exp(torch.logsumexp(allowed, 0) - torch.logsumexp(logits[0].float(), 0)).item())
        row = {**{k: case[k] for k in ['id', 'episode_id', 'study', 'role', 'expected', 'truth', 'factors', 'oracle']},
               'full_top1_id': top, 'full_top1_text': self.p.runtime.tokenizer.decode([top]),
               'full_prediction': pred, 'format_valid': pred is not None,
               'full_correct': bool(pred is not None and case['expected'] is not None and codes[pred] == case['expected']),
               'semantic_one': int(pred == 1), 'semantic_yes': int(pred == 1),
               'semantic_logit_difference': float((allowed[1] - allowed[0]).item()),
               'restricted_probabilities': probs.cpu().tolist(), 'valid_probability_mass': fullmass,
               'answer_logits': allowed.cpu().tolist(), 'answer_token_ids': ids, 'tokens': enc['tokens'],
               'authoritative_input_sha256': enc['authoritative_input_sha256'], 'probe_delivery': hooks.delivery}
        self.rows.append(row); self.file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'); self.file.flush()
        return row

    def close(self): self.file.close()


def execute(cp):
    from neuron_lab.round6_explicit_discovery import technical
    cfg, reg, manifest = check(cp)
    if any((ROOT / f'logs/summary/round9_P0_{s}.json').exists() for s in cfg['execution_order']):
        raise RuntimeError('A measured R9 phase exists. No silent rerun or post-output source repair.')
    assert sum(s['scientific_cap'] for s in cfg['studies'].values()) + cfg['technical_call_cap'] <= cfg['P0_round_actual_forward_cap']
    encodings = read(ENC)
    budget = {'started_unix': time.time(), 'calls': 0, 'completed': []}; save_json(ROOT / 'runs/round9_global_budget.json', budget)
    runtime = ModelRuntime(audit_path=ROOT / 'logs/summary/round9_P0_model_load.json')
    assert runtime.cfg['revision'] == cfg['model_revision']
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    tr = Run('round9_P0_technical', cfg); tp = Probe(runtime, tr, cfg, cfg['technical_call_cap'], budget)
    try:
        tech = technical(tp, manifest['technical'], cfg)
        assert tech['actual_forward_calls'] == cfg['expected_shared_technical_calls']
        tr.finish({**tech, 'model_revision': cfg['model_revision'], 'measurement_seed': cfg['seed'],
                   'shared_scope': 'same frozen backbone, hook/readout instrumentation for all five sequential P0s'})
    except BaseException as e:
        tr.finish({'error_type': type(e).__name__, 'error': str(e), 'actual_forward_calls': tp.calls}, 'FAIL'); raise
    finally:
        tp.close(); budget['calls'] += tp.calls; save_json(ROOT / 'runs/round9_global_budget.json', budget)
    for study in cfg['execution_order']:
        run = Run(f'round9_P0_{study}', cfg); p = Probe(runtime, run, cfg, cfg['studies'][study]['scientific_cap'], budget); j = Journal(p, encodings)
        outcome = None
        try:
            for c in prepared_cases(manifest, study, anchors=True): j.call(c)
            quality = anchor_quality(j.rows, cfg)
            print(json.dumps({'event': 'R9_anchor_gate', 'study': study, 'quality': quality, 'calls': p.calls}), flush=True)
            if quality['passed']:
                for i, c in enumerate(prepared_cases(manifest, study)):
                    j.call(c)
                    if (i + 1) % 64 == 0:
                        print(json.dumps({'event': 'R9_main_progress', 'study': study, 'science_calls': p.calls, 'main_completed': i + 1}), flush=True)
            outcome = analyze(study, j.rows, cfg); p.check_frozen()
            for rel, h in reg['protected_prior_sha256'].items(): assert sha(ROOT / rel) == h, rel
            j.close()
            summary = {'schema_version': 1, 'study_id': f'R9_{study}', 'model_id': cfg['model_id'], 'model_revision': cfg['model_revision'],
                'measurement_seed': cfg['seed'], 'corpus_seed': cfg['corpus_seed'], 'device': runtime.cfg['device'],
                'primary_dtype': cfg['primary_dtype'], 'actual_forward_calls': p.calls, 'completed_forward_calls': p.completed,
                'shared_technical_forward_calls': tech['actual_forward_calls'], 'technical_call_count_is_not_per_study': True,
                'rows': len(j.rows), 'first50': p.first50, 'frozen_parameters_verified': True, 'use_cache': False,
                'registration_sha256': sha(REG), 'manifest_sha256': sha(MAN), 'encoded_inputs_sha256': sha(ENC),
                'raw_rows_sha256': sha(run.path / 'rows.jsonl'), 'protected_prior_files_verified': len(reg['protected_prior_sha256']),
                'peak_CPU_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                'outcome': outcome, 'P1': 'NOT_EVALUATED', 'P2': 'NOT_EVALUATED'}
            run.finish(summary)
            print(json.dumps({'event': 'R9_P0_finished', 'study': study, 'run_id': run.id, 'calls': p.calls,
                              'gate': outcome['gate'], 'classification': outcome['classification']}), flush=True)
        except BaseException as e:
            j.close(); run.finish({'study_id': f'R9_{study}', 'actual_forward_calls': p.calls, 'completed_forward_calls': p.completed,
                'rows': len(j.rows), 'error_type': type(e).__name__, 'error': str(e), 'outcome': outcome}, 'FAIL'); raise
        finally:
            p.close(); budget['calls'] += p.calls; budget['completed'].append({'study': study, 'run_id': run.id, 'calls': p.calls})
            save_json(ROOT / 'runs/round9_global_budget.json', budget)
    print(json.dumps({'event': 'R9_round_complete', 'actual_forward_calls_including_shared_technical': budget['calls']}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--config', default='configs/round9_P0.yaml')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true'); group.add_argument('--check', action='store_true'); group.add_argument('--execute', action='store_true')
    args = ap.parse_args(); cp = (ROOT / args.config).resolve(); assert cp.parent == ROOT / 'configs'
    if args.prepare: prepare(cp)
    elif args.check:
        c, r, m = check(cp); print(json.dumps({'status': 'PASS', 'operative_sources': len(r['operative_source_sha256']), 'model_forwards': 0}))
    else: execute(cp)
