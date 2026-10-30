#!/usr/bin/env python3
"""Register then run calibration and fresh A/B/C/E P0 with one GPU owner.

Every actual model forward, including cached decoding, is counted. A generated
response is a separate unit; it is never counted as one forward by convention.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import resource
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from neuron_lab.runtime import ModelRuntime, Run, save_json
from neuron_lab.round6_engine import Probe
from neuron_lab.round6_explicit_corpus import make_episode
from neuron_lab.round6_explicit_discovery import technical
from neuron_lab.round10b_repair import build_manifest, prepared_cases, option_audit, parse_generated_body, generation_boundary, calibration_futility
from neuron_lab.round10b_analysis import analyze, anchor_quality, accuracy
from neuron_lab.input_contracts import token_fingerprint

CONFIG = ROOT / 'configs/round10b_repair.yaml'
REG = ROOT / 'logs/summary/round10b_registration.json'
MAN = ROOT / 'runs/round10b_manifest.json'
ENC = ROOT / 'runs/round10b_encoded_inputs.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generation_config(runtime, cfg):
    from transformers import GenerationConfig
    original = runtime.model.generation_config
    # A fresh neutral config avoids inheriting the model's sampling defaults
    # or its reserved-special-token suppression. EOS defines completion only.
    return GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
        max_new_tokens=cfg['max_new_tokens'], temperature=None, top_p=None, top_k=None,
        bos_token_id=original.bos_token_id, eos_token_id=original.eos_token_id,
        pad_token_id=original.pad_token_id, suppress_tokens=None,
        begin_suppress_tokens=None, forced_bos_token_id=None, forced_eos_token_id=None,
        bad_words_ids=None, repetition_penalty=1.0, no_repeat_ngram_size=0)


def sources():
    return ['src/neuron_lab/round10b_repair.py', 'src/neuron_lab/round10b_analysis.py',
            'src/neuron_lab/round10_repair.py', 'src/neuron_lab/round10_analysis.py',
            'scripts/run_round10b_repair.py', 'tests/test_round10b_repair.py',
            'configs/round10b_repair.yaml', 'src/neuron_lab/runtime.py',
            'src/neuron_lab/round6_engine.py', 'src/neuron_lab/round6_explicit_corpus.py',
            'src/neuron_lab/round6_explicit_discovery.py', 'src/neuron_lab/round9_worlds.py',
            'src/neuron_lab/input_contracts.py', 'configs/model.yaml']


def protected_prior():
    reg = read(ROOT / 'logs/summary/round9_P0_registration.json')
    frozen = read(ROOT / 'logs/summary/round9_measurement_freeze.json')
    result = {**reg['protected_prior_sha256'], **frozen['protected_measured_sha256']}
    for pattern in ['logs/summary/round9*.json', 'reports/ROUND9*.md',
                    'src/neuron_lab/round9*.py', 'scripts/*round9*.py',
                    'configs/round9*.yaml', 'figures/round9*.svg', 'figures/round9*metadata.json']:
        for p in ROOT.glob(pattern):
            key = p.relative_to(ROOT).as_posix()
            if key in result:
                assert sha(p) == result[key], key
            result[key] = sha(p)
    measured_r10 = read(ROOT/'logs/summary/round10_registration.json')
    result.update(measured_r10['operative_source_sha256'])
    for pattern in ['logs/summary/round10_*.json', 'reports/ROUND10_*.md',
                    'scripts/run_round10_repair.py', 'scripts/smoke_round10_generation.py',
                    'configs/round10_*.yaml', 'tests/test_round10_repair.py']:
        for p in ROOT.glob(pattern):
            if p.name == 'round10_progress.json':
                continue  # Cumulative status remains explicitly mutable.
            key = p.relative_to(ROOT).as_posix()
            if key in result:
                assert sha(p) == result[key], key
            result[key] = sha(p)
    return result


def encode(tokenizer, case, maximum):
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': case['text']}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    assert len(ids) <= maximum, 'No truncation of scientific input'
    assert rendered.endswith('<|channel>thought\n<channel|>'), 'Unexpected generation channel'
    return {'input_ids': ids, 'attention_mask': [1]*len(ids), 'tokens': len(ids),
            'authoritative_input_sha256': token_fingerprint(ids)}


def prepare():
    from transformers import AutoTokenizer
    if REG.exists():
        raise RuntimeError('R10b is already registered. Do not overwrite a measured protocol.')
    started = time.monotonic(); cfg = yaml.safe_load(CONFIG.read_text())
    if cfg['status'] != 'PEER_REVIEWED_BEFORE_SCIENTIFIC_OUTPUTS':
        raise RuntimeError('Complete actual independent and cross design review before registration')
    peer = read(ROOT / 'logs/summary/round10b_peer_review.json')
    assert peer['design_review_status'] == 'BOTH_INDEPENDENT_AND_CROSS_COMPLETE'
    old = read(ROOT / 'runs/round9_P0_manifest.json')
    manifest = build_manifest(cfg, old)
    manifest['technical'] = [make_episode(cfg['corpus_seed']+990000+i, f'R10b-tech-{i}', 'A', 'technical') for i in range(2)]
    local = Path((ROOT / 'runs/model_path.txt').read_text().strip())
    assert local.name == cfg['model_revision']
    tok = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
    assert all(len(tok.encode(code, add_special_tokens=False)) == 1 for code in ['A', 'B'])
    old_encoded = read(ROOT / 'runs/round9_P0_encoded_inputs.json')
    old_input_rows = len(old_encoded)
    historical_hashes = {e['authoritative_input_sha256'] for e in old_encoded.values()}
    del old_encoded
    prior_r10_encoded = read(ROOT/'runs/round10_encoded_inputs.json')
    prior_r10_input_rows = len(prior_r10_encoded)
    historical_hashes.update(e['authoritative_input_sha256'] for e in prior_r10_encoded.values())
    del prior_r10_encoded
    encodings = {}; contracts = {}; all_hashes = set(); max_tokens = 0
    for study, detail in manifest['studies'].items():
        contracts[study] = {}
        for split in detail['splits']:
            cases = [*prepared_cases(manifest, study, split, 'anchor'),
                     *prepared_cases(manifest, study, split, 'main')]
            audit = option_audit(cases)
            counts = Counter(c['role'] for c in cases)
            assert counts['anchor'] == cfg['studies'][study]['anchor_rows_per_world']*len(detail['splits'][split])
            assert counts['main'] == (16 if split == 'calibration' else cfg['studies'][study]['P0_main_rows_per_world'])*len(detail['splits'][split])
            for case in cases:
                assert case['id'] not in encodings
                enc = encode(tok, case, cfg['max_input_tokens'])
                assert enc['authoritative_input_sha256'] not in historical_hashes, 'R9 input reused'
                assert enc['authoritative_input_sha256'] not in all_hashes, 'Scientific input duplicated'
                all_hashes.add(enc['authoritative_input_sha256'])
                encodings[case['id']] = enc
                max_tokens = max(max_tokens, enc['tokens'])
            contracts[study][split] = {**audit, 'role_counts': dict(counts),
                'independent_worlds': len(detail['splits'][split])}
            if time.monotonic()-started > cfg['CPU_prepare_seconds_cap']:
                raise TimeoutError('CPU preparation cap')
    protected = protected_prior()
    for path, expected in protected.items():
        assert sha(ROOT / path) == expected, path
    save_json(MAN, manifest); save_json(ENC, encodings)
    assert MAN.stat().st_size+ENC.stat().st_size < cfg['CPU_prepare_disk_gb_cap']*1e9
    max_responses = sum(v[s]['presentations'] for v in contracts.values() for s in ['calibration', 'P0'])
    assert max_responses*cfg['max_new_tokens']+cfg['technical_call_cap'] <= cfg['max_actual_forward_calls']
    assert cfg['prior_actual_forward_entries'] + cfg['max_actual_forward_calls'] <= cfg['cumulative_forward_cap']
    reg = dict(schema_version=1, status='REGISTERED_BEFORE_R10b_SCIENTIFIC_OUTPUTS',
        recorded_at_UTC=datetime.now(timezone.utc).isoformat(), registered_scientific_outputs_seen=0,
        earlier_instrumentation_only_model_forwards=108,
        earlier_calibration_forward_entries=7684,
        earlier_R10_total_forward_entries=7792,
        config_sha256=sha(CONFIG), operative_source_sha256={p: sha(ROOT / p) for p in sources()},
        protected_prior_sha256=protected, manifest_sha256=sha(MAN), encoded_inputs_sha256=sha(ENC),
        prior_R9_manifest_sha256=sha(ROOT / 'runs/round9_P0_manifest.json'),
        prior_R10_manifest_sha256=sha(ROOT / 'runs/round10_manifest.json'),
        prior_R10_input_rows_checked=prior_r10_input_rows,
        prior_R9_input_rows_checked=old_input_rows,
        prior_R9_and_R10_unique_input_fingerprints=len(historical_hashes),
        peer_design_review_sha256=sha(ROOT / 'logs/summary/round10b_peer_review.json'),
        maximum_input_tokens=max_tokens, contracts=contracts,
        maximum_generated_responses=max_responses,
        maximum_generation_forwards=max_responses*cfg['max_new_tokens'],
        CPU_seconds=time.monotonic()-started, CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        world_exclusion='All R9 AND R10 used/reserved canonical A/E matrices and B reward-transform orbits excluded. C probabilities disjoint from both earlier pools and across R10b splits. Every new principal token input unique and outside both earlier stored input sets.',
        scientific_protocol=cfg)
    save_json(REG, reg)
    print(json.dumps({'status': 'REGISTERED', 'responses_cap': max_responses, 'input_rows': len(encodings),
                      'maximum_input_tokens': max_tokens, 'CPU_seconds': reg['CPU_seconds'], 'new_model_forwards': 0}), flush=True)


class CounterHook:
    def __init__(self, runtime, cfg, budget):
        self.runtime = runtime; self.cfg = cfg; self.budget = budget
        self.first = None; self.want_first = False
        self.code_ids = [runtime.tokenizer.encode(c, add_special_tokens=False)[0] for c in ['A', 'B']]
        self.pre = runtime.model.register_forward_pre_hook(self.before, with_kwargs=True)
        self.post = runtime.model.register_forward_hook(self.after)

    def before(self, module, args, kwargs):
        if self.budget['calls'] >= self.cfg['max_actual_forward_calls']:
            raise RuntimeError('R10b global actual-forward cap')
        if time.time()-self.budget['started_unix'] >= self.cfg['max_seconds']:
            raise TimeoutError('R10b global wall-time cap')
        self.budget['calls'] += 1

    def after(self, module, args, output):
        if self.want_first and self.first is None:
            logits = output.logits[0, -1, :].float()
            top = int(logits.argmax().item())
            mass = float((logits[self.code_ids].logsumexp(0)-logits.logsumexp(0)).exp().item())
            self.first = {'full_vocab_first_token_id': top,
                          'full_vocab_first_token_text': self.runtime.tokenizer.decode([top], clean_up_tokenization_spaces=False),
                          'first_token_AB_probability_mass_diagnostic': mass}

    def close(self):
        self.pre.remove(); self.post.remove()


@torch.inference_mode()
def cache_technical(runtime, counter, encodings, cases):
    start = counter.budget['calls']; details = []
    for case in cases:
        enc = encodings[case['id']]
        ids = torch.tensor([enc['input_ids']], device=runtime.cfg['device'])
        mask = torch.ones_like(ids)
        cached = runtime.model(input_ids=ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
        bare = runtime.model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
        errors = []
        for step in range(4):
            a = cached.logits[0, -1].float(); b = bare.logits[0, -1].float()
            err = float((a-b).abs().max().item()); errors.append(err)
            assert int(a.argmax()) == int(b.argmax()) and err <= .001, 'Cached/full-prefix mismatch'
            if step == 3:
                break
            token = a.argmax().reshape(1, 1)
            ids = torch.cat([ids, token], dim=1); mask = torch.ones_like(ids)
            past = cached.past_key_values
            cached = runtime.model(input_ids=token, attention_mask=mask, past_key_values=past,
                                   use_cache=True, logits_to_keep=1)
            bare = runtime.model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
        details.append({'input_id': case['id'], 'FP32_full_vocab_max_absolute_errors': errors,
                        'all_greedy_argmax_equal': True})
        del cached, bare
    return {'status': 'PASS', 'actual_forward_calls': counter.budget['calls']-start,
            'scope': '2 fixed calibration inputs x prefill/3 decoded steps, full-prefix versus KV cache; not universal equivalence',
            'records': details}


class Journal:
    def __init__(self, runtime, run, cfg, counter, encodings):
        self.runtime = runtime; self.run = run; self.cfg = cfg; self.counter = counter
        self.encodings = encodings; self.rows = []
        self.generation_config = generation_config(runtime, cfg)
        self.special_ids = set(runtime.tokenizer.all_special_ids)
        self.file = (run.path / 'rows.jsonl').open('x')

    @torch.inference_mode()
    def call(self, case):
        self.run.check_budget()
        enc = self.encodings[case['id']]
        inputs = {k: torch.tensor([enc[k]], device=self.runtime.cfg['device']) for k in ['input_ids', 'attention_mask']}
        before = self.counter.budget['calls']; started = time.monotonic()
        self.counter.first = None; self.counter.want_first = True
        output = self.runtime.model.generate(**inputs,
            generation_config=self.generation_config, logits_to_keep=1)
        self.counter.want_first = False
        ids = output[0, enc['tokens']:].cpu().tolist()
        assert ids and ids[0] == self.counter.first['full_vocab_first_token_id'], 'Generation is not full-vocabulary greedy at its first step'
        eos = self.runtime.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        complete, exhausted, body_ids = generation_boundary(ids, eos, self.cfg['max_new_tokens'])
        calls = self.counter.budget['calls']-before
        assert calls == len(ids), 'Actual cached forward/token counter mismatch'
        raw_text = self.runtime.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        internal_special = [i for i in body_ids if i in self.special_ids]
        text = self.runtime.tokenizer.decode(body_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        parsed = parse_generated_body(text, exhausted=exhausted, internal_special=bool(internal_special))
        code = parsed['code']; semantic = case['semantic_codes'].index(code) if code is not None else None
        row = {**case, 'generated_token_ids': ids, 'generated_text': text,
               'generated_text_with_terminal_special': raw_text,
               'internal_special_token_ids': internal_special,
               'thought_marker_present': '<|channel>thought' in text,
               'parse_status': parsed['status'], 'parsed_code': code, 'semantic_choice': semantic,
               'single_line_contract_compliant': code is not None and bool(re.fullmatch(r'FINAL: *[AB]', text.strip())),
               'correct': (semantic == case['truth']) if semantic is not None else None,
               'generation_complete_EOS': complete, 'token_cap_exhausted': exhausted,
               'actual_model_forwards': calls, 'generated_tokens': len(ids),
               'input_tokens': enc['tokens'], 'authoritative_input_sha256': enc['authoritative_input_sha256'],
               'elapsed_seconds': time.monotonic()-started, **self.counter.first}
        self.rows.append(row)
        self.file.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); self.file.flush()
        self.counter.budget['generated_responses'] += 1
        save_json(ROOT / 'runs/round10b_global_budget.json', self.counter.budget)
        if len(self.rows) % 16 == 0:
            print(json.dumps({'event': 'R10b_generation_progress', 'stage': self.run.meta['stage'],
                'responses': len(self.rows), 'generated_tokens': sum(r['generated_tokens'] for r in self.rows),
                'parse_status': dict(Counter(r['parse_status'] for r in self.rows)),
                'global_actual_forwards': self.counter.budget['calls']}), flush=True)
        del output
        return row

    def close(self):
        self.file.close()


def execute():
    cfg = yaml.safe_load(CONFIG.read_text()); reg = read(REG)
    assert sha(CONFIG) == reg['config_sha256'] and sha(MAN) == reg['manifest_sha256'] and sha(ENC) == reg['encoded_inputs_sha256']
    for path, expected in {**reg['operative_source_sha256'], **reg['protected_prior_sha256']}.items():
        assert sha(ROOT / path) == expected, path
    if (ROOT / 'runs/round10b_global_budget.json').exists():
        raise RuntimeError('R10b execution exists; no silent repeat or resume under altered definitions')
    manifest = read(MAN); encodings = read(ENC)
    budget = {'calls': 0, 'generated_responses': 0, 'started_unix': time.time(), 'completed': []}
    assert cfg['prior_measured_GPU_seconds'] + cfg['max_seconds'] <= cfg['cumulative_GPU_seconds_cap']
    save_json(ROOT / 'runs/round10b_global_budget.json', budget)
    runtime = ModelRuntime(audit_path=ROOT / 'logs/summary/round10b_model_load.json')
    assert runtime.cfg['revision'] == cfg['model_revision']
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    tr = Run('round10b_technical', cfg); p = Probe(runtime, tr, cfg, cfg['technical_call_cap'], budget)
    try:
        result = technical(p, manifest['technical'], cfg)
        assert result['actual_forward_calls'] == 34
        tr.finish({**result, 'model_revision': cfg['model_revision'], 'measurement_seed': cfg['seed']})
    except BaseException as e:
        tr.finish({'error_type': type(e).__name__, 'error': str(e), 'actual_forward_calls': p.calls}, 'FAIL'); raise
    finally:
        p.close(); budget['calls'] += p.calls
        save_json(ROOT / 'runs/round10b_global_budget.json', budget)
    counter = CounterHook(runtime, cfg, budget)
    versions = {n: p._version for n, p in runtime.model.named_parameters()}
    save_json(ROOT / 'logs/summary/round10b_effective_generation_config.json', {
        'status': 'RECORDED_BEFORE_CALIBRATION', 'model_default': runtime.model.generation_config.to_dict(),
        'actual_neutral_greedy': generation_config(runtime, cfg).to_dict(),
        'interpretation': 'No A/B constraint, no top-k/top-p, no forced token, no inherited reserved-token suppression.'})
    try:
        cache_run = Run('round10b_cache_technical', cfg)
        fixed = [next(prepared_cases(manifest, s, 'calibration', 'anchor')) for s in ['A', 'C']]
        try:
            result = cache_technical(runtime, counter, encodings, fixed)
            cache_run.finish(result)
        except BaseException as e:
            cache_run.finish({'error_type': type(e).__name__, 'error': str(e)}, 'FAIL'); raise
        assert budget['calls'] <= cfg['technical_call_cap']
        admitted = {}
        for split in ['calibration', 'P0']:
            for study in cfg['execution_order']:
                if split == 'P0' and not admitted[study]:
                    print(json.dumps({'event': 'R10b_P0_not_admitted', 'study': study, 'reason': 'CALIBRATION_NOT_READY'}), flush=True)
                    continue
                run = Run(f'round10b_{split}_{study}', cfg); before = budget['calls']
                j = Journal(runtime, run, cfg, counter, encodings)
                try:
                    anchor_plan = list(prepared_cases(manifest, study, split, 'anchor'))
                    futility = None
                    for c in anchor_plan:
                        j.call(c)
                        if split == 'calibration':
                            futility = calibration_futility(j.rows, anchor_plan, cfg)
                            if futility['stop']:
                                break
                    quality = anchor_quality(j.rows, cfg)
                    sentinel_futility = None
                    if (split == 'calibration' and not (futility and futility['stop'])) or (split != 'calibration' and quality['passed']):
                        main_plan = list(prepared_cases(manifest, study, split, 'main'))
                        for c in main_plan:
                            j.call(c)
                            if split == 'calibration':
                                seen_main = [r for r in j.rows if r['role'] == 'main']
                                invalid = sum(r['semantic_choice'] is None for r in seen_main)
                                upper = (len(main_plan)-invalid)/len(main_plan)
                                if upper < cfg['main_parse_min']:
                                    sentinel_futility = {'planned': len(main_plan), 'observed': len(seen_main),
                                                         'best_possible_final_parse_fraction': upper}
                                    break
                    if split == 'calibration':
                        sentinels = [r for r in j.rows if r['role'] == 'main']
                        parsed_fraction = accuracy(sentinels)['parsed_fraction'] if sentinels else None
                        admitted[study] = (not (futility and futility['stop']) and not sentinel_futility
                            and quality['passed'] and parsed_fraction is not None and parsed_fraction >= cfg['main_parse_min'])
                        outcome = {'gate': 'CALIBRATION_READY' if admitted[study] else 'CALIBRATION_NOT_READY',
                                   'readiness': quality, 'sentinel_parse_fraction': parsed_fraction,
                                   'elementary_anchor_futility': futility,
                                   'sentinel_parse_futility': sentinel_futility,
                                   'anchor_rows_planned': len(anchor_plan),
                                   'anchor_rows_completed': sum(r['role'] == 'anchor' for r in j.rows),
                                   'sentinel_rows_planned': 16*cfg['studies'][study]['splits']['calibration'],
                                   'single_line_contract_compliant': sum(r['single_line_contract_compliant'] for r in j.rows),
                                   'sentinel_task_accuracy_not_used_for_admission': True,
                                   'scientific_hypothesis_result': 'NOT_EVALUATED'}
                    else:
                        outcome = analyze(study, j.rows, cfg)
                    j.file.flush(); runtime.verify_sources()
                    assert all(p._version == versions[n] for n, p in runtime.model.named_parameters()), 'Model weights mutated'
                    summary = {'model_id': cfg['model_id'], 'model_revision': cfg['model_revision'],
                               'primary_dtype': cfg['primary_dtype'], 'measurement_seed': cfg['seed'],
                               'corpus_seed': cfg['corpus_seed'], 'split': split, 'study': study,
                               'actual_forward_calls': budget['calls']-before, 'generated_responses': len(j.rows),
                               'generated_tokens': sum(r['generated_tokens'] for r in j.rows),
                               'raw_rows_sha256': sha(run.path / 'rows.jsonl'), 'registration_sha256': sha(REG),
                               'outcome': outcome, 'P1': 'NOT_EVALUATED', 'P2': 'NOT_EVALUATED'}
                    run.finish(summary); budget['completed'].append({'split': split, 'study': study,
                        'responses': len(j.rows), 'forward_calls': budget['calls']-before, 'gate': outcome['gate']})
                    print(json.dumps({'event': 'R10b_phase_complete', 'split': split, 'study': study,
                        'responses': len(j.rows), 'actual_forward_calls': budget['calls']-before, 'outcome': outcome}), flush=True)
                except BaseException as e:
                    run.finish({'error_type': type(e).__name__, 'error': str(e),
                                'actual_forward_calls': budget['calls']-before, 'generated_responses': len(j.rows)}, 'FAIL')
                    raise
                finally:
                    j.close(); save_json(ROOT / 'runs/round10b_global_budget.json', budget)
        for path, expected in reg['protected_prior_sha256'].items():
            assert sha(ROOT / path) == expected, path
        budget['ended_unix'] = time.time()
        budget['elapsed_seconds'] = budget['ended_unix']-budget['started_unix']
        print(json.dumps({'event': 'R10b_execution_complete', **budget, 'calibration_admitted': admitted}), flush=True)
    finally:
        counter.close(); save_json(ROOT / 'runs/round10b_global_budget.json', budget)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--prepare', action='store_true'); ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    if args.prepare == args.execute:
        ap.error('Select exactly one of --prepare or --execute')
    prepare() if args.prepare else execute()
