#!/usr/bin/env python3
"""Technical-only validation while scientific repair is under peer review."""
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from neuron_lab.runtime import ModelRuntime, Run, save_json
from neuron_lab.round6_engine import Probe
from neuron_lab.round6_explicit_corpus import make_episode
from neuron_lab.round6_explicit_discovery import technical
from run_round10_repair import CounterHook, cache_technical, encode, protected_prior, sha


def main():
    cfg = yaml.safe_load((ROOT / 'configs/round10_generation_smoke.yaml').read_text())
    dest = ROOT / 'logs/summary/round10_generation_smoke.json'
    if dest.exists():
        raise RuntimeError('Technical smoke already recorded; no silent overwrite')
    protected = protected_prior()
    runtime = ModelRuntime(audit_path=ROOT / 'logs/summary/round10_smoke_model_load.json')
    run = Run('round10_generation_smoke', cfg)
    import time
    budget = {'calls': 0, 'started_unix': time.time()}
    p = Probe(runtime, run, cfg, cfg['technical_call_cap'], budget)
    rows = []
    hook_charged = False
    try:
        hook = technical(p, [make_episode(101990+i, f'R10-smoke-{i}', 'A', 'technical') for i in range(2)], cfg)
        p.close(); budget['calls'] += p.calls
        hook_charged = True
        counter = CounterHook(runtime, cfg, budget)
        cases = [{'id': f'R10-generation-smoke-{i}', 'text': text} for i, text in enumerate([
            'Calculate 2 + 2. A means 4, B means 5. Finish with exactly FINAL: A or FINAL: B.',
            'A bag has red and blue objects. How many distinct colors? A means one, B means two. Finish with exactly FINAL: A or FINAL: B.'
        ])]
        enc = {c['id']: encode(runtime.tokenizer, c, cfg['max_input_tokens']) for c in cases}
        try:
            cached = cache_technical(runtime, counter, enc, cases)
            for c in cases:
                inp = {k: torch.tensor([enc[c['id']][k]], device=runtime.cfg['device']) for k in ['input_ids', 'attention_mask']}
                before = budget['calls']
                with torch.inference_mode():
                    out = runtime.model.generate(**inp, do_sample=False, max_new_tokens=4, use_cache=True,
                        temperature=None, top_p=None, top_k=None, logits_to_keep=1)
                ids = out[0, enc[c['id']]['tokens']:].cpu().tolist()
                assert budget['calls']-before == len(ids)
                rows.append({'id': c['id'], 'actual_forward_calls': budget['calls']-before,
                             'generated_token_ids': ids, 'text': runtime.tokenizer.decode(ids, skip_special_tokens=True)})
        finally:
            counter.close()
        for path, expected in protected.items():
            assert sha(ROOT / path) == expected, path
        save_json(run.path / 'generation_smoke_rows.json', rows)
        run.finish({'actual_forward_calls': budget['calls'], 'scientific_response_count': 0,
                    'hook_parity': hook, 'KV_cache_parity': cached,
                    'generated_response_counter_equal_token_count': True,
                    'generated_response_count': len(rows), 'protected_prior_files_verified': len(protected),
                    'generation_smoke_rows_sha256': sha(run.path / 'generation_smoke_rows.json'),
                    'model_revision': cfg['model_revision'], 'measurement_seed': cfg['seed'],
                    'primary_dtype': 'FP32_upcast_of_same_stored_BF16_values',
                    'claim_limit': 'Technical pipeline only, not scientific calibration or hypothesis evaluation.'})
        print(json.dumps({'status': 'PASS', 'actual_forward_calls': budget['calls'], 'scientific_response_count': 0}), flush=True)
    except BaseException as e:
        p.close()
        run.finish({'error_type': type(e).__name__, 'error': str(e),
                    'actual_forward_calls': budget['calls'] + (0 if hook_charged else p.calls)}, 'FAIL')
        raise


if __name__ == '__main__':
    main()
