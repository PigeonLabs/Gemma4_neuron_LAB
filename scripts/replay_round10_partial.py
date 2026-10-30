#!/usr/bin/env python3
"""Replay the preserved interrupted R10 attempt without rewriting its FAIL."""
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from neuron_lab.round10_repair import prepared_cases, generation_boundary, parse_generated_body


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    from transformers import AutoTokenizer
    cfg=yaml.safe_load((ROOT/'configs/round10_repair.yaml').read_text())
    reg=read(ROOT/'logs/summary/round10_registration.json')
    for p,h in {**reg['operative_source_sha256'],**reg['protected_prior_sha256']}.items():assert sha(ROOT/p)==h,p
    abort=read(ROOT/'logs/summary/round10_attempt1_abort.json')
    summary=read(ROOT/'logs/summary/round10_calibration_A.json')
    assert summary['status']=='FAIL' and summary['error_type']=='KeyboardInterrupt'
    assert sha(ROOT/'logs/summary/round10_calibration_A.json')==abort['original_failed_run_summary_sha256']
    raw=ROOT/'runs'/summary['run_id']/'rows.jsonl';assert sha(raw)==abort['raw_rows_sha256']
    rows=[json.loads(s) for s in raw.read_text().splitlines()]
    expected=list(prepared_cases(read(ROOT/'runs/round10_manifest.json'),'A','calibration','anchor'))
    tok=AutoTokenizer.from_pretrained(Path((ROOT/'runs/model_path.txt').read_text().strip()),local_files_only=True,trust_remote_code=False)
    eos=read(ROOT/'logs/summary/round10_effective_generation_config.json')['actual_neutral_greedy']['eos_token_id']
    special=set(tok.all_special_ids)
    for row,case in zip(rows,expected):
        assert all(row[k]==v for k,v in case.items())
        ids=row['generated_token_ids'];complete,exhausted,body=generation_boundary(ids,eos,cfg['max_new_tokens'])
        text=tok.decode(body,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        p=parse_generated_body(text,exhausted=exhausted,internal_special=any(i in special for i in body))
        assert p['status']==row['parse_status'] and p['code']==row['parsed_code']
        assert text==row['generated_text'] and row['generation_complete_EOS']==complete
        assert row['actual_model_forwards']==len(ids)==row['generated_tokens']
    recorded=sum(r['actual_model_forwards'] for r in rows)
    assert len(rows)==42 and recorded==7609
    assert summary['actual_forward_calls']==7684 and summary['actual_forward_calls']-recorded==75
    budget=read(ROOT/'runs/round10_global_budget.json')
    assert budget['calls']==summary['actual_forward_calls']+50==7734
    assert budget['generated_responses']==42
    assert expected[len(rows)]['id']==abort['next_in_flight_case_id']
    assert budget['calls']+58==abort['all_R10_forward_entries_so_far']==7792
    out={'schema_version':1,'status':'PASS','new_model_forwards':0,'completed_responses_verified':42,
        'completed_response_forwards':7609,'interrupted_unfinished_response_entries':75,
        'all_R10_forward_entries':7792,'interrupted_response_not_a_natural_token_cap':True,
        'all_P0':'NOT_EVALUATED','original_FAIL_and_sources_preserved':True,
        'original_rows_sha256':sha(raw),'script_sha256':sha(Path(__file__)),
        'limitation':'CPU token/input/counter replay, not a second model evaluation. The incomplete response has no completed generation to classify.'}
    (ROOT/'logs/summary/round10b_prior_attempt_replay.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))


if __name__=='__main__':main()
