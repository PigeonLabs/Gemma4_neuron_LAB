#!/usr/bin/env python3
"""Independently replay recorded R6 oracles, logit arithmetic and accounting on CPU.

No experiment corpus/render/margin helpers are imported. This script performs
no model forward and keeps raw prompts, response rows and tensors local.
"""
from __future__ import annotations
from collections import defaultdict
import argparse,hashlib,json,math
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
STAGES=('round6_A_discovery','round6x_A_discovery','round6x_B_discovery','round6x_C_discovery')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(p):return json.loads(p.read_text())


def oracle(ep,c,row):
    study=ep['study'];role=row['role'];selector=row['selector'];book=c['book'];ctx=row['context']
    if study=='A':
        key={'recipient':'receiver','donor_same':'donor','donor_opposite':'donor','semantic_sham':'sham',
             'independent_rescue_same':'rescue','independent_rescue_opposite':'rescue'}[role]
        if role.endswith('opposite'):selector=1-selector
        if role not in ('recipient','semantic_sham'):book=c['donor_book']
    elif study=='B':key='receiver'
    else:
        key={'recipient':f"receiver{c['function']}",'semantic_sham':f"sham{c['function']}",'intermediate_donor':'donor',
             'same_intermediate_other_g_donor':'otherg','independent_correct_intermediate_rescue':'rescue_correct',
             'independent_wrong_intermediate_rescue':'rescue_wrong'}[role]
        selector=1
        if role not in ('recipient','semantic_sham'):book=c['donor_book']
    w=ep['worlds'][key];x=c['query'];direct=w['h'][x];compose=w['g'][w['f'][x]]
    if study=='A' and ctx=='agreement':compose=direct
    if study=='B' and ctx in ('conflict1','conflict2'):compose=c['wrong1' if ctx=='conflict1' else 'wrong2']
    if row.get('fact_edit'):
        if row['fact_edit']=='direct':direct=c['donor_selected']
        else:compose=c['donor_selected']
    if row['witness']=='direct_only':selector=0
    if row['witness']=='composition_only' or study=='C':selector=1
    z=row.get('explicit_z')
    if z is None:z=direct if selector==0 else compose
    return next(k for k,v in book.items() if v==z),book


def logodds(values,correct):
    i='ABCD'.index(correct);others=[float(x) for k,x in enumerate(values) if k!=i];a=max(others)
    return float(values[i])-a-math.log(sum(math.exp(v-a) for v in others))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-incomplete',action='store_true');args=ap.parse_args()
    from transformers import AutoTokenizer
    local=Path((ROOT/'runs/model_path.txt').read_text().strip())
    tokenizer=AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False)
    records=[];summed=0
    for stage in STAGES:
        path=ROOT/f'logs/summary/{stage}.json'
        if not path.exists():
            if args.allow_incomplete:continue
            raise FileNotFoundError(path)
        s=load(path);assert s['status']=='PASS',(stage,s['status'])
        run=ROOT/'runs'/s['run_id'];raw=run/'rows.jsonl';rows=[json.loads(line) for line in raw.read_text().splitlines()]
        assert sha(raw)==s['raw_rows_sha256']
        assert len(rows)==s['rows']==s['actual_forward_calls']==s['completed_forward_calls']
        reg=load(run/'registration.json');assert canonical(reg)==s['registration_sha256']
        for rel,h in {**reg['source_sha256'],**reg['protected_prior_sha256']}.items():assert sha(ROOT/rel)==h,(stage,rel)
        manifest=load(run/'manifest.json');episodes={e['id']:e for eps in manifest.values() for e in eps}
        for eps in manifest.values():assert len({e['id'] for e in eps})==len(eps)
        # The full registered manifest remains local; verify its independent digest.
        cfg=yaml.safe_load((ROOT/('configs/round6_explicit_execution.yaml' if stage.startswith('round6x') else 'configs/round6_execution.yaml')).read_text())
        assert canonical(load(ROOT/cfg['manifest_path']))==s['manifest_canonical_sha256']==reg['manifest_canonical_sha256']
        encodings=load(run/'encoded_inputs.json');input_hashes={}
        for text,enc in encodings.items():
            h=hashlib.sha256(text.encode()).hexdigest();assert h==enc['text_sha256']
            tok=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,return_dict=True,add_generation_prompt=True,enable_thinking=False)
            assert list(tok['input_ids'])==enc['input_ids'] and list(tok['attention_mask'])==enc['attention_mask']
            assert len(enc['input_ids'])==enc['tokens']<=cfg['max_input_tokens']
            assert enc['regions']['query']==list(range(enc['query_start'],enc['tokens']))
            input_hashes[h]=enc
        groups=defaultdict(list);unique=set();max_margin_error=0.;max_softcap_error=0.
        for row in rows:
            ep=episodes[row['episode_id']];assert ep['split']=='discovery',row['phase']
            c=next(c for c in ep['cases'] if c['id']==row['case_id'])
            expected,book=oracle(ep,c,row);assert expected==row['expected'] and book==row['codebook']
            values=row['post_softcap_logits'];pred='ABCD'[max(range(4),key=lambda i:values[i])]
            assert pred==row['prediction'] and row['correct']==(pred==expected)
            err=abs(logodds(values,expected)-row['correct_margin']);max_margin_error=max(max_margin_error,err);assert err<1e-10
            for a,b in zip(row['raw_head_logits'],values):
                err=abs(math.tanh(a/30.)*30.-b);max_softcap_error=max(max_softcap_error,err);assert err<1e-5
            enc=input_hashes[row['text_sha256']];assert row['tokens']==enc['tokens'] and row['query_start']==enc['query_start']
            ident=tuple(row.get(k) for k in ('episode_id','case_id','phase','role','arm','context','witness','explicit_z','fact_edit','selector'))
            assert ident not in unique,ident;unique.add(ident)
            if 'readiness_' in row['arm']:groups[row['arm'].split('readiness_',1)[1]].append(row)
        readiness=s['outcome']['readiness'];metrics={}
        for name,rr in groups.items():
            correct=sum(r['correct'] for r in rr);accuracy=correct/len(rr)
            assert accuracy==readiness['accuracy'][name] and len(rr)==readiness['counts'][name]
            metrics[name]={'correct':correct,'n':len(rr),'accuracy':accuracy,'episodes':len({r['episode_id'] for r in rr}),
                           'full_vocab_top1_is_code':sum(r['full_top1_code'] is not None for r in rr),
                           'full_vocab_top1_correct':sum(r['full_top1_code']==r['expected'] for r in rr)}
        passed=all(v['accuracy']>=(cfg['code_accuracy_min'] if k=='code_only' else cfg['healthy_accuracy_min']) for k,v in metrics.items())
        assert passed==readiness['passed']
        if not passed:
            assert s['outcome']['gate']=='STOPPED_AT_TASK_READINESS'
            assert s['calibration']==s['confirmation']=='NOT_EVALUATED'
            assert all('readiness_' in r['arm'] for r in rows)
        tech=load(ROOT/f'logs/summary/{stage}_technical.json')
        assert tech['status']=='PASS' and tech['actual_forward_calls']==s['technical_forward_calls_separate']==34
        assert len(tech['records'])==4 and all(all(r['checks'].values()) for r in tech['records'])
        meta=load(run/'metadata.json');assert meta['config']==cfg
        assert meta['environment']['gpu'] and meta['environment']['packages']['torch']
        total=len(rows)+tech['actual_forward_calls'];summed+=total
        records.append({'stage':stage,'run_id':s['run_id'],'gate':s['outcome']['gate'],'scientific_calls':len(rows),'technical_calls':tech['actual_forward_calls'],
            'total_actual_forwards':total,'rows_sha256':sha(raw),'registration_sha256':canonical(reg),'summary_sha256':sha(path),
            'readiness':metrics,'independent_logit_margin_max_error':max_margin_error,'softcap_formula_max_error':max_softcap_error,
            'tokenized_unique_inputs_checked':len(encodings),'protected_prior_files_checked':len(reg['protected_prior_sha256']),
            'confirmation_rows':sum(r['phase']=='confirmation' for r in rows),'CPU_model_forwards':0})
    if not args.allow_incomplete:
        global_budget=load(ROOT/'runs/round6_global_budget.json');assert global_budget['calls']==summed
        assert len(global_budget['completed_phases'])==4
    result={'schema_version':1,'status':'PASS','scope':'all recorded discovery rows, oracles, encoded token IDs, readout arithmetic, technical checks and protected files; no confirmation outcomes exist',
            'total_actual_forwards':summed,'new_model_forwards':0,'records':records,'calibration':'NOT_EVALUATED','confirmation':'NOT_EVALUATED',
            'assessment':'Ready within reviewed scope; mechanisms NOT_IDENTIFIABLE when prerequisites fail'}
    (ROOT/'logs/summary/round6_execution_replay.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':'PASS','stages':len(records),'rows':sum(r['scientific_calls'] for r in records),'total_actual_forwards':summed,'new_model_forwards':0}))


if __name__=='__main__':main()
