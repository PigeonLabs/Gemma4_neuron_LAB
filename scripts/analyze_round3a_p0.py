#!/usr/bin/env python3
"""Read-only raw replay of registered R3-A numerical qualification (CPU)."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from neuron_lab.round3a import qualify
from neuron_lab.round3a_corpus import build_manifest

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def compute():
    p=ROOT/'logs/summary/round3a_p0.json';s=json.loads(p.read_text())
    if s['status']!='PASS':raise ValueError('No completed P0 run')
    cfg=yaml.safe_load((ROOT/'configs/round3a.yaml').read_text())
    run=ROOT/'runs'/s['run_id'];checks={}
    for name,h in s['raw_sha256'].items():
        checks['raw_'+name]=digest(run/name)==h
    for name,h in s['registration']['implementation_sha256'].items():
        checks['registered_'+name]=digest(ROOT/name)==h
    for name,h in s['protected_older_artifacts_sha256'].items():
        checks['protected_'+name]=digest(ROOT/name)==h
    checks['manifest']=digest(ROOT/cfg['manifest'])==s['registration']['manifest_sha256']
    checks['manifest_reproduces']=json.loads((ROOT/cfg['manifest']).read_text())==build_manifest(cfg)
    rows=[json.loads(line) for line in (run/'P0_interventions.jsonl').read_text().splitlines()]
    expected={c['id']:c for c in build_manifest(cfg)['cases'] if c['split']=='qualification'}
    checks['condition_count']=len(rows)==s['condition_rows']==576
    counts=Counter((r['dtype'],r['case_id']) for r in rows)
    checks['complete_arms']=len(counts)==64 and set(counts.values())=={9}
    checks['correct_factorial_codes']=all(all(r[k]==expected[r['case_id']][k] for k in ('r','c','code','task','difficulty','family_id')) for r in rows)
    checks['margin_from_recorded_label_logits']=all(abs(r['semantic_margin']-r['c']*(r['label_logits'][r['code'][0]]-r['label_logits'][r['code'][1]]))<1e-12 for r in rows)
    checks['correctness_margin_and_oracle']=all(r['correctness_margin']==r['r']*r['semantic_margin'] and r['forced_choice_correct']==(r['correctness_margin']>0) for r in rows)
    params=json.loads((run/'geometry.json').read_text())
    checks['registered_shared_nominal_dose']=all(r['nominal_delta']==r['sign']*params[r['candidate_id']]['delta'] for r in rows if r['arm']=='candidate')
    measured=qualify(rows,cfg,all(s['technical_checks'].values()))
    checks['registered_gate_replay']=measured==s['qualification']
    checks['P0_budget']=s['forward_calls']<=cfg['qualification']['max_forward_calls'] and s['budget_elapsed_seconds']<=cfg['qualification']['max_seconds']
    baseline={}
    for dtype in ('bfloat16','float32'):
        clean=[r for r in rows if r['dtype']==dtype and r['arm']=='clean']
        wrong={r['family_id'] for r in clean if not r['forced_choice_correct']}
        baseline[dtype]={'n_cases':len(clean),'forced_choice_correct':sum(r['forced_choice_correct'] for r in clean),
            'raw_top1_correct':sum(r['raw_top1_correct'] for r in clean),'families_with_any_wrong_variant':len(wrong),
            'candidate_top1_change_rows':sum(r.get('top1_changed',False) for r in rows if r['dtype']==dtype and r['arm']=='candidate')}
    grouped=defaultdict(dict)
    for row in measured['signed_effects']:
        grouped[row['candidate_id']][(row['dtype'],row['case_id'])]=row
    descriptive=[]
    for cid,g in grouped.items():
        for dtype in ('bfloat16','float32'):
            ds=[r['d'] for (dt,_),r in g.items() if dt==dtype]
            descriptive.append({'candidate_id':cid,'dtype':dtype,'n_signed_cases':len(ds),'zero_signed_effect_cases':sum(d==0 for d in ds),
                'median_abs_signed_effect_nats':float(np.median(abs(np.array(ds)))),'max_abs_signed_effect_nats':float(max(abs(np.array(ds))))})
    if not all(checks.values()):raise ValueError('P0 replay failed: '+str([k for k,v in checks.items() if not v]))
    return {'schema_version':1,'status':'PASS','scope':'CPU replay of recorded real-model P0; no new inference',
        'run_id':s['run_id'],'source_sha256':{str(p.relative_to(ROOT)):digest(p),'configs/round3a.yaml':digest(ROOT/'configs/round3a.yaml')},
        'checks':checks,'qualification_status':measured['status'],'eligible_candidates':measured['eligible_candidates'],
        'baseline':baseline,'effect_resolution_descriptive':descriptive,
        'inference_unit':'8 source families, 4 paired derivatives each; 32 derivatives are not 32 independent families',
        'new_GPU_inference_executed':False}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check',action='store_true');args=ap.parse_args()
    result=compute();out=ROOT/'logs/summary/round3a_p0_analysis.json'
    if args.check:
        if json.loads(out.read_text())!=result:raise SystemExit('P0 analysis is stale')
    else:out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'qualification_status':result['qualification_status'],'checks':len(result['checks']),'new_GPU_inference':False}))
