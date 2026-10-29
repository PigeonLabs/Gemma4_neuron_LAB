#!/usr/bin/env python3
"""Registered P0 on the actual pinned 12B model. CPU --prepare/--check only.

BF16 native hooks are validated before measurement. One model is first run in
BF16, then its same parameter/buffer values are upcast to FP32, in one process.
No discovery is performed by this command, regardless of numerical outcome.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from neuron_lab.runtime import ModelRuntime,Run,save_json
from neuron_lab.round2_ef import probe_logits,cpu_observation,geometry
from neuron_lab.round3a_corpus import build_manifest
from neuron_lab.round3a import option_metrics,qualify
from run_round2_ef import audit_probe,ForwardBudget

IMPLEMENTATION=['configs/round3a.yaml','scripts/run_round3a_p0.py','src/neuron_lab/round3a.py',
 'src/neuron_lab/round3a_corpus.py','src/neuron_lab/runtime.py','src/neuron_lab/smoke.py',
 'src/neuron_lab/round2_ef.py','scripts/run_round2_ef.py']
PROTECTED=['configs/round2_ef_frozen.json','logs/summary/round2_ef_analysis.json',
 'logs/summary/round2_ef_diagnostics.json','logs/summary/analysis.json','logs/summary/synthesis.json',
 'configs/hypotheses.json','catalog/public/neurons.json']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args): return subprocess.check_output(['git','--git-dir=runs/git-metadata','--work-tree=.',*args],cwd=ROOT,text=True).strip()

def append(path,row):
    with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

class TimedBudget(ForwardBudget):
    def tick(self,*args):
        super().tick(*args)
        if self.count==50:
            self.first_50_seconds=time.monotonic()-self.started
            print(json.dumps({'event':'first_50_forwards','seconds':self.first_50_seconds,'scope':'includes smoke/generation validation'}),flush=True)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    cfg=yaml.safe_load((ROOT/'configs/round3a.yaml').read_text());manifest=build_manifest(cfg)
    path=ROOT/cfg['manifest']
    if a.prepare: save_json(path,manifest)
    if a.prepare or a.check:
        if json.loads(path.read_text())!=manifest:raise ValueError('Manifest does not reproduce')
        from transformers import AutoTokenizer
        local=Path((ROOT/'runs/model_path.txt').read_text().strip());tok=AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False)
        lengths=[len(tok.apply_chat_template([{'role':'user','content':c['text']}],tokenize=True,return_dict=True,add_generation_prompt=True,enable_thinking=False)['input_ids']) for c in manifest['cases']]
        if max(lengths)>cfg['max_input_tokens']:raise ValueError('Prompt over token cap')
        for label in 'ABCD':
            if len(tok.encode(label,add_special_tokens=False))!=1:raise ValueError('Non-single-token code')
        print(json.dumps({'status':'PASS','cases':len(lengths),'max_tokens':max(lengths),'manifest_sha256':sha(path),'inference_executed':False}));return
    if json.loads(path.read_text())!=manifest:raise ValueError('Manifest changed')
    if (ROOT/'logs/summary/round3a_p0.json').exists():raise RuntimeError('P0 result exists; no silent rerun/overwrite')
    fingerprint={name:sha(ROOT/name) for name in IMPLEMENTATION}
    for name,h in fingerprint.items():
        if hashlib.sha256(subprocess.check_output(['git','--git-dir=runs/git-metadata','--work-tree=.','show',f'HEAD:{name}'],cwd=ROOT)).hexdigest()!=h:
            raise RuntimeError(f'Uncommitted registered implementation: {name}')
    protected={name:sha(ROOT/name) for name in PROTECTED}
    runtime=ModelRuntime(audit_path=ROOT/'logs/summary/round3a_p0_model_load.json')
    budget=TimedBudget(runtime,cfg['qualification']);run=Run('round3a_p0',{**cfg,**cfg['qualification']})
    run.meta['registration']={'git_commit':git('rev-parse','HEAD'),'implementation_sha256':fingerprint,
        'manifest_sha256':sha(path),'protected_sha256':protected,'new_study_after_R2':True}
    cases=[c for c in manifest['cases'] if c['split']=='qualification'];rows=[]
    technical={};checks_by_dtype={};all_baselines={};params={}
    try:
        from neuron_lab.smoke import smoke
        smoke_result=smoke(runtime,stage='round3a_p0_smoke')
        technical['original_BF16_smoke']=smoke_result['status']=='PASS'
        labels={l:runtime.tokenizer.encode(l,add_special_tokens=False)[0] for l in 'ABCD'}
        hook_audit=audit_probe(runtime,cases,labels);technical['BF16_aligned_hook_audit']=all(hook_audit['checks'].values())
        save_json(run.path/'hook_audit.json',hook_audit)
        for dtype in ('bfloat16','float32'):
            if dtype=='float32':
                # Parameter/buffer values are upcast; BF16-rounded stored values are retained.
                runtime.model.float();torch.cuda.empty_cache()
                if {p.dtype for p in runtime.model.parameters()}!={torch.float32}:raise ValueError('FP32 upcast failed')
            checks={};baselines={}
            for j,case in enumerate(cases):
                run.check_budget();inputs=runtime.inputs(case['text']);logits,obs=probe_logits(runtime,inputs)
                baselines[case['id']]={'logits':logits.cpu(),'observed':cpu_observation(obs)}
                base_row={**{k:case[k] for k in ('id','family_id','task','difficulty','code','r','c')},'case_id':case['id'],
                    'dtype':dtype,'arm':'clean','input_tokens':inputs['input_ids'].shape[-1],
                    'position_index':inputs['input_ids'].shape[-1]-1,'selected_token_id':int(inputs['input_ids'][0,-1]),
                    **option_metrics(logits,case,labels)}
                rows.append(base_row);append(run.path/'P0_interventions.jsonl',base_row)
                if j<3:
                    bare=runtime.logits(inputs);checks[f'{j}_read_only_exact']=torch.equal(bare,logits)
                    cand=cfg['candidates'][j];value=float(obs[cand['layer']]['act'][cand['neuron']])
                    restored,_=probe_logits(runtime,inputs,[{'kind':'technical_restore','layer':cand['layer'],'neuron':cand['neuron'],'value':value}])
                    checks[f'{j}_restore_exact']=torch.equal(bare,restored)
                    checks[f'{j}_removal_exact']=torch.equal(bare,runtime.logits(inputs))
            all_baselines[dtype]=baselines
            if dtype=='bfloat16':
                for cand in cfg['candidates']:
                    values=torch.tensor([float(baselines[c['id']]['observed'][cand['layer']]['act'][cand['neuron']]) for c in cases],dtype=torch.float64)
                    sd=float(values.std(unbiased=True))
                    if sd<=0:raise ValueError('Qualification candidate has zero natural variance')
                    params[cand['id']]={**cand,'std':sd,'delta':cfg['sigma']*sd,'min':float(values.min()),'max':float(values.max())}
                save_json(run.path/'geometry.json',params)
            start=time.monotonic();before=budget.count
            for j,case in enumerate(cases):
                inputs=runtime.inputs(case['text']);base=baselines[case['id']]
                for cid,param in params.items():
                    layer,n=param['layer'],param['neuron']
                    for sign in (-1,1):
                        run.check_budget();amount=sign*param['delta']
                        logits,obs=probe_logits(runtime,inputs,[{'kind':'add','layer':layer,'neuron':n,'value':amount}])
                        actual=float(obs[layer]['after_act'][n])-float(base['observed'][layer]['act'][n])
                        row={**{k:case[k] for k in ('family_id','task','difficulty','code','r','c')},'case_id':case['id'],
                            'dtype':dtype,'arm':'candidate','candidate_id':cid,'layer':layer,'neuron':n,'sign':sign,
                            'nominal_delta':amount,'actual_delta':actual,**option_metrics(logits,case,labels),
                            'top1_changed':int(logits.argmax())!=int(base['logits'].argmax()),
                            **geometry(base['observed'],obs,layer)}
                        checks[f'{case["id"]}_{cid}_{sign}_finite']=bool(logits.isfinite().all())
                        rows.append(row);append(run.path/'P0_interventions.jsonl',row)
                if (j+1)%4==0:print(json.dumps({'event':'P0_progress','dtype':dtype,'cases':j+1,'total':len(cases),'forwards':budget.count}),flush=True)
            checks_by_dtype[dtype]=checks
            technical[dtype+'_checks']=all(checks.values())
            print(json.dumps({'event':'dtype_complete','dtype':dtype,'intervention_seconds':time.monotonic()-start,'intervention_forwards':budget.count-before}),flush=True)
        torch.save(all_baselines,run.path/'baseline_cache.pt')
        save_json(run.path/'technical_checks.json',checks_by_dtype)
        if any(sha(ROOT/name)!=h for name,h in protected.items()):raise RuntimeError('Protected older result changed')
        result=qualify(rows,cfg,all(technical.values()))
        save_json(run.path/'qualification_analysis.json',result)
        outcome=run.finish({'model_revision':cfg['model_revision'],'seed':cfg['seed'],'device':runtime.cfg['device'],
            'dtype_order':['bfloat16','float32'],'same_stored_weight_values_upcast':True,'condition_rows':len(rows),
            'forward_calls':budget.count,'budget_elapsed_seconds':time.monotonic()-budget.started,
            'first_50_forward_seconds_including_smoke':getattr(budget,'first_50_seconds',None),
            'n_source_families':8,'n_variants':32,'label_token_ids':labels,
            'technical_checks':technical,'qualification':result,'geometry':params,
            'registration':run.meta['registration'],'protected_older_artifacts_sha256':protected,
            'parameter_dtypes_after_upcast':sorted({str(p.dtype) for p in runtime.model.parameters()}),
            'raw_sha256':{name:sha(run.path/name) for name in ('P0_interventions.jsonl','baseline_cache.pt','geometry.json','hook_audit.json','technical_checks.json','qualification_analysis.json')},
            'R3A_discovery':'NOT_EVALUATED','R3A_confirmation':'NOT_EVALUATED',
            'execution_scope':'P0 numerical qualification only; PASS execution is not scientific hypothesis support'})
        print(json.dumps({'event':'P0_complete','run_id':run.id,'execution_status':outcome['status'],'qualification_status':result['status'],'eligible_candidates':result['eligible_candidates'],'forwards':budget.count}),flush=True)
    except Exception as exc:
        run.finish({'error':str(exc),'model_revision':cfg['model_revision'],'forward_calls':budget.count},status='FAIL');raise
    finally:budget.handle.remove()

if __name__=='__main__':main()
