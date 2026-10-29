#!/usr/bin/env python3
"""Register and execute one bounded R7 study without changing historical runs."""
from __future__ import annotations
import argparse,hashlib,importlib,json,random,resource,sys,time
from pathlib import Path
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neuron_lab.runtime import save_json,source_hashes,Run,ModelRuntime
from neuron_lab.round6_engine import Probe,encode
from neuron_lab.round7_tools import Journal


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(p):return json.loads(p.read_text())


def paths(cfg):
    sid=cfg['study_id'];assert sid.startswith('R7_')
    letter=sid.split('_')[1];assert letter in 'ABCDE'
    stage=f'round7_{letter}_P0'
    return letter,stage,ROOT/f'runs/{stage}_manifest.json',ROOT/f'logs/summary/{stage}_registration.json'


def prepare(cp):
    from transformers import AutoTokenizer
    cfg=yaml.safe_load(cp.read_text());letter,stage,mp,rp=paths(cfg);mod=importlib.import_module(cfg['scientific_module'])
    if rp.exists():raise RuntimeError('Registration exists; use --check; never overwrite measured definitions')
    manifest=mod.build_manifest(cfg)
    from neuron_lab.round6_explicit_corpus import make_episode
    manifest['technical']=[make_episode(cfg['corpus_seed']+99000000+i,f'{stage}-technical-{i}','A','technical') for i in range(2)]
    if mp.exists():assert load(mp)==manifest
    else:save_json(mp,manifest)
    local=Path((ROOT/'runs/model_path.txt').read_text().strip());assert local.name==cfg['model_revision']
    tokenizer=AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False)
    assert all(len(tokenizer.encode(x,add_special_tokens=False))==1 for x in 'ABCD')
    cases=list(mod.prepared_cases(manifest,cfg));maximum=0;ident=set()
    for c in cases:
        assert c['id'] not in ident;ident.add(c['id'])
        enc=encode(tokenizer,c['text'],cfg['max_input_tokens']);maximum=max(maximum,enc['tokens'])
    assert len(cases)==cfg['expected_scientific_calls_if_quality_passes']
    operative={p:h for p,h in source_hashes().items() if p.startswith('src/') and ('/round7_' not in p or p in cfg['operative_R7_sources'])}
    operative.update({str(cp.relative_to(ROOT)):sha(cp),'scripts/run_round7.py':sha(ROOT/'scripts/run_round7.py')})
    for rel in cfg.get('CPU_test_sources',[]):operative[rel]=sha(ROOT/rel)
    protected=load(ROOT/'logs/summary/round7_baseline_audit.json')['protected_prior_sha256']
    for p,h in protected.items():assert sha(ROOT/p)==h,p
    reg={'schema_version':1,'study_id':cfg['study_id'],'phase':'P0','status':'REGISTERED_BEFORE_STUDY_MODEL_OUTPUTS','model_revision':cfg['model_revision'],
         'config_path':str(cp.relative_to(ROOT)),'config_sha256':sha(cp),'manifest_sha256':canonical(manifest),'operative_source_sha256':operative,
         'protected_prior_sha256':protected,'prepared_scientific_inputs':len(cases),'maximum_tokens':maximum,'study_model_outputs_seen':0,
         'split_world_counts':{k:len(v) for k,v in manifest['splits'].items()},'analysis_scope':cfg['P0_scope'],'scientific_protocol':cfg,
         'technical_inputs':'two independent instrument-only worlds; technical task correctness does not choose scientific cases'}
    save_json(rp,reg);print(json.dumps({'status':'PREPARED','study':letter,'science_cases':len(cases),'maximum_tokens':maximum,'actual_model_calls':0}))


def check(cp):
    cfg=yaml.safe_load(cp.read_text());letter,stage,mp,rp=paths(cfg);reg=load(rp);manifest=load(mp)
    assert sha(cp)==reg['config_sha256'] and canonical(manifest)==reg['manifest_sha256']
    for p,h in {**reg['operative_source_sha256'],**reg['protected_prior_sha256']}.items():assert sha(ROOT/p)==h,p
    mod=importlib.import_module(cfg['scientific_module']);expected=mod.build_manifest(cfg)
    assert expected['splits']==manifest['splits']
    return cfg,reg,manifest


def execute(cp):
    from neuron_lab.round6_explicit_discovery import technical
    cfg,reg,manifest=check(cp);letter,stage,mp,rp=paths(cfg)
    if (ROOT/f'logs/summary/{stage}.json').exists():raise RuntimeError('Measured phase exists; no silent rerun')
    gp=ROOT/'runs/round7_global_budget.json';budget=load(gp) if gp.exists() else {'started_unix':time.time(),'calls':0,'completed':[]}
    assert budget['calls']+cfg['phase_actual_forward_cap']+cfg['technical_call_cap']<=cfg['P0_round_actual_forward_cap'],'P0 round envelope'
    save_json(gp,budget)
    runtime=ModelRuntime(audit_path=ROOT/f'logs/summary/{stage}_model_load.json')
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    tr=Run(stage+'_technical',cfg);tp=Probe(runtime,tr,cfg,cfg['technical_call_cap'],budget)
    try:
        tech=technical(tp,manifest['technical'],cfg);tr.finish({**tech,'study_id':cfg['study_id'],'model_revision':cfg['model_revision'],'measurement_seed':seed})
    except BaseException as exc:
        tr.finish({'error_type':type(exc).__name__,'error':str(exc),'actual_forward_calls':tp.calls},'FAIL');raise
    finally:tp.close();budget['calls']+=tp.calls;save_json(gp,budget)
    run=Run(stage,cfg);p=Probe(runtime,run,cfg,cfg['phase_actual_forward_cap'],budget);j=Journal(p,cfg)
    save_json(run.path/'manifest.json',manifest);save_json(run.path/'registration.json',reg);outcome=None
    try:
        mod=importlib.import_module(cfg['scientific_module']);outcome=mod.run(j,manifest['splits']['P0'],cfg)
        j.close();save_json(run.path/'encoded_inputs.json',j.encodings);p.check_frozen()
        for rel,h in reg['protected_prior_sha256'].items():assert sha(ROOT/rel)==h,rel
        summary={'schema_version':1,'study_id':cfg['study_id'],'model_id':cfg['model_id'],'model_revision':cfg['model_revision'],
            'primary_dtype':cfg['primary_dtype'],'measurement_seed':seed,'corpus_seed':cfg['corpus_seed'],'loading_seed':runtime.cfg['seed'],
            'actual_forward_calls':p.calls,'completed_forward_calls':p.completed,'technical_forward_calls_separate':tech['actual_forward_calls'],
            'rows':len(j.rows),'first50':p.first50,'local_module_evaluations':p.local_evaluations,'manifest_sha256':canonical(manifest),
            'registration_sha256':canonical(reg),'raw_rows_sha256':sha(run.path/'rows.jsonl'),'frozen_parameters_verified':True,'use_cache':False,
            'protected_prior_files_verified':len(reg['protected_prior_sha256']),'peak_CPU_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'outcome':outcome,'source_sha256':source_hashes(),'P1':'NOT_EVALUATED','P2':'NOT_EVALUATED'}
        run.finish(summary);print(json.dumps({'event':'R7_P0_finished','study':letter,'run_id':run.id,'calls':p.calls,'gate':outcome['gate'],'classification':outcome['classification']}),flush=True)
    except BaseException as exc:
        j.close();run.finish({'study_id':cfg['study_id'],'actual_forward_calls':p.calls,'completed_forward_calls':p.completed,'rows':len(j.rows),
            'error_type':type(exc).__name__,'error':str(exc),'outcome':outcome},'FAIL');raise
    finally:
        p.close();budget['calls']+=p.calls;budget['completed'].append({'study':letter,'run_id':run.id,'calls':p.calls});save_json(gp,budget)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True);ap.add_argument('--prepare',action='store_true');ap.add_argument('--check',action='store_true');ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    assert sum([a.prepare,a.check,a.execute])==1;cp=(ROOT/a.config).resolve();assert cp.parent==ROOT/'configs'
    if a.prepare:prepare(cp)
    elif a.check:
        cfg,r,m=check(cp);print(json.dumps({'status':'PASS','study':cfg['study_id'],'operative_sources':len(r['operative_source_sha256']),'protected':len(r['protected_prior_sha256'])}))
    else:execute(cp)

if __name__=='__main__':main()
