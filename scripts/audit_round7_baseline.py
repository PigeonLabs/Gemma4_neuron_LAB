#!/usr/bin/env python3
"""R7 reset audit: CPU-only arithmetic and raw evidence, preserving past studies."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def rank(x):
    out=np.empty(len(x));order=np.argsort(x,kind='stable');start=0
    while start<len(x):
        end=start+1
        while end<len(x) and x[order[end]]==x[order[start]]:end+=1
        out[order[start:end]]=(start+end-1)/2;start=end
    return out

s=read('logs/summary/round2_ef_analysis.json');e=s['phases']['confirmation']['E'];rr=e['family_predictions']
y=np.array([r['y'] for r in rr]);p0=np.array([r['predictions']['baseline_geometry'] for r in rr]);p1=np.array([r['predictions']['baseline_geometry_controls'] for r in rr])
gain=(y-p0)**2-(y-p1)**2;variance=np.mean((y-y.mean())**2)
delta=float(gain.mean()/variance);final=float(1-np.mean((y-p1)**2)/variance)
assert abs(delta-e['primary']['delta_r2'])<1e-12 and abs(final-e['models']['baseline_geometry_controls']['r2'])<1e-12
by_task={t:float(gain[[r['task']==t for r in rr]].sum()/gain.sum()) for t in sorted({r['task'] for r in rr})}
f=s['phases']['confirmation']['F'];fr=[r for r in f['family_contrasts'] if r['mode']=='translate']
damage=float(np.mean([r['damage'] for r in fr]));interaction=float(np.mean([r['interaction'] for r in fr]))
assert damage==f['groups']['translate']['stats']['damage']['mean'] and not f['groups']['translate']['damage_gate_passed']
r3=read('logs/summary/round3a_p0.json');numerical=[]
for c in r3['qualification']['candidates']:
    bf=np.array([r['BF16'] for r in c['paired_effects']]);fp=np.array([r['FP32'] for r in c['paired_effects']]);rho=float(np.corrcoef(rank(bf),rank(fp))[0,1]);nontrivial=int((abs(fp)>=.02).sum())
    assert nontrivial==c['n_nontrivial_fp32']==0 and abs(rho-c['spearman_signed_effects'])<1e-12
    numerical.append({'id':c['id'],'FP32_mean_absolute_signed_effect':float(np.abs(fp).mean()),'BF16_mean_absolute_signed_effect':float(np.abs(bf).mean()),'signed_spearman':rho,'nontrivial_FP32_cases':nontrivial,'variants':len(fp),'independent_families':8,'eligible':False})
concentration={}
for stage in ('steering','validation','expanded'):
    ss=read(f'logs/summary/{stage}.json');raw=f"runs/{ss['run_id']}/interventions.jsonl";assert sha(raw)==ss['raw_sha256']
    rows=[json.loads(x) for x in (ROOT/raw).read_text().splitlines()];assert len(rows)==ss['intervention_count']
    flips=[r for r in rows if r['role']=='corpus' and r['top1_changed']]
    concentration[ss['split']]={'changed_condition_rows':len(flips),'unique_changed_prompts':len({r['prompt_id'] for r in flips}),'by_prompt':dict(Counter(r['prompt_id'] for r in flips)),'raw_sha256':sha(raw),'semantic_correctness':'NOT_EVALUATED; source logs did not save an answer oracle for the top1 event'}
    expected=read('logs/summary/round2_diagnostics.json')['splits'][ss['split']]['top1_change']
    assert concentration[ss['split']]['changed_condition_rows']==expected['condition_rows'] and concentration[ss['split']]['by_prompt']==expected['by_prompt']
r6=read('logs/summary/round6_execution_results.json');assert r6['total_actual_forwards']==744 and all(x['gate']=='STOPPED_AT_TASK_READINESS' for x in r6['studies'])
checks={}
for name in ('R2','R3','R5','R1_diagnostics'):
    path=f'runs/round7_reaudit_{name}.log';log=(ROOT/path).read_text();assert '"status": "PASS"' in log
    checks[name]={'status':'PASS','log_sha256':sha(path)}
protected={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for pat in ('configs/round[2356]*','logs/summary/round[2356]*','reports/ROUND[2356]*','figures/round[2356]*') for p in ROOT.glob(pat) if p.is_file()}
result={'schema_version':1,'status':'PASS','analysis_type':'new R7 CPU re-audit; past registered results unchanged; no model execution',
 'goal_objective_sha256':hashlib.sha256(Path('/home/jeong/.codex/attachments/44e4e250-07d3-4bfd-800a-57074b8ec2e9/goal-objective.md').read_bytes()).hexdigest(),
 'previous_goal_turn_classification':'progress: completed and published R6 A/B/C feasibility outcomes, independent replay and CP8 receipts',
 'R2_E':{'delta_R2':delta,'final_R2':final,'MSE_reduction_fraction_by_task':by_task,'scope':'conditional on candidate-observed dose; not candidate-blind prediction','dose_matching_source':'scripts/run_round2_ef.py:149-172'},
 'R2_F':{'translate_damage':damage,'interaction':interaction,'damage_gate':False,'recovery_fraction':'NOT_IDENTIFIABLE'},'R3_A':numerical,
 'initial_top1_concentration':concentration,'R5':{'gate':read('logs/summary/round5_discovery.json')['analysis']['gate_outcome'],'raw_rows_rechecked':1632,'actual_forwards':1680},
 'R6':{'actual_forwards':744,'outcome':'all registered task-readiness gates failed','causal_hypotheses':'NOT_IDENTIFIABLE'},
 'existing_raw_audit_commands':checks,'protected_prior_sha256':protected,'new_model_forwards':0,
 'inference_limit':'past observations motivate new questions; none is new independent confirmation or universal neuron null'}
(ROOT/'logs/summary/round7_baseline_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':'PASS','E_delta_R2':delta,'string_MSE_share':by_task['string_rule'],'R3_qualified':0,'initial_test_flips':concentration['test']['changed_condition_rows'],'initial_test_unique_prompts':concentration['test']['unique_changed_prompts'],'protected_prior_files':len(protected),'new_model_forwards':0}))
