"""Explicitly posthoc task-identifiability and behavior diagnostics, no gate changes."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def main():
 out={'schema_version':1,'scope':'POSTHOC on registered P0 only; no independent confirmation; no candidate/gate revision','study_sources':{}}
 for l in 'ABCD':
  sp=ROOT/f'logs/summary/round7_{l}_P0.json';s=read(sp);out['study_sources'][l]={'run':s['run_id'],'summary_sha256':hashlib.sha256(sp.read_bytes()).hexdigest()};rows=[json.loads(x) for x in (ROOT/'runs'/s['run_id']/'rows.jsonl').read_text().splitlines()]
  if l=='D':
   m=read(ROOT/'runs'/s['run_id']/'manifest.json');agree=0;total=0;by=defaultdict(lambda:{'rows':0,'correct':0,'predicts_forward':0,'full_vocab_correct':0})
   for w in m['splits']['P0']:
    z=('north','south','east','west');mapping=dict(zip(z,[w['g'][k] for k in ('red','blue','green','gold')]))
    for q in (0,1):
     x=z[q];agree+=mapping[x]==next(k for k,v in mapping.items() if v==x);total+=1
   for r in rows:
    if r['arm']!='no_prefix_inverse':continue
    o=r['oracle'];forward=o['mapping'][o['query']];same=forward==o['correct'];g=by['forward_equals_inverse' if same else 'forward_differs_from_inverse'];g['rows']+=1;g['correct']+=bool(r['correct']);g['full_vocab_correct']+=r['full_top1_code']==r['expected'];fc=next(k for k,v in o['codebook'].items() if v==forward);g['predicts_forward']+=r['prediction']==fc
   out['D']={'abstract_queries_with_same_forward_and_inverse_answer':agree,'abstract_queries':total,'inverse_anchor_behavior':dict(by),'interpretation':'same-answer queries cannot distinguish operations; no revision of original readiness stop'}
  if l=='C':
   by=defaultdict(lambda:{'rows':0,'correct_restricted':0,'correct_full_vocab':0,'predicts_obsolete_restricted':0,'predicts_obsolete_full_vocab':0})
   for r in rows:
    if r['phase']!='P0':continue
    f=r['factors'];g=by[f['task']+'_'+f['history'][:-1]+'_'+f['ordering']];o=r['oracle'];old=o['old1'] if f['history'].endswith('1') else o['old2'];oc=next(k for k,v in o['codebook'].items() if v==old)
    g['rows']+=1;g['correct_restricted']+=bool(r['correct']);g['correct_full_vocab']+=r['full_top1_code']==r['expected'];g['predicts_obsolete_restricted']+=r['prediction']==oc;g['predicts_obsolete_full_vocab']+=r['full_top1_code']==oc
   out['C']={'behavior':dict(by),'interpretation':'stale logit influence is distinct from a stale final answer or a composition-selective mechanism'}
  if l in 'AB':
   by=defaultdict(lambda:{'rows':0,'correct_restricted':0,'correct_full_vocab':0,'full_vocab_is_code':0})
   for r in rows:
    if r['phase'] not in ('P0_quality','P0_instrument'):continue
    k=r['arm'];g=by[k];g['rows']+=1;g['correct_restricted']+=bool(r['correct']);g['correct_full_vocab']+=r['full_top1_code']==r['expected'];g['full_vocab_is_code']+=r['full_top1_code'] is not None
   out[l]={'quality_and_instrument_behavior':dict(by)}
 p=ROOT/'logs/summary/round7_posthoc_task_diagnostics.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
