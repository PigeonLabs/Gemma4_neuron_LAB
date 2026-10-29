"""Independent CPU replay of R7 raw readouts/oracles/counters and frozen files.

No scientific module is imported. Does not rerun the model or relabel old gates.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import hashlib,json,sys
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def eq(a,b,tol=1e-8):assert np.allclose(a,b,rtol=0,atol=tol),(a,b)
def value(row,which):return row['post_softcap_logits']['ABCD'.index(which)]
def cm(row):
 x=np.asarray(row['post_softcap_logits'],float);i='ABCD'.index(row['expected']);xx=np.delete(x,i);m=max(xx)
 return float(x[i]-m-np.log(np.exp(xx-m).sum()))
def pair(row,a,b):return value(row,a)-value(row,b)
def coding(book,v):return next(k for k,q in book.items() if q==v)
def mean(x):return float(np.mean(x))
def ci(v,cfg):
 x=np.asarray(v,float);rng=np.random.default_rng(cfg['bootstrap_seed']);m=x[rng.integers(0,len(x),(cfg['bootstrap_resamples'],len(x)))].mean(1)
 return {'point':mean(x),'ci95':np.quantile(m,[.025,.975]).tolist(),'episodes':len(x)}
def verify_stats(episodes,stats,cfg):
 for key,stat in stats.items():
  rr=ci([e[key] for e in episodes],cfg);eq(rr['point'],stat['point']);eq(rr['ci95'],stat['ci95']);assert rr['episodes']==stat['episodes']

def oracle(row,w):
 f=row['factors'];o=row['oracle'];letter=row['study'];code_anchor='code' in row['arm']
 if letter=='A':
  q=w['queries'][f['query_index']];codeonly=f['code_only'];r=f['anchor'] if f['anchor'] is not None else f['evidence_rule'];spec=w['rules'][r]
  v=codeonly if codeonly is not None else int(q['x'][spec['index']])^spec['invert'];book={'A':f['book_index'],'B':1-f['book_index']}
  assert o['h0']==q['h0'] and o['h1']==q['h1'];return coding(book,v)
 if code_anchor:return coding(w['books'][f['codebook_index']],('north','south','east','west')[w['seed']%4])
 book=w['books'][f['codebook_index']]
 if letter=='B':
  values={x:w['g'][w['f'][x]] for x in w['queries']};q=f['query_index'];donor=f['donor']
  if donor:
   k=w['queries'][q if donor=='same' else 1-q];values[k]=next(z for z in ('north','south','east','west') if z not in values.values())
  assert o['values']==values;return coding(book,values[w['queries'][q]])
 if letter=='C':
  v=w['g'][w['f'][w['queries'][f['query_index']]]];alts=[z for z in ('north','south','east','west') if z!=v]
  assert o['old1']==alts[0] and o['old2']==alts[1] and o['f']==w['f'] and o['g']==w['g']
  return coding(book,v)
 if letter=='D':
  z=('north','south','east','west');m=dict(zip(z,[w['g'][k] for k in ('red','blue','green','gold')]));x=z[f['query_index']]
  v=m[x] if f['task']=='forward' else next(k for k,v in m.items() if v==x);assert o['mapping']==m;return coding(book,v)
 if letter=='E':
  q=w['seed']%2;v=w['g'][('red','blue')[q]] if f['task']=='lookup' else [w['g'][k] for k in ('red','blue','green','gold')][q+1]
  return coding(book,v)
 raise ValueError(letter)

def B_episodes(rows):
 groups=defaultdict(dict);episodes=defaultdict(lambda:defaultdict(list));acc=defaultdict(lambda:defaultdict(list))
 for r in rows:
  if r['phase']!='P0':continue
  f=r['factors'];k=(r['episode_id'],f['timing'],f['distance'],f['query_index']);groups[k][r['arm']]=r
 for (wid,timing,dist,q),rr in groups.items():
  c=rr['clean'];base=c['expected'];same=rr['donor_same']['expected'];by={name:pair(r,same,base) for name,r in rr.items()}
  select=by['main_same']-by['main_other'];control=.5*(by['pre_same']-by['pre_other']+by['row1_same']-by['row1_other'])
  episodes[wid][timing+'_'+dist].append(select);episodes[wid]['control_excess'].append(select-control)
  for name,r in rr.items():
   if not name.startswith('donor'):acc[wid][name].append(bool(r['correct']))
 return [{'id':wid,**{k:mean(v) for k,v in ep.items()},'accuracy':{k:mean(v) for k,v in acc[wid].items()}} for wid,ep in episodes.items()]

def C_episodes(rows):
 groups=defaultdict(dict);episodes=defaultdict(lambda:defaultdict(list))
 for r in rows:
  if r['phase']!='P0':continue
  f=r['factors'];groups[(r['episode_id'],f['query_index'],f['codebook_index'],f['task'],f['ordering'])][f['history']]=r
 effects={}
 for (wid,q,bi,task,ordering),rr in groups.items():
  o=next(iter(rr.values()))['oracle'];a,b=coding(o['codebook'],o['old1']),coding(o['codebook'],o['old2']);s={k:pair(r,a,b) for k,r in rr.items()}
  eff=s['related1']-s['related2']-s['unrelated1']+s['unrelated2'];effects[wid,q,bi,task,ordering]=eff;episodes[wid][task+'_'+ordering].append(eff)
 for (wid,q,bi,task,ordering),eff in effects.items():
  if task=='compose':episodes[wid]['interaction_'+ordering].append(eff-effects[wid,q,bi,'direct',ordering])
 return [{'id':wid,**{k:mean(v) for k,v in ep.items()}} for wid,ep in episodes.items()]

def D_episodes(rows):
 groups=defaultdict(dict);episodes=defaultdict(lambda:defaultdict(list));effects={}
 for r in rows:
  if r['phase']!='P0':continue
  f=r['factors'];groups[(r['episode_id'],f['query_index'],f['codebook_index'],f['task'],f['placement'])][f['orientation']]=r
 for (wid,q,bi,task,place),rr in groups.items():
  s=(cm(rr['ascending'])-cm(rr['descending']))*(1 if task=='forward' else -1);effects[wid,q,bi,task,place]=s
  episodes[wid][task+'_'+place].append(s);episodes[wid][task+'_book'+str(bi)].append(s)
 for (wid,q,bi,task,place),s in effects.items():
  if task=='forward':episodes[wid]['interaction_'+place].append(.5*(s+effects[wid,q,bi,'inverse',place]))
 return [{'id':wid,**{k:mean(v) for k,v in ep.items()}} for wid,ep in episodes.items()]

def e_replay(path,rows,summary,cfg):
 clean={r['id']:r for r in rows if r['arm'] in ('P0_clean','calibration_clean')};deltas=defaultdict(list)
 for r in rows:
  if r['arm'].startswith(('cal_native','cal_resid','P0_source','P0_target')):
   fam='native_mlp' if 'native_mlp' in r['arm'] else 'native_attn' if 'native_attn' in r['arm'] else 'resid';deltas[r['id'],fam].append(cm(r)-cm(clean[r['id']]))
 if not deltas:return {'calibration_effect_recomputed':'NOT_EVALUATED; stopped before reactions','prospective_predictions':'NOT_EVALUATED'}
 env={f:mean([abs(v) for (rid,ff),dd in deltas.items() if ff==f and rid.startswith('r7E-calibration') for v in dd]) for f in ('native_mlp','native_attn','resid')}
 for k,v in summary['outcome'].get('calibration_mean_absolute_effect',{}).items():eq(env[k],v)
 if not (path/'prediction_cases.json').exists():return {'calibration_effect_recomputed':env,'prospective_predictions':'NOT_EVALUATED'}
 models=read(path/'frozen_calibration_models.json')
 for target,mm in models['models'].items():
  for name,model in mm.items():
   tr=models['training'];x=np.asarray([r['features']+([r['source']] if name=='augmented' else []) for r in tr]);y=np.asarray([r['y'][target] for r in tr]);mu=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1.;z=np.c_[np.ones(len(x)),(x-mu)/sd];pen=np.eye(z.shape[1])*cfg['ridge_lambda'];pen[0,0]=0
   eq(np.linalg.solve(z.T@z+pen,z.T@y),model['beta']);eq(mu,model['mean']);eq(sd,model['sd'])
 predictions=read(path/'prospective_predictions.json');cases=read(path/'prediction_cases.json');ps={r['id']:r for r in predictions}
 for c in cases:
  for fam in ('native_mlp','native_attn','resid'):eq(c['y'][fam],mean(np.abs(deltas[c['id'],fam])))
  pp=ps[c['id']];idx=next(i for i,r in enumerate(rows) if r['id']==c['id'] and r['arm'].startswith('P0_target'));assert idx==pp['completed_calls_before_target']
  for target in ('native_attn','resid'):
   for model in ('baseline','augmented'):
    f=models['models'][target][model];features=c['features']+([c['source']] if model=='augmented' else []);pred=float(np.r_[1.,(np.asarray(features)-f['mean'])/f['sd']]@np.asarray(f['beta']));eq(pred,c['predictions'][target][model]);eq(pred,pp['predictions'][target][model])
 denom={t:float(np.var([c['y'][t] for c in cases])) for t in ('native_attn','resid')};ee=defaultdict(lambda:defaultdict(list))
 for c in cases:
  for t in denom:
   y=c['y'][t];p=c['predictions'][t];d=((y-p['baseline'])**2-(y-p['augmented'])**2)/denom[t];ee[c['world']]['primary'].append(d);ee[c['world']][c['task']].append(d)
 episodes=[{'id':w,**{k:mean(v) for k,v in e.items()}} for w,e in ee.items()];verify_stats(episodes,summary['outcome']['statistics'],cfg)
 return {'calibration_effect_recomputed':env,'prospective_predictions_verified':len(cases),'worlds':len(episodes),'statistics_recomputed':True}

def main():
 records=[];total=0;tech=0
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained((ROOT/'runs/model_path.txt').read_text().strip(),local_files_only=True)
 soft=read(ROOT/'runs/runtime_config.json')['text_config']['final_logit_softcapping']
 for letter in 'ABCDE':
  sp=ROOT/f'logs/summary/round7_{letter}_P0.json'
  if not sp.exists():continue
  summary=read(sp);assert summary['status']=='PASS';cfg=yaml.safe_load((ROOT/f'configs/round7_{letter}.yaml').read_text());reg=read(ROOT/f'logs/summary/round7_{letter}_P0_registration.json');path=ROOT/'runs'/summary['run_id'];manifest=read(path/'manifest.json');assert canonical(manifest)==summary['manifest_sha256']==reg['manifest_sha256'];assert canonical(reg)==summary['registration_sha256']
  for p,h in {**reg['operative_source_sha256'],**reg['protected_prior_sha256']}.items():assert sha(ROOT/p)==h,p
  assert sha(path/'rows.jsonl')==summary['raw_rows_sha256'];rows=[json.loads(s) for s in (path/'rows.jsonl').read_text().splitlines()];worlds={w['id']:w for split in manifest['splits'].values() for w in split}
  if letter=='B':
   # Instrument-only input is identified in its own recorded encoding; scientific oracles below use registered worlds.
   worlds.update({})
  encoded=read(path/'encoded_inputs.json')
  for text,enc in encoded.items():
   assert hashlib.sha256(text.encode()).hexdigest()==enc['text_sha256']
   serial=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,return_dict=True,add_generation_prompt=True,enable_thinking=False)
   assert serial['input_ids']==enc['input_ids'] and serial['attention_mask']==enc['attention_mask']
  encbyhash={e['text_sha256']:e for e in encoded.values()};assert len(rows)==summary['rows']==summary['completed_forward_calls']==summary['actual_forward_calls']
  unique=set()
  for rr in rows:
   u=(rr['id'],rr['arm'],rr['phase']);assert u not in unique,u;unique.add(u);enc=encbyhash[rr['text_sha256']];assert rr['tokens']==enc['tokens'] and rr['query_start']==enc['query_start'];assert rr['prediction']=='ABCD'[int(np.argmax(rr['post_softcap_logits']))];eq(np.tanh(np.asarray(rr['raw_head_logits'],np.float32)/soft)*soft,rr['post_softcap_logits'],4e-6)
   if rr['phase']!='P0_instrument':assert oracle(rr,worlds[rr['episode_id']])==rr['expected'],rr['id']
   if rr['has_unique_target_answer']:assert rr['correct']==(rr['prediction']==rr['expected']);eq(cm(rr),rr['correct_margin'])
   else:assert rr['correct'] is None and rr['correct_margin'] is None
   for d in rr['delivery']:assert d['positions']==len(enc['regions'][d['scope']]) and np.isfinite(d['delta_norm'])
  if letter=='B':
   import torch
   cached_pairs=0;maximum=0.
   for w in manifest['splits']['P0']:
    for distance in ('short','long'):
     for d0,d1 in [('same','other'),('other','same')]:
      a=path/f"{w['id']}-q0-b0-late-{distance}-d{d0}_captured.pt";b=path/f"{w['id']}-q1-b0-late-{distance}-d{d1}_captured.pt"
      if not a.exists():continue
      aa=torch.load(a,map_location='cpu',weights_only=True);bb=torch.load(b,map_location='cpu',weights_only=True);assert aa.keys()==bb.keys()
      for k in aa:
       dd=float((aa[k]-bb[k]).abs().max());maximum=max(maximum,dd);assert dd<=1e-5,('pre-query future-key leakage',w['id'],k,dd)
      cached_pairs+=1
   assert cached_pairs==len(manifest['splits']['P0'])*4
   ins={r['arm']:r for r in rows if r['phase']=='P0_instrument'}
   assert max(abs(a-b) for a,b in zip(ins['null_graph_A']['raw_head_logits'],ins['null_graph_B']['raw_head_logits']))<=cfg['seal_parity_tolerance']
   for enc in encoded.values():
    if 'B_suffix' not in enc['regions']:continue
    rr=enc['regions'];end=max(rr['carrier']);assert rr['B_suffix']==list(range(end+1,enc['tokens']))
    for role,allowed in [('main',rr['carrier']),('pre',rr['pre_anchor']),('row1',rr['row1_token']),('null',[])]:assert rr['B_block_'+role]==[i for i in range(end+1) if i not in allowed]
  qq=defaultdict(list)
  for rr in rows:
   if rr['phase']!='P0_quality':continue
   if letter=='A':k='code_anchor' if rr['arm']=='code_anchor' else f"rule_{rr['factors']['anchor']}_{rr['factors']['proximity']}"
   elif rr['arm']=='code_anchor':k='code'
   elif letter=='B':k=rr['factors']['distance']
   else:k=rr['factors']['task']
   qq[k].append(bool(rr['correct']))
  for k,v in summary['outcome'].get('readiness',{}).items():assert len(qq[k])==v['rows'] and sum(qq[k])==v['correct'];eq(mean(qq[k]),v['accuracy'])
  ep=None
  if summary['outcome']['gate']=='P0_COMPLETED' and letter in 'BCD':
   ep={'B':B_episodes,'C':C_episodes,'D':D_episodes}[letter](rows);verify_stats(ep,summary['outcome']['statistics'],cfg)
   old={e['id']:e for e in summary['outcome']['episodes']}
   for e in ep:
    for k,v in e.items():
     if k=='id':continue
     if isinstance(v,dict):
      for kk,vv in v.items():eq(vv,old[e['id']][k][kk])
     else:eq(v,old[e['id']][k])
  extra=e_replay(path,rows,summary,cfg) if letter=='E' else {'query_blind_cache_pairs':cached_pairs,'query_blind_cache_max_difference':maximum} if letter=='B' else {}
  technical=read(ROOT/f'logs/summary/round7_{letter}_P0_technical.json');assert technical['status']=='PASS' and technical['actual_forward_calls']==34
  total+=len(rows);tech+=34
  records.append({'study':letter,'run_id':summary['run_id'],'raw_rows':len(rows),'gate':summary['outcome']['gate'],'classification_preserved':summary['outcome']['classification'],'raw_readout_oracle_token_scope_hash_checks':'PASS','episode_statistics_recomputed':len(ep) if ep else 0,**extra})
 budget=read(ROOT/'runs/round7_global_budget.json');assert budget['calls']==total+tech
 result={'schema_version':1,'status':'PASS','scope':'independent CPU arithmetic/oracle/frozen-file replay; not an independent GPU rerun','scientific_forwards':total,'technical_forwards':tech,'total_actual_forwards':total+tech,'studies':records,'A_protocol_defect_preserved':read(ROOT/'logs/summary/round7_input_audit.json')['A_anchor_inconsistent_rows'],'prior_files_preserved':91}
 (ROOT/'logs/summary/round7_replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
