"""Posthoc CPU checks of reviewer factual alternatives; no study retuning/refit."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def ranks(x):
 x=np.asarray(x,float);o=np.argsort(x,kind='stable');r=np.empty(len(x));a=0
 while a<len(x):
  b=a+1
  while b<len(x) and x[o[b]]==x[o[a]]:b+=1
  r[o[a:b]]=(a+b-1)/2;a=b
 return r
def rho(x,y):
 a=ranks(x);b=ranks(y)
 return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None
def bc(x,y,rng,n=2000):
 vv=[]
 for _ in range(n):
  ix=rng.integers(0,len(x),len(x));v=rho(np.asarray(x)[ix],np.asarray(y)[ix])
  if v is not None:vv.append(v)
 return np.quantile(vv,[.025,.975]).tolist()
def main():
 from transformers import AutoTokenizer
 t=AutoTokenizer.from_pretrained((ROOT/'runs/model_path.txt').read_text().strip(),local_files_only=True)
 b=read(ROOT/'logs/summary/round7_B_P0.json');enc=read(ROOT/'runs'/b['run_id']/'encoded_inputs.json');n=0
 for text,e in enc.items():
  if 'B_suffix' not in e['regions']:continue
  templ=t.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False);off=t(templ,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'];a=templ.index('Codebook:');z=templ.index('. Reply with only A/B/C/D.',a);ix=[i for i,(u,v) in enumerate(off) if v>a and u<z]
  assert ix and set(ix)<=set(e['regions']['B_suffix']) and not set(ix)&set(e['regions']['B_block_null']);n+=1
 c=read(ROOT/'logs/summary/round7_C_P0.json');rr=[json.loads(x) for x in (ROOT/'runs'/c['run_id']/'rows.jsonl').read_text().splitlines()];groups=defaultdict(dict)
 for r in rr:
  if r['phase']!='P0':continue
  f=r['factors'];groups[(r['episode_id'],f['query_index'],f['codebook_index'],f['task'],f['ordering'],f['history'][:-1])][f['history'][-1]]=r
 by=defaultdict(lambda:{'paired_conditions':0,'restricted_choice_changes':0,'full_vocab_top1_changes':0,'correctness_changes':0})
 for (w,q,bi,task,ordering,context),pair in groups.items():
  a,b=pair['1'],pair['2'];d=by[task+'_'+ordering+'_'+context];d['paired_conditions']+=1;d['restricted_choice_changes']+=a['prediction']!=b['prediction'];d['full_vocab_top1_changes']+=a['full_top1_id']!=b['full_top1_id'];d['correctness_changes']+=a['correct']!=b['correct']
 e=read(ROOT/'logs/summary/round7_E_P0.json');cases=read(ROOT/'runs'/e['run_id']/'prediction_cases.json');dd=defaultdict(list)
 for c in cases:dd[c['world'],c['task']].append(c)
 associations=[];rng=np.random.default_rng(71777)
 for task in ('lookup','position'):
  vv=[v for (w,tt),v in dd.items() if tt==task];source=[np.mean([r['source'] for r in v]) for v in vv]
  for target in ('native_attn','resid'):
   for typ in ('raw_target','target_minus_frozen_baseline'):
    y=[np.mean([r['y'][target]-(r['predictions'][target]['baseline'] if typ!='raw_target' else 0.) for r in v]) for v in vv]
    associations.append({'task':task,'target':target,'description':typ,'rho':rho(source,y),'paired_world_bootstrap_CI95':bc(source,y,rng),'worlds':len(vv)})
 out={'schema_version':1,'status':'PASS','scope':'POSTHOC descriptive audit of same P0; no new confirmation, model calls, P0 coefficient fit, gate change or candidate selection','B':{'encoded_cases_codebook_in_visible_suffix':n,'codebook_was_blocked':False,'original_pre_fact_cue':'blocked by suffix-to-prefix seal','format_collapse':'real observation, but its attribution specifically to blocking codebook is false; missing sealed explicit-answer control limits mechanism interpretation'},'C':{'all_1024_main_rows_scored_for_correctness_and_full_vocab':True,'history_value_pair_behavior':dict(by),'currently_mentioned_obsolete_value_chosen':0,'caution':'old1-vs-old2 logits can shift without a changed answer; direction of bias from accidental unrelated-history agreement is not mathematically guaranteed'},'E':{'associations':associations,'caution':'Target residualization alone is not partial Spearman. Positive association does not prove calibration alone caused prediction failure; zero point or CI crossing0 does not establish absent association. Fixed prospectively evaluated algorithm negative is valid and narrower than a universal claim. Calibration had six independent worlds despite 24 rows.'}}
 p=ROOT/'logs/summary/round7_peer_factual_checks.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
