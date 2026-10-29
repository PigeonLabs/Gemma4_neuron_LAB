"""POSTHOC audit of actual rendered views, beyond latent-world seed uniqueness."""
from pathlib import Path
from collections import defaultdict
import sys,json,hashlib,itertools
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neuron_lab import round7_rule_footprint as A,round7_carrier as B,round7_history as C,round7_order_priming as D,round7_transfer as E
MODS=dict(A=A,B=B,C=C,D=D,E=E)
def texts(l,m,w):
 if l=='A':return {m.case(w,q,b,r,ns)['text'] for q,b,r,ns in itertools.product(range(6),range(2),range(2),['TARGET','OTHER'])}
 if l=='B':return {m.material(w,q,0,t,d)['text'] for q,t,d in itertools.product(range(2),['early','late'],['short','long'])}
 if l=='C':return {m.material(w,q,b,t,h,o)['text'] for q,b,t,h,o in itertools.product(range(2),range(2),['direct','compose'],['related1','related2','unrelated1','unrelated2'],['before','after'])}
 if l=='D':return {m.material(w,q,b,t,o,p)['text'] for q,b,t,o,p in itertools.product(range(2),range(2),['forward','inverse'],['ascending','descending'],['before','after'])}
 return {m.material(w,t,b)['text'] for t,b in itertools.product(['lookup','position'],range(2))}
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def main():
 out={};Bviews=None
 for l,m in MODS.items():
  splits=json.loads((ROOT/f'runs/round7_{l}_P0_manifest.json').read_text())['splits'];views={};alltext={}
  for sp,ww in splits.items():
   pairs=[(w['id'],texts(l,m,w)) for w in ww];views[sp]=[digest('\n'.join(sorted(t))) for wid,t in pairs];alltext[sp]={digest(s) for wid,t in pairs for s in t}
   if l=='B' and sp=='P0':Bviews={wid:digest('\n'.join(sorted(t))) for wid,t in pairs}
  a=views['P0'];out[l]={'P0_worlds':len(a),'P0_unique_rendered_world_views':len(set(a)),'P0_unique_main_prompt_texts':len(alltext['P0']),'reserved_prompt_overlap_with_P0':{sp:len(alltext['P0'] & ss) for sp,ss in alltext.items() if sp!='P0'},'reserved_entire_world_view_overlap_with_P0':{sp:len(set(a)&set(vv)) for sp,vv in views.items() if sp!='P0'}}
 b=json.loads((ROOT/'logs/summary/round7_B_P0.json').read_text());cfg=yaml.safe_load((ROOT/'configs/round7_B.yaml').read_text());groups=defaultdict(list)
 for e in b['outcome']['episodes']:groups[Bviews[e['id']]].append(e)
 sensitivity={}
 for k in ('late_short','late_long','control_excess'):
  x=np.asarray([np.mean([e[k] for e in ee]) for ee in groups.values()]);rng=np.random.default_rng(cfg['bootstrap_seed']);means=x[rng.integers(0,len(x),(cfg['bootstrap_resamples'],len(x)))].mean(1)
  sensitivity[k]={'point':float(x.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'unique_rendered_views':len(x),'scope':'POSTHOC equal weight per unique rendered world view; original registered 12-world result preserved'}
 res={'schema_version':1,'scope':'CPU POSTHOC finite-support rendered-input audit; no reserved split model outputs inspected','generator_sha256':digest(Path(__file__).read_text()),'studies':out,'B_unique_view_sensitivity':sensitivity,'interpretation':'Distinct full latent worlds do not guarantee every rendered prompt is unique. B has 12 registered IID latent draws but 11 unique rendered views, so also report unique-view sensitivity; no 12 unique-world claim. No P1/P2 was evaluated. E calibration and P0 prompt overlap is zero. Future generators must register rendered-input split deduplication before any confirmation claim; existing manifests and outcomes remain unchanged.'}
 (ROOT/'logs/summary/round7_rendered_split_audit.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n');print(json.dumps(res,ensure_ascii=False))
if __name__=='__main__':main()
