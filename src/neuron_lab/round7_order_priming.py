"""R7-D: answer-independent prefix ordering; no abstract syntax claim."""
from __future__ import annotations
from collections import defaultdict
import json
import numpy as np
from .round7_lookup_worlds import manifest,code,code_case,Z,encode_quality
from .round6_engine import margin,bootstrap

def build_manifest(cfg):return manifest(cfg)
def material(w,q=0,bi=0,task='forward',orientation='none',placement='before'):
    mapping=dict(zip(Z,[w['g'][k] for k in ('red','blue','green','gold')]))
    x=Z[q];answer=mapping[x] if task=='forward' else next(k for k,v in mapping.items() if v==x);book=w['books'][bi]
    seq=['alpha','beta','gamma','delta'];seq=seq if orientation!='descending' else list(reversed(seq))
    prefix='PREFIX_BEGIN:\n '+' '.join(seq*12)+'\nPREFIX_END\n' if orientation!='none' else ''
    target='TARGET_BEGIN:\nMapping table:\n'+'\n'.join(f'{k} -> {mapping[k]}' for k in Z)+'\nTARGET_END\n'
    facts=prefix+target if placement=='before' else target+prefix
    cue=('FORWARD: find the value to the right of the named key.' if task=='forward' else 'INVERSE: find the key to the left of the named value.')+' The prefix words carry no facts about the mapping.'
    text='Read the arbitrary mapping literally.\nCUE_BEGIN:\n'+cue+'\nFACTS_BEGIN:\n'+facts+'END_FACTS\nNEW_QUERY:\n'+f'Input: {x}. Codebook: '+'; '.join(f'{a}={b}' for a,b in book.items())+'. Reply only A/B/C/D.'
    return {'id':f'{w["id"]}-q{q}-b{bi}-{task}-{orientation}-{placement}','episode_id':w['id'],'study':'D','text':text,'expected':code(book,answer),
            'sections':{'prefix':('PREFIX_BEGIN:\n','\nPREFIX_END')} if orientation!='none' else {},
            'factors':{'query_index':q,'codebook_index':bi,'task':task,'orientation':orientation,'placement':placement},'oracle':{'mapping':mapping,'correct':answer,'query':x,'codebook':book}}
def prepared_cases(m,cfg):
    for w in m['splits']['P0']:
        for q in (0,1):
            for bi in (0,1):
                for task in ('forward','inverse'):yield material(w,q,bi,task)
        for bi in (0,1):yield code_case(w,bi,'D')
    for w in m['splits']['P0']:
        for q in (0,1):
            for bi in (0,1):
                for task in ('forward','inverse'):
                    for placement in ('before','after'):
                        for orientation in ('ascending','descending'):yield material(w,q,bi,task,orientation,placement)
def run(j,worlds,cfg):
    qual=defaultdict(list)
    for w in worlds:
        for q in (0,1):
            for bi in (0,1):
                for task in ('forward','inverse'):
                    rr,_,_=j.call(material(w,q,bi,task),arm='no_prefix_'+task,phase='P0_quality');qual[task].append(rr)
        for bi in (0,1):rr,_,_=j.call(code_case(w,bi,'D'),arm='code_anchor',phase='P0_quality');qual['code'].append(rr)
    readiness=encode_quality(qual);eligible=all(v['accuracy']>=(cfg['code_accuracy_min'] if k=='code' else cfg['anchor_accuracy_min']) for k,v in readiness.items())
    print(json.dumps({'event':'R7_D_quality','readiness':readiness,'eligible':eligible,'calls':j.p.calls}),flush=True)
    if not eligible:return {'gate':'STOPPED_AT_FORWARD_INVERSE_READINESS','classification':'Inconclusive','scientific_status':'NOT_IDENTIFIABLE','readiness':readiness}
    episodes=[]
    for w in worlds:
        vals=defaultdict(list)
        for q in (0,1):
            for bi in (0,1):
                for placement in ('before','after'):
                    contrasts={}
                    for task in ('forward','inverse'):
                        by={};tokens=[]
                        for orientation in ('ascending','descending'):
                            c=material(w,q,bi,task,orientation,placement);enc=j.enc(c);tokens.append(sorted(enc['input_ids'][i] for i in enc['regions']['prefix']))
                            rr,_,_=j.call(c,arm=orientation+'_'+placement);by[orientation]=margin(rr,c['expected'])
                        assert tokens[0]==tokens[1],'Prefix token multiset differs'
                        effect=by['ascending']-by['descending'];effect*=1 if task=='forward' else -1
                        contrasts[task]=effect;vals[task+'_'+placement].append(effect);vals[task+'_book'+str(bi)].append(effect)
                    vals['interaction_'+placement].append(.5*(contrasts['forward']+contrasts['inverse']))
        episodes.append({'id':w['id'],**{k:float(np.mean(v)) for k,v in vals.items()}})
        print(json.dumps({'event':'R7_D_episode','id':w['id'],'calls':j.p.calls}),flush=True)
    stats={k:bootstrap([e[k] for e in episodes],cfg) for k in episodes[0] if k!='id'}
    needed=['forward_before','inverse_before','forward_book0','forward_book1','inverse_book0','inverse_book1']
    pos=all(stats[k]['ci95'][0]>cfg['practical_margin_nats'] for k in needed)
    neg=any(stats[k]['ci95'][1]<cfg['practical_margin_nats'] for k in ('forward_before','inverse_before'))
    return {'gate':'P0_COMPLETED','classification':'Positive' if pos else 'Negative' if neg else 'Inconclusive','scientific_status':'EXPLORATORY_ORDERING_TREATMENT_ONLY','readiness':readiness,'statistics':stats,'episodes':episodes,'scope':'token ordering treatment; no isolation of abstract structure from n-grams or evidence of extra target computation','P1':'NOT_EVALUATED','P2':'NOT_EVALUATED'}
