"""R7-B: path-complete sealed-graph, query-blind indexed transport P0."""
from __future__ import annotations
from collections import defaultdict
import copy,json
import numpy as np
import torch
from .round7_lookup_worlds import manifest,code,code_case,Z,encode_quality
from .round6_engine import bootstrap,pair_score
from .runtime import save_json

def build_manifest(cfg):return manifest(cfg)
def material(w,q=0,bi=0,timing='late',distance='short',donor=None):
    values={x:w['g'][w['f'][x]] for x in w['queries']};x=w['queries'][q];old=values[x]
    if donor is not None:
        changed=x if donor=='same' else w['queries'][1-q]
        values[changed]=next(v for v in Z if v not in values.values())
    prefix='The two mappings are independent. Use their values literally.\nCUE_BEGIN:\n'+(f'Later you will be asked for {x}.\n' if timing=='early' else 'The key will be revealed only after the neutral marker.\n')
    facts='PRE_ANCHOR_BEGIN:\n.\nPRE_ANCHOR_END\n'
    for i,k in enumerate(w['queries']):
        line=f'{k} -> {values[k]}'
        facts+=('ROW1_BEGIN:\n'+line+'\nROW1_END' if i==0 else line)+'\n'
    facts+=('zero '*(8 if distance=='short' else 1000))+'\nCARRIER_BEGIN:\n.\nCARRIER_END'
    book=w['books'][bi];text=prefix+'FACTS_BEGIN:\n'+facts+'\nEND_FACTS\nNEW_QUERY:\n'+f'Key: {x}. Codebook: '+'; '.join(f'{a}={b}' for a,b in book.items())+'. Reply with only A/B/C/D.'
    return {'id':f'{w["id"]}-q{q}-b{bi}-{timing}-{distance}-d{donor}','episode_id':w['id'],'study':'B','text':text,'expected':code(book,values[x]),
            'sections':{'pre_anchor':('PRE_ANCHOR_BEGIN:\n','\nPRE_ANCHOR_END'),'row1':('ROW1_BEGIN:\n','\nROW1_END'),'carrier':('CARRIER_BEGIN:\n','\nCARRIER_END')},
            'factors':{'query_index':q,'codebook_index':bi,'timing':timing,'distance':distance,'donor':donor},'oracle':{'values':values,'query':x,'original':old,'correct':values[x],'codebook':book}}
def configure(j,c):
    enc=j.enc(c);r=enc['regions'];assert len(r['carrier'])==len(r['pre_anchor'])==1
    r['row1_token']=[max(r['row1'])];end=max(r['carrier']);r['B_suffix']=list(range(end+1,enc['tokens']))
    for name,region in [('main','carrier'),('pre','pre_anchor'),('row1','row1_token'),('null',None)]:r['B_block_'+name]=[i for i in range(end+1) if region is None or i not in r[region]]
    return enc

def graph_qualification(n=9,depth=4):
    """CPU reachability: all prefix facts reach suffix only through allowed role."""
    facts={2,3};carrier=6;suffix=set(range(7,n));prefix=set(range(7));out={}
    for allowed in ({carrier},set(),{0},{3}):
        reach=set(facts)-allowed
        for _ in range(depth):
            nxt=set(reach)
            for q in range(n):
                for k in range(q+1):
                    if q in suffix and k in prefix and k not in allowed:continue
                    if k in reach and q not in allowed:nxt.add(q)
            reach=nxt-allowed
        out[str(sorted(allowed))]=not bool(reach & suffix)
    assert all(out.values());return out

def edges(role):return [{'queries':'B_suffix','keys':'B_block_'+role}]
def patches(h,region,mode='patch'):
    return [{'key':f'resid:{l}','scope':region,'source_scope':region,'mode':mode,**({'source':h} if mode=='patch' else {})} for l in range(47)]
def prepared_cases(m,cfg):
    ww=m['splits']['P0'];a=material(ww[0]);b=material(ww[1]);
    for i,c in enumerate((a,b,a,a)):yield {**c,'id':c['id']+'-instrument-'+str(i)} # CPU count only; real journal tags phase/arm
    for w in ww:
        for q in (0,1):
            for distance in ('short','long'):yield material(w,q,0,'late',distance)
        for bi in (0,1):yield code_case(w,bi,'B')
    for w in ww:
        for timing in ('late','early'):
            for distance in ('short','long'):
                for q in (0,1):
                    for d in ('same','other'):yield material(w,q,0,timing,distance,d)
                    for i in range(10):
                        c=material(w,q,0,timing,distance);yield {**c,'id':c['id']+f'-arm{i}'}

def run(j,worlds,cfg):
    graph=graph_qualification();save_json(j.p.run.path/'CPU_graph_qualification.json',graph)
    # Instrument-only worlds are disjoint from the 12 scientific worlds.
    from .round7_lookup_worlds import make
    w=make(cfg['corpus_seed']+91000000,'B-instrument');a=material(w);b=material(w,donor='same');configure(j,a);configure(j,b)
    try:
        n0,_,_=j.call(a,arm='null_graph_A',edges=edges('null'),phase='P0_instrument')
        n1,_,_=j.call(b,arm='null_graph_B',edges=edges('null'),phase='P0_instrument')
        clean,_,_=j.call(a,arm='instrument_clean',phase='P0_instrument')
        ident,_,_=j.call(a,arm='instrument_identity',specs=patches({},'carrier','identity'),phase='P0_instrument')
        floor=max(abs(x-y) for x,y in zip(n0['raw_head_logits'],n1['raw_head_logits']))
        parity=max(abs(x-y) for x,y in zip(clean['raw_head_logits'],ident['raw_head_logits']))
        assert floor<=cfg['seal_parity_tolerance'] and parity<=cfg['seal_parity_tolerance'],(floor,parity)
        assert len(n0['edge_delivery'])==48 and any(e['removed_previously_allowed_edges'] for e in n0['edge_delivery'])
    except (RuntimeError,AssertionError) as exc:
        return {'gate':'STOPPED_AT_ATTENTION_EDGE_QUALIFICATION','classification':'Inconclusive','scientific_status':'NOT_IDENTIFIABLE','instrument_error':str(exc),'P1':'NOT_EVALUATED','P2':'NOT_EVALUATED'}
    qualification={'null_graph_raw_head_max_difference':floor,'identity_max_difference':parity,'CPU_reachability':graph,'status':'PASS'}
    save_json(j.p.run.path/'seal_qualification.json',qualification)
    qual=defaultdict(list)
    for w in worlds:
        for q in (0,1):
            for distance in ('short','long'):
                c=material(w,q,0,'late',distance);configure(j,c);row,_,_=j.call(c,arm='clean_lookup_anchor',phase='P0_quality');qual[distance].append(row)
        for bi in (0,1):row,_,_=j.call(code_case(w,bi,'B'),arm='code_anchor',phase='P0_quality');qual['code'].append(row)
    readiness=encode_quality(qual);eligible=all(v['accuracy']>=(cfg['code_accuracy_min'] if k=='code' else cfg['anchor_accuracy_min']) for k,v in readiness.items())
    print(json.dumps({'event':'R7_B_quality','readiness':readiness,'eligible':eligible,'calls':j.p.calls}),flush=True)
    if not eligible:return {'gate':'STOPPED_AT_LOOKUP_OR_CODE_ANCHOR','classification':'Inconclusive','scientific_status':'NOT_IDENTIFIABLE','readiness':readiness,'qualification':qualification}
    episodes=[];keys=[f'resid:{l}' for l in range(47)];regions=('carrier','pre_anchor','row1_token')
    for w in worlds:
        scores=defaultdict(list);accuracies=defaultdict(list)
        for timing in ('late','early'):
            for distance in ('short','long'):
                for q in (0,1):
                    c=material(w,q,0,timing,distance);configure(j,c);donors={}
                    for d in ('same','other'):
                        dc=material(w,q,0,timing,distance,d);configure(j,dc)
                        rr,hh,_=j.call(dc,arm='donor_'+d,capture=keys,regions=regions);donors[d]=(dc,hh.cache)
                        torch.save(hh.cache,j.p.run.path/(dc['id']+'_captured.pt'))
                    same=code(c['oracle']['codebook'],donors['same'][0]['oracle']['correct']);base=c['expected'];by={}
                    arms=[('clean',None,None,None),('seal','main',None,None),('null','null',None,None),('identity','main','carrier','identity'),
                          ('main_same','main','carrier','same'),('main_other','main','carrier','other'),('pre_same','pre','pre_anchor','same'),('pre_other','pre','pre_anchor','other'),('row1_same','row1','row1_token','same'),('row1_other','row1','row1_token','other')]
                    for name,role,region,d in arms:
                        specs=() if d is None else patches({},region,'identity') if d=='identity' else patches(donors[d][1],region)
                        rr,_,_=j.call(c,arm=name,specs=specs,edges=edges(role) if role else ())
                        by[name]=pair_score(rr,same,base);accuracies[name].append(bool(rr['correct']))
                    if max(abs(x-y) for x,y in zip(j.rows[-9]['raw_head_logits'],j.rows[-7]['raw_head_logits']))>cfg['seal_parity_tolerance']:raise RuntimeError('Sealed identity parity failed')
                    selectivity=by['main_same']-by['main_other'];ctrl=.5*((by['pre_same']-by['pre_other'])+(by['row1_same']-by['row1_other']))
                    scores[timing+'_'+distance].append(selectivity);scores['control_excess'].append(selectivity-ctrl)
        episodes.append({'id':w['id'],**{k:float(np.mean(v)) for k,v in scores.items()},'accuracy':{k:float(np.mean(v)) for k,v in accuracies.items()}})
        print(json.dumps({'event':'R7_B_episode','id':w['id'],'calls':j.p.calls}),flush=True)
    stats={k:bootstrap([e[k] for e in episodes],cfg) for k in ('late_short','late_long','early_short','early_long','control_excess')}
    late_good=all(stats[k]['ci95'][0]>cfg['practical_margin_nats'] for k in ('late_short','late_long','control_excess'))
    negative=any(stats[k]['ci95'][1]<cfg['practical_margin_nats'] for k in ('late_short','late_long'))
    return {'gate':'P0_COMPLETED','classification':'Positive' if late_good else 'Negative' if negative else 'Inconclusive','scientific_status':'EXPLORATORY_MODIFIED_GRAPH_ONLY','qualification':qualification,'readiness':readiness,'statistics':stats,'episodes':episodes,'scope':'all-layer token trajectory replacement demonstrates conditional transport, not compact state or natural necessity','P1':'NOT_EVALUATED','P2':'NOT_EVALUATED'}
