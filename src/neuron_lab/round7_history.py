"""R7-C: obsolete history influence with exactly matched final tables."""
from __future__ import annotations
from collections import defaultdict
import json
import numpy as np
from .round7_lookup_worlds import manifest,code,code_case,Z,encode_quality
from .round6_engine import pair_score,bootstrap

def build_manifest(cfg):return manifest(cfg)
def material(w,q=0,bi=0,task='compose',history='none',ordering='before'):
    x=w['queries'][q];y=w['f'][x];answer=w['g'][y];alts=[z for z in Z if z!=answer];old1,old2=alts[:2];book=w['books'][bi]
    hist='';key=y if history.startswith('related') else next(k for k in w['g_order'] if k!=y)
    if history!='none':
        stale=old1 if history.endswith('1') else old2
        hist=f'ARCHIVED_HISTORY_BEGIN:\nAn old, CANCELLED record stated G({key}) = {stale}. This record is obsolete and must be ignored.\nARCHIVED_HISTORY_END\n'
    final='FINAL_BEGIN:\nThese FINAL tables are authoritative. All archived records are cancelled.\nF:\n'+'\n'.join(f'{k} -> {w["f"][k]}' for k in w['order'])+'\nG:\n'+'\n'.join(f'{k} -> {w["g"][k]}' for k in w['g_order'])+'\nFINAL_END'
    facts=(hist+final) if ordering=='before' else (final+'\n'+hist)
    cue='Use only the FINAL tables. '+('Look up the person in F, then that color in G.' if task=='compose' else 'Look up the color directly in G.')
    query=x if task=='compose' else y
    text='All mappings are arbitrary. Read them literally.\nCUE_BEGIN:\n'+cue+'\nFACTS_BEGIN:\n'+facts+'\nEND_FACTS\nNEW_QUERY:\nQuery: '+query+'. Codebook: '+'; '.join(f'{a}={b}' for a,b in book.items())+'. Reply only A/B/C/D.'
    return {'id':f'{w["id"]}-q{q}-b{bi}-{task}-{history}-{ordering}','episode_id':w['id'],'study':'C','text':text,'expected':code(book,answer),
            'factors':{'query_index':q,'codebook_index':bi,'task':task,'history':history,'ordering':ordering},
            'oracle':{'correct':answer,'old1':old1,'old2':old2,'history_key':key,'query_key':y,'codebook':book,'f':w['f'],'g':w['g'],'final_text':final}}
def prepared_cases(m,cfg):
    for w in m['splits']['P0']:
        for q in (0,1):
            for bi in (0,1):
                for task in ('direct','compose'):yield material(w,q,bi,task)
        for bi in (0,1):yield code_case(w,bi,'C')
    for w in m['splits']['P0']:
        for q in (0,1):
            for bi in (0,1):
                for task in ('direct','compose'):
                    for ordering in ('before','after'):
                        for history in ('related1','related2','unrelated1','unrelated2'):yield material(w,q,bi,task,history,ordering)
def run(j,worlds,cfg):
    quality=defaultdict(list)
    for w in worlds:
        for q in (0,1):
            for bi in (0,1):
                for task in ('direct','compose'):
                    rr,_,_=j.call(material(w,q,bi,task),arm='final_only_'+task,phase='P0_quality');quality[task].append(rr)
        for bi in (0,1):rr,_,_=j.call(code_case(w,bi,'C'),arm='code_anchor',phase='P0_quality');quality['code'].append(rr)
    readiness=encode_quality(quality);eligible=all(v['accuracy']>=(cfg['code_accuracy_min'] if k=='code' else cfg['anchor_accuracy_min']) for k,v in readiness.items())
    print(json.dumps({'event':'R7_C_quality','readiness':readiness,'eligible':eligible,'calls':j.p.calls}),flush=True)
    if not eligible:return {'gate':'STOPPED_AT_FINAL_STATE_READINESS','classification':'Inconclusive','scientific_status':'NOT_IDENTIFIABLE','readiness':readiness}
    episodes=[]
    for w in worlds:
        vals=defaultdict(list)
        for q in (0,1):
            for bi in (0,1):
                effects={}
                for task in ('direct','compose'):
                    for ordering in ('before','after'):
                        by={}
                        for history in ('related1','related2','unrelated1','unrelated2'):
                            c=material(w,q,bi,task,history,ordering);rr,_,_=j.call(c,arm=history+'_'+ordering)
                            o=c['oracle'];by[history]=pair_score(rr,code(o['codebook'],o['old1']),code(o['codebook'],o['old2']))
                        effects[task,ordering]=by['related1']-by['related2']-by['unrelated1']+by['unrelated2']
                        vals[task+'_'+ordering].append(effects[task,ordering])
                for ordering in ('before','after'):vals['interaction_'+ordering].append(effects['compose',ordering]-effects['direct',ordering])
        episodes.append({'id':w['id'],**{k:float(np.mean(v)) for k,v in vals.items()}})
        print(json.dumps({'event':'R7_C_episode','id':w['id'],'calls':j.p.calls}),flush=True)
    stats={k:bootstrap([e[k] for e in episodes],cfg) for k in episodes[0] if k!='id'}
    pos=all(stats[k]['ci95'][0]>cfg['practical_margin_nats'] for k in ('compose_before','interaction_before','compose_after','interaction_after'))
    neg=stats['interaction_before']['ci95'][1]<cfg['practical_margin_nats'] or stats['interaction_after']['ci95'][1]<cfg['practical_margin_nats']
    return {'gate':'P0_COMPLETED','classification':'Positive' if pos else 'Negative' if neg else 'Inconclusive','scientific_status':'EXPLORATORY_HISTORY_TREATMENT_ONLY','readiness':readiness,'statistics':stats,'episodes':episodes,'scope':'obsolete-history treatment at matched final text, not localization or a novel editing benchmark','P1':'NOT_EVALUATED','P2':'NOT_EVALUATED'}
