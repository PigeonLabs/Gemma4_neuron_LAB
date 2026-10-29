"""R6-B discovery: qualify a selective persistent lesion before backup claims.

Only new B discovery worlds choose L/R; whole-output masks are labelled as
components, never as MLP neurons. No calibration/confirmation is read here.
"""
from __future__ import annotations
from collections import defaultdict
import json
import numpy as np
from .runtime import save_json
from .round6_explicit_corpus import code_for
from .round6_explicit_discovery import spec
from .round6_engine import margin,pair_score,native_candidates,random_panel


WITNESSES=(('direct_only','direct_only'),('composition_only','composition_only'),('code_only',None),('agreement',None))
CONTEXTS=('agreement','conflict1','conflict2')


def primary_specs(mask):
    if 'coordinates' in mask:return [spec(mask,'zero',scope='all')]
    return [{'key':k,'scope':'all','mode':'zero'} for k in mask['keys']]


def primary_metrics(pairs,cfg):
    grouped=defaultdict(list)
    for name,base,row in pairs:grouped[name].append((base,row))
    loss={k:float(np.mean([int(b['correct'])-int(r['correct']) for b,r in v])) for k,v in grouped.items()}
    code_abs=float(np.mean([abs(b['correct_margin']-r['correct_margin']) for b,r in grouped['code_only']]))
    eligible=(loss['direct_only']>=.20 and loss['composition_only']<=.10 and loss['code_only']<=cfg['code_accuracy_damage_max'] and
              code_abs<=cfg['code_mean_absolute_margin_change_max_nats'] and loss['agreement']<=.10)
    score=loss['direct_only']-loss['composition_only']-abs(loss['code_only'])-loss['agreement']-max(0.,code_abs-.20)
    return {'accuracy_loss':loss,'code_mean_absolute_margin_change':code_abs,'eligible':bool(eligible),'selection_score':float(score)}


def interaction(base,lesioned,cfg):
    # D_R is loss of correct-answer 4-way log odds; the authority stays h.
    d={ctx:{p:base[ctx,p]['correct_margin']-lesioned[ctx,p]['correct_margin'] for p in (0,1)} for ctx in CONTEXTS}
    return float((d['agreement'][1]-d['agreement'][0])-np.mean([d[c][1]-d[c][0] for c in CONTEXTS[1:]]))


def discover_backup(j,episodes):
    cfg=j.cfg;scfg=cfg['studies']['B'];start=j.p.calls
    keys=[f'native_{kind}:{l}' for kind in ('mlp','attn') for l in range(48)]
    base=[];reduced=defaultdict(list);healthy=defaultdict(list)
    for ep in episodes[:8]:
        for c in [c for c in ep['cases'] if c['codebook_index']==0]:
            means={}
            for name,witness in WITNESSES:
                kwargs={'context':'agreement','witness':witness}
                if name=='code_only':kwargs['explicit_z']=c['original']
                capture=keys if name in ('direct_only','composition_only') else ()
                row,h,_=j.call(ep,c,arm='B_readiness_'+name,capture=capture,**kwargs)
                base.append((ep,c,name,kwargs,row));healthy[name].append(row['correct'])
                if capture:means[name]={k:v[0].mean(0).numpy() for (k,r),v in h.cache.items()}
                del h
            for key in keys:reduced[key].append(means['direct_only'][key]-means['composition_only'][key])
            for context in CONTEXTS[1:]:
                row,_,_=j.call(ep,c,arm='B_readiness_'+context,context=context)
                healthy[context].append(row['correct'])
        print(json.dumps({'event':'B_readiness_episode','episode':ep['id'],'calls':j.p.calls}),flush=True)
    ready={'accuracy':{k:float(np.mean(v)) for k,v in healthy.items()},'counts':{k:len(v) for k,v in healthy.items()}}
    ready['passed']=all(v>=(cfg['code_accuracy_min'] if k=='code_only' else cfg['healthy_accuracy_min']) for k,v in ready['accuracy'].items())
    save_json(j.p.run.path/'task_readiness.json',ready)
    print(json.dumps({'event':'R6_task_readiness','study':'B',**ready,'calls':j.p.calls}),flush=True)
    if not ready['passed']:return {'gate':'STOPPED_AT_TASK_READINESS','readiness':ready,'calibration':'NOT_EVALUATED','confirmation':'NOT_EVALUATED'}
    masks=[];evaluated=[]
    def evaluate(mask):
        pairs=[]
        for ep,c,name,kwargs,row in base:
            changed,_,_=j.call(ep,c,arm='B_primary_'+mask['id']+'_'+name,specs=primary_specs(mask),**kwargs)
            pairs.append((name,row,changed))
        result={'mask':mask,**primary_metrics(pairs,cfg)};evaluated.append(result)
        save_json(j.p.run.path/'primary_candidates.json',evaluated)
        print(json.dumps({'event':'B_primary_mask','id':mask['id'],'metrics':{k:v for k,v in result.items() if k!='mask'},'calls':j.p.calls}),flush=True)
        return result
    for bi,layers in enumerate(cfg['coarse_layer_bins']):
        for kind in ('mlp','attn'):
            mask={'id':f'{kind}_bin{bi}_all','kind':kind,'layers':layers,'keys':[f'{kind}:{l}' for l in layers],
                  'unit':'whole_MLP_component_outputs' if kind=='mlp' else 'whole_attention_component_outputs','scope':'all'}
            masks.append(mask);evaluate(mask)
    best=sorted(evaluated,key=lambda r:(-r['selection_score'],r['mask']['id']))[0]['mask']
    fine=[]
    for l in best['layers']:
        mask={'id':f"{best['kind']}_L{l}_all",'kind':best['kind'],'layers':[l],'keys':[f"{best['kind']}:{l}"],
              'unit':best['unit'],'scope':'all'}
        fine.append(evaluate(mask))
    layer=sorted(fine,key=lambda r:(-r['selection_score'],r['mask']['id']))[0]['mask']['layers'][0]
    kind=best['kind'];key=f'native_{kind}:{layer}';delta=np.asarray(reduced[key])
    for group in native_candidates(j.p.inv,kind,layer,np.abs(delta).mean(0),delta.mean(0)):
        group={**group,'scope':'all'};evaluate(group)
    qualified=[r for r in evaluated if r['eligible']]
    if not qualified:
        assert j.p.calls-start<=scfg['discovery_cap']
        return {'gate':scfg['no_primary'],'readiness':ready,'primary_candidates':evaluated,'primary_candidates_tested':len(evaluated),
                'primary_lesion':'NOT_QUALIFIED','backup':'NOT_EVALUATED','calibration':'NOT_EVALUATED','confirmation':'NOT_EVALUATED'}
    chosen=sorted(qualified,key=lambda r:(-r['selection_score'],r['mask']['id']))[0]
    L=chosen['mask'];lspec=primary_specs(L)
    save_json(j.p.run.path/'primary_selection.json',chosen)
    del base,reduced
    sites=[f'native_{kind}:{l}' for l in scfg['backup_search_layers'] for kind in ('mlp','attn')]
    cases=[];deltas=defaultdict(list)
    for ep in episodes[:4]:
        c=ep['cases'][0];bl={};cache={}
        for context in CONTEXTS:
            for phys in (0,1):
                row,h,_=j.call(ep,c,arm=f'B_backup_source_{context}_{phys}',context=context,specs=lspec if phys else (),capture=sites)
                bl[context,phys]=row;cache[context,phys]=h.cache
            for key in sites:deltas[key].append((cache[context,1][key,'query']-cache[context,0][key,'query'])[0].numpy())
        cases.append((ep,c,bl,cache))
    pool=[]
    for l in scfg['backup_search_layers']:
        for kind in ('mlp','attn'):
            key=f'native_{kind}:{l}'
            if 'keys' in L and f'{kind}:{l}' in L['keys']:continue
            absolute=np.mean([np.abs(x).mean(0) for x in deltas[key]],0)
            coherent=np.mean([x.mean(0) for x in deltas[key]],0)
            if L.get('key')==key:
                absolute[L['coordinates']]=float('-inf');coherent[L['coordinates']]=0
            ranked=native_candidates(j.p.inv,kind,l,absolute,coherent)
            wanted=[g for g in ranked if g['id'].endswith('_absolute')]
            if kind=='attn':wanted=wanted[:1]
            for g in wanted:
                if L.get('key')==key:assert not set(g['coordinates'])&set(L['coordinates'])
                pool.append(g)
    backup=[]
    def test_group(g,full=True):
        values=[]
        for ep,c,bl,cache in cases:
            lesions={}
            for context in CONTEXTS:
                for phys in (0,1):
                    row,_,_=j.call(ep,c,arm=f"B_R_{g['id']}_{context}_{phys}",context=context,
                                  specs=(lspec if phys else [])+[spec(g,'zero')])
                    lesions[context,phys]=row
            values.append(interaction(bl,lesions,cfg))
        return {'group':g,'interaction':float(np.mean(values)),'episode_values':values}
    for g in pool:backup.append(test_group(g))
    save_json(j.p.run.path/'backup_candidates.json',backup)
    qualified=[r for r in backup if r['interaction']>=cfg['practical_margin_nats']]
    if not qualified:
        assert j.p.calls-start<=scfg['discovery_cap']
        return {'gate':scfg['no_backup'],'readiness':ready,'primary':chosen,'backup_candidates':backup,
                'calibration':'NOT_EVALUATED','confirmation':'NOT_EVALUATED'}
    selected=sorted(qualified,key=lambda r:(-r['interaction'],r['group']['id']))[0];R=selected['group']
    panel=random_panel(R,j.p.inv,cfg)
    controls=[test_group(g) for g in panel]
    identity=[];guards=[]
    for ep,c,bl,cache in cases:
        # Source is a naturally occurring wrong2 composition state under the same L.
        # Compare both specific wrong labels: arbitrary degradation cannot suffice.
        main,_,_=j.call(ep,c,arm='B_identity_wrong2',context='conflict1',specs=lspec+[spec(R,source=cache['conflict2',1])])
        sham,_,_=j.call(ep,c,arm='B_identity_same',context='conflict1',specs=lspec+[spec(R,source=cache['conflict1',1])])
        code2=code_for(c['book'],c['wrong2']);code1=code_for(c['book'],c['wrong1'])
        identity.append(pair_score(main,code2,code1)-pair_score(sham,code2,code1))
        for phys in (0,1):
            bas,_,_=j.call(ep,c,arm='B_code_guard_base',explicit_z=c['original'],specs=lspec if phys else ())
            for mode in ('zero','patch'):
                gspec=spec(R,mode,cache['conflict2',phys] if mode=='patch' else None)
                row,_,_=j.call(ep,c,arm='B_code_guard_'+mode,explicit_z=c['original'],specs=(lspec if phys else [])+[gspec])
                guards.append({'accuracy_damage':int(bas['correct'])-int(row['correct']),'margin_abs':abs(bas['correct_margin']-row['correct_margin'])})
    theta=selected['interaction']-np.mean([r['interaction'] for r in controls])
    guard={k:float(np.mean([r[k] for r in guards])) for k in ('accuracy_damage','margin_abs')}
    passed=theta>=cfg['practical_margin_nats'] and np.mean(identity)>=cfg['practical_margin_nats'] and guard['accuracy_damage']<=cfg['code_accuracy_damage_max'] and guard['margin_abs']<=cfg['code_mean_absolute_margin_change_max_nats']
    assert j.p.calls-start<=scfg['discovery_cap']
    result={'gate':'DISCOVERY_PASSED_REQUIRES_CALIBRATION' if passed else scfg['no_backup'],'readiness':ready,'primary':chosen,
            'backup_candidates':backup,'group':R,'panel':panel,'random_controls':controls,'net_interaction':float(theta),
            'specific_wrong2_identity_contrast':float(np.mean(identity)),'code_guard':guard,'calibration':'NOT_EVALUATED','confirmation':'NOT_EVALUATED'}
    save_json(j.p.run.path/'discovery_selection.json',result)
    return result
