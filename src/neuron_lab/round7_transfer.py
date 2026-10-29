"""R7-E: prospective MLP susceptibility prediction of unseen intervention families."""
from __future__ import annotations
from collections import defaultdict
import hashlib,json
import numpy as np
import torch
import torch.nn.functional as F
from .round7_lookup_worlds import manifest,code,code_case,Z,encode_quality
from .round6_engine import margin,bootstrap
from .runtime import save_json

FAMILIES=('native_mlp','native_attn','resid');SEEDS=(8101,8102,8103,8104)
def build_manifest(cfg):return manifest(cfg)
def material(w,task='lookup',bi=0):
    q=w['seed']%2;book=w['books'][bi];vals=[w['g'][k] for k in ('red','blue','green','gold')]
    if task=='lookup':
        key=('red','blue')[q];facts='Mapping table:\n'+'\n'.join(f'{k} -> {w["g"][k]}' for k in w['g_order']);cue='Look up the named color in the table.';query=key;answer=w['g'][key]
    else:
        facts='Sequence: '+'; '.join(f'{i+1}: {v}' for i,v in enumerate(vals))+'.';cue='Return the value at the specified position; positions start at 1.';query=f'Position {q+2}';answer=vals[q+1]
    text='Use the explicitly given data.\nCUE_BEGIN:\n'+cue+'\nFACTS_BEGIN:\n'+facts+'\nEND_FACTS\nNEW_QUERY:\n'+query+'. Codebook: '+'; '.join(f'{a}={b}' for a,b in book.items())+'. Reply with only A/B/C/D.'
    return {'id':f'{w["id"]}-{task}-b{bi}','episode_id':w['id'],'study':'E','text':text,'expected':code(book,answer),'factors':{'task':task,'codebook_index':bi},'oracle':{'correct':answer,'codebook':book}}
def prepared_cases(m,cfg):
    # All 13 arms use identical input IDs but differ in an explicitly journalled intervention.
    for split in ('calibration','P0'):
        for w in m['splits'][split]:
            for task in ('lookup','position'):
                for bi in (0,1):
                    for arm in range(13):
                        c=material(w,task,bi);yield {**c,'id':c['id']+f'-prepared{arm}'}
    for w in m['splits']['P0']:
        for bi in (0,1):yield code_case(w,bi,'E')
def controls(inv,cfg):
    out={}
    for family in FAMILIES:
        for seed in SEEDS:
            rng=np.random.default_rng(seed);key=f'{family}:23'
            if family=='resid':
                width=inv.layers[23].hidden_size if hasattr(inv.layers[23],'hidden_size') else inv.projection('native_mlp:23').out_features
                u=rng.normal(size=width);u/=np.linalg.norm(u);out[family,seed]={'key':key,'unit':u.astype(np.float32)}
            else:
                width=inv.projection(key).in_features;coords=rng.choice(width,min(cfg['native_coordinate_count'],width),replace=False).tolist()
                out[family,seed]={'key':key,'coordinates':coords}
    return out

def intervention(panel,family,seed,clean,cfg):
    c=panel[family,seed]
    if family!='resid':return [{'key':c['key'],'scope':'last','mode':'zero','coordinates':c['coordinates'],'attenuation':cfg['native_attenuation']}]
    key=c['key'];x=clean[(key,'last')];u=torch.from_numpy(c['unit']).view(1,1,-1)
    target=x+cfg['residual_relative_dose']*x.norm()*u
    return [{'key':key,'scope':'last','source_scope':'last','source':{(key,'last'):target}}]

@torch.inference_mode()
def readout_probe(j,c,rr,h,cfg):
    model=j.p.runtime.model;norm=model.model.language_model.norm;head=model.get_output_embeddings();ids=j.p.ids
    x=h[('resid:47','last')].to(head.weight.device);weight=head.weight[ids];soft=model.config.get_text_config().final_logit_softcapping
    def replay(x):
        j.p.local_evaluations+=1
        if j.p.local_evaluations>cfg['max_local_module_evaluations']:raise RuntimeError('LOCAL_READOUT_BUDGET')
        raw=F.linear(norm(x),weight,None).view(-1,4).float();return raw,torch.tanh(raw/soft)*soft
    raw,post=replay(x);parity=float((raw[0]-torch.tensor(rr['raw_head_logits'],device=x.device)).abs().max())
    if parity>cfg['readout_parity_tolerance']:raise RuntimeError(f'READOUT_PARITY {parity}')
    i='ABCD'.index(c['expected']);others=[k for k in range(4) if k!=i]
    def local_margin(p):return float(p[0,i]-torch.logsumexp(p[0,others],0))
    base=local_margin(post);effects=[]
    for seed in cfg['readout_probe_seeds']:
        rng=np.random.default_rng(seed);u=torch.tensor(rng.normal(size=x.shape[-1]),device=x.device,dtype=x.dtype);u/=u.norm()
        _,y=replay(x+cfg['residual_relative_dose']*x.norm()*u);effects.append(local_margin(y)-base)
    return {'parity':parity,'readout_gain':float(np.mean(np.abs(effects))),'readout_effects':effects}

def feature(task,rr,h,readout):
    return [float(task=='position'),margin(rr,rr['expected']),float(rr['tokens']),readout['readout_gain'],*[float(np.log(h[(k,'last')].norm().item()+1e-12)) for k in ('native_mlp:23','native_attn:23','resid:23')]]
def fit(x,y,lam):
    x=np.asarray(x,float);y=np.asarray(y,float);mean=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1.;z=(x-mean)/sd;z=np.c_[np.ones(len(z)),z]
    pen=np.eye(z.shape[1])*lam;pen[0,0]=0;beta=np.linalg.solve(z.T@z+pen,z.T@y)
    return {'mean':mean.tolist(),'sd':sd.tolist(),'beta':beta.tolist()}
def predict(f,x):return float(np.r_[1.,(np.asarray(x)-f['mean'])/f['sd']]@np.asarray(f['beta']))

def analyze(cases,cfg):
    # Predictions were emitted before target forwards; no refitting on P0 labels.
    by=defaultdict(list)
    denom={target:float(np.var([c['y'][target] for c in cases])) for target in ('native_attn','resid')}
    if any(v<=cfg['minimum_target_variance'] for v in denom.values()):return {'classification':'Inconclusive','gate':'STOPPED_AT_PREDICTION_TARGET_VARIANCE','variance':denom}
    for c in cases:
        improvements=[]
        for target in ('native_attn','resid'):
            y=c['y'][target];a=c['predictions'][target];d=((y-a['baseline'])**2-(y-a['augmented'])**2)/denom[target]
            by[c['world']].append({'task':c['task'],'target':target,'gain':float(d)})
    episodes=[]
    for w,rows in by.items():episodes.append({'id':w,'primary':float(np.mean([r['gain'] for r in rows])),**{task:float(np.mean([r['gain'] for r in rows if r['task']==task])) for task in ('lookup','position')}})
    stats={k:bootstrap([e[k] for e in episodes],cfg) for k in ('primary','lookup','position')}
    # Input matching negative control: frozen calibration coefficients, within-task+codebook world shuffle only.
    rng=np.random.default_rng(cfg['shuffle_seed']);null=[]
    for _ in range(cfg['shuffle_resamples']):
        shuffled={}
        for task in ('lookup','position'):
            for bi in (0,1):
                ix=[i for i,c in enumerate(cases) if c['task']==task and c['book']==bi];perm=rng.permutation(ix)
                shuffled.update({i:cases[int(k)]['source'] for i,k in zip(ix,perm)})
        imp=[]
        for i,c in enumerate(cases):
            for target in ('native_attn','resid'):
                p=predict(c['fitted_models'][target]['augmented'],c['features']+[shuffled[i]]);y=c['y'][target];b=c['predictions'][target]['baseline']
                imp.append(((y-b)**2-(y-p)**2)/denom[target])
        null.append(float(np.mean(imp)))
    q95=float(np.quantile(null,.95));positive=stats['primary']['ci95'][0]>cfg['practical_delta_R2'] and all(stats[k]['point']>0 for k in ('lookup','position')) and stats['primary']['point']>q95
    negative=stats['primary']['ci95'][1]<cfg['practical_delta_R2']
    return {'classification':'Positive' if positive else 'Negative' if negative else 'Inconclusive','gate':'P0_COMPLETED','statistics':stats,'variance':denom,'source_shuffle_95th_gain':q95,'episodes':episodes,'scope':'prospective input-level prediction only; no inference of a common causal bottleneck'}

def run(j,worlds,cfg):
    import pathlib
    full=json.loads((j.p.run.path/'manifest.json').read_text());cal=full['splits']['calibration'];panel=controls(j.p.inv,cfg)
    save_json(j.p.run.path/'frozen_random_controls.json',{f'{f}_{s}':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()} for (f,s),c in panel.items()})
    keys=['native_mlp:23','native_attn:23','resid:23','resid:47'];training=[];calqual=defaultdict(list);effects=defaultdict(list)
    for w in cal:
        for task in ('lookup','position'):
            for bi in (0,1):
                c=material(w,task,bi);rr,hh,_=j.call(c,arm='calibration_clean',capture=keys,regions=('last',),phase='independent_calibration');calqual[task].append(rr)
                try:rp=readout_probe(j,c,rr,hh.cache,cfg)
                except RuntimeError as exc:return {'gate':'STOPPED_AT_READOUT_QUALIFICATION','classification':'Inconclusive','scientific_status':'NOT_IDENTIFIABLE','error':str(exc)}
                baseline=margin(rr,c['expected']);ys={}
                for family in FAMILIES:
                    deltas=[]
                    for seed in SEEDS:
                        out,_,_=j.call(c,arm=f'cal_{family}_{seed}',specs=intervention(panel,family,seed,hh.cache,cfg),phase='independent_calibration')
                        deltas.append(margin(out,c['expected'])-baseline)
                    ys[family]=float(np.mean(np.abs(deltas)));effects[family].extend(deltas)
                training.append({'features':feature(task,rr,hh.cache,rp),'source':ys['native_mlp'],'y':ys,'world':w['id']})
    readiness=encode_quality(calqual);envelope={k:float(np.mean(np.abs(v))) for k,v in effects.items()}
    print(json.dumps({'event':'R7_E_calibration','accuracy':readiness,'envelope':envelope,'calls':j.p.calls}),flush=True)
    if any(v['accuracy']<cfg['anchor_accuracy_min'] for v in readiness.values()) or any(not cfg['calibration_effect_min']<=v<=cfg['calibration_effect_max'] for v in envelope.values()):return {'gate':'STOPPED_AT_INDEPENDENT_CALIBRATION','classification':'Inconclusive','scientific_status':'NOT_IDENTIFIABLE','calibration_readiness':readiness,'calibration_mean_absolute_effect':envelope}
    models={}
    for target in ('native_attn','resid'):
        ys=[r['y'][target] for r in training];models[target]={'baseline':fit([r['features'] for r in training],ys,cfg['ridge_lambda']),'augmented':fit([r['features']+[r['source']] for r in training],ys,cfg['ridge_lambda'])}
    save_json(j.p.run.path/'frozen_calibration_models.json',{'models':models,'training':training,'P0_target_outputs_seen':0})
    clean={};quality=defaultdict(list)
    for w in worlds:
        for task in ('lookup','position'):
            for bi in (0,1):
                c=material(w,task,bi);rr,hh,_=j.call(c,arm='P0_clean',capture=keys,regions=('last',),phase='P0_quality');quality[task].append(rr);clean[c['id']]=(c,rr,hh.cache)
                torch.save(hh.cache,j.p.run.path/(c['id']+'_clean_capture.pt'))
        for bi in (0,1):rr,_,_=j.call(code_case(w,bi,'E'),arm='code_anchor',phase='P0_quality');quality['code'].append(rr)
    readiness=encode_quality(quality);eligible=all(v['accuracy']>=(cfg['code_accuracy_min'] if k=='code' else cfg['anchor_accuracy_min']) for k,v in readiness.items())
    print(json.dumps({'event':'R7_E_quality','readiness':readiness,'eligible':eligible,'calls':j.p.calls}),flush=True)
    if not eligible:return {'gate':'STOPPED_AT_TASK_OR_CODE_READINESS','classification':'Inconclusive','scientific_status':'NOT_IDENTIFIABLE','readiness':readiness,'calibration_mean_absolute_effect':envelope}
    cases=[];predictions=[]
    for c,rr,h in clean.values():
        task=c['factors']['task'];base=margin(rr,c['expected']);rp=readout_probe(j,c,rr,h,cfg);x=feature(task,rr,h,rp);d=[]
        for seed in SEEDS:
            out,_,_=j.call(c,arm=f'P0_source_native_mlp_{seed}',specs=intervention(panel,'native_mlp',seed,h,cfg));d.append(margin(out,c['expected'])-base)
        source=float(np.mean(np.abs(d)));pred={target:{'baseline':predict(f['baseline'],x),'augmented':predict(f['augmented'],x+[source])} for target,f in models.items()}
        predictions.append({'id':c['id'],'completed_calls_before_target':j.p.completed,'target_outputs_for_this_case_seen':0,'features':x,'source':source,'predictions':pred})
        save_json(j.p.run.path/'prospective_predictions.json',predictions)
        ys={'native_mlp':source}
        for target in ('native_attn','resid'):
            dd=[]
            for seed in SEEDS:
                out,_,_=j.call(c,arm=f'P0_target_{target}_{seed}',specs=intervention(panel,target,seed,h,cfg));dd.append(margin(out,c['expected'])-base)
            ys[target]=float(np.mean(np.abs(dd)))
        cases.append({'id':c['id'],'world':c['episode_id'],'task':task,'book':c['factors']['codebook_index'],'features':x,'source':source,'predictions':pred,'y':ys,'readout':rp,'fitted_models':models})
        print(json.dumps({'event':'R7_E_case','id':c['id'],'calls':j.p.calls}),flush=True)
    save_json(j.p.run.path/'prediction_cases.json',cases);analysis=analyze(cases,cfg)
    return {**analysis,'scientific_status':'EXPLORATORY_PROSPECTIVE_PREDICTION','readiness':readiness,'calibration_mean_absolute_effect':envelope,'calibration_worlds':len(cal),'P0_worlds':len(worlds),'P1':'NOT_EVALUATED','P2':'NOT_EVALUATED'}
