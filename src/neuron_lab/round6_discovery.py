"""Registered bounded discovery for selector and reusable-intermediate studies."""
from __future__ import annotations
from collections import defaultdict
import json
import numpy as np
import torch
from .runtime import save_json
from .round6_corpus import render,code_for
from .round6_engine import encode,margin,pair_score,native_candidates,random_panel


class Journal:
    def __init__(self,probe,cfg,study):
        self.p=probe;self.cfg=cfg;self.study=study;self.rows=[]
        self.file=(probe.run.path/"rows.jsonl").open("x")
        self.encodings={}

    def enc(self,material):
        text=material["text"]
        if text not in self.encodings:self.encodings[text]=encode(self.p.runtime.tokenizer,text,self.cfg["max_input_tokens"])
        return self.encodings[text]

    def call(self,ep,c,role="recipient",arm="source",context="conflict",witness=None,explicit_z=None,
             specs=(),capture=(),regions=("query",),norm_capture_layer=None,norm_reference=None,full=False,phase="discovery",fact_edit=None):
        material=render(ep,c,role,context,witness,explicit_z,fact_edit)
        enc=self.enc(material)
        row,hooks,logits=self.p.forward(enc,specs,capture,regions,norm_capture_layer,norm_reference,full)
        row={**row,"episode_id":ep["id"],"case_id":c["id"],"phase":phase,"role":role,"arm":arm,
            "context":context,"witness":witness,"expected":material["expected"],"correct":row["prediction"]==material["expected"],
            "explicit_z":explicit_z,"fact_edit":fact_edit,
            "original":c["original"],"target":c.get("target"),"codebook":material["codebook"],
            "text_sha256":enc["text_sha256"],"selector":c["selector"],"codebook_index":c["codebook_index"],"query_index":c["query_index"]}
        row["correct_margin"]=margin(row,material["expected"])
        self.rows.append(row);self.file.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n");self.file.flush()
        return row,hooks,logits

    def close(self):self.file.close()


def spec(group,mode="patch",source=None,scope=None,attenuation=None):
    x={k:group[k] for k in ("key","coordinates") if k in group}
    x.update(mode=mode,scope=scope or group.get("scope","query"))
    if source is not None:x["source"]=source
    if attenuation is not None:x["attenuation"]=attenuation
    elif "attenuation" in group:x["attenuation"]=group["attenuation"]
    return x


def effect(row,c,study):
    target=code_for(c["book"],c["target"])
    if study=="A":return margin(row,target)
    return pair_score(row,target,code_for(c["book"],c["donor_selected"]))


def roles(study):
    return ("donor_opposite","donor_same") if study=="A" else ("intermediate_donor","semantic_sham")


def technical(probe,episodes,cfg):
    records=[];samples=[];p=probe;runtime=p.runtime
    for dtype in (torch.bfloat16,torch.float32):
        if dtype==torch.float32:
            sampled={n:t.flatten()[::max(1,t.numel()//32)][:32].float().cpu() for n,t in runtime.model.named_parameters()}
            runtime.model.float()
            assert all(t.dtype==dtype and torch.equal(sampled[n],t.flatten()[::max(1,t.numel()//32)][:32].cpu()) for n,t in runtime.model.named_parameters())
        p.versions={n:t._version for n,t in runtime.model.named_parameters()}
        assert runtime.model.get_input_embeddings().weight.data_ptr()==runtime.model.get_output_embeddings().weight.data_ptr()
        for ep in episodes:
            c=ep["cases"][0];material=render(ep,c);enc=encode(runtime.tokenizer,material["text"],cfg["max_input_tokens"])
            keys=["native_mlp:15","native_attn:15","mlp:15","attn:15"]
            _,_,bare=p.forward(enc,full=True)
            _,read,readonly=p.forward(enc,capture=keys,regions=("query","cue","facts","all"),norm_capture_layer=15,full=True)
            _,_,identity=p.forward(enc,[{"key":k,"scope":"all","mode":"identity"} for k in keys],full=True)
            g=[{"key":k,"coordinates":[0,2,4],"scope":"query","mode":"zero"} for k in keys[:2]]
            _,qh,_=p.forward(enc,g,full=True)
            ga=[{**s,"scope":"all"} for s in g]
            _,ah,_=p.forward(enc,ga,full=True)
            restored_specs=ga+[{"key":k,"scope":"all","coordinates":[0,2,4],"source":read.cache,"mode":"patch"} for k in keys[:2]]
            _,_,restored=p.forward(enc,restored_specs,full=True)
            _,_,out_identity=p.forward(enc,[{"key":k,"scope":"all","source":read.cache,"mode":"patch"} for k in keys[2:]],full=True)
            _,_,removed=p.forward(enc,full=True)
            checks={"read_only_exact":torch.equal(bare,readonly),"identity_exact":torch.equal(bare,identity),
                "same_forward_restoration_exact":torch.equal(bare,restored),"tuple_and_tensor_output_identity_exact":torch.equal(bare,out_identity),
                "removed_exact":torch.equal(bare,removed),"nonzero_query_delivery":all(x["delta_norm"]>0 for x in qh.delivery),
                "nonzero_all_token_delivery":all(x["delta_norm"]>0 for x in ah.delivery),"tied_head":True,"softcap_replay_exact":True}
            norm_error=None
            if dtype==torch.float32:
                _,_,replay=p.forward(enc,norm_reference=read.denominators,full=True)
                norm_error=float((bare-replay).abs().max().item());checks["RMSNorm_replay_within_1e-5"]=norm_error<=1e-5
            if not all(checks.values()):raise RuntimeError(f"R6 technical gate failed: {checks}")
            records.append({"dtype":str(dtype),"episode":ep["id"],"tokens":enc["tokens"],"checks":checks,"norm_replay_max_logit_error":norm_error})
            p.check_frozen()
        print(json.dumps({"event":"R6_technical_dtype_pass","dtype":str(dtype),"calls":p.calls}),flush=True)
    if p.calls>cfg["technical_call_cap"]:raise RuntimeError("Technical call cap")
    runtime.smoke_passed=True
    return {"status":"PASS","actual_forward_calls":p.calls,"records":records,"parameter_upcast":"all parameters; BF16-stored sampled values exactly preserved"}


def readiness(j,episodes,study):
    scores=defaultdict(list)
    for ep in episodes[:8]:
        if study=="A":
            cases=[c for c in ep["cases"] if c["selector"]==0]
            for c in cases:
                for name,witness,selector in (("conflict_direct",None,0),("conflict_compose",None,1),
                                             ("direct_only","direct_only",0),("composition_only","composition_only",1)):
                    cc={**c,"selector":selector}
                    row,_,_=j.call(ep,cc,arm="readiness_"+name,witness=witness)
                    scores[name].append(row["correct"])
        else:
            for c in ep["cases"]:
                row,_,_=j.call(ep,c,arm="readiness_composition")
                scores["composition"].append(row["correct"])
        for c in [x for x in ep["cases"] if x["selector"]==0] if study=="A" else [x for x in ep["cases"] if x["function"]==0]:
            for z in (c["original"],c["target"]):
                row,_,_=j.call(ep,c,arm="readiness_code_only",explicit_z=z)
                scores["code_only"].append(row["correct"])
    result={k:float(np.mean(v)) for k,v in scores.items()}
    passed=all(v>=(j.cfg["code_accuracy_min"] if k=="code_only" else j.cfg["healthy_accuracy_min"]) for k,v in result.items())
    return {"passed":passed,"accuracy":result,"counts":{k:len(v) for k,v in scores.items()}}


def discovery(j,episodes,study):
    cfg=j.cfg;start=j.p.calls;ready=readiness(j,episodes,study)
    save_json(j.p.run.path/"task_readiness.json",ready)
    print(json.dumps({"event":"R6_task_readiness","study":study,**ready,"calls":j.p.calls}),flush=True)
    if not ready["passed"]:return {"gate":"STOPPED_AT_TASK_READINESS","readiness":ready}
    mains,controls=roles(study)
    regions=("facts","cue","query") if study=="A" else ("facts","query")
    outputkeys=[f"{kind}:{l}" for kind in ("mlp","attn") for l in range(48)]
    coarse=defaultdict(list)
    for ep in episodes[:4]:
        for c in [x for x in ep["cases"] if x["codebook_index"]==0]:
            rr,rh,_=j.call(ep,c,arm="coarse_source_recipient",capture=outputkeys,regions=regions)
            rp,ph,_=j.call(ep,c,role=mains,arm="coarse_source_main",capture=outputkeys,regions=regions)
            rs,sh,_=j.call(ep,c,role=controls,arm="coarse_source_control",capture=outputkeys,regions=regions)
            for bi,layers in enumerate(cfg["coarse_layer_bins"]):
                for kind in ("mlp","attn"):
                    for region in regions:
                        results=[]
                        for label,source in (("main",ph.cache),("control",sh.cache)):
                            specs=[{"key":f"{kind}:{l}","scope":region,"source":source,"mode":"patch"} for l in layers]
                            row,_,_=j.call(ep,c,arm=f"coarse_b{bi}_{kind}_{region}_{label}",specs=specs)
                            results.append(effect(row,c,study))
                        coarse[(bi,kind,region)].append(results[0]-results[1])
            del rh,ph,sh
        print(json.dumps({"event":"coarse_episode_complete","study":study,"episode":ep["id"],"calls":j.p.calls}),flush=True)
    summary=[{"bin":b,"kind":k,"region":r,"score":float(np.mean(v))} for (b,k,r),v in coarse.items()]
    summary.sort(key=lambda r:(-r["score"],r["bin"],r["kind"],r["region"]))
    save_json(j.p.run.path/"coarse_results.json",summary)
    query=[r for r in summary if r["region"]=="query"];winner=query[0]
    if winner["score"]<cfg["practical_margin_nats"]:
        return {"gate":"STOPPED_AT_QUERY_LOCALIZATION_GATE","readiness":ready,"coarse":summary,"winner":winner}
    layers=cfg["coarse_layer_bins"][winner["bin"]];kind=winner["kind"]
    capture=[f"{k}:{l}" for l in layers for k in (kind,f"native_{kind}")]
    fine_sources=[];layer_scores=defaultdict(list)
    for ep in episodes[:4]:
        for c in [x for x in ep["cases"] if x["codebook_index"]==0]:
            r,rh,_=j.call(ep,c,arm="fine_source_recipient",capture=capture,regions=("query",))
            _,ph,_=j.call(ep,c,role=mains,arm="fine_source_main",capture=capture,regions=("query",))
            _,sh,_=j.call(ep,c,role=controls,arm="fine_source_control",capture=capture,regions=("query",))
            fine_sources.append((ep,c,r,rh.cache,ph.cache,sh.cache))
            for l in layers:
                values=[]
                for label,source in (("main",ph.cache),("control",sh.cache)):
                    row,_,_=j.call(ep,c,arm=f"single_layer_{kind}_{l}_{label}",specs=[{"key":f"{kind}:{l}","scope":"query","source":source}])
                    values.append(effect(row,c,study))
                layer_scores[l].append(values[0]-values[1])
    layer=max(layers,key=lambda l:(np.mean(layer_scores[l]),-l));nkey=f"native_{kind}:{layer}"
    delta=[];coherent=[]
    for ep,c,r,rc,pc,sc in fine_sources:
        d=(pc[nkey,"query"]-sc[nkey,"query"])[0].numpy()
        delta.append(np.abs(d).mean(0));coherent.append(d.mean(0)*(1 if study=="C" or c["selector"]==0 else -1))
    groups=native_candidates(j.p.inv,kind,layer,np.mean(delta,0),np.mean(coherent,0))
    groupstats=[]
    for g in groups:
        values=[];choice=[];damage=[]
        for ep,c,r,rc,pc,sc in fine_sources:
            a,_,_=j.call(ep,c,arm=g["id"]+"_main",specs=[spec(g,source=pc)])
            b,_,_=j.call(ep,c,arm=g["id"]+"_control",specs=[spec(g,source=sc)])
            z,_,_=j.call(ep,c,arm=g["id"]+"_zero_query",specs=[spec(g,"zero")])
            j.call(ep,c,arm=g["id"]+"_zero_all_diagnostic",specs=[spec(g,"zero",scope="all")])
            values.append(effect(a,c,study)-effect(b,c,study));target=code_for(c["book"],c["target"])
            choice.append(int(a["prediction"]==target)-int(b["prediction"]==target));damage.append(int(r["correct"])-int(z["correct"]))
        groupstats.append({"group":g,"score":float(np.mean(values)),"choice_increase":float(np.mean(choice)),"necessity_damage":float(np.mean(damage))})
    save_json(j.p.run.path/"native_candidates.json",groupstats)
    eligible=[x for x in groupstats if x["score"]>=cfg["practical_margin_nats"] and x["choice_increase"]>=cfg["practical_behavior_points"] and x["necessity_damage"]>=cfg["practical_behavior_points"]]
    del fine_sources
    if not eligible:
        return {"gate":cfg["studies"][study]["no_qualified_group"],"readiness":ready,"coarse":summary,"native_candidates":groupstats}
    chosen=sorted(eligible,key=lambda x:(-x["score"],x["group"]["id"]))[0];group=chosen["group"]
    panel=random_panel(group,j.p.inv,cfg)
    save_json(j.p.run.path/"discovery_selection.json",{"group":group,"panel":panel,"selection_scope":"first discovery worlds only; no reselection"})
    screen=[]
    for ep in episodes[8:12]:
        for c in [x for x in ep["cases"] if x["codebook_index"]==0]:
            base,_,_=j.call(ep,c,arm="screen_source_recipient")
            _,p,_=j.call(ep,c,role=mains,arm="screen_source_main",capture=[nkey])
            _,s,_=j.call(ep,c,role=controls,arm="screen_source_control",capture=[nkey])
            scores=[];choices=[];damages=[]
            for g in [group,*panel]:
                a,_,_=j.call(ep,c,arm="screen_"+g["id"]+"_main",specs=[spec(g,source=p.cache)])
                b,_,_=j.call(ep,c,arm="screen_"+g["id"]+"_control",specs=[spec(g,source=s.cache)])
                z,_,_=j.call(ep,c,arm="screen_"+g["id"]+"_zero",specs=[spec(g,"zero")])
                scores.append(effect(a,c,study)-effect(b,c,study));target=code_for(c["book"],c["target"])
                choices.append(int(a["prediction"]==target)-int(b["prediction"]==target));damages.append(int(base["correct"])-int(z["correct"]))
            screen.append({"episode":ep["id"],"score":scores[0]-np.mean(scores[1:]),"choice":choices[0],"necessity":damages[0]})
    guards=[]
    for ep in episodes[4:8]:
        c=ep["cases"][0]
        _,source,_=j.call(ep,c,role=controls,arm="guard_source",capture=[nkey])
        for zvalue in (c["original"],c["target"]):
            baseline,_,_=j.call(ep,c,arm="guard_clean",explicit_z=zvalue)
            for g in [group,*panel]:
                for mode in ("zero","patch"):
                    row,_,_=j.call(ep,c,arm=f"guard_{g['id']}_{mode}",explicit_z=zvalue,
                                  specs=[spec(g,mode,source.cache if mode=="patch" else None)])
                    if g["id"]==group["id"]:guards.append({"accuracy_damage":int(baseline["correct"])-int(row["correct"]),"margin_abs":abs(baseline["correct_margin"]-row["correct_margin"])})
    screened={k:float(np.mean([r[k] for r in screen])) for k in ("score","choice","necessity")}
    guard={k:float(np.mean([r[k] for r in guards])) for k in ("accuracy_damage","margin_abs")}
    passed=(screened["score"]>=cfg["practical_margin_nats"] and screened["choice"]>=cfg["practical_behavior_points"] and screened["necessity"]>=cfg["practical_behavior_points"] and
            guard["accuracy_damage"]<=cfg["code_accuracy_damage_max"] and guard["margin_abs"]<=cfg["code_mean_absolute_margin_change_max_nats"])
    gate="DISCOVERY_PASSED_REQUIRES_CALIBRATION" if passed else cfg["studies"][study]["no_qualified_group"]
    if j.p.calls-start>cfg["studies"][study]["discovery_cap"]:raise RuntimeError("DISCOVERY_BUDGET_EXCEEDED")
    return {"gate":gate,"readiness":ready,"coarse":summary,"native_candidates":groupstats,"group":group,"panel":panel,
            "screen":screened,"code_guard":guard,"discovery_only":True,"calibration":"NOT_EVALUATED","confirmation":"NOT_EVALUATED"}
