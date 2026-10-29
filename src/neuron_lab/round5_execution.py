"""R5 technical qualification and the fixed discovery-lesion gate.

This executable opens neither calibration nor confirmation outcomes. A successful
lesion gate hands its frozen candidates to the remaining registered R5 stages.
"""
from __future__ import annotations
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from .runtime import ROOT, Run, ModelRuntime, save_json, source_hashes
from .round5_corpus import render, discovery_cases, digest
from .round5_hooks import R5Hooks
from .round5_analysis import summarize


def encode(tokenizer, text, maximum):
    messages=[{"role":"user","content":text}]
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    authoritative=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=True,
        add_generation_prompt=True,enable_thinking=False)
    offsets=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    if authoritative["input_ids"]!=offsets["input_ids"]:
        raise ValueError("Offset serialization differs from authoritative chat template")
    if rendered.count("\nNEW_QUERY:\n")!=1:
        raise ValueError("Ambiguous query boundary")
    char_start=rendered.index("\nNEW_QUERY:\n")
    query_start=next(i for i,(_,end) in enumerate(offsets["offset_mapping"]) if end>char_start)
    length=len(authoritative["input_ids"])
    if not 0<query_start<length<=maximum:
        raise ValueError(f"Invalid untruncated input or boundary: {query_start}/{length}/{maximum}")
    return {"input_ids":authoritative["input_ids"],"attention_mask":authoritative["attention_mask"],
            "query_start":query_start,"tokens":length,"text_sha256":hashlib.sha256(text.encode()).hexdigest()}


class Probe:
    def __init__(self,runtime,run,cfg):
        self.runtime,self.run,self.cfg=runtime,run,cfg
        self.calls=0
        self.completed=0
        self.started=time.monotonic()
        self.first50=None
        self.raw=None
        self.label_ids=[runtime.tokenizer.encode(c,add_special_tokens=False)[0] for c in cfg["label_codes"]]
        if any(len(runtime.tokenizer.encode(c,add_special_tokens=False))!=1 for c in cfg["label_codes"]):
            raise ValueError("Labels must be single tokens")
        self.armed=False
        self.handle=runtime.model.register_forward_pre_hook(self._before,with_kwargs=True)
        self.head_handle=runtime.model.get_output_embeddings().register_forward_hook(self._head)
        self.versions=None
        self.latest_context={}

    def _head(self,module,args,output):
        self.raw=output[:,-1,:].detach()

    def _before(self,module,args,kwargs):
        if not self.armed or kwargs.get("use_cache") is not False:
            raise RuntimeError("Uncounted forward or cache enabled")
        if self.calls>=self.cfg["max_actual_forward_calls"]:
            raise RuntimeError("Actual full-forward cap reached")
        self.run.check_budget()
        self.calls+=1

    def freeze_versions(self):
        self.versions={n:p._version for n,p in self.runtime.model.named_parameters()}

    def check_frozen(self):
        if any(p.requires_grad for p in self.runtime.model.parameters()):
            raise RuntimeError("Model no longer frozen")
        if self.versions!={n:p._version for n,p in self.runtime.model.named_parameters()}:
            raise RuntimeError("Parameter mutation detected")

    @torch.inference_mode()
    def forward(self,encoded,groups=(),capture_layers=(),restore=False):
        model=self.runtime.model
        inputs={k:torch.tensor([encoded[k]],device=self.runtime.cfg["device"]) for k in ("input_ids","attention_mask")}
        self.raw=None
        self.armed=True
        try:
            with R5Hooks(self.runtime.modules,encoded["query_start"],groups,capture_layers,restore) as hooks:
                logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1,:]
            if self.raw is None or not torch.isfinite(logits).all() or not torch.isfinite(self.raw).all():
                raise RuntimeError("Nonfinite or missing readout")
            expected_layers=sorted(set(capture_layers)|{g["layer"] for g in groups})
            if hooks.events!=expected_layers:
                raise RuntimeError(f"Unexpected native hook delivery {hooks.events}/{expected_layers}")
            if any(d["zero_delivered"] is False for d in hooks.delivery):
                raise RuntimeError("Zero clamp not delivered")
            # Prefix equality is an invariant for a query-only mask. Joint masks
            # are checked in the technical fixture with the query mask first.
            if len(groups)==1 and groups[0].get("scope")=="query" and not hooks.delivery[0]["prefix_unchanged"]:
                raise RuntimeError("Query mask changed prefix activations")
            softcap=model.config.get_text_config().final_logit_softcapping
            replay=torch.tanh(self.raw/softcap)*softcap
            if not torch.equal(logits,replay):
                raise RuntimeError("Raw tied-head to softcap replay parity failed")
            self.completed+=1
            if self.completed==50:
                torch.cuda.synchronize()
                self.first50={"completed_forwards":50,"seconds":time.monotonic()-self.started,
                              "last_tokens":encoded["tokens"],"dtype":str(logits.dtype)}
                save_json(self.run.path/"first50.json",self.first50)
                print(json.dumps({"event":"first_50_completed",**self.first50}),flush=True)
            log=logits[0].float()
            four=log[self.label_ids]
            raw_four=self.raw[0,self.label_ids].float().cpu().tolist()
            top=int(log.argmax().item())
            prediction=self.cfg["label_codes"][int(four.argmax().item())]
            result={"prediction":prediction,"full_top1_id":top,
                    "full_top1_text":self.runtime.tokenizer.decode([top]),
                    "full_top1_is_code":top in self.label_ids,
                    "post_softcap_logits":four.cpu().tolist(),"raw_head_logits":raw_four,
                    "tokens":encoded["tokens"],"query_start":encoded["query_start"]}
            return result,hooks,logits.detach().clone()
        finally:
            self.armed=False
            self.raw=None

    def close(self):
        self.handle.remove()
        self.head_handle.remove()


def technical(probe,manifest,cfg):
    runtime=probe.runtime
    records=[]
    coords=cfg["technical_test_coordinates"]
    encodings=[encode(runtime.tokenizer,render(e,e["test"][0],statuses=["2"]*8),cfg["max_input_tokens"])
               for e in manifest["splits"]["technical"]]
    for dtype in (torch.bfloat16,torch.float32):
        if dtype==torch.float32:
            # Convert every parameter, preserving BF16-stored values and tying.
            sampled={n:p.flatten()[::max(1,p.numel()//32)][:32].float().cpu()
                     for n,p in runtime.model.named_parameters()}
            runtime.model.float()
            if any(p.dtype!=torch.float32 or not torch.equal(sampled[n],p.flatten()[::max(1,p.numel()//32)][:32].cpu())
                   for n,p in runtime.model.named_parameters()):
                raise RuntimeError("Whole-model upcast did not preserve stored BF16 values")
        if runtime.model.get_input_embeddings().weight.data_ptr()!=runtime.model.get_output_embeddings().weight.data_ptr():
            raise RuntimeError("Tied input/output embeddings no longer tied")
        probe.freeze_versions()
        for ei,encoded in enumerate(encodings):
            all_group={"layer":15,"coordinates":coords,"scope":"all"}
            query_group={"layer":15,"coordinates":coords,"scope":"query"}
            other={"layer":15,"coordinates":[1,3,5],"scope":"all"}
            _,_,base=probe.forward(encoded)
            checks={}
            _,read,read_logits=probe.forward(encoded,capture_layers=[15])
            checks["read_only_exact"]=torch.equal(base,read_logits)
            _,_,identity=probe.forward(encoded,groups=[{**all_group,"attenuation":0.}])
            checks["identity_exact"]=torch.equal(base,identity)
            _,all_hook,all_logits=probe.forward(encoded,groups=[all_group])
            checks["all_token_zero"]=all_hook.delivery[0]["zero_delivered"] and all_hook.delivery[0]["positions"]==encoded["tokens"]
            _,query_hook,_=probe.forward(encoded,groups=[query_group])
            checks["query_only_zero_and_prefix_unchanged"]=query_hook.delivery[0]["zero_delivered"] and query_hook.delivery[0]["prefix_unchanged"]
            _,joint,_=probe.forward(encoded,groups=[query_group,other])
            checks["joint_same_module_delivery"]=len(joint.delivery)==2 and all(d["zero_delivered"] for d in joint.delivery)
            _,restored_hook,restored=probe.forward(encoded,groups=[all_group],restore=True)
            checks["same_forward_restoration_exact"]=torch.equal(base,restored) and restored_hook.delivery[0]["zero_delivered"]
            _,_,removed=probe.forward(encoded)
            checks["removal_exact"]=torch.equal(base,removed)
            checks["finite_capture"]=bool(torch.isfinite(read.means[15]).all())
            checks["softcap_replay_exact"]=True # enforced on every forward
            checks["tied_embeddings"]=True
            checks["use_cache_false"]=True
            if not all(checks.values()):
                raise RuntimeError(f"R5 technical failure: {checks}")
            records.append({"dtype":str(dtype),"technical_episode":ei,"tokens":encoded["tokens"],
                            "checks":checks,"diagnostic_max_zero_logit_change":float((base-all_logits).abs().max().item())})
            probe.check_frozen()
            if probe.calls>cfg["technical_max_forwards"]:
                raise RuntimeError("Technical call cap exceeded")
        print(json.dumps({"event":"technical_dtype_passed","dtype":str(dtype),"calls":probe.calls}),flush=True)
    runtime.smoke_passed=True
    return {"status":"PASS","calls":probe.calls,"checks":records,"full_model_upcast":True,
            "scope":"actual 12B native hook qualification; not scientific evidence of recovery"}


def rank_groups(means,cfg):
    groups=[]
    for layer in cfg["discovery"]["layer_pool"]:
        direct=torch.stack(means["direct_only"][layer]).mean(0)
        comp=torch.stack(means["composition_only"][layer]).mean(0)
        score=(comp-direct)/(comp+direct+1e-8)
        for size in cfg["discovery"]["group_sizes"]:
            for orientation,sign in (("composition",-1),("direct",1)):
                order=np.lexsort((np.arange(score.numel()),sign*score.numpy()))
                selected=order[:size].tolist()
                groups.append({"id":f"L{layer:02d}_K{size:03d}_{orientation}","layer":layer,
                               "coordinates":selected,"scope":"all","orientation":orientation,
                               "selection":"clean discovery witness query mean-absolute activation contrast"})
    if len(groups)!=cfg["discovery"]["candidate_count"]:
        raise ValueError("Registered candidate count mismatch")
    return groups


def execute(cfg,manifest,registration):
    model_cfg=yaml.safe_load((ROOT/"configs/model.yaml").read_text())
    if model_cfg["revision"]!=cfg["model_revision"] or model_cfg["model_id"]!=cfg["model_id"]:
        raise RuntimeError("Execution and model-loader identities differ")
    if digest(manifest)!=registration["manifest_canonical_sha256"]:
        raise RuntimeError("Manifest differs from pre-data registration")
    for name,sha in registration["source_sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError(f"Source changed since pre-data registration: {name}")
    if (ROOT/"logs/summary/round5_discovery.json").exists():
        raise RuntimeError("R5 discovery already has a recorded result; no silent rerun")
    runtime=ModelRuntime(audit_path=ROOT/"logs/summary/round5_model_load.json")
    random.seed(cfg["seed"]);np.random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
    run=Run("round5_discovery",cfg)
    probe=Probe(runtime,run,cfg)
    rows=[];groups=[];tech=None
    save_json(run.path/"input_manifest.json",manifest)
    save_json(run.path/"registration.json",registration)
    row_file=(run.path/"rows.jsonl").open("x")
    def record(case,arm,result):
        correct=result["prediction"]==case["case"]["correct_code"]
        ix=cfg["label_codes"].index(case["case"]["correct_code"])
        logits=torch.tensor(result["post_softcap_logits"],dtype=torch.float64)
        margin=float(logits[ix]-torch.logsumexp(torch.cat([logits[:ix],logits[ix+1:]]),0))
        row={"arm":arm,"case_id":case["id"],"episode_id":case["episode_id"],"witness":case["witness"],
             "features":case["features"],"correct_code":case["case"]["correct_code"],"correct":correct,
             "margin":margin,**result}
        rows.append(row);row_file.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n");row_file.flush()
    try:
        tech=technical(probe,manifest,cfg)
        save_json(run.path/"technical.json",tech)
        save_json(ROOT/"logs/summary/round5_technical.json",{**tech,"run_id":run.id,
                  "model_revision":cfg["model_revision"],"source_sha256":source_hashes()})
        runtime.verify_sources()
        cases=discovery_cases(manifest,cfg)
        encoded={c["id"]:encode(runtime.tokenizer,c["text"],cfg["max_input_tokens"]) for c in cases}
        save_json(run.path/"encoded_discovery.json",encoded)
        layers=cfg["discovery"]["layer_pool"]
        means={w:{l:[] for l in layers} for w in ("direct_only","composition_only")}
        for i,case in enumerate(cases):
            capture=layers if case["witness"] in means else []
            result,hooks,_=probe.forward(encoded[case["id"]],capture_layers=capture)
            record(case,"healthy",result)
            for l,v in hooks.means.items():
                means[case["witness"]][l].append(v)
            if (i+1)%24==0:
                print(json.dumps({"event":"healthy_progress","cases":i+1,"calls":probe.calls}),flush=True)
        healthy=summarize(rows,[],cfg)
        if healthy["task_readiness_passed"]:
            groups=rank_groups(means,cfg)
            save_json(run.path/"discovery_candidates.json",groups)
            torch.save(means,run.path/"clean_query_absmeans.pt")
            for gi,g in enumerate(groups):
                for case in cases:
                    result,_,_=probe.forward(encoded[case["id"]],groups=[g])
                    record(case,g["id"],result)
                probe.check_frozen()
                partial=summarize(rows,groups[:gi+1],cfg)
                save_json(run.path/"progress.json",{"finished_candidates":gi+1,"calls":probe.calls,"summary":partial})
                print(json.dumps({"event":"candidate_completed","candidate":g["id"],"index":gi+1,
                                  "calls":probe.calls,"dual_accuracy":partial["candidates"][-1]["accuracy"]["dual_route"]}),flush=True)
        analysis=summarize(rows,groups,cfg)
        probe.check_frozen();runtime.verify_sources()
        for name,sha in registration["protected_prior_sha256"].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
                raise RuntimeError(f"Prior result changed during R5: {name}")
        full_top1={w:float(np.mean([r["full_top1_id"]==probe.label_ids[cfg["label_codes"].index(r["correct_code"])]
                                  for r in rows if r["arm"]=="healthy" and r["witness"]==w]))
                   for w in cfg["discovery"]["witnesses"]}
        summary={"schema_version":1,"study_id":cfg["study_id"],"model_id":cfg["model_id"],
            "model_revision":cfg["model_revision"],"primary_dtype":cfg["primary_dtype"],
            "loading_seed":runtime.cfg["seed"],"measurement_seed":cfg["seed"],"corpus_seed":cfg["corpus_seed"],
            "manifest_canonical_sha256":digest(manifest),"registration_sha256":digest(registration),
            "actual_forward_calls":probe.calls,"completed_forward_calls":probe.completed,
            "technical_forward_calls":tech["calls"],"scientific_rows":len(rows),"first50":probe.first50,
            "discovery_episodes":len(manifest["splits"]["discovery"]),"candidate_count_evaluated":len(groups),
            "healthy_unrestricted_top1_accuracy":full_top1,"analysis":analysis,
            "calibration":"NOT_EVALUATED","confirmation":"NOT_EVALUATED",
            "feedback_recovery":"NOT_EVALUATED","query_time_recruitment":"NOT_EVALUATED",
            "norm_replay":"NOT_EVALUATED","protected_prior_results_unchanged":True,
            "raw_rows_sha256":hashlib.sha256((run.path/"rows.jsonl").read_bytes()).hexdigest(),
            "source_sha256":source_hashes(),"frozen_parameters_verified":True,"use_cache":False,
            "scope":"bounded discovery gate; candidate selection uncertainty is not confirmation"}
        row_file.close()
        return run.finish(summary)
    except BaseException as error:
        row_file.close()
        failure={"schema_version":1,"study_id":cfg["study_id"],"model_revision":cfg["model_revision"],
            "actual_forward_calls":probe.calls,"completed_forward_calls":probe.completed,"scientific_rows":len(rows),
            "error_type":type(error).__name__,"error":str(error),"technical":tech,
            "feedback_recovery":"NOT_EVALUATED","query_time_recruitment":"NOT_EVALUATED",
            "gate_outcome":"STOPPED_AT_TECHNICAL_OR_EXECUTION_FAILURE"}
        run.finish(failure,"FAIL")
        raise
    finally:
        probe.close()
