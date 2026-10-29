#!/usr/bin/env python3
"""Freeze fresh R6 sources, then run one registered study phase at a time."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import resource
import time
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from neuron_lab.round6_explicit_corpus import build_manifest,digest,render
from neuron_lab.round6_engine import encode
from neuron_lab.runtime import save_json,source_hashes,Run,ModelRuntime

CFG=ROOT/"configs/round6_explicit_execution.yaml"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(study,phase):
    from transformers import AutoTokenizer
    cfg=yaml.safe_load(CFG.read_text());mpath=ROOT/cfg["manifest_path"]
    manifest=build_manifest(cfg)
    if mpath.exists():
        assert json.loads(mpath.read_text())==manifest,"Fresh manifest changed; no overwrite"
    else:save_json(mpath,manifest)
    regpath=ROOT/f"logs/summary/round6x_{study}_{phase}_registration.json"
    if regpath.exists():raise RuntimeError("Registration already exists; use --check")
    local=Path((ROOT/"runs/model_path.txt").read_text().strip());assert local.name==cfg["model_revision"]
    tokenizer=AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False)
    assert all(len(tokenizer.encode(c,add_special_tokens=False))==1 for c in "ABCD")
    maximum=0;checked=0
    roles=cfg["studies"][study].get("source_worlds",["recipient"])
    for ep in manifest["studies"][study]["technical"]+manifest["studies"][study]["discovery"]:
        for c in ep["cases"]:
            baseline=encode(tokenizer,render(ep,c)["text"],cfg["max_input_tokens"])
            for role in roles:
                encoded=encode(tokenizer,render(ep,c,role)["text"],cfg["max_input_tokens"])
                for region in ("query","facts","cue"):
                    assert len(encoded["regions"][region])==len(baseline["regions"][region]),(study,ep["id"],role,region)
                maximum=max(maximum,encoded["tokens"]);checked+=1
    protected={p.relative_to(ROOT).as_posix():sha(p) for pattern in ["configs/round2*","configs/round3*","configs/round5*",
        "logs/summary/round2*","logs/summary/round3*","logs/summary/round5*","reports/ROUND2*","reports/ROUND3*","reports/ROUND5*",
        "logs/summary/round6_A*","configs/round6_execution.yaml","configs/round6_plan.yaml","logs/summary/round6_peer_review.json","logs/summary/round6_margin_diagnostics.json",
        "logs/summary/round6_verification.json","logs/summary/round6_publication.json"] for p in ROOT.glob(pattern) if p.is_file()}
    sources={p:h for p,h in source_hashes().items() if p.startswith("src/") or p=="configs/model.yaml" or p=="configs/round6_explicit_execution.yaml" or p=="scripts/run_round6_explicit.py"}
    reg={"schema_version":1,"status":"REGISTERED_BEFORE_PHASE_MODEL_OUTPUTS","study":study,"phase":phase,
        "model_id":cfg["model_id"],"model_revision":cfg["model_revision"],"config_sha256":sha(CFG),
        "manifest_canonical_sha256":digest(manifest),"source_sha256":sources,"protected_prior_sha256":protected,
        "technical_discovery_tokenized_inputs":checked,"maximum_tokens":maximum,
        "split_counts":{k:len(v) for k,v in manifest["studies"][study].items()},
        "current_phase_model_calls_at_registration":0,
        "execution_scope":"registered bounded discovery feasibility phase; later calibration/confirmation requires separate fixed-candidate/source registration",
        "scientific_protocol":cfg["studies"][study],"resource_caps":{k:cfg[k] for k in ["max_actual_forward_calls","max_seconds","max_run_disk_gb","max_local_module_evaluations"]}}
    save_json(regpath,reg)
    return reg


def check(study,phase):
    cfg=yaml.safe_load(CFG.read_text());reg=json.loads((ROOT/f"logs/summary/round6x_{study}_{phase}_registration.json").read_text())
    manifest=json.loads((ROOT/cfg["manifest_path"]).read_text())
    assert manifest==build_manifest(cfg) and digest(manifest)==reg["manifest_canonical_sha256"]
    for p,h in {**reg["source_sha256"],**reg["protected_prior_sha256"]}.items():assert sha(ROOT/p)==h,p
    return cfg,reg,manifest


def execute(study,phase):
    from neuron_lab.round6_engine import Probe
    from neuron_lab.round6_explicit_discovery import Journal,technical,discovery
    cfg,reg,manifest=check(study,phase)
    stage=f"round6x_{study}_{phase}"
    if (ROOT/f"logs/summary/{stage}.json").exists():raise RuntimeError("Recorded phase exists; no silent rerun")
    gpath=ROOT/"runs/round6_global_budget.json"
    global_state=json.loads(gpath.read_text()) if gpath.exists() else {"started_unix":time.time(),"calls":0,"completed_phases":[]}
    save_json(gpath,global_state)
    model_cfg=yaml.safe_load((ROOT/"configs/model.yaml").read_text())
    assert model_cfg["model_id"]==cfg["model_id"] and model_cfg["revision"]==cfg["model_revision"]
    runtime=ModelRuntime(audit_path=ROOT/f"logs/summary/round6x_{study}_{phase}_model_load.json")
    seed=cfg["seed"]+ord(study)-ord("A");random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    tr=Run(f"round6x_{study}_{phase}_technical",cfg);tp=Probe(runtime,tr,cfg,cfg["technical_call_cap"],global_state)
    try:
        tech=technical(tp,manifest["studies"]["A"]["technical"],cfg)
        tr.finish({**tech,"study":study,"model_revision":cfg["model_revision"],"measurement_seed":seed})
    except BaseException as error:
        tr.finish({"study":study,"error_type":type(error).__name__,"error":str(error),"actual_forward_calls":tp.calls},"FAIL")
        raise
    finally:
        global_state["calls"]+=tp.calls;save_json(gpath,global_state);tp.close()
    run=Run(stage,cfg);p=Probe(runtime,run,cfg,cfg["studies"][study]["max_actual_forward_calls"],global_state)
    journal=Journal(p,cfg,study)
    save_json(run.path/"registration.json",reg)
    save_json(run.path/"manifest.json",manifest["studies"][study])
    outcome=None
    try:
        if phase=="discovery":
            if study=="B":
                from neuron_lab.round6_backup import discover_backup
                outcome=discover_backup(journal,manifest["studies"][study]["discovery"])
            else:outcome=discovery(journal,manifest["studies"][study]["discovery"],study)
        else:
            from neuron_lab.round6_confirmation import confirm
            outcome=confirm(journal,manifest["studies"][study],study)
        journal.close();p.check_frozen()
        save_json(run.path/"encoded_inputs.json",journal.encodings)
        for path,h in reg["protected_prior_sha256"].items():assert sha(ROOT/path)==h,path
        summary={"schema_version":1,"study":study,"model_id":cfg["model_id"],"model_revision":cfg["model_revision"],
            "primary_dtype":cfg["primary_dtype"],"measurement_seed":seed,"loading_seed":runtime.cfg["seed"],"corpus_seed":cfg["corpus_seed"],
            "actual_forward_calls":p.calls,"completed_forward_calls":p.completed,"technical_forward_calls_separate":tech["actual_forward_calls"],
            "rows":len(journal.rows),"first50":p.first50,"local_module_evaluations":p.local_evaluations,
            "manifest_canonical_sha256":digest(manifest),"registration_sha256":digest(reg),
            "raw_rows_sha256":sha(run.path/"rows.jsonl"),"outcome":outcome,"calibration":"NOT_EVALUATED" if phase=="discovery" else outcome.get("calibration"),
            "confirmation":"NOT_EVALUATED" if phase=="discovery" else outcome.get("confirmation"),
            "source_sha256":source_hashes(),"frozen_parameters_verified":True,"use_cache":False,"protected_prior_files_verified":len(reg["protected_prior_sha256"])}
        summary["peak_CPU_RSS_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        result=run.finish(summary)
        print(json.dumps({"event":"phase_finished","study":study,"phase":phase,"run_id":run.id,"calls":p.calls,"gate":outcome["gate"]}),flush=True)
        return result
    except BaseException as error:
        journal.close()
        run.finish({"study":study,"actual_forward_calls":p.calls,"completed_forward_calls":p.completed,"rows":len(journal.rows),
            "error_type":type(error).__name__,"error":str(error),"outcome":outcome,"gate":"EXECUTION_FAILURE","confirmation":"NOT_EVALUATED"},"FAIL")
        raise
    finally:
        p.close();global_state["calls"]+=p.calls;global_state["completed_phases"].append({"study":study,"phase":phase,"run_id":run.id,"calls":p.calls})
        save_json(gpath,global_state)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--study",choices=list("ABC"),required=True)
    parser.add_argument("--phase",choices=["discovery","confirmation"],default="discovery")
    parser.add_argument("--prepare",action="store_true");parser.add_argument("--check",action="store_true");parser.add_argument("--execute",action="store_true")
    args=parser.parse_args();assert sum([args.prepare,args.check,args.execute])==1
    if args.prepare:
        reg=prepare(args.study,args.phase);print(json.dumps({"status":"PREPARED","study":args.study,"max_tokens":reg["maximum_tokens"],"model_calls":0}))
    elif args.check:
        cfg,reg,manifest=check(args.study,args.phase);print(json.dumps({"status":"PASS","study":args.study,"sources":len(reg["source_sha256"]),"protected":len(reg["protected_prior_sha256"])}))
    else:execute(args.study,args.phase)


if __name__=="__main__":main()
