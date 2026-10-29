#!/usr/bin/env python3
"""Replay actual R5 discovery records on CPU; never performs model inference."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from neuron_lab.round5_corpus import build_manifest,discovery_cases,digest
from neuron_lab.round5_analysis import summarize

DEST=ROOT/"logs/summary/round5_replay.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    cfg=yaml.safe_load((ROOT/"configs/round5_execution.yaml").read_text())
    result=json.loads((ROOT/"logs/summary/round5_discovery.json").read_text())
    if result["status"]!="PASS":
        raise ValueError("Execution failure needs an explicit failure audit")
    path=ROOT/"runs"/result["run_id"]
    manifest=json.loads((path/"input_manifest.json").read_text())
    registration=json.loads((path/"registration.json").read_text())
    assert manifest==build_manifest(cfg)
    assert digest(manifest)==result["manifest_canonical_sha256"]==registration["manifest_canonical_sha256"]
    assert digest(registration)==result["registration_sha256"]
    for name,value in {**registration["source_sha256"],**registration["protected_prior_sha256"]}.items():
        assert sha(ROOT/name)==value,name
    assert sha(path/"rows.jsonl")==result["raw_rows_sha256"]
    rows=[json.loads(x) for x in (path/"rows.jsonl").read_text().splitlines()]
    cases={c["id"]:c for c in discovery_cases(manifest,cfg)}
    assert len({(r["arm"],r["case_id"]) for r in rows})==len(rows)
    groups=json.loads((path/"discovery_candidates.json").read_text()) if (path/"discovery_candidates.json").exists() else []
    if groups:
        means=torch.load(path/"clean_query_absmeans.pt",map_location="cpu",weights_only=True)
        for g in groups:
            layer=g["layer"]
            # Rebuild signed ranking independently of the GPU-runner function.
            # Torch and numpy reductions can differ in last-bit tie order. Reuse
            # exact saved torch reductions, then independently lexsort coordinates.
            direct=torch.stack(means["direct_only"][layer]).mean(0).numpy()
            comp=torch.stack(means["composition_only"][layer]).mean(0).numpy()
            score=(comp-direct)/(comp+direct+np.float32(1e-8))
            order=np.lexsort((np.arange(len(score)),(-1 if g["orientation"]=="composition" else 1)*score))
            assert g["coordinates"]==order[:len(g["coordinates"])].tolist(),g["id"]
    margins=[];softcap_errors=[]
    for r in rows:
        c=cases[r["case_id"]]
        assert r["episode_id"]==c["episode_id"] and r["witness"]==c["witness"] and r["features"]==c["features"]
        assert r["correct_code"]==c["case"]["correct_code"]
        four=np.asarray(r["post_softcap_logits"],dtype=float)
        assert len(four)==4 and np.isfinite(four).all()
        assert r["prediction"]==cfg["label_codes"][int(np.argmax(four))]
        assert r["correct"]==(r["prediction"]==r["correct_code"])
        ix=cfg["label_codes"].index(r["correct_code"])
        other=np.delete(four,ix);maximum=float(other.max())
        margin=float(four[ix]-maximum-math.log(sum(math.exp(float(x)-maximum) for x in other)))
        margins.append(abs(margin-r["margin"]))
        raw=np.asarray(r["raw_head_logits"],dtype=np.float32)
        post=np.tanh(raw/np.float32(30))*np.float32(30)
        softcap_errors.append(float(np.abs(post-four).max()))
    assert max(margins,default=0)<1e-10
    assert max(softcap_errors,default=0)<1e-5
    counts=Counter(r["arm"] for r in rows)
    assert set(counts)=={"healthy",*[g["id"] for g in groups]}
    assert all(v==len(cases) for v in counts.values())
    assert len(rows)==result["scientific_rows"]
    assert result["actual_forward_calls"]==result["completed_forward_calls"]==len(rows)+result["technical_forward_calls"]
    technical=json.loads((path/"technical.json").read_text())
    assert technical["status"]=="PASS" and technical["calls"]==48
    assert len(technical["checks"])==6 and all(all(x["checks"].values()) for x in technical["checks"])
    replay=summarize(rows,groups,cfg)
    assert replay==result["analysis"]
    # Direct descriptive reductions, without the shared summary function.
    for candidate in result["analysis"]["candidates"]:
        arm=[r for r in rows if r["arm"]==candidate["id"]]
        for w in cfg["discovery"]["witnesses"]:
            a=[r for r in arm if r["witness"]==w]
            clean=[r for r in rows if r["arm"]=="healthy" and r["witness"]==w]
            assert abs(sum(r["correct"] for r in a)/len(a)-candidate["accuracy"][w])<1e-12
            assert abs((sum(r["correct"] for r in clean)-sum(r["correct"] for r in a))/len(a)-candidate["damage"][w]["point"])<1e-12
    for name in ("calibration","confirmation","feedback_recovery","query_time_recruitment","norm_replay"):
        assert result[name]=="NOT_EVALUATED"
    remote=json.loads((ROOT/"runs/round5_registration_remote_receipt.json").read_text())
    assert remote["file_hashes_verified"] and remote["independently_read_back_files"]==11
    return {"schema_version":1,"status":"PASS","run_id":result["run_id"],
        "scope":"CPU record replay of actual 12B R5 results, not independent model replication",
        "scientific_rows_checked":len(rows),"actual_forwards_reconciled":result["actual_forward_calls"],
        "episodes":result["discovery_episodes"],"candidate_groups":len(groups),"selection_reproduced":True,
        "maximum_margin_replay_error":max(margins,default=0),"maximum_softcap_CPU_GPU_difference":max(softcap_errors,default=0),
        "protected_prior_files_verified":len(registration["protected_prior_sha256"]),
        "registration_source_files_verified":len(registration["source_sha256"]),
        "pre_inference_public_registration_commit":remote["remote_main_sha"],
        "source_result_sha256":sha(ROOT/"logs/summary/round5_discovery.json"),
        "raw_rows_sha256":result["raw_rows_sha256"],"technical_identity_and_delivery_passed":True,
        "calibration_and_confirmation_remain_unmeasured":True,"gate_outcome":replay["gate_outcome"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--check",action="store_true");args=parser.parse_args()
    result=verify()
    if args.check:
        assert json.loads(DEST.read_text())==result,"Replay receipt is stale"
    else:
        DEST.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"status":"PASS","rows":result["scientific_rows_checked"],"forwards":result["actual_forwards_reconciled"],"gate":result["gate_outcome"]}))


if __name__=="__main__":
    main()
