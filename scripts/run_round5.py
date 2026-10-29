#!/usr/bin/env python3
"""Prepare/freeze fresh R5 inputs or execute the registered first gate."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from neuron_lab.round5_corpus import build_manifest,discovery_cases,render,digest

CONFIG=ROOT/"configs/round5_execution.yaml"
REGISTRATION=ROOT/"logs/summary/round5_registration.json"
SOURCES=["configs/round5_execution.yaml","configs/round5_plan.yaml","src/neuron_lab/round5_design.py",
    "src/neuron_lab/round5_corpus.py","src/neuron_lab/round5_analysis.py","src/neuron_lab/round5_hooks.py",
    "src/neuron_lab/round5_execution.py","src/neuron_lab/runtime.py","configs/model.yaml",
    "scripts/run_round5.py","tests/test_round5_execution.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    from transformers import AutoTokenizer
    from neuron_lab.round5_execution import encode
    cfg=yaml.safe_load(CONFIG.read_text())
    expected=sha(ROOT/cfg["parent_design"])
    if cfg["parent_design_sha256"]=="TO_BE_PREPARED":
        CONFIG.write_text(CONFIG.read_text().replace("TO_BE_PREPARED",expected))
        cfg=yaml.safe_load(CONFIG.read_text())
    if cfg["parent_design_sha256"]!=expected:
        raise RuntimeError("Reviewed parent design changed")
    if REGISTRATION.exists():
        raise RuntimeError("Registration exists; use --check, never overwrite")
    path=ROOT/cfg["manifest_path"]
    if path.exists():
        raise RuntimeError("Manifest exists without registration; inspect before proceeding")
    manifest=build_manifest(cfg)
    local=Path((ROOT/"runs/model_path.txt").read_text().strip())
    if local.name!=cfg["model_revision"]:
        raise RuntimeError("Wrong cached model revision")
    tokenizer=AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False)
    code_ids={c:tokenizer.encode(c,add_special_tokens=False) for c in cfg["label_codes"]}
    if any(len(v)!=1 for v in code_ids.values()) or len({v[0] for v in code_ids.values()})!=4:
        raise RuntimeError("Four distinct single-token codes required")
    lengths=[encode(tokenizer,c["text"],cfg["max_input_tokens"])["tokens"] for c in discovery_cases(manifest,cfg)]
    for e in manifest["splits"]["technical"]:
        lengths.append(encode(tokenizer,render(e,e["test"][0],statuses=["2"]*8),cfg["max_input_tokens"])["tokens"])
    protected=[p for pattern in ("configs/round2*","configs/round3*","logs/summary/round2*",
                 "logs/summary/round3*","reports/ROUND2*","reports/ROUND3*",
                 "logs/summary/round5_design*","logs/summary/round5_verification.json","logs/summary/round5_publication.json")
               for p in ROOT.glob(pattern) if p.is_file()]
    registration={"schema_version":1,"study_id":cfg["study_id"],"status":"REGISTERED_BEFORE_MODEL_INFERENCE",
        "scope":"technical and discovery-lesion eligibility gate; conditional later stages require their candidate freeze",
        "model_id":cfg["model_id"],"model_revision":cfg["model_revision"],
        "manifest_path":cfg["manifest_path"],"manifest_canonical_sha256":digest(manifest),
        "split_counts":{s:len(es) for s,es in manifest["splits"].items()},
        "technical_and_discovery_max_tokens":max(lengths),"technical_and_discovery_min_tokens":min(lengths),
        "calibration_and_confirmation_outcomes":"UNOPENED",
        "source_sha256":{p:sha(ROOT/p) for p in SOURCES},
        "protected_prior_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(protected))},
        "label_token_ids":code_ids,"discovery_cases":len(lengths)-cfg["technical_prompts"],
        "scientific_model_forwards_at_registration":0,
        "full_budget":cfg["budget_explanation"],
        "amendment_before_data":"Four code-map-only witnesses add 408 calls, within the unchanged 32000 cap. Historical reviewed design is preserved.",
        "selection_and_gates":cfg["discovery"],"novelty_claim":"UNVERIFIED; no first-ever claim"}
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    REGISTRATION.write_text(json.dumps(registration,ensure_ascii=False,indent=2)+"\n")
    return registration


def check():
    cfg=yaml.safe_load(CONFIG.read_text())
    registration=json.loads(REGISTRATION.read_text())
    manifest=json.loads((ROOT/cfg["manifest_path"]).read_text())
    if digest(manifest)!=registration["manifest_canonical_sha256"] or manifest!=build_manifest(cfg):
        raise RuntimeError("Registered fresh inputs are not reproducible")
    for path,value in {**registration["source_sha256"],**registration["protected_prior_sha256"]}.items():
        if sha(ROOT/path)!=value:
            raise RuntimeError(f"Registered file changed: {path}")
    if cfg["budget_explanation"]["discovery_screen_cases"]*cfg["budget_explanation"]["discovery_screen_arms"]!=cfg["budget_explanation"]["discovery_screen_calls"]:
        raise RuntimeError("Call accounting mismatch")
    if cfg["budget_explanation"]["full_study_planned_calls_with_code_guard"]>cfg["max_actual_forward_calls"]:
        raise RuntimeError("Registered study exceeds cap")
    return cfg,manifest,registration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--execute-discovery",action="store_true")
    args=parser.parse_args()
    if sum((args.prepare,args.check,args.execute_discovery))!=1:
        parser.error("Choose exactly one operation")
    if args.prepare:
        reg=prepare()
        print(json.dumps({"status":"PREPARED","manifest_sha256":reg["manifest_canonical_sha256"],
                          "max_tokens":reg["technical_and_discovery_max_tokens"],"model_forwards":0}))
    else:
        cfg,manifest,registration=check()
        if args.check:
            print(json.dumps({"status":"PASS","registration_sources_verified":len(registration["source_sha256"]),
                              "protected_prior_files":len(registration["protected_prior_sha256"])}))
        else:
            from neuron_lab.round5_execution import execute
            result=execute(cfg,manifest,registration)
            print(json.dumps({"status":result["status"],"run_id":result["run_id"],
                              "calls":result["actual_forward_calls"],"gate":result["analysis"]["gate_outcome"]}))


if __name__=="__main__":
    main()
