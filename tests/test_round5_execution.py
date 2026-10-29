"""CPU fixtures check scientific gate logic and causal-mask delivery, not 12B results."""
from pathlib import Path
import numpy as np
import torch
import yaml
from neuron_lab.round5_analysis import auc,paired_interval,summarize
from neuron_lab.round5_corpus import build_manifest,discovery_cases,render
from neuron_lab.round5_hooks import R5Hooks
from neuron_lab.round5_execution import rank_groups

ROOT=Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT/"configs/round5_execution.yaml").read_text())


def test_fresh_episode_assignment_and_witnesses():
    cfg=config();manifest=build_manifest(cfg)
    assert manifest==build_manifest(cfg)
    cases=discovery_cases(manifest,cfg)
    assert len(cases)==96 and len({c["id"] for c in cases})==96
    for c in cases:
        if c["witness"]=="code_map_only":
            assert f"Z={c['case']['answer']}" in c["text"] and "\nh:" not in c["text"]
    e=manifest["splits"]["technical"][0]
    text=render(e,e["test"][0],statuses=["2"]*8)
    assert text.count("diagnostic-pass=2")==8
    assert "not your previous output" in text


def test_native_mask_scope_and_cleanup():
    linear=torch.nn.Linear(6,2,bias=False)
    x=torch.arange(30,dtype=torch.float32).reshape(1,5,6)
    base=linear(x)
    with R5Hooks({0:linear},3,[{"layer":0,"coordinates":[1,4],"scope":"query"}]) as hooks:
        changed=linear(x)
    assert torch.equal(changed[:,:3],base[:,:3]) and not torch.equal(changed[:,3:],base[:,3:])
    assert hooks.delivery[0]["zero_delivered"] and hooks.delivery[0]["prefix_unchanged"]
    assert torch.equal(linear(x),base) and torch.equal(x,torch.arange(30).reshape(1,5,6).float())
    with R5Hooks({0:linear},3,[{"layer":0,"coordinates":[1],"scope":"all"}],restore=True) as hooks:
        restored=linear(x)
    assert torch.equal(restored,base) and hooks.delivery[0]["positions"]==5


def test_episode_uncertainty_and_auc_ties():
    assert auc([0,1,2,3],[0,0,1,1])==1
    assert auc([1,1],[0,1])==.5 and auc([0,1],[0,0]) is None
    result=paired_interval([.5]*8,1000,1)
    assert result["point"]==.5 and result["ci95"]==[.5,.5] and result["episodes"]==8


def test_ranking_uses_signed_witness_contrast():
    cfg=config()
    means={w:{l:[torch.ones(200)] for l in cfg["discovery"]["layer_pool"]}
           for w in ("direct_only","composition_only")}
    for l in cfg["discovery"]["layer_pool"]:
        means["composition_only"][l][0][:50]=10
        means["direct_only"][l][0][100:150]=10
    groups=rank_groups(means,cfg)
    assert len(groups)==16
    assert groups[0]["coordinates"]==list(range(32))
    assert groups[1]["coordinates"]==list(range(100,132))


def test_gate_does_not_promote_non_damaging_candidates():
    cfg=config();manifest=build_manifest(cfg)
    cases=discovery_cases(manifest,cfg)
    groups=[{"id":"c","layer":15,"coordinates":[0],"orientation":"composition"}]
    rows=[{"case_id":c["id"],"episode_id":c["episode_id"],"witness":c["witness"],
           "features":c["features"],"arm":a,"correct":True} for c in cases for a in ("healthy","c")]
    result=summarize(rows,groups,cfg)
    assert result["task_readiness_passed"]
    assert result["selected_pair"] is None and result["eligible_pair_count"]==0
    assert result["gate_outcome"]=="STOPPED_AT_DISCOVERY_LESION_GATE"


def test_inadequate_task_prevents_lesion_interpretation():
    cfg=config();cases=discovery_cases(build_manifest(cfg),cfg)
    rows=[{"case_id":c["id"],"episode_id":c["episode_id"],"witness":c["witness"],
           "features":c["features"],"arm":"healthy","correct":c["witness"]!="code_map_only"} for c in cases]
    result=summarize(rows,[],cfg)
    assert result["gate_outcome"]=="STOPPED_AT_TASK_READINESS"


def test_opposite_selective_eligible_pair_is_selected():
    cfg=config();cases=discovery_cases(build_manifest(cfg),cfg)
    groups=[{"id":"a","layer":15,"coordinates":list(range(32)),"orientation":"composition"},
            {"id":"b","layer":15,"coordinates":list(range(32,64)),"orientation":"direct"}]
    rows=[]
    dual_ids=[c["id"] for c in cases if c["witness"]=="dual_route"]
    single_ids={w:[c["id"] for c in cases if c["witness"]==w] for w in ("direct_only","composition_only")}
    for c in cases:
        index=dual_ids.index(c["id"]) if c["witness"]=="dual_route" else 0
        features=dict.fromkeys(cfg["discovery"]["error_features"],index/23)
        for arm in ("healthy","a","b"):
            error=(c["witness"]=="dual_route" and (index<6 if arm=="a" else index>=18) and arm!="healthy")
            if c["witness"] in single_ids:
                error=arm==({"composition_only":"a","direct_only":"b"}[c["witness"]]) and single_ids[c["witness"]].index(c["id"])<12
            rows.append({"case_id":c["id"],"episode_id":c["episode_id"],"witness":c["witness"],
                         "features":features,"arm":arm,"correct":not error})
    result=summarize(rows,groups,cfg)
    assert result["eligible_pair_count"]==1 and result["selected_pair"]["A"]=="a"
    assert result["gate_outcome"]=="DISCOVERY_PAIR_SELECTED"
