"""CPU tests of counterfactual identities and intervention delivery, not 12B results."""
from pathlib import Path
import math
import json
import torch
import yaml
from neuron_lab.round6_corpus import make_episode,build_manifest,render,answer,code_for,validate_episode
from neuron_lab.round6_engine import Hooks,margin

ROOT=Path(__file__).resolve().parents[1]


def cfg():return yaml.safe_load((ROOT/"configs/round6_execution.yaml").read_text())


def test_all_sources_differ_from_recipient_answers_and_letters():
    for seed in range(20):
        e=make_episode(seed,"cpu-A","A","discovery")
        for c in e["cases"]:
            r=render(e,c);s=render(e,c,"donor_same");o=render(e,c,"donor_opposite")
            assert len({r["semantic_answer"],c["target"],s["semantic_answer"],o["semantic_answer"]})==4
            assert len({r["expected"],code_for(c["book"],c["target"]),s["expected"],o["expected"]})==4
            assert s["codebook"]==o["codebook"] and s["semantic_answer"]!=o["semantic_answer"]


def test_intermediate_is_used_with_receiver_function_not_donor_answer():
    for seed in range(20):
        e=make_episode(seed,"cpu-C","C","confirmation")
        for c in e["cases"]:
            w=e["worlds"][f"receiver{c['function']}"]
            assert w["g"][c["intermediate"]]==c["target"]!=answer(e,c,"intermediate_donor")
            assert answer(e,c,"independent_correct_intermediate_rescue")==answer(e,c,"independent_wrong_intermediate_rescue")
            assert "r:" not in render(e,c)["text"]


def test_backup_authority_does_not_change_with_substitute():
    e=make_episode(19,"cpu-B","B","discovery")
    for c in e["cases"]:
        assert {answer(e,c,context=x) for x in ("agreement","conflict1","conflict2")}=={c["original"]}
        assert answer(e,c,context="conflict1",witness="composition_only")==c["wrong1"]
        assert answer(e,c,context="conflict2",witness="composition_only")==c["wrong2"]


def test_no_source_world_leakage_between_splits_or_studies():
    from neuron_lab.round6_corpus import digest
    m=build_manifest(cfg());ids=[];worlds=[]
    for splits in m["studies"].values():
        for episodes in splits.values():
            for e in episodes:ids.append(e["id"]);worlds.append(digest(e["worlds"]))
    assert len(ids)==len(set(ids))==len(set(worlds))
    assert m==build_manifest(cfg())
    assert json.loads(json.dumps(m))==m


def test_query_patch_changes_only_registered_native_coordinates():
    class Inv:
        sites={"native_mlp:0":(torch.nn.Linear(6,3,bias=False),True)};norms={}
    inv=Inv();m=inv.sites["native_mlp:0"][0];x=torch.arange(24).reshape(1,4,6).float()
    enc={"tokens":4,"query_start":2,"regions":{"query":[2,3],"all":[0,1,2,3]}}
    with Hooks(inv,enc,capture=["native_mlp:0"],regions=("query",)) as h:base=m(x)
    source={k:v+10 for k,v in h.cache.items()}
    with Hooks(inv,enc,[{"key":"native_mlp:0","coordinates":[1,3],"source":source,"scope":"query"}]) as h:
        changed=m(x)
    expected=x.clone();expected[:,2:,[1,3]]+=10
    assert torch.equal(changed,m(expected)) and torch.equal(changed[:,:2],base[:,:2])
    assert torch.equal(m(x),base) and not m._forward_pre_hooks


def test_tuple_output_and_norm_reference_are_not_mutations():
    class Tuple(torch.nn.Module):
        def forward(self,x):return x*2,None
    class Norm(torch.nn.Module):
        eps=1e-6;with_scale=True
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(6),requires_grad=False)
        def forward(self,x):return (x.float()*torch.pow(x.float().pow(2).mean(-1,keepdim=True)+self.eps,-.5))*self.weight
    class Inv:
        sites={"attn:0":(Tuple(),False)};norms={"norm":(Norm(),0)}
    inv=Inv();x=torch.arange(24).reshape(1,4,6).float()
    enc={"tokens":4,"query_start":2,"regions":{"query":[2,3],"all":[0,1,2,3]}}
    with Hooks(inv,enc,capture=["attn:0"],norm_capture_layer=0) as h:
        v=inv.sites["attn:0"][0](x);base=inv.norms["norm"][0](x)
    with Hooks(inv,enc,[{"key":"attn:0","scope":"query","mode":"zero"}],norm_reference=h.denominators):
        z=inv.sites["attn:0"][0](x);replay=inv.norms["norm"][0](x)
    assert isinstance(z,tuple) and z[1] is None and not z[0][:,2:].count_nonzero()
    assert torch.equal(base,replay) and not inv.norms["norm"][0]._forward_hooks


def test_global_and_conditional_call_envelopes_fit():
    c=cfg()
    assert sum(s["max_actual_forward_calls"] for s in c["studies"].values())+3*c["technical_call_cap"]<=c["max_actual_forward_calls"]
    for s in c["studies"].values():
        extra=s.get("additional_code_guards_and_norm_calls_cap",s.get("additional_guards_and_norm_calls_cap"))
        assert s["discovery_cap"]+s["confirmation_max_base_calls"]+s["calibration_max_base_calls"]+extra<=s["max_actual_forward_calls"]


def test_readout_margin_is_4way_log_odds():
    row={"post_softcap_logits":[1.,2.,3.,4.]}
    assert abs(margin(row,"D")-(4-math.log(sum(math.exp(x) for x in [1.,2.,3.]))))<1e-12
