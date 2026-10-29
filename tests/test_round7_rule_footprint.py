"""CPU-only rule logic and split checks, not a model P0."""
import json,yaml
from pathlib import Path
from neuron_lab.round7_rule_footprint import make_world,build_manifest,case,prepared_cases,rule
ROOT=Path(__file__).resolve().parents[1]


def test_logical_dependence_is_balanced_across_distance_and_disjoint_from_demonstrations():
    for seed in range(100):
        w=make_world(seed,'cpu')
        assert len(w['queries'])==6
        assert not set(w['initial'])&{q['x'] for q in w['queries']}
        for q in w['queries']:
            assert q['distance']==sum(a!=b for a,b in zip(q['x'],w['evidence']))
            assert q['dependent']==(q['h0']!=q['h1'])
        for proximity in ('near','far'):
            qq=[q for q in w['queries'] if q['proximity']==proximity]
            assert sum(q['dependent'] for q in qq)==2 and len(qq)==3
        assert {rule(x,w['rules'][0]) for x in w['initial']}=={0,1}
        assert all(rule(x,w['rules'][0])==rule(x,w['rules'][1]) for x in w['initial'])


def test_unrelated_namespace_is_not_assigned_a_known_target_answer():
    w=make_world(33,'cpu')
    for r in (0,1):
        for bi in (0,1):
            c=case(w,0,bi,r,'OTHER');assert not c['has_unique_target_answer']
            t=case(w,0,bi,r,'TARGET');assert t['has_unique_target_answer']
            assert t['oracle']['book'][t['expected']]==rule(w['queries'][0]['x'],w['rules'][r])
    assert {case(w,0,bi,code_only=w['seed']%2)['expected'] for bi in (0,1)}=={'A','B'}


def test_manifest_is_unique_roundtrippable_and_all_calls_are_enumerated():
    cfg=yaml.safe_load((ROOT/'configs/round7_A.yaml').read_text());m=build_manifest(cfg)
    signatures=[w['signature'] for ww in m['splits'].values() for w in ww]
    assert len(signatures)==len(set(signatures)) and json.loads(json.dumps(m))==m
    cases=list(prepared_cases(m,cfg));assert len(cases)==1296==cfg['expected_scientific_calls_if_quality_passes']
    assert len({c['id'] for c in cases})==len(cases)
    assert cfg['phase_actual_forward_cap']+cfg['technical_call_cap']<=cfg['P0_round_actual_forward_cap']
