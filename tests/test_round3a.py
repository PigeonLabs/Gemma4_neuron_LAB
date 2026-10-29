"""CPU toy checks only, not actual 12B experimental outcomes."""
import numpy as np
import pytest
import torch
import yaml
from neuron_lab.runtime import ROOT
from neuron_lab.round3a_corpus import build_manifest,validate_manifest
from neuron_lab.round3a import option_metrics,factorial,qualify,ranks

def config():return yaml.safe_load((ROOT/'configs/round3a.yaml').read_text())

def test_manifest_independent_sources_and_complete_factorials():
    cfg=config();m=build_manifest(cfg)
    assert m==build_manifest(cfg)
    assert len(m['cases'])==32+128+192+512
    for row in m['cases']:
        assert row['correct_label']==row['code'][0 if row['r']*row['c']==1 else 1]
    bad=[dict(r) for r in m['cases']];bad[0]['split']='confirmation'
    with pytest.raises(ValueError):validate_manifest(bad,cfg)

def test_content_axis_is_independent_of_correctness_label():
    labels={'A':0,'B':1,'C':2,'D':3}
    row={'code':'AB','r':1,'c':1,'correct_label':'A'}
    a=option_metrics(torch.tensor([[5.,2.,0.,0.]]),row,labels)
    b=option_metrics(torch.tensor([[2.,5.,0.,0.]]),{**row,'c':-1,'correct_label':'B'},labels)
    c=option_metrics(torch.tensor([[5.,2.,0.,0.]]),{**row,'r':-1,'correct_label':'B'},labels)
    assert a['semantic_margin']==b['semantic_margin']==c['semantic_margin']==3
    assert c['correctness_margin']==-3

def test_factorial_recovers_components_without_label_tautology():
    rows=[{'r':r,'c':c,'d':2+3*r+5*c+7*r*c} for r in (1,-1) for c in (1,-1)]
    assert factorial(rows)==dict(C=2.,G=3.,L=5.,J=7.)
    with pytest.raises(ValueError):factorial(rows[:3])
    assert np.allclose(ranks([2,1,1,4]),[2,.5,.5,3])

def test_confidence_can_look_like_rule_gain_and_is_removed_by_frozen_beta():
    rows=[{'r':r,'c':c,'d':.4*(10*r),'clean_semantic_margin':10*r} for r in (1,-1) for c in (1,-1)]
    assert factorial(rows)['G']==4
    assert factorial(rows,beta=.4)==dict(C=0.,G=0.,L=0.,J=0.)

def test_precision_gate_fails_reversed_effects_and_never_implies_no_function():
    cfg=config();cases=[c for c in build_manifest(cfg)['cases'] if c['split']=='qualification']
    rows=[]
    for dtype in ('bfloat16','float32'):
        for cid in ['H001','H002','H003','H004']:
            for j,c in enumerate(cases):
                effect=(j+1)/100 * (1 if dtype=='float32' else -1)
                for sign in (-1,1):
                    rows.append({**c,'case_id':c['id'],'dtype':dtype,'arm':'candidate','candidate_id':cid,'sign':sign,'semantic_margin':sign*effect,'actual_delta':sign*.5})
    bad=qualify(rows,cfg,True)
    assert bad['status']=='STOPPED_AT_NUMERICAL_QUALIFICATION' and not bad['eligible_candidates']
    for row in rows:
        if row['dtype']=='bfloat16':row['semantic_margin']*=-1
    assert qualify(rows,cfg,True)['eligible_candidates']==['H001','H002','H003','H004']
    assert qualify(rows,cfg,False)['status']=='STOPPED_AT_NUMERICAL_QUALIFICATION'
