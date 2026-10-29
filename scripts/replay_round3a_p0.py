#!/usr/bin/env python3
"""Replay numerical gates from public paired effects, without model/raw cache."""
import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from neuron_lab.round3a import ranks,correlation

def main():
    s=json.loads((ROOT/'logs/summary/round3a_p0.json').read_text())
    cfg=yaml.safe_load((ROOT/'configs/round3a.yaml').read_text());q=cfg['qualification']
    checks=[];eligible=[]
    for row in s['qualification']['candidates']:
        paired=row['paired_effects'];a=np.array([r['BF16'] for r in paired]);b=np.array([r['FP32'] for r in paired])
        assert len(a)==32 and len({r['case_id'] for r in paired})==32
        mask=abs(b)>=q['nontrivial_fp32_effect_floor_nats'];n=int(mask.sum())
        sign=float(np.mean(np.sign(a[mask])==np.sign(b[mask]))) if n else None
        rho=correlation(ranks(a),ranks(b));mae=float(np.mean(abs(a-b)))
        bound=max(q['maximum_mean_absolute_error_floor_nats'],q['maximum_mean_absolute_error_relative']*float(np.mean(abs(b))))
        actual={'n_nontrivial_fp32':n,'sign_agreement_nontrivial':sign,'spearman_signed_effects':rho,
            'mean_absolute_error_nats':mae,'mean_absolute_error_bound_nats':bound,
            'absolute_error_99th_percentile_nats':float(np.quantile(abs(a-b),.99)),
            'mean_abs_signed_effect_BF16':float(np.mean(abs(a))),'mean_abs_signed_effect_FP32':float(np.mean(abs(b)))}
        for k,v in actual.items():
            assert (row[k] is None if v is None else abs(row[k]-v)<1e-15),(row['id'],k)
        criteria={'technical':all(s['technical_checks'].values()),
            'nonzero_delivery':all(v>=q['actual_nonzero_fraction_minimum'] for v in row['nonzero_delivery_fraction'].values()),
            'sufficient_nontrivial_cases':n>=q['minimum_nontrivial_signed_cases'],
            'sign_agreement':sign is not None and sign>=q['minimum_sign_agreement'],
            'rank_stability':rho is not None and rho>=q['minimum_spearman'],'mean_absolute_error':mae<=bound}
        assert criteria==row['checks'] and all(criteria.values())==row['eligible']
        if row['eligible']:eligible.append(row['id'])
        checks.append(row['id'])
    assert eligible==s['qualification']['eligible_candidates']
    assert s['qualification']['status']==('PASS' if eligible and all(s['technical_checks'].values()) else 'STOPPED_AT_NUMERICAL_QUALIFICATION')
    print(json.dumps({'status':'PASS','candidate_gate_replays':checks,'eligible_candidates':eligible,
        'qualification_status':s['qualification']['status'],'scope':'public paired effects only; upstream inference, delivery fractions and hook checks not independently revalidated','new_GPU_inference':False}))

if __name__=='__main__':main()
