#!/usr/bin/env python3
"""CPU-only existence proof; no torch, model load, decoding or GPU calls."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations, permutations
import json
from pathlib import Path
import platform
import resource
import sys
import time

import yaml

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def hall(rows):
    for size in range(1,len(rows)+1):
        for subset in combinations(range(len(rows)),size):
            union=set().union(*(rows[i] for i in subset))
            if len(union)<size:return {'tasks':list(subset),'neighbors':sorted(union),'minimum_witness_size':size}
    return None


def overlap_profile(rows):
    return sorted((list(k),v) for k,v in Counter((min(len(a),len(b)),max(len(a),len(b)),len(a&b))
        for a,b in combinations(rows,2)).items())


def columns(rows):return [{i for i,r in enumerate(rows) if j in r} for j in range(len(rows))]


def main():
    started=time.monotonic();cp=ROOT/'configs/round10b_next_cpu_design.yaml'
    cfg=yaml.safe_load(cp.read_text())
    base=[{0},{1},{0,1},{2,3},{3,4},{2,5},{6},{7}]
    flip=[set(r) for r in base];flip[2]={1,2};flip[3]={0,3}
    sham=[set(r) for r in base];sham[4]={2,4};sham[5]={3,5}
    duplicate_feasible=[{0},{1},{2,3},{2,3},{0,4},{1,5},{6},{7}]
    graphs={'base_infeasible':base,'switch_feasible':flip,'sham_infeasible':sham,
            'duplicate_but_feasible_auxiliary':duplicate_feasible}
    evaluations=0;results={}
    for name,rows in graphs.items():
        assert len(rows)==cfg['max_vertices_per_side']
        match=None
        for p in permutations(range(8)):
            evaluations+=1
            if all(p[t] in rows[t] for t in range(8)):
                match=list(p);break
        witness=hall(rows)
        assert (match is not None)==(witness is None)
        assert list(map(len,rows))==list(map(len,base))
        assert list(map(len,columns(rows)))==list(map(len,columns(base)))
        assert rows[6:]==base[6:] and rows[6]|rows[7]=={6,7}
        duplicates=len(rows)-len({tuple(sorted(r)) for r in rows})
        if name!='duplicate_but_feasible_auxiliary':
            assert duplicates==0
            assert overlap_profile(rows)==overlap_profile(base)
            assert overlap_profile(columns(rows))==overlap_profile(columns(base))
        if name in ['base_infeasible','sham_infeasible']:
            assert witness['minimum_witness_size']==3
        if name in ['switch_feasible','duplicate_but_feasible_auxiliary']:assert match is not None
        results[name]={'rows':[sorted(r) for r in rows],'exclusive_feasible':match is not None,
            'unlimited_reuse_feasible':all(rows),'matching_witness':match,'Hall_witness':witness,
            'duplicate_rows':duplicates,'row_degrees':list(map(len,rows)),
            'column_degrees':list(map(len,columns(rows))),
            'typed_pairwise_row_overlap_histogram':overlap_profile(rows),
            'typed_pairwise_column_overlap_histogram':overlap_profile(columns(rows))}
        assert time.monotonic()-started<cfg['max_seconds']
    switch_changes=[(i,j) for i in range(8) for j in base[i]^flip[i]]
    sham_changes=[(i,j) for i in range(8) for j in base[i]^sham[i]]
    assert len(switch_changes)==len(sham_changes)==4
    for changed in [switch_changes,sham_changes]:
        assert len({i for i,j in changed})==len({j for i,j in changed})==2
        assert all(len(base[i])==2 for i,j in changed)
        assert all(len(columns(base)[j])==2 for i,j in changed)
        assert all(i not in [6,7] for i,j in changed)
    assert evaluations<=cfg['max_permutation_checks']
    out={'schema_version':1,'status':'CPU_PROTOTYPE_PASS_NOT_GPU_GO','recorded_at_UTC':datetime.now(timezone.utc).isoformat(),
        'new_model_forwards':0,'GPU_used':False,'device':'cpu','dtype':'boolean_and_integer','seed':cfg['seed'],
        'model_revision':'NOT_APPLICABLE_CPU_ONLY','split':'CPU_CONSTRUCTION_NOT_MODEL_DISCOVERY_OR_TEST',
        'input_sha256':hashlib.sha256(json.dumps({k:[sorted(r) for r in v] for k,v in graphs.items()},sort_keys=True).encode()).hexdigest(),
        'independent_prototype_families':1,'config_sha256':sha(cp),'source_sha256':sha(Path(__file__)),
        'environment':{'python':sys.version.split()[0],'platform':platform.platform()},
        'results':results,'switch_changed_edges':[list(x) for x in switch_changes],
        'sham_changed_edges':[list(x) for x in sham_changes],
        'verified':['All degrees and fixed query rows/union exactly preserved',
            'Primary graphs have no duplicate rows; all1/2-task Hall conditions pass',
            'Global infeasibility witnessed only from minimum3-task set',
            'True switch and sham change4edge entries across2rows/2columns, with identical2/2 endpoint degree types',
            'Primary typed pairwise row/column neighbor-overlap histograms identical',
            'Feasibility differs under exclusivity; every graph feasible with unlimited reuse',
            'Duplicated degree2 rows can coexist with a valid complete assignment in auxiliary control',
            'Independent exhaustive permutation matching and subset-Hall oracle agree'],
        'permutation_checks':evaluations,'CPU_seconds':time.monotonic()-started,
        'CPU_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'remaining_limits':['Only one structural family; relabelings are NOT independent worlds',
            'Component-size profiles and other higher-order structure are not controlled; global-feasibility-only causal attribution remains unproven',
            'No rendered/token-length matched prompt corpus or model data exists',
            'Auxiliary duplicate-feasible graph is not fully matched on every primary structural statistic',
            'Novelty requires full-text comparison; no neural mechanism or GPU Go established',
            'Original A/C/E main remain NOT_EVALUATED; this does not repair or rerun their scientific endpoint']}
    path=ROOT/'logs/summary/round10b_next_cpu_design.json';raw=json.dumps(out,ensure_ascii=False,indent=2)+'\n'
    assert len(raw.encode())<=cfg['max_output_bytes'];path.write_text(raw)
    print(json.dumps({k:out[k] for k in ['status','new_model_forwards','independent_prototype_families','permutation_checks','CPU_seconds']}))


if __name__=='__main__':main()
