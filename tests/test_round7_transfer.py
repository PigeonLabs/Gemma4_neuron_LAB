import numpy as np
from neuron_lab.round7_transfer import fit,predict,prepared_cases,build_manifest,material
from neuron_lab.round7_lookup_worlds import make

def test_E_frozen_prediction_and_oracle():
    f=fit([[0.,1.],[1.,2.],[2.,3.]],[1.,2.,3.],1.)
    assert np.isfinite(predict(f,[3.,4.]))
    w=make(33,'unit')
    for task in ['lookup','position']:
        a=material(w,task);assert a['oracle']['codebook'][a['expected']]==a['oracle']['correct']

def test_E_call_envelope_and_split_disjointness():
    import yaml
    from pathlib import Path
    c=yaml.safe_load(Path('configs/round7_E.yaml').read_text());m=build_manifest(c);cases=list(prepared_cases(m,c))
    assert len(cases)==1392
    assert len({v['id'] for v in cases})==len(cases)
    assert not ({w['signature'] for w in m['splits']['P0']} & {w['signature'] for w in m['splits']['calibration']})
