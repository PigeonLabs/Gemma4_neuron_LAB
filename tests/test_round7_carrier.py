from neuron_lab.round7_carrier import graph_qualification,material,build_manifest,prepared_cases
from neuron_lab.round7_lookup_worlds import make

def test_B_graph_cut_and_key_selectivity():
    assert all(graph_qualification().values())
    w=make(55,'unit');c=material(w);same=material(w,donor='same');other=material(w,donor='other')
    assert c['oracle']['correct']!=same['oracle']['correct']
    assert c['oracle']['correct']==other['oracle']['correct']
    assert len([k for k in c['oracle']['values'] if c['oracle']['values'][k]!=same['oracle']['values'][k]])==1

def test_B_scientific_count_and_reserved_worlds():
    import yaml
    from pathlib import Path
    cfg=yaml.safe_load(Path('configs/round7_B.yaml').read_text());m=build_manifest(cfg);cases=list(prepared_cases(m,cfg))
    assert len(cases)==cfg['expected_scientific_calls_if_quality_passes']
    assert len({c['id'] for c in cases})==len(cases)
