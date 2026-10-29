import yaml
from pathlib import Path
from neuron_lab.round7_lookup_worlds import make
from neuron_lab import round7_history as C,round7_order_priming as D

def test_C_identical_final_state_and_counterfactual_history():
    w=make(62,'unit');aa=[C.material(w,history=h,ordering=o) for h in ['related1','related2','unrelated1','unrelated2'] for o in ['before','after']]
    assert len({a['oracle']['final_text'] for a in aa})==1
    assert len({a['expected'] for a in aa})==1
    assert len({a['text'] for a in aa})==8

def test_CD_counts_no_input_selection():
    for letter,mod in [('C',C),('D',D)]:
        cfg=yaml.safe_load(Path(f'configs/round7_{letter}.yaml').read_text());m=mod.build_manifest(cfg);cases=list(mod.prepared_cases(m,cfg))
        assert len(cases)==cfg['expected_scientific_calls_if_quality_passes']
        assert len({c['id'] for c in cases})==len(cases)

def test_D_target_unknown_to_prefix_and_task_oracle():
    w=make(41,'unit');a=D.material(w,task='inverse',orientation='ascending');b=D.material(make(43,'other'),orientation='ascending')
    prefix=lambda c:c['text'].split('PREFIX_BEGIN:\n')[1].split('\nPREFIX_END')[0]
    assert prefix(a)==prefix(b)
    m=a['oracle']['mapping'];assert m[a['oracle']['correct']]==a['oracle']['query']
