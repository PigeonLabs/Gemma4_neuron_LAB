"""Synthetic CPU checks of B gate arithmetic; no model or semantic claims."""
import pytest
from neuron_lab.round6_backup import primary_metrics,interaction,primary_specs

CFG={'code_accuracy_damage_max':.05,'code_mean_absolute_margin_change_max_nats':.2}


def row(correct=True,margin=4.):return {'correct':correct,'correct_margin':margin}


def test_primary_requires_real_selective_damage_and_preserved_code():
    pairs=[]
    for name in ('direct_only','composition_only','code_only','agreement'):
        for i in range(10):pairs.append((name,row(),row(not(name=='direct_only' and i<3))))
    assert primary_metrics(pairs,CFG)['eligible']
    nonspecific=[(name,b,row(False) if name=='composition_only' else r) for name,b,r in pairs]
    assert not primary_metrics(nonspecific,CFG)['eligible']
    decoder=[(name,b,row(r['correct'],3.) if name=='code_only' else r) for name,b,r in pairs]
    assert not primary_metrics(decoder,CFG)['eligible']


def test_interaction_is_lesion_and_context_difference_of_damage():
    base={(c,p):row(margin=4.) for c in ('agreement','conflict1','conflict2') for p in (0,1)}
    changed={k:row(margin=3.) for k in base}
    changed['agreement',1]=row(margin=1.)
    assert interaction(base,changed,CFG)==pytest.approx(2.)
    changed['conflict1',1]=row(margin=1.);changed['conflict2',1]=row(margin=1.)
    assert interaction(base,changed,CFG)==pytest.approx(0.)


def test_primary_mask_is_persistent_on_all_positions():
    assert primary_specs({'keys':['mlp:1','mlp:2']})==[{'key':'mlp:1','scope':'all','mode':'zero'},{'key':'mlp:2','scope':'all','mode':'zero'}]
    assert primary_specs({'key':'native_mlp:2','coordinates':[0,3]})[0]['scope']=='all'
