import pytest
from neuron_lab.input_contracts import InputContractError,token_fingerprint,validate_splits,validate_deterministic_rule_anchor
from neuron_lab.round7_rule_footprint import make_world,rule

def test_reject_actual_R7_A_anchor_failure_mode():
    w=make_world(42,'synthetic')
    examples=[(x,rule(x,w['rules'][0])) for x in w['initial']]+[(w['evidence'],rule(w['evidence'],w['rules'][0]))]
    assert validate_deterministic_rule_anchor(w['rules'][0],examples)['status']=='PASS'
    with pytest.raises(InputContractError,match='contradicts'):
        validate_deterministic_rule_anchor(w['rules'][1],examples)

def test_new_latent_id_does_not_hide_repeated_model_input():
    h=token_fingerprint([2,25,19,8]);episodes={'discovery':[{'id':'fresh-a','observable_signature':'state-a','principal_input_sha256':[h,h]}],
       'confirmation':[{'id':'fresh-b','observable_signature':'different-latent-world','principal_input_sha256':[h]}]}
    with pytest.raises(InputContractError,match='input repeats'):validate_splits(episodes)

def test_new_tokens_do_not_hide_identical_observed_world():
    episodes={'discovery':[{'id':'a','observable_signature':'same-visible-facts','principal_input_sha256':[token_fingerprint([2,3])]}],
       'confirmation':[{'id':'b','observable_signature':'same-visible-facts','principal_input_sha256':[token_fingerprint([2,4])]}]}
    with pytest.raises(InputContractError,match='Observed world repeats'):validate_splits(episodes)

def test_repeated_arms_stay_inside_one_world_and_unseen_split_passes():
    h=token_fingerprint([1,2]);v=validate_splits({'discovery':[{'id':'a','observable_signature':'a','principal_input_sha256':[h,h]}],
       'confirmation':[{'id':'b','observable_signature':'b','principal_input_sha256':[token_fingerprint([1,3])]}]})
    assert v['unique_observable_worlds']==2 and v['unique_principal_inputs']==2
