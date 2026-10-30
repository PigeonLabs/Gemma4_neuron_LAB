"""Independent oracle checks and anti-pseudoreplication contracts, CPU only."""
from collections import Counter
from fractions import Fraction as F
import itertools as it
import random
from pathlib import Path
import yaml
import pytest
from neuron_lab.round9_worlds import (
    winning_policy_count, policy_valid, canonical_matrix, perfect_matchings,
    hall_violations, transformed_rewards, posterior, higher_order_distribution,
    build_manifest, prepared_cases, CASES,
)
from neuron_lab.round9_analysis import joint_from_prefixes, low_order_error, analyze


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def prepared():
    cfg = yaml.safe_load((ROOT / 'configs/round9_P0.yaml').read_text())
    return cfg, build_manifest(cfg)


def test_information_oracle_matches_independent_policy_enumeration():
    rng = random.Random(91)
    for _ in range(100):
        m = [[rng.randrange(2) for _ in range(3)] for _ in range(4)]
        groups = [0, 0, 1, 1]
        enumerated = 0
        for a, b in it.product(range(3), repeat=2):
            if all(m[s][a if s < 2 else b] for s in range(4)):
                enumerated += 1
        assert winning_policy_count(m, groups) == enumerated
    m = [[1, 0], [1, 0], [0, 1], [0, 1]]
    assert winning_policy_count(m, [0, 0, 1, 1]) == 1
    assert winning_policy_count(m, [0, 1, 0, 1]) == 0
    assert policy_valid(m, [0, 0, 1, 1], [0, 0, 1, 1])
    assert not policy_valid(m, [0, 1, 0, 1], [0, 0, 1, 1])


def test_all_generated_information_controls_and_structural_uniqueness(prepared):
    cfg, manifest = prepared; keys = set()
    for worlds in manifest['studies']['A']['splits'].values():
        for w in worlds:
            m = w['matrices']['mixed']
            assert winning_policy_count(m, w['partial_good']) > 0
            assert winning_policy_count(m, w['partial_bad']) == 0
            assert Counter(w['partial_good']) == Counter(w['partial_bad'])
            assert winning_policy_count(w['matrices']['universal'], [0] * 6) > 0
            assert winning_policy_count(w['matrices']['impossible'], list(range(6))) == 0
            for v in w['matrices'].values():
                k = canonical_matrix(v); assert k not in keys; keys.add(k)


def test_reward_potential_transform_preserves_every_complete_path(prepared):
    cfg, manifest = prepared
    for worlds in manifest['studies']['B']['splits'].values():
        for w in worlds:
            totals = [sum(x) for x in w['paths']]
            assert totals[0] != totals[1]
            for arm in ['base', 'front_U', 'front_V']:
                new = transformed_rewards(w, arm)
                assert [sum(x) for x in new] == totals
                assert min(sum(new, [])) >= 0
            front = transformed_rewards(w, 'front_U'); back = transformed_rewards(w, 'front_V')
            assert front[0][0] > back[0][0] and front[1][0] < back[1][0]


def test_sampling_oracle_by_independent_conditioned_sample_space(prepared):
    cfg, manifest = prepared
    for w in manifest['studies']['C']['splits']['P0']:
        p0 = F(w['low_p']); n = len(w['record']); k = sum(w['record'])
        for mirror in range(2):
            seq = [1-b if mirror else b for b in w['record']]
            target_bit = 0 if mirror else 1
            assert seq[-1] == target_bit and seq[:-1].count(target_bit) == k - 1
            likelihoods = []
            for p in [p0, 1-p0]:
                raw = F(1)
                for b in seq: raw *= p if b else 1-p
                accepted = sum((p ** sum(x) * (1-p) ** (n-sum(x)) for x in it.product([0,1], repeat=n)
                                if x.count(target_bit) == k), F(0))
                likelihoods.append((raw, raw / accepted))
            ordinary = likelihoods[1][0] / sum(x[0] for x in likelihoods)
            selected = likelihoods[1][1] / sum(x[1] for x in likelihoods)
            assert ordinary == posterior(w, 'fixed', mirror) == posterior(w, 'optional_stop', mirror) == posterior(w, 'random_filter', mirror)
            assert selected == posterior(w, 'value_filter', mirror) == F(1,2)


def test_resource_oracles_hall_and_permutation_agree_and_degrees_preserved(prepared):
    cfg, manifest = prepared; keys=set()
    for worlds in manifest['studies']['E']['splits'].values():
        for w in worlds:
            bad, good = w['matrices']
            assert not perfect_matchings(bad) and perfect_matchings(good)
            assert hall_violations(bad) and not hall_violations(good)
            assert list(map(sum,bad)) == list(map(sum,good))
            assert list(map(sum,zip(*bad))) == list(map(sum,zip(*good)))
            assert sum(a != b for x,y in zip(bad,good) for a,b in zip(x,y)) == 4
            for m in [bad,good]:
                k=canonical_matrix(m);assert k not in keys;keys.add(k)
            union = {j for i in w['marked'] for j,x in enumerate(bad[i]) if x}
            assert len(union) == 2 and len(w['marked']) == 3


def test_parity_joint_reconstruction_and_marginal_error_without_model():
    for q in [F(1,8), F(1,2), F(7,8)]:
        truth=higher_order_distribution(q); rows=[]
        for prefix in ['', '0', '1', '00', '01', '10', '11']:
            masses=[sum(p for s,p in truth.items() if s.startswith(prefix+b)) for b in '01']
            rows.append({'factors':{'prefix':prefix},'restricted_probabilities':[float(p/sum(masses)) for p in masses], 'valid_probability_mass':1.})
        actual, mass=joint_from_prefixes(rows)
        assert actual == {s:float(p) for s,p in truth.items()}
        assert abs(mass-1) < 1e-12 and low_order_error(actual) == 0


def test_manifest_counts_and_anchor_response_balance(prepared):
    cfg, manifest=prepared
    for s in 'ABCDE':
        anchors=list(prepared_cases(manifest,s,anchors=True)); mains=list(prepared_cases(manifest,s))
        assert len(anchors)==cfg['studies'][s]['anchors']
        assert len(mains)==cfg['studies'][s]['main']
        assert set(c['expected'] for c in anchors)=={'A','B'}
        assert min(Counter(c['expected'] for c in anchors).values()) >= len(anchors)*.25
        assert len({c['id'] for c in anchors+mains})==len(anchors+mains)
        assert all(c['text'].count('\nNEW_QUERY:\n')==1 for c in anchors+mains)
    assert sum(x['scientific_cap'] for x in cfg['studies'].values())+cfg['expected_shared_technical_calls']==3646


def test_failed_anchors_do_not_become_negative_main_evidence(prepared):
    cfg,manifest=prepared;c=next(prepared_cases(manifest,'A',anchors=True))
    row={**c,'full_correct':False,'format_valid':True}
    out=analyze('A',[row],cfg)
    assert out['classification']=='Inconclusive' and out['main']=='NOT_EVALUATED'
    assert out['identifiability']=='NOT_IDENTIFIABLE'
