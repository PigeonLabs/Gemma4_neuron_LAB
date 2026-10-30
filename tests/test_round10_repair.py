"""Independent CPU checks of the repaired question and response contracts."""
from collections import Counter
from fractions import Fraction as F
from itertools import product, permutations
import random
import re

import numpy as np

from neuron_lab.round9_worlds import _information_world, _reward_world, _resource_world, transformed_rewards
from neuron_lab.round10_repair import (
    parse_final, information_cases, reward_cases, sampling_cases, resource_cases,
    option_audit, winning_policy_count, parse_generated_body, prepared_cases, generation_boundary,
)
from neuron_lab.round10_analysis import cell_bounds, contrast, bounds_estimate, analyze


def world(study):
    if study == 'C':
        return dict(id='test-C', study='C', split='P0', low_p='2/7', record=[1, 0, 1], anchor_scale=1)
    w = {'A': _information_world, 'B': _reward_world, 'E': _resource_world}[study](random.Random(101+ord(study)), 'test-'+study, set())
    return {**w, 'split': 'P0'}


def test_parser_does_not_treat_bare_letters_or_conflicts_as_verdicts():
    assert parse_final('some calculation\nFINAL: B\n') == {'code': 'B', 'status': 'VALID'}
    assert parse_final('FINAL: A', exhausted=True)['code'] is None
    assert parse_final('calculation\r\nFINAL:A\r\n')['code'] == 'A'
    assert parse_final('```text\na calculation\n```\nFINAL: B')['code'] == 'B'
    for text in ['A', 'A and B, probably A', 'Example FINAL: A', 'FINAL: A\nFINAL: B',
                 'FINAL: A\nActually, reconsider.', '```\nFINAL: A\n```', '```\nFINAL: A', '"FINAL: A"',
                 'FINAL: A or FINAL: B', 'final: B', 'FINAL: B.']:
        assert parse_final(text)['code'] is None, text


def test_channel_controls_cannot_be_erased_to_recover_a_verdict():
    assert parse_generated_body('FINAL: A', internal_special=True)['code'] is None
    assert parse_generated_body('FINAL: A', exhausted=True, internal_special=True)['status'] == 'TOKEN_CAP'
    assert parse_generated_body('calculation\nFINAL: A')['code'] == 'A'


def test_natural_eos_at_the_exact_cap_is_not_truncation():
    assert generation_boundary([9, 8, 106], [1, 106, 50], 3) == (True, False, [9, 8])
    assert generation_boundary([9, 8, 7], [1, 106, 50], 3) == (False, True, [9, 8, 7])


def test_calibration_checks_all_collection_forms_without_filtering_by_accuracy():
    w = {**world('C'), 'split': 'calibration'}
    manifest = {'studies': {'C': {'splits': {'calibration': [w]}}}}
    rows = list(prepared_cases(manifest, 'C', 'calibration', 'main'))
    assert len(rows) == 16
    assert {c['factors']['process'] for c in rows} == {'fixed', 'optional_stop', 'value_filter', 'random_filter'}
    assert all(c['factors']['likelihoods'] == 0 for c in rows if c['factors']['process'] == 'value_filter')


def test_position_and_mapping_balance_is_per_question_not_just_global():
    for study, generator in [('A', information_cases), ('B', reward_cases), ('C', sampling_cases), ('E', resource_cases)]:
        w = world(study)
        for anchor in [False, True]:
            rows = list(generator(w, anchors=anchor))
            assert option_audit(rows)['every_question_mapping_and_position_fully_crossed']
            assert len({c['id'] for c in rows}) == len(rows)
        if study == 'C':
            rows = list(generator(w, anchors=True))
            assert Counter(r['truth'] for r in rows) == {0: 16, 1: 16}


def test_a_guarantees_against_independent_enumeration_of_signal_policies():
    w = world('A')
    assert 'independent auditor' in list(information_cases(w))[0]['text']
    for m in w['matrices'].values():
        for groups in [[0]*6, list(range(6)), w['partial_good'], w['partial_bad']]:
            signals = sorted(set(groups)); count = 0
            for actions in product(range(4), repeat=len(signals)):
                lookup = dict(zip(signals, actions))
                count += all(m[s][lookup[groups[s]]] for s in range(6))
            assert count == winning_policy_count(m, groups)


def test_b_sum_anchor_has_no_route_selection_command_and_transforms_preserve_sums():
    w = world('B')
    for row in reward_cases(w, anchors=True):
        assert 'Choose route' not in row['text'] and 'Which route has' not in row['text']
        if row['factors']['anchor_kind'] == 'path_sum':
            assert row['meanings'] == ['NO', 'YES']
        else:
            assert row['meanings'] == ['route U', 'route V']
    for allocation in ['base', 'front_U', 'front_V']:
        rows = transformed_rewards(w, allocation)
        assert [sum(r) for r in rows] == [sum(r) for r in w['paths']]
        assert all(v >= 0 for r in rows for v in r)


def test_b_serialized_credits_and_option_mapping_reconstruct_the_answer():
    w = world('B')
    for row in [*reward_cases(w, anchors=True), *reward_cases(w)]:
        facts = row['text'].split('FACTS:\n')[1].split('\n\nQUESTION:')[0]
        q = row['text'].split('QUESTION:\n')[1].split('\n\nCODE MEANINGS:')[0]
        sums = {}
        for route, numbers in re.findall(r'Route ([UV]): ([^\n]+)', facts):
            sums[route] = sum(int(n) for n in re.findall(r'credit (\d+)', numbers))
        assert set(sums) == {'U', 'V'}
        if 'total_comparison' == row['factors'].get('anchor_kind'):
            claimed = re.search(r'route U = (\d+); route V = (\d+)', facts)
            assert list(map(int, claimed.groups())) == [sums['U'], sums['V']]
        binary = re.search(r'Is the sum for route ([UV]) exactly (\d+)\?', q)
        meaning = ('YES' if sums[binary[1]] == int(binary[2]) else 'NO') if binary else ('route V' if sums['V'] > sums['U'] else 'route U')
        block = row['text'].split('CODE MEANINGS:\n')[1].split('\n\n')[0]
        code = next(line.split(' = ')[0] for line in block.splitlines() if line.split(' = ')[1] == meaning)
        assert code == row['expected']


def test_c_likelihood_oracle_by_finite_sample_space_and_rejection_conditioning():
    w = world('C'); n = len(w['record']); k = sum(w['record'])
    for mirror in [0, 1]:
        seq = [1-b if mirror else b for b in w['record']]
        likelihoods = []
        for p in [F(w['low_p']), 1-F(w['low_p'])]:
            probs = {bits: p**sum(bits)*(1-p)**(n-sum(bits)) for bits in product([0, 1], repeat=n)}
            direct = probs[tuple(seq)]
            bit = 0 if mirror else 1
            accepted = sum(prob for bits, prob in probs.items() if bits.count(bit) == k)
            likelihoods.append((direct, direct/accepted))
        direct_high = likelihoods[1][0]/sum(v[0] for v in likelihoods)
        selected_high = likelihoods[1][1]/sum(v[1] for v in likelihoods)
        for row in sampling_cases(w):
            if row['factors']['mirror'] == mirror:
                expected = selected_high if row['factors']['process'] == 'value_filter' else direct_high
                assert F(row['oracle']['posterior_HIGH']) == expected
    for row in sampling_cases(w, anchors=True):
        if row['factors']['anchor_kind'] != 'Bayes_weights':
            continue
        weights = list(map(F, row['oracle']['weights']))
        assert weights[1]/sum(weights) == F(row['oracle']['posterior_HIGH'])


def test_e_global_matching_against_exhaustive_task_resource_assignments():
    w = world('E')
    for row in resource_cases(w):
        f = row['factors']; m = w['matrices'][int(f['graph'] == 'feasible')]
        if f['query'] == 'global':
            independent = all(any(r) for r in m) if f['reuse'] else any(all(m[t][p[t]] for t in range(6)) for p in permutations(range(6)))
        else:
            independent = sum(any(m[t][r] for t in w['marked']) for r in range(6)) >= len(w['marked'])
        assert row['truth'] == int(independent)


def test_missing_rows_give_partial_bounds_and_never_become_no():
    rows = [{'semantic_choice': 1}, {'semantic_choice': 0}, {'semantic_choice': None}]
    assert np.allclose(cell_bounds(rows), [1/3, 2/3])
    unknown = [{'semantic_choice': None}]
    assert contrast([(1, unknown), (-1, unknown)]) == [-1, 1]
    estimate = bounds_estimate([[-1, 1]]*4, {'bootstrap_seed': 1, 'bootstrap_resamples': 100})
    assert estimate['ci95_outer'] == [-1, 1] and not estimate['fully_observed']


def test_readiness_failure_cannot_be_a_scientific_negative():
    row = dict(role='anchor', factors={'anchor_kind': 'Bayes_weights'}, semantic_choice=None, truth=1,
               parse_status='NO_FINAL_MARKER')
    out = analyze('C', [row], {'anchor_accuracy_min': .85, 'code_anchor_accuracy_min': .95, 'anchor_parse_min': .95})
    assert out['classification'] == 'Inconclusive' and out['primary'] is None
    assert out['main'] == 'NOT_EVALUATED'
