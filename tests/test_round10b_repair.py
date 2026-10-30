import json
from pathlib import Path

from neuron_lab.round10b_repair import STRICT_FORMAT, prepared_cases, calibration_futility


def test_strict_instruction_does_not_force_or_insert_an_assistant_answer_prefix():
    old = json.loads(Path('runs/round10_manifest.json').read_text())
    w = old['studies']['A']['splits']['calibration'][0]
    manifest = {'studies': {'A': {'splits': {'calibration': [w]}}}}
    case = next(prepared_cases(manifest, 'A', 'calibration', 'anchor'))
    assert case['text'].endswith(STRICT_FORMAT)
    assert 'You may calculate briefly' not in case['text']
    assert 'CODE MEANINGS:' in case['text'] and 'QUESTION:' in case['text']
    assert 'FINAL:<code>' in case['text']


def test_futility_is_a_provable_bound_on_the_declared_final_denominator():
    planned = [{'factors': {'anchor_kind': 'order'}} for _ in range(32)]
    cfg = dict(code_anchor_accuracy_min=.95, anchor_accuracy_min=.85, anchor_parse_min=.95)
    observed = [dict(factors={'anchor_kind': 'order'}, semantic_choice=None, truth=1) for _ in range(2)]
    out = calibration_futility(observed, planned, cfg)
    assert out['stop'] and out['groups']['order']['best_possible_parse_fraction'] == 30/32


def test_order_anchor_asks_direct_announcement_not_possible_signal_inference():
    old = json.loads(Path('runs/round10_manifest.json').read_text())
    w = old['studies']['A']['splits']['calibration'][0]
    manifest = {'studies': {'A': {'splits': {'calibration': [w]}}}}
    rows = [c for c in prepared_cases(manifest, 'A', 'calibration', 'anchor')
            if c['factors']['anchor_kind'] == 'order' and c['truth'] == 0]
    assert rows and all('DIRECTLY ANNOUNCED' in c['text'] for c in rows)
    assert all('Does the player know the exact state before' not in c['text'] for c in rows)


def test_calibration_futility_never_requires_the_main_hypothesis_to_be_correct():
    planned = [{'factors': {'anchor_kind': 'payoff'}} for _ in range(32)]
    cfg = dict(code_anchor_accuracy_min=.95, anchor_accuracy_min=.85, anchor_parse_min=.95)
    observed = [dict(factors={'anchor_kind': 'payoff'}, semantic_choice=1, truth=1)]
    assert calibration_futility(observed, planned, cfg)['stop'] is False


def test_calibration_round_robin_spreads_worlds_and_anchor_kinds_before_second_view():
    old = json.loads(Path('runs/round10_manifest.json').read_text())
    cases = list(prepared_cases(old, 'A', 'calibration', 'anchor'))
    first = cases[:16]
    assert len({(c['episode_id'], c['factors']['anchor_kind']) for c in first}) == 16
    assert len({c['id'] for c in cases}) == len(cases) == 128
    assert all('Output no other text.' in c['text'] for c in cases)
