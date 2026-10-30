"""Generated-final behavior and conservative missing-answer bounds for R10."""
from __future__ import annotations

from collections import defaultdict, Counter
import numpy as np


def choose(rows, **factors):
    return [r for r in rows if all(r['factors'].get(k) == v for k, v in factors.items())]


def cell_bounds(rows, semantic=1):
    assert rows
    yes = sum(r['semantic_choice'] == semantic for r in rows)
    missing = sum(r['semantic_choice'] is None for r in rows)
    return np.array([yes/len(rows), (yes+missing)/len(rows)])


def contrast(cells):
    """cells=(coefficient, all registered rows of one family/cell)."""
    low = high = 0.
    for weight, rows in cells:
        lo, hi = cell_bounds(rows)
        low += weight*(lo if weight >= 0 else hi)
        high += weight*(hi if weight >= 0 else lo)
    return [low, high]


def bounds_estimate(values, cfg):
    x = np.asarray(values, dtype=float)
    assert x.ndim == 2 and x.shape[1] == 2 and len(x) and np.isfinite(x).all()
    assert np.all(x[:, 0] <= x[:, 1]+1e-12)
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    indices = rng.integers(0, len(x), size=(cfg['bootstrap_resamples'], len(x)))
    samples = x[indices].mean(1)
    point = x.mean(0)
    return {'identified_interval': point.tolist(),
            'ci95_outer': [float(np.quantile(samples[:, 0], .025)), float(np.quantile(samples[:, 1], .975))],
            'families': len(x), 'fully_observed': bool(np.allclose(x[:, 0], x[:, 1], atol=0, rtol=0)),
            'unit': 'world/family, with all code/display/query repeats paired',
            'scope': 'exploratory finite-generator family bootstrap and worst-case missing-answer bounds; no population equivalence theorem'}


def accuracy(rows):
    correct = sum(r['semantic_choice'] is not None and r['semantic_choice'] == r['truth'] for r in rows)
    wrong = sum(r['semantic_choice'] is not None and r['semantic_choice'] != r['truth'] for r in rows)
    unknown = sum(r['semantic_choice'] is None for r in rows)
    return {'n': len(rows), 'correct': correct, 'known_wrong': wrong, 'unparsed': unknown,
            'confirmed_correct_fraction': correct/len(rows),
            'accuracy_bounds': [correct/len(rows), (correct+unknown)/len(rows)],
            'parsed_fraction': (correct+wrong)/len(rows),
            'parsed_only_accuracy_diagnostic': correct/(correct+wrong) if correct+wrong else None}


def anchor_quality(rows, cfg):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['factors']['anchor_kind']].append(row)
    groups = {k: accuracy(v) for k, v in grouped.items()}
    passed = all(v['confirmed_correct_fraction'] >= (cfg['code_anchor_accuracy_min'] if k == 'code' else cfg['anchor_accuracy_min'])
                 and v['parsed_fraction'] >= cfg['anchor_parse_min'] for k, v in groups.items())
    return {'passed': passed, 'groups': groups,
            'interpretation': 'A readiness failure is not a scientific Negative. Missing answers remain distinct from known wrong answers.'}


def major_readout_quality(study, rows):
    names = {'A': ['structure', 'partition', 'query'],
             'B': ['allocation', 'totals', 'reverse_listing'],
             'C': ['process', 'mirror', 'likelihoods'],
             'E': ['graph', 'reuse', 'query']}[study] + ['codebook', 'option_order']
    out = {}
    for name in names:
        for value in sorted({r['factors'][name] for r in rows}, key=str):
            out[f'{name}={value}'] = accuracy(choose(rows, **{name: value}))
    return out


def analyze(study, rows, cfg):
    anchor = [r for r in rows if r['role'] == 'anchor']
    main = [r for r in rows if r['role'] == 'main']
    readiness = anchor_quality(anchor, cfg)
    base = {'readiness': readiness, 'P1': 'NOT_EVALUATED', 'P2': 'NOT_EVALUATED',
            'parse_status': dict(Counter(r['parse_status'] for r in rows))}
    if not main:
        return {**base, 'classification': 'Inconclusive', 'gate': 'STOPPED_AT_REPAIRED_ANCHORS',
                'estimand_valid': False, 'primary': None, 'main': 'NOT_EVALUATED'}
    assert readiness['passed']
    by = defaultdict(list)
    for row in main:
        by[row['episode_id']].append(row)
    estimates = defaultdict(list); controls = []
    for fid, rr in by.items():
        if study == 'A':
            estimates['false_approval_excess'].append(contrast([
                (1, choose(rr, structure='mixed', partition='useless')),
                (-1, choose(rr, structure='impossible', partition='useless'))]))
            estimates['useful_minus_useless'].append(contrast([
                (1, choose(rr, structure='mixed', partition='useful')),
                (-1, choose(rr, structure='mixed', partition='useless'))]))
            for query in ['existence', 'given_policy']:
                estimates['query_'+query].append(contrast([
                    (1, choose(rr, structure='mixed', partition='useless', query=query)),
                    (-1, choose(rr, structure='impossible', partition='useless', query=query))]))
            controls.extend(r for r in rr if r['factors']['structure'] != 'mixed')
        elif study == 'B':
            # semantic1 = V, so P(U|frontU)-P(U|frontV) has reversed signs.
            for total in range(2):
                for reverse in range(2):
                    estimates[f'total{total}_reverse{reverse}'].append(contrast([
                        (1, choose(rr, allocation='front_V', totals=total, reverse_listing=reverse)),
                        (-1, choose(rr, allocation='front_U', totals=total, reverse_listing=reverse))]))
            controls.extend(choose(rr, allocation='base'))
        elif study == 'C':
            for explicit in range(2):
                estimates[f'selection_likelihood{explicit}'].append(contrast([
                    (1, choose(rr, process='value_filter', likelihoods=explicit)),
                    *[(-1/3, choose(rr, process=p, likelihoods=explicit))
                      for p in ['fixed', 'optional_stop', 'random_filter']]]))
                for p in ['optional_stop', 'random_filter']:
                    estimates[f'{p}_minus_fixed_likelihood{explicit}'].append(contrast([
                        (1, choose(rr, process=p, likelihoods=explicit)),
                        (-1, choose(rr, process='fixed', likelihoods=explicit))]))
            controls.extend(choose(rr, likelihoods=1))
        elif study == 'E':
            estimates['false_global_minus_local'].append(contrast([
                (1, choose(rr, graph='infeasible', reuse=0, query='global')),
                (-1, choose(rr, graph='infeasible', reuse=0, query='marked_union'))]))
            for reuse in range(2):
                estimates[f'feasible_minus_infeasible_reuse{reuse}'].append(contrast([
                    (1, choose(rr, graph='feasible', reuse=reuse, query='global')),
                    (-1, choose(rr, graph='infeasible', reuse=reuse, query='global'))]))
            controls.extend(r for r in rr if r['factors']['query'] == 'global' and
                            (r['factors']['reuse'] == 1 or r['factors']['graph'] == 'feasible'))
        else:
            raise ValueError(study)
        for code in range(2):
            cr = choose(rr, codebook=code)
            if study == 'A':
                cells = [(1, choose(cr, structure='mixed', partition='useless')),
                         (-1, choose(cr, structure='impossible', partition='useless'))]
            elif study == 'B':
                cells = [(1, choose(cr, allocation='front_V', totals=0)),
                         (-1, choose(cr, allocation='front_U', totals=0))]
            elif study == 'C':
                cells = [(1, choose(cr, process='value_filter', likelihoods=0)),
                         *[(-1/3, choose(cr, process=p, likelihoods=0))
                           for p in ['fixed', 'optional_stop', 'random_filter']]]
            else:
                cells = [(1, choose(cr, graph='infeasible', reuse=0, query='global')),
                         (-1, choose(cr, graph='infeasible', reuse=0, query='marked_union'))]
            estimates[f'primary_code{code}'].append(contrast(cells))
    result = {k: bounds_estimate(v, cfg) for k, v in estimates.items()}
    threshold = cfg['practical_behavior_effect']
    readout = major_readout_quality(study, main)
    valid = accuracy(main)['parsed_fraction'] >= cfg['main_parse_min'] and all(
        q['parsed_fraction'] >= cfg['main_parse_min'] for q in readout.values())
    if study == 'A':
        primary = result['false_approval_excess']
        positive = primary['ci95_outer'][0] > threshold and accuracy(controls)['confirmed_correct_fraction'] >= .85
        positive &= all(result['query_'+q]['identified_interval'][0] > 0 for q in ['existence', 'given_policy'])
        positive &= result['useful_minus_useless']['identified_interval'][1] < .80
        negative = primary['ci95_outer'][1] < threshold
    elif study == 'B':
        primary = {k: result[k] for k in ['total0_reverse0', 'total0_reverse1']}
        positive = all(v['ci95_outer'][0] > threshold for v in primary.values())
        front = [r for r in main if r['factors']['allocation'] != 'base']
        positive &= accuracy(choose(front, totals=1))['accuracy_bounds'][0] > accuracy(choose(front, totals=0))['accuracy_bounds'][1]
        negative = any(v['ci95_outer'][1] < threshold for v in primary.values())
    elif study == 'C':
        primary = result['selection_likelihood0']
        invariant = all(-.10 <= result[p+'_minus_fixed_likelihood0']['ci95_outer'][0] and
                        result[p+'_minus_fixed_likelihood0']['ci95_outer'][1] <= .10
                        for p in ['optional_stop', 'random_filter'])
        positive = primary['ci95_outer'][0] > threshold and invariant and accuracy(choose(main, likelihoods=0))['confirmed_correct_fraction'] >= .85
        negative = primary['ci95_outer'][1] < threshold
    else:
        primary = result['false_global_minus_local']
        local = choose(main, query='marked_union')
        positive = primary['ci95_outer'][0] > threshold and accuracy(local)['confirmed_correct_fraction'] >= .85
        positive &= accuracy(controls)['confirmed_correct_fraction'] >= .85
        negative = primary['ci95_outer'][1] < threshold
    positive &= all(result[f'primary_code{code}']['identified_interval'][0] > 0 for code in range(2))
    classification = 'Positive' if positive else 'Negative' if negative else 'Inconclusive'
    return {**base, 'classification': classification if valid else 'Inconclusive',
            'gate': 'P0_MEASURED' if valid else 'STOPPED_AT_FINAL_PARSE', 'estimand_valid': valid,
            'primary': primary if valid else None, 'all_contrasts_diagnostic_if_invalid': result, 'main_quality': accuracy(main),
            'major_factor_quality': readout,
            'internal_control_response_count': sum(bool(r['internal_special_token_ids']) for r in main),
            'thought_marker_response_count': sum(r['thought_marker_present'] for r in main),
            'controls_quality': accuracy(controls), 'independent_families': len(by),
            'answer_code_quality': {c: accuracy([r for r in main if r['expected'] == c]) for c in ['A', 'B']},
            'option_position_quality': {str(p): accuracy([r for r in main if r['gold_display_position'] == p]) for p in [0, 1]},
            'claim_limit': 'Generated-final decisions under a repaired, newly registered protocol; not a rerun of the original estimator or identification of an internal algorithm.'}
