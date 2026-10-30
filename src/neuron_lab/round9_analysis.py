"""Frozen exploratory R9 analyses; never chooses inputs or a new hypothesis."""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction
from itertools import combinations, product
import numpy as np


def estimate(values, cfg):
    x = np.asarray(values, dtype=float)
    assert len(x) and np.isfinite(x).all()
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    means = x[rng.integers(0, len(x), (cfg['bootstrap_resamples'], len(x)))].mean(1)
    return {'point': float(x.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist(),
            'ci90': np.quantile(means, [.05, .95]).tolist(), 'families': len(x),
            'unit': 'family', 'scope': 'exploratory finite-generator bootstrap; zero event endpoints do not prove population equivalence'}


def mean(rows, key):
    assert rows, 'Empty registered cell'
    return float(np.mean([r[key] for r in rows]))


def select(rows, **factors):
    return [r for r in rows if all(r['factors'].get(k) == v for k, v in factors.items())]


def anchor_quality(rows, cfg):
    groups = defaultdict(list)
    for r in rows:
        groups[r['factors']['anchor_kind']].append(r)
    counts = {k: {'correct': sum(r['full_correct'] for r in v), 'n': len(v),
                  'accuracy': mean(v, 'full_correct'), 'format_valid': mean(v, 'format_valid')}
              for k, v in groups.items()}
    return {'passed': all(v['accuracy'] >= (cfg['code_anchor_accuracy_min'] if k == 'code' else cfg['anchor_accuracy_min'])
                          for k, v in counts.items()), 'groups': counts,
            'criterion': 'full-vocabulary first token equals the exact correct answer code; restricted scores never rescue invalid output'}


def information_analysis(families, cfg):
    effects = []; margins = []; discriminations = []; controls = []; details = []
    byquery = defaultdict(list); bycode = defaultdict(list)
    for fid, rows in families.items():
        bad = select(rows, structure='mixed', partition='useless')
        impossible = select(rows, structure='impossible', partition='useless')
        good = select(rows, structure='mixed', partition='useful')
        theta = mean(bad, 'semantic_yes') - mean(impossible, 'semantic_yes')
        effects.append(theta); margins.append(mean(bad, 'semantic_logit_difference') - mean(impossible, 'semantic_logit_difference'))
        discriminations.append(mean(good, 'semantic_yes') - mean(bad, 'semantic_yes'))
        control = [r for r in rows if r['factors']['structure'] != 'mixed']
        controls.append(mean(control, 'full_correct'))
        for q in ['existence', 'given_policy']:
            byquery[q].append(mean(select(bad, query=q), 'semantic_yes') - mean(select(impossible, query=q), 'semantic_yes'))
        for c in [0, 1]:
            bycode[str(c)].append(mean(select(bad, codebook=c), 'semantic_yes') - mean(select(impossible, codebook=c), 'semantic_yes'))
        details.append({'family': fid, 'mixed_useless_false_approval': mean(bad, 'semantic_yes'),
                        'mixed_useful_approval': mean(good, 'semantic_yes'), 'control_accuracy': controls[-1]})
    effect, margin = estimate(effects, cfg), estimate(margins, cfg)
    concordant = all(np.mean(v) > 0 for v in [*byquery.values(), *bycode.values()])
    positive = (effect['point'] >= .1 and margin['ci95'][0] > .1 and np.mean(discriminations) < .8
                and np.mean(controls) >= .85 and concordant)
    negative = effect['ci95'][1] < .1
    return {'classification': 'Positive' if positive else 'Negative' if negative else 'Inconclusive',
            'false_approval_excess': effect, 'YES_margin_excess_nats': margin,
            'compatible_minus_incompatible_approval': estimate(discriminations, cfg),
            'control_accuracy': float(np.mean(controls)), 'query_contrasts': {k: estimate(v, cfg) for k, v in byquery.items()},
            'code_contrasts': {k: estimate(v, cfg) for k, v in bycode.items()}, 'family_details': details,
            'claim_limit': 'Input-partition effect or selective strategy-evaluation error; no unique latent strategy-fusion circuit identified.'}


def reward_analysis(families, cfg):
    behavior = defaultdict(list); margin = defaultdict(list); errors = defaultdict(list); details = []
    for fid, rows in families.items():
        for totals, reverse in product(range(2), repeat=2):
            u = select(rows, allocation='front_U', totals=totals, reverse_listing=reverse)
            v = select(rows, allocation='front_V', totals=totals, reverse_listing=reverse)
            key = f'totals{totals}_reverse{reverse}'
            behavior[key].append(mean(v, 'semantic_one') - mean(u, 'semantic_one'))  # P(U|frontU)-P(U|frontV)
            margin[key].append(mean(v, 'semantic_logit_difference') - mean(u, 'semantic_logit_difference'))
            errors[str(totals)].append(1 - mean(u + v, 'full_correct'))
        details.append({'family': fid, 'accuracy': mean(rows, 'full_correct')})
    be = {k: estimate(v, cfg) for k, v in behavior.items()}; ma = {k: estimate(v, cfg) for k, v in margin.items()}
    keys = ['totals0_reverse0', 'totals0_reverse1']
    reduction = np.mean(errors['1']) < np.mean(errors['0'])
    positive = all(be[k]['point'] >= .1 and ma[k]['ci95'][0] > .1 for k in keys) and reduction
    negative = any(be[k]['ci95'][1] < .1 for k in keys)
    return {'classification': 'Positive' if positive else 'Negative' if negative else 'Inconclusive',
            'directional_U_choice_effect': be, 'directional_U_margin_effect_nats': ma,
            'error_rate_without_totals': float(np.mean(errors['0'])), 'error_rate_with_totals': float(np.mean(errors['1'])),
            'family_details': details, 'claim_limit': 'All path sums are fixed. A logical-time effect is not unique evidence for discounting or shallow search.'}


def sampling_analysis(families, cfg):
    effects = defaultdict(list); invariants = defaultdict(list); accuracy = defaultdict(list); details = []
    for fid, rows in families.items():
        for explicit in range(2):
            chosen = select(rows, likelihoods=explicit)
            selected = select(chosen, process='value_filter')
            inv = [r for r in chosen if r['factors']['process'] != 'value_filter']
            effects[str(explicit)].append(mean(selected, 'semantic_one') - mean(inv, 'semantic_one'))
            fixed = select(chosen, process='fixed')
            for process in ['optional_stop', 'random_filter']:
                invariants[f'likelihood{explicit}_{process}'].append(mean(select(chosen, process=process), 'semantic_one') - mean(fixed, 'semantic_one'))
            for code in range(2):
                accuracy[f'likelihood{explicit}_code{code}'].append(mean(select(chosen, codebook=code), 'full_correct'))
        details.append({'family': fid, 'accuracy': mean(rows, 'full_correct')})
    ef = {k: estimate(v, cfg) for k, v in effects.items()}; iv = {k: estimate(v, cfg) for k, v in invariants.items()}
    ac = {k: float(np.mean(v)) for k, v in accuracy.items()}
    stable = all(abs(v['point']) <= .1 for k, v in iv.items() if k.startswith('likelihood0'))
    positive = ef['0']['ci95'][0] > .1 and stable and all(ac[f'likelihood0_code{c}'] >= .85 for c in range(2))
    negative = ef['0']['ci95'][1] < .1 or max(abs(v['point']) for k, v in iv.items() if k.startswith('likelihood0')) >= ef['0']['point']
    return {'classification': 'Positive' if positive else 'Negative' if negative else 'Inconclusive',
            'selective_prior_choice_contrast': ef, 'likelihood_preserving_arm_choice_differences': iv,
            'accuracy_by_likelihood_and_code': ac, 'invariance_P0_screen_passed': stable,
            'equivalence_status': 'DESCRIPTIVE_P0_SCREEN_ONLY; not a confirmed population equivalence result',
            'family_details': details, 'claim_limit': 'Exact fraction decisions under specified procedures; code softmax is not a subjective posterior.'}


def joint_from_prefixes(rows):
    nodes = {r['factors']['prefix']: r for r in rows}
    assert set(nodes) == {'', '0', '1', '00', '01', '10', '11'}
    joint = {}
    for bits in product('01', repeat=3):
        prefix = ''; p = 1.
        for bit in bits:
            p *= nodes[prefix]['restricted_probabilities'][int(bit)]; prefix += bit
        joint[prefix] = p
    assert abs(sum(joint.values()) - 1) < 1e-10
    weighted_mass = 0.
    for prefix, row in nodes.items():
        weight = sum(p for s, p in joint.items() if s.startswith(prefix))
        weighted_mass += weight * row['valid_probability_mass'] / 3
    return joint, weighted_mass


def low_order_error(joint):
    errors = []
    for k in [1, 2]:
        for axes in combinations(range(3), k):
            for values in product('01', repeat=k):
                probability = sum(p for s, p in joint.items() if all(s[a] == v for a, v in zip(axes, values)))
                errors.append(abs(probability - 1 / (2 ** k)))
    return max(errors)


def higher_order_analysis(families, reference_rows, cfg):
    details = []; selective = 0; condition_details = []
    for fid, rows in families.items():
        conditions = defaultdict(list)
        for r in rows:
            f = r['factors']; conditions[(f['q'], f['representation'], f['codebook'])].append(r)
        qualifies = []
        for (q, rep, code), v in conditions.items():
            joint, valid_mass = joint_from_prefixes(v)
            oracle = {s: float(Fraction(p)) for s, p in v[0]['oracle']['encoded_joint'].items()}
            actual_even = sum(p for s, p in joint.items() if sum(map(int, s)) % 2 == 0)
            target_even = sum(p for s, p in oracle.items() if sum(map(int, s)) % 2 == 0)
            low, high = low_order_error(joint), abs(actual_even - target_even)
            ok = low <= .05 and high >= .15 and valid_mass >= .8
            qualifies.append(ok)
            condition_details.append({'family': fid, 'q': q, 'representation': rep, 'codebook': code,
                'low_order_max_error': low, 'parity_mass_error': high, 'grammar_weighted_valid_mass': valid_mass,
                'joint': joint, 'target_joint': oracle, 'selective_pattern': ok})
        assert len(qualifies) == 8
        qualifies_family = all(qualifies); selective += qualifies_family
        details.append({'family': fid, 'all_eight_conditions_selective': qualifies_family})
    refs = []
    for rep, code in product(['rule', 'table'], range(2)):
        v = [r for r in reference_rows if r['factors']['representation'] == rep and r['factors']['codebook'] == code]
        joint, mass = joint_from_prefixes(v)
        refs.append({'representation': rep, 'codebook': code, 'low_order_error': low_order_error(joint), 'weighted_valid_mass': mass})
    return {'classification': 'Positive' if selective >= 6 else 'Negative',
            'selective_q_families': selective, 'q_families': len(families), 'conditions': condition_details,
            'family_details': details, 'uniform_reference': refs,
            'claim_limit': 'Exact temperature1 restricted binary-grammar decoder distribution; no inference about free generation or internal randomness.'}


def resource_analysis(families, cfg):
    effects = []; local = []; controls = []; details = []; bycode = defaultdict(list)
    for fid, rows in families.items():
        global_bad = select(rows, graph='infeasible', reuse=0, query='global')
        local_bad = select(rows, graph='infeasible', reuse=0, query='marked_union')
        effects.append(mean(global_bad, 'semantic_yes') - mean(local_bad, 'semantic_yes'))
        local.append(mean(select(rows, query='marked_union'), 'full_correct'))
        control = [r for r in rows if r['factors']['query'] == 'global' and (r['factors']['graph'] == 'feasible' or r['factors']['reuse'] == 1)]
        controls.append(mean(control, 'full_correct'))
        for c in range(2):
            bycode[str(c)].append(mean(select(global_bad, codebook=c), 'semantic_yes') - mean(select(local_bad, codebook=c), 'semantic_yes'))
        details.append({'family': fid, 'false_global_approval': mean(global_bad, 'semantic_yes'), 'marked_union_accuracy': local[-1]})
    ef = estimate(effects, cfg)
    positive = ef['ci95'][0] > .1 and np.mean(local) >= .85 and np.mean(controls) >= .85 and all(np.mean(v) > 0 for v in bycode.values())
    negative = ef['ci95'][1] < .1 or np.mean(local) < .85 or np.mean(controls) < .85
    return {'classification': 'Positive' if positive else 'Negative' if negative else 'Inconclusive',
            'false_global_minus_local_approval': ef, 'marked_union_accuracy': float(np.mean(local)),
            'feasible_or_reusable_control_accuracy': float(np.mean(controls)),
            'code_contrasts': {k: estimate(v, cfg) for k, v in bycode.items()}, 'family_details': details,
            'claim_limit': 'Selective response to degree-preserving rewiring under resource exclusion; generic matching errors alone are not novelty.'}


ANALYSES = {'A': information_analysis, 'B': reward_analysis, 'C': sampling_analysis, 'E': resource_analysis}


def analyze(study, rows, cfg):
    anchor = [r for r in rows if r['role'] == 'anchor']; main = [r for r in rows if r['role'] == 'main']
    quality = anchor_quality(anchor, cfg)
    if not quality['passed']:
        assert not main
        return {'gate': 'STOPPED_AT_BASIC_ANCHORS', 'classification': 'Inconclusive', 'identifiability': 'NOT_IDENTIFIABLE',
                'readiness': quality, 'main': 'NOT_EVALUATED', 'paper_verdict': 'No-Go at current P0'}
    allmain = [r for r in rows if r['role'] != 'anchor']
    assert len(allmain) == cfg['studies'][study]['main']
    format_rate = mean(allmain, 'format_valid')
    families = defaultdict(list)
    for r in main:
        families[r['episode_id']].append(r)
    out = (higher_order_analysis(families, [r for r in rows if r['role'] == 'reference'], cfg) if study == 'D'
           else ANALYSES[study](families, cfg))
    out.update(readiness=quality, full_top1_format_rate=format_rate, main_rows=len(allmain), independent_families=len(families))
    if format_rate < cfg['main_full_top1_format_min']:
        out.update(gate='STOPPED_AT_FULL_VOCABULARY_FORMAT', classification='Inconclusive', identifiability='NOT_IDENTIFIABLE')
    else:
        out.update(gate='P0_MEASURED', identifiability='INPUT_INTERVENTION_ONLY')
    out['paper_verdict'] = 'Hold; requires fresh predictive P1/P2 and novelty adjudication' if out['classification'] == 'Positive' else 'No-Go at current P0'
    return out
