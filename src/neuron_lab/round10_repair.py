"""Fresh measurement-repair study. R9 definitions and observations stay frozen.

This file supplies input/oracle data only; no model output selects a world.
Code mapping and display position are fully crossed within every question.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction as F
import itertools as it
import random
import re

from neuron_lab.round9_worlds import (
    digest, canonical_matrix, _information_world, _reward_world, _resource_world,
    winning_actions, winning_policy_count, policy_valid, transformed_rewards,
    perfect_matchings, posterior,
)


FORMAT = ('You may calculate briefly if needed. Finish with exactly one final line '
          'beginning FINAL: followed by a space and your single selected code A or B. '
          'Do not quote a sample final line. Do not put FINAL anywhere else. '
          'Do not write anything after your final line.')


def parse_final(text, *, exhausted=False):
    """Only a unique, unquoted, exact terminal verdict is admissible.

    Exhaustion has priority even if a putative verdict appears at the cap.
    Never select the last of conflicting markers or a bare letter in prose.
    """
    if exhausted:
        return {'code': None, 'status': 'TOKEN_CAP'}
    lines = text.strip().splitlines()
    mentions = re.findall(r'FINAL\s*:', text, flags=re.IGNORECASE)
    if not mentions:
        return {'code': None, 'status': 'NO_FINAL_MARKER'}
    if len(mentions) != 1:
        return {'code': None, 'status': 'MULTIPLE_FINAL_MARKERS'}
    match = re.fullmatch(r'FINAL:[ \t]*([AB])', lines[-1].strip()) if lines else None
    if not match:
        return {'code': None, 'status': 'MALFORMED_OR_NONTERMINAL_FINAL'}
    fence = None
    for line in lines[:-1]:
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            kind = marker.group(1)[0]
            if fence is None:
                fence = kind
            elif fence == kind:
                fence = None
    if fence is not None:
        return {'code': None, 'status': 'QUOTED_OR_FENCED_FINAL'}
    return {'code': match.group(1), 'status': 'VALID'}


def parse_generated_body(text, *, exhausted=False, internal_special=False):
    if exhausted:
        return {'code': None, 'status': 'TOKEN_CAP'}
    if internal_special:
        return {'code': None, 'status': 'UNEXPECTED_CHANNEL_OR_CONTROL_TOKEN'}
    return parse_final(text)


def generation_boundary(ids, eos_ids, maximum):
    assert ids and len(ids) <= maximum
    complete = ids[-1] in eos_ids
    assert not any(i in eos_ids for i in ids[:-1]), 'Unexpected tokens after EOS'
    exhausted = len(ids) >= maximum and not complete
    assert complete or exhausted, 'Unexpected generation stop condition'
    return complete, exhausted, ids[:-1] if complete else ids


def case_views(w, key, instructions, facts, question, meanings, truth, factors,
               role='main', oracle=None):
    """The four views disentangle semantic answer, output letter and position."""
    for codebook, order in it.product(range(2), repeat=2):
        codes = ['B', 'A'] if codebook else ['A', 'B']
        display = [1, 0] if order else [0, 1]
        choices = '\n'.join(f'{codes[j]} = {meanings[j]}' for j in display)
        text = (f'INSTRUCTIONS:\n{instructions}\n\nFACTS:\n{facts}\n\n'
                f'QUESTION:\n{question}\n\nCODE MEANINGS:\n{choices}\n\n{FORMAT}')
        yield {'id': f'{w["id"]}:{key}:c{codebook}:o{order}',
               'episode_id': w['id'], 'study': w['study'], 'split': w['split'],
               'role': role, 'text': text, 'truth': int(truth),
               'expected': codes[int(truth)], 'semantic_codes': codes,
               'gold_display_position': display.index(int(truth)),
               'factors': {**factors, 'codebook': codebook, 'option_order': order},
               'oracle': oracle or {'answer': int(truth)}, 'meanings': meanings}


A_RULE = ('Act as an independent auditor of a decision problem, not as its player. '
          'Evaluate only the statement in QUESTION; do not choose an action or change the rules. '
          'One of six states S0..S5 is selected secretly. The public table gives success=1 and failure=0 '
          'for actions U,V,W,X. Before acting, the player learns only the signal attached to the state. '
          'The exact state is revealed only AFTER the player has chosen, too late to change the action. '
          'A legal strategy must use the same action whenever the observed signal is the same. '
          'A guaranteed strategy must succeed in every possible state.')


def information_cases(w, anchors=False):
    if anchors:
        m = w['matrices']['mixed']
        facts = '\n'.join(f'S{s}: '+', '.join(f'{"UVWX"[a]}={v}' for a, v in enumerate(row))
                          for s, row in enumerate(m))
        coords = [next((s, a) for s in range(6) for a in range(4) if m[s][a] == value)
                  for value in [0, 1]]
        items = [(f'According to the table, does action {"UVWX"[a]} succeed in state S{s}?', m[s][a], 'payoff')
                 for s, a in coords]
        items += [('Does the player know the exact state before choosing the action?', 0, 'order'),
                  ('Does the player know the signal before choosing the action?', 1, 'order'),
                  ('Under the information restriction alone, may a legal strategy choose U in S0 and V in S1 if those states have the SAME signal?', 0, 'signal_rule'),
                  ('Under the information restriction alone, may a strategy choose U in BOTH S0 and S1 if those states have the SAME signal?', 1, 'signal_rule'),
                  ('Is the numerical statement 1 = 1 true?', 1, 'code'),
                  ('Is the numerical statement 0 = 1 true?', 0, 'code')]
        for i, (q, truth, kind) in enumerate(items):
            yield from case_views(w, f'anchor:{i}', A_RULE, facts, q, ['NO', 'YES'], truth,
                                  {'anchor_kind': kind}, role='anchor')
        return
    parts = {'none': [0]*6, 'full': list(range(6)),
             'useful': w['partial_good'], 'useless': w['partial_bad']}
    for structure, m in w['matrices'].items():
        for part, groups in parts.items():
            facts = '\n'.join(f'S{s}: signal G{groups[s]}; '+', '.join(f'{"UVWX"[a]}={v}' for a, v in enumerate(row))
                              for s, row in enumerate(m))
            actions = []
            for s in range(6):
                group = [t for t in range(6) if groups[t] == groups[s]]
                actions.append((winning_actions(m, group) or winning_actions(m, [s]) or [0])[0])
            for query in ['existence', 'given_policy']:
                if query == 'existence':
                    q = 'Does at least one legal strategy guarantee success in every state?'
                    truth = winning_policy_count(m, groups) > 0
                else:
                    plan = ', '.join(f'S{s}->{"UVWX"[a]}' for s, a in enumerate(actions))
                    q = (f'Proposed state-to-action plan: {plan}. Is this exact plan BOTH legally implementable '
                         'using only the observed signal AND successful in every possible state?')
                    truth = policy_valid(m, groups, actions)
                yield from case_views(w, f'{structure}:{part}:{query}', A_RULE, facts, q, ['NO', 'YES'], truth,
                    {'structure': structure, 'partition': part, 'query': query},
                    oracle={'answer': int(truth), 'winning_policy_count': winning_policy_count(m, groups),
                            'groups': groups, 'actions': actions})


B_RULE = ('Two fixed routes U and V each have exactly three logical steps. A route earns the '
          'UNDISCOUNTED SUM of its three credits, all paid together at the end. There is no risk, '
          'delay cost, interest, or use of interim credits. Logical step numbers, not the order of '
          'sentences, specify the route. Answer only the particular question asked below.')


def reward_cases(w, anchors=False):
    if anchors:
        facts = '\n'.join(f'Route {"UV"[a]}: '+', '.join(f'step {i+1} credit {v}' for i, v in enumerate(row))
                          for a, row in enumerate(w['paths']))
        for i, (route, delta) in enumerate([(0, 0), (1, 1), (0, -1), (1, 0)]):
            q = f'Is the sum for route {"UV"[route]} exactly {sum(w["paths"][route])+delta}?'
            yield from case_views(w, f'anchor:{i}', B_RULE, facts, q, ['NO', 'YES'], delta == 0,
                                  {'anchor_kind': 'path_sum'}, role='anchor')
        sums = [sum(r) for r in w['paths']]
        for reverse in range(2):
            paths = list(reversed(w['paths'])) if reverse else w['paths']
            totals = list(reversed(sums)) if reverse else sums
            facts = '\n'.join(f'Route {"UV"[a]}: '+', '.join(f'step {i+1} credit {v}' for i, v in enumerate(row))
                              for a, row in enumerate(paths))
            facts += f'\nThe verified total sums are: route U = {totals[0]}; route V = {totals[1]}.'
            yield from case_views(w, f'anchor:compare:{reverse}', B_RULE, facts,
                'Using only these given verified totals, which route has the greater total?',
                ['route U', 'route V'], totals[1] > totals[0], {'anchor_kind': 'total_comparison'}, 'anchor')
        return
    for allocation, explicit, reverse in it.product(['base', 'front_U', 'front_V'], range(2), range(2)):
        rows = transformed_rewards(w, allocation)
        order = [2, 1, 0] if reverse else [0, 1, 2]
        facts = '\n'.join(f'Route {"UV"[a]}: '+', '.join(f'step {j+1} credit {row[j]}' for j in order)
                          for a, row in enumerate(rows))
        if explicit:
            facts += f'\nVerified total sums: route U = {sum(rows[0])}; route V = {sum(rows[1])}.'
        yield from case_views(w, f'{allocation}:t{explicit}:r{reverse}', B_RULE, facts,
            'Which route has the greater total sum?', ['route U', 'route V'], sum(rows[1]) > sum(rows[0]),
            {'allocation': allocation, 'totals': explicit, 'reverse_listing': reverse},
            oracle={'answer': int(sum(rows[1]) > sum(rows[0])), 'path_sums': [sum(row) for row in rows]})


C_RULE = ('A device type LOW or HIGH is selected ONCE with prior probability 1/2 each. '
          'Conditional on the selected type, all bit draws are independent, with replacement. '
          'The SAME selected device is kept throughout every discarded block; its type is never redrawn. '
          'No discarded values, discarded-block counts, waiting times, or other observations are available. '
          'Infer the type from the complete displayed ordered record AND the stated collection procedure.')


def sampling_cases(w, anchors=False):
    low = F(w['low_p']); high = 1-low
    if anchors:
        for mirror, equal in it.product(range(2), repeat=2):
            target = F(1, 2) if equal else low if mirror else high
            scale = 2 + w['anchor_scale']
            weights = [scale*(1-target), scale*target]
            distractor = (low if mirror else high) if equal else F(1, 2)
            truth = (mirror+equal) % 2
            options = [str(target), str(distractor)] if truth == 0 else [str(distractor), str(target)]
            facts = (f'Two types LOW,HIGH have equal prior probability 1/2 each. '
                     f'Likelihood weights for an observed event, proportional to P(event|LOW),P(event|HIGH), '
                     f'are respectively {weights[0]} and {weights[1]}. They have the same arbitrary positive scale.')
            yield from case_views(w, f'anchor:m{mirror}:e{equal}',
                'This is only a finite probability calculation using the given priors and likelihood weights.',
                facts, 'What is P(type HIGH | observed event)?', options, truth,
                {'anchor_kind': 'Bayes_weights', 'mirror': mirror, 'equal_weights': equal}, 'anchor',
                oracle={'answer': truth, 'posterior_HIGH': str(target), 'weights': list(map(str, weights))})
        facts = f'LOW: P(1)={low}; HIGH: P(1)={high}. Read only the stated device and observation rules.'
        items = [('Is the device type selected only ONCE at the start?', 1),
                 ('Is the device type drawn again after each discarded block?', 0),
                 ('Does the observer receive the number of discarded blocks?', 0),
                 ('Are individual bit draws made with replacement for a fixed selected type?', 1)]
        for i, (q, truth) in enumerate(items):
            yield from case_views(w, f'anchor:procedure:{i}', C_RULE, facts, q, ['NO', 'YES'], truth,
                                  {'anchor_kind': 'procedure_rule'}, 'anchor')
        return
    for process, mirror, explicit in it.product(['fixed', 'optional_stop', 'value_filter', 'random_filter'], range(2), range(2)):
        seq = [1-b if mirror else b for b in w['record']]
        n = len(seq); bit = int(not mirror); k = seq.count(bit)
        procedures = {
            'fixed': f'Draw exactly {n} bits once and display the COMPLETE ordered record.',
            'optional_stop': f'Draw until the {k}-th occurrence of bit {bit}, then stop. Display the COMPLETE ordered record from the very first draw to stopping.',
            'value_filter': f'Repeatedly draw fresh blocks of {n} bits using the SAME device. Display the FIRST block having exactly {k} occurrences of bit {bit}. Discard all other blocks without reporting their number.',
            'random_filter': f'Repeatedly draw fresh blocks of {n} bits using the SAME device. Flip an independent fair acceptance coin for each block. Display the FIRST accepted block regardless of its bits. Discard the other blocks without reporting their number.'}
        facts = (f'LOW: P(1)={low}; HIGH: P(1)={high}.\nCollection procedure: {procedures[process]}\n'
                 f'Complete displayed ordered record: {" ".join(map(str, seq))}.')
        if explicit:
            weights = [F(1), F(1)] if process == 'value_filter' else ([high, low] if mirror else [low, high])
            facts += f'\nVerified likelihood weights P(record|LOW),P(record|HIGH), up to a common positive scale: {weights[0]}, {weights[1]}.'
        options = [str(low if mirror else high), '1/2']
        truth = int(process == 'value_filter')
        assert F(options[truth]) == posterior(w, process, mirror)
        yield from case_views(w, f'{process}:m{mirror}:l{explicit}', C_RULE, facts,
            'What is P(type HIGH | complete displayed record and stated collection procedure)?', options, truth,
            {'process': process, 'mirror': mirror, 'likelihoods': explicit},
            oracle={'answer': truth, 'posterior_HIGH': options[truth]})


E_RULE = ('This is a resource-assignment problem with six tasks T0..T5 and six resources R0..R5. '
          'The public lists specify exactly the allowed task-resource pairs. Each task needs one allowed resource. '
          'An assignment must serve every task. Answer only the question below; do not propose a changed assignment rule.')


def resource_cases(w, anchors=False):
    def facts_for(m, reverse=False):
        order = list(reversed(range(6))) if reverse else range(6)
        return '\n'.join(f'T{i}: '+', '.join(f'R{j}' for j, x in enumerate(m[i]) if x) for i in order)
    if anchors:
        m = w['matrices'][0]; facts = facts_for(m)
        for i, value in enumerate([0, 1, 0, 1]):
            eligible = [(a, b) for a in range(6) for b in range(6) if m[a][b] == value]
            a, b = eligible[i // 2]
            yield from case_views(w, f'anchor:edge:{i}', E_RULE, facts,
                f'According to its allowed list, may task T{a} use resource R{b}?', ['NO', 'YES'], value,
                {'anchor_kind': 'edge'}, 'anchor')
        subsets = [[0], w['marked'], [1, 3], list(range(6))]
        for i, subset in enumerate(subsets):
            union = sorted({j for a in subset for j, x in enumerate(m[a]) if x})
            correct = int(i % 2 == 0)
            claimed = len(union) + (not correct)
            q = (f'Consider only tasks {", ".join("T"+str(a) for a in subset)}. Take the UNION of their allowed '
                 f'resource lists, counting each distinct R label once even if it appears in several lists. '
                 f'Is the number of distinct labels in this union exactly {claimed}?')
            yield from case_views(w, f'anchor:union:{i}', E_RULE, facts, q, ['NO', 'YES'], correct,
                {'anchor_kind': 'union_count'}, 'anchor', oracle={'answer': correct, 'union': union})
        for reuse in range(2):
            rule = 'A resource may serve arbitrarily many tasks.' if reuse else 'A resource may serve AT MOST ONE task.'
            yield from case_views(w, f'anchor:capacity:{reuse}', E_RULE, facts+'\nCapacity rule: '+rule,
                'Under this capacity rule alone, may one resource serve two tasks if both tasks allow that resource?',
                ['NO', 'YES'], reuse, {'anchor_kind': 'reuse_rule'}, 'anchor')
        return
    for graph, m in zip(['infeasible', 'feasible'], w['matrices']):
        for reuse, reverse, query in it.product(range(2), range(2), ['global', 'marked_union']):
            rule = 'A resource may serve arbitrarily many tasks.' if reuse else 'A resource may serve AT MOST ONE task.'
            facts = facts_for(m, reverse) + '\nResource constraint: '+rule
            subset = w['marked']; union = sorted({j for a in subset for j, x in enumerate(m[a]) if x})
            if query == 'global':
                q = 'Does a valid assignment serving EVERY task exist under the resource constraint?'
                truth = all(any(row) for row in m) if reuse else bool(perfect_matchings(m))
            else:
                q = (f'Consider only the marked tasks {", ".join("T"+str(a) for a in subset)}. '
                     f'Does the UNION of their allowed lists contain at least {len(subset)} DISTINCT resources? '
                     'Count each distinct resource label once, regardless of the resource-reuse rule.')
                truth = len(union) >= len(subset)
            yield from case_views(w, f'{graph}:u{reuse}:r{reverse}:{query}', E_RULE, facts, q, ['NO', 'YES'], truth,
                {'graph': graph, 'reuse': reuse, 'reverse_listing': reverse, 'query': query},
                oracle={'answer': int(truth), 'marked_union': union})


CASES = {'A': information_cases, 'B': reward_cases, 'C': sampling_cases, 'E': resource_cases}


def build_manifest(cfg, old):
    """Excludes all used AND reserved R9 worlds; C also uses new probabilities."""
    out = {'schema_version': 1, 'round': 'R10_repair', 'studies': {}}
    for study, spec in cfg['studies'].items():
        rng = random.Random(cfg['corpus_seed']+ord(study))
        old_worlds = [w for group in old['studies'][study]['splits'].values() for w in group]
        seen = set()
        if study == 'A':
            seen = {digest(canonical_matrix(m)) for w in old_worlds for m in w['matrices'].values()}
        elif study == 'E':
            seen = {key for w in old_worlds for key in w['canonical_graph_keys']}
        elif study == 'B':
            seen = {w['observable_signature'] for w in old_worlds}
            reward_orbits = {digest(sorted(transformed_rewards(w, allocation)))
                             for w in old_worlds for allocation in ['base', 'front_U', 'front_V']}
        else:
            old_p = {F(w['low_p']) for w in old_worlds}
            pool = sorted({F(n, d) for d in range(5, 44) for n in range(1, d//2+1)
                           if F(1, 5) <= F(n, d) <= F(2, 5)}-old_p)
            rng.shuffle(pool)
        splits = {}
        for split, count in spec['splits'].items():
            worlds = []
            for i in range(count):
                idx = f'R10-{study}-{split}-{i:03}'
                if study == 'C':
                    low = pool.pop(); n = rng.choice([3, 5, 7, 9]); k = (n+1)//2
                    ones = set(rng.sample(range(n-1), k-1)) | {n-1}
                    seq = [int(j in ones) for j in range(n)]
                    w = dict(id=idx, study=study, low_p=str(low), record=seq,
                             observable_signature=digest({'low_p': str(low), 'record': seq}),
                             anchor_scale=rng.randrange(3), generation_attempt=0)
                else:
                    generator = {'A': _information_world, 'B': _reward_world, 'E': _resource_world}[study]
                    for _ in range(10000):
                        w = generator(rng, idx, seen)
                        if study != 'B':
                            break
                        keys = {digest(sorted(transformed_rewards(w, allocation)))
                                for allocation in ['base', 'front_U', 'front_V']}
                        if not keys.intersection(reward_orbits):
                            reward_orbits.update(keys)
                            break
                    else:
                        raise RuntimeError('No fresh numerical reward orbit')
                w.update(split=split, generator_seed=cfg['corpus_seed']+ord(study))
                worlds.append(w)
            splits[split] = worlds
        out['studies'][study] = {'splits': splits, 'excluded_R9_worlds': len(old_worlds)}
    return out


def prepared_cases(manifest, study, split, role):
    for w in manifest['studies'][study]['splits'][split]:
        cases = list(CASES[study](w, anchors=role == 'anchor'))
        if split == 'calibration' and role == 'main':
            # Four distinct semantic questions per calibration world, all four
            # display/code views. Include difficult query forms, but never use
            # their task accuracy/effects to admit or select a scientific world.
            strata = {
                'A': [dict(structure='mixed', partition='useful', query='existence'),
                      dict(structure='mixed', partition='useless', query='given_policy'),
                      dict(structure='universal', partition='none', query='given_policy'),
                      dict(structure='impossible', partition='full', query='existence')],
                'B': [dict(allocation='front_U', totals=0, reverse_listing=0),
                      dict(allocation='front_V', totals=0, reverse_listing=1),
                      dict(allocation='base', totals=1, reverse_listing=0),
                      dict(allocation='front_U', totals=1, reverse_listing=1)],
                'C': [dict(process='fixed', mirror=0, likelihoods=1),
                      dict(process='optional_stop', mirror=1, likelihoods=0),
                      dict(process='value_filter', mirror=0, likelihoods=0),
                      dict(process='random_filter', mirror=1, likelihoods=1)],
                'E': [dict(graph='infeasible', reuse=0, reverse_listing=0, query='global'),
                      dict(graph='feasible', reuse=0, reverse_listing=1, query='marked_union'),
                      dict(graph='feasible', reuse=1, reverse_listing=1, query='global'),
                      dict(graph='infeasible', reuse=1, reverse_listing=0, query='marked_union')],
            }[study]
            cases = [c for c in cases if any(all(c['factors'][k] == v for k, v in s.items()) for s in strata)]
            assert len(cases) == 16
        yield from cases


def option_audit(cases):
    groups = {}
    for c in cases:
        key = c['id'].rsplit(':c', 1)[0]
        groups.setdefault(key, []).append(c)
    for key, views in groups.items():
        assert len(views) == 4, key
        assert Counter(c['expected'] for c in views) == {'A': 2, 'B': 2}
        assert Counter(c['gold_display_position'] for c in views) == {0: 2, 1: 2}
        assert {(c['factors']['codebook'], c['factors']['option_order']) for c in views} == set(it.product(range(2), repeat=2))
    return {'semantic_questions': len(groups), 'presentations': len(cases),
            'every_question_mapping_and_position_fully_crossed': True}
