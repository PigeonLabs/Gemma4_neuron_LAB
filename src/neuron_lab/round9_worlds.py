"""Fresh R9 input interventions and exact CPU oracles (not model results).

All codebooks, renderings and questions are repeated observations of a family.
The generators reject observable duplicates before any model output is seen.
"""
from __future__ import annotations

import hashlib
import itertools as it
import json
import random
from fractions import Fraction as F


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def frame(instructions, facts, question):
    return f'CUE_BEGIN:\n{instructions}\nFACTS_BEGIN:\n{facts}\nEND_FACTS\nNEW_QUERY:\n{question}'


def binary_case(world, suffix, instructions, facts, question, truth, swap, factors, role='main'):
    codes = ['B', 'A'] if swap else ['A', 'B']
    text = frame(instructions, facts, question + f'\nAnswer only one code. {codes[1]} = YES; {codes[0]} = NO.')
    return {'id': world['id'] + ':' + suffix, 'episode_id': world['id'], 'study': world['study'],
            'role': role, 'text': text, 'expected': codes[int(bool(truth))],
            'semantic_codes': codes, 'truth': int(bool(truth)), 'prefix': '',
            'factors': factors, 'oracle': {'answer': int(bool(truth))}}


def canonical_matrix(matrix):
    """Exact bipartite isomorphism key: permute columns, sort row multiset."""
    n = len(matrix[0])
    return min(tuple(sorted(tuple(row[j] for j in p) for row in matrix)) for p in it.permutations(range(n)))


def winning_actions(matrix, states):
    return [a for a in range(len(matrix[0])) if all(matrix[s][a] for s in states)]


def winning_policy_count(matrix, groups):
    count = 1
    for g in sorted(set(groups)):
        count *= len(winning_actions(matrix, [s for s, x in enumerate(groups) if x == g]))
    return count


def policy_valid(matrix, groups, actions):
    return all(matrix[s][actions[s]] for s in range(len(matrix))) and all(
        actions[s] == actions[t] for s in range(len(matrix)) for t in range(len(matrix)) if groups[s] == groups[t])


def _partitions(n):
    for rest in it.combinations(range(1, n), n // 2 - 1):
        left = {0, *rest}
        yield [0 if s in left else 1 for s in range(n)]


def _information_world(rng, idx, seen):
    for attempt in range(10000):
        m = [[int(rng.random() < .48) for _ in range(4)] for _ in range(6)]
        if not all(any(row) for row in m) or winning_actions(m, range(6)):
            continue
        parts = list(_partitions(6))
        good = [p for p in parts if winning_policy_count(m, p)]
        bad = [p for p in parts if not winning_policy_count(m, p)]
        if not good or not bad:
            continue
        goodp, badp = rng.choice(good), rng.choice(bad)
        universal = [r[:] for r in m]
        col = rng.randrange(4)
        for r in universal:
            r[col] = 1
        impossible = [r[:] for r in m]
        impossible[rng.randrange(6)] = [0] * 4
        variants = {'mixed': m, 'universal': universal, 'impossible': impossible}
        keys = [digest(canonical_matrix(x)) for x in variants.values()]
        if len(set(keys)) != 3 or any(k in seen for k in keys):
            continue
        seen.update(keys)
        return {'id': idx, 'study': 'A', 'matrices': variants, 'partial_good': goodp, 'partial_bad': badp,
                'observable_signature': digest({'variants': sorted(keys), 'good': goodp, 'bad': badp}),
                'generation_attempt': attempt}
    raise RuntimeError('No fresh information-partition world within generator budget')


def information_cases(w, anchors=False):
    instructions = ('One state S0..S5 is chosen secretly. You must choose exactly one action U,V,W,X. '
        'The table is public and lists success (1) or failure (0). Only the signal is revealed BEFORE choosing. '
        'The exact state is revealed AFTER choosing, too late to change the action. '
        'A legal strategy must choose the same action for states with the same observed signal. '
        'Guaranteed success means success for every possible state.')
    if anchors:
        m = w['matrices']['mixed']
        facts = '\n'.join(f'S{s}: ' + ' '.join(f'{"UVWX"[a]}={v}' for a, v in enumerate(row)) for s, row in enumerate(m))
        queries = [(f'In state S{s}, does action {"UVWX"[a]} succeed?', bool(m[s][a]), 'payoff')
                   for s, a in [(0, 0), (2, 1), (4, 2), (5, 3)]]
        queries += [('Is the exact state known before the action is chosen?', False, 'order'),
                    ('Is the signal known before the action is chosen?', True, 'order'),
                    ('For this code check, is 1 equal to 1?', True, 'code'),
                    ('For this code check, is 0 equal to 1?', False, 'code')]
        for i, (q, truth, kind) in enumerate(queries):
            yield binary_case(w, f'anchor:{i}', instructions, facts, q, truth, (i // 2) % 2, {'anchor_kind': kind}, 'anchor')
        return
    parts = {'none': [0] * 6, 'full': list(range(6)), 'useful': w['partial_good'], 'useless': w['partial_bad']}
    for structure, m in w['matrices'].items():
        for part, groups in parts.items():
            actions = []
            for s in range(6):
                group = [t for t in range(6) if groups[t] == groups[s]]
                group_winners = winning_actions(m, group)
                single = winning_actions(m, [s])
                actions.append((group_winners or single or [0])[0])
            facts = '\n'.join(f'S{s}: signal G{groups[s]}; ' + ' '.join(f'{"UVWX"[a]}={v}' for a, v in enumerate(row)) for s, row in enumerate(m))
            policy = ', '.join(f'S{s}->{"UVWX"[a]}' for s, a in enumerate(actions))
            for query in ['existence', 'given_policy']:
                if query == 'existence':
                    q = 'Does any legal strategy guarantee success in every state?'
                    truth = winning_policy_count(m, groups) > 0
                else:
                    q = f'Proposed state-to-action plan: {policy}. Can this exact plan be legally implemented using the available signal AND succeed in every state?'
                    truth = policy_valid(m, groups, actions)
                for swap in range(2):
                    yield binary_case(w, f'{structure}:{part}:{query}:c{swap}', instructions, facts, q, truth, swap,
                                      {'structure': structure, 'partition': part, 'query': query, 'codebook': swap,
                                       'group_sizes': sorted(groups.count(g) for g in set(groups)),
                                       'winning_policy_count': winning_policy_count(m, groups)})


def _reward_world(rng, idx, seen):
    for attempt in range(10000):
        x = [rng.randrange(5, 16) for _ in range(3)]
        y = [rng.randrange(5, 16) for _ in range(2)]
        y.append(sum(x) + rng.choice([-3, -2, -1, 1, 2, 3]) - sum(y))
        if not 5 <= y[-1] <= 15:
            continue
        k = rng.randrange(2, min(x[0], x[-1], y[0], y[-1]))
        key = digest({'paths': sorted([x, y]), 'shift': k})
        if key in seen:
            continue
        seen.add(key)
        return {'id': idx, 'study': 'B', 'paths': [x, y], 'shift': k, 'observable_signature': key,
                'generation_attempt': attempt}
    raise RuntimeError('Reward generator budget')


def transformed_rewards(w, allocation):
    k = {'base': 0, 'front_U': w['shift'], 'front_V': -w['shift']}[allocation]
    phi = [[0, k, k, 0], [0, -k, -k, 0]]
    return [[v + p[i + 1] - p[i] for i, v in enumerate(row)] for row, p in zip(w['paths'], phi)]


def reward_cases(w, anchors=False):
    instructions = ('Choose route U or route V to maximize the UNDISCOUNTED sum of its three edge credits. '
        'All credits are paid together only at the end. There is no risk, delay cost, interest, or use of interim credits. '
        'Both routes have exactly three edges, in logical steps 1,2,3. Text listing order does not change logical step numbers.')
    basefacts = '\n'.join(f'Route {"UV"[a]}: ' + ', '.join(f'step {i + 1} credit {v}' for i, v in enumerate(row)) for a, row in enumerate(w['paths']))
    if anchors:
        for i, (a, delta) in enumerate([(0, 0), (1, 1), (0, -1), (1, 0)]):
            truth = delta == 0
            q = f'Does route {"UV"[a]} have total credit {sum(w["paths"][a]) + delta}?'
            yield binary_case(w, f'anchor:{i}', instructions, basefacts, q, truth, i % 2, {'anchor_kind': 'path_sum'}, 'anchor')
        return
    for allocation in ['base', 'front_U', 'front_V']:
        rows = transformed_rewards(w, allocation)
        assert all(sum(x) == sum(y) for x, y in zip(rows, w['paths'])) and min(sum(rows, [])) >= 0
        for explicit in range(2):
            for reverse in range(2):
                order = [2, 1, 0] if reverse else [0, 1, 2]
                facts = '\n'.join(f'Route {"UV"[a]}: ' + ', '.join(f'step {i + 1} credit {row[i]}' for i in order) for a, row in enumerate(rows))
                if explicit:
                    facts += '\nVerified totals: ' + '; '.join(f'route {"UV"[a]} = {sum(row)}' for a, row in enumerate(rows)) + '.'
                for swap in range(2):
                    codes = ['B', 'A'] if swap else ['A', 'B']
                    q = f'Which route has greater total credit? Answer only one code. {codes[0]} = route U; {codes[1]} = route V.'
                    expected = int(sum(rows[1]) > sum(rows[0]))
                    yield {'id': w['id'] + f':{allocation}:t{explicit}:r{reverse}:c{swap}', 'episode_id': w['id'], 'study': 'B',
                           'role': 'main', 'text': frame(instructions, facts, q), 'expected': codes[expected], 'truth': expected,
                           'semantic_codes': codes, 'prefix': '', 'factors': {'allocation': allocation, 'totals': explicit,
                               'reverse_listing': reverse, 'codebook': swap}, 'oracle': {'path_sums': [sum(x) for x in rows],
                               'potentials_endpoints': [0, 0], 'answer': expected}}


def _sampling_world(rng, idx, seen):
    for attempt in range(10000):
        p = rng.choice([F(1, 5), F(1, 4), F(1, 3), F(2, 5)])
        n = rng.choice([3, 5, 7, 9])
        k = (n + 1) // 2
        ones = set(rng.sample(range(n - 1), k - 1)) | {n - 1}
        seq = [int(i in ones) for i in range(n)]
        key = digest({'low_p': str(p), 'record': seq})
        if key in seen:
            continue
        seen.add(key)
        return {'id': idx, 'study': 'C', 'low_p': str(p), 'record': seq, 'observable_signature': key,
                'generation_attempt': attempt}
    raise RuntimeError('Sampling generator budget')


def posterior(w, process, mirror=0):
    low = F(w['low_p']); seq = [1 - b if mirror else b for b in w['record']]
    n, k = len(seq), sum(seq)
    if process == 'value_filter':
        weights = [F(1), F(1)]
    else:
        weights = [p ** k * (1 - p) ** (n - k) for p in [low, 1 - low]]
    return weights[1] / sum(weights)


def sampling_cases(w, anchors=False):
    low = F(w['low_p']); high = 1 - low
    if anchors:
        for i, mirror in enumerate([0, 1]):
            for process in ['fixed', 'value_filter']:
                seq = [1 - b if mirror else b for b in w['record']]
                weights = [F(1), F(1)] if process == 'value_filter' else ([high, low] if mirror else [low, high])
                facts = f'Type LOW has P(1)={low}; type HIGH has P(1)={high}. Prior probability of each type is 1/2.\nRecord: {" ".join(map(str,seq))}.\nLikelihood weights for this observed record, proportional to its probabilities under LOW,HIGH: {weights[0]}, {weights[1]}.'
                target = posterior(w, process, mirror)
                other = F(1, 2) if target != F(1, 2) else (low if mirror else high)
                swap = (i + int(process == 'fixed')) % 2
                codes = ['B', 'A'] if swap else ['A', 'B']
                q = f'Using the given prior and likelihood weights, what is P(type HIGH | record)? {codes[0]} = {target}; {codes[1]} = {other}. Answer only one code.'
                yield {'id': w['id'] + f':anchor:{mirror}:{process}', 'episode_id': w['id'], 'study': 'C', 'role': 'anchor',
                       'text': frame('This is a finite probability calculation. Both weights have the same arbitrary positive scale.', facts, q),
                       'expected': codes[0], 'truth': 0, 'semantic_codes': codes, 'prefix': '',
                       'factors': {'anchor_kind': 'Bayes_weights', 'process': process, 'mirror': mirror},
                       'oracle': {'posterior_HIGH': str(target)}}
        return
    instructions = ('A device type LOW or HIGH is chosen once, with equal prior probability 1/2 each. '
        'Conditional on its type, all bit draws are independent with replacement. '
        'The SAME chosen device is used throughout any discarded blocks; it is not redrawn. '
        'No discarded results, counts, waiting times, or other measurements are given. '
        'Infer the probability of type HIGH conditional on the displayed record AND the stated collection procedure.')
    for process in ['fixed', 'optional_stop', 'value_filter', 'random_filter']:
        for mirror in range(2):
            seq = [1 - b if mirror else b for b in w['record']]
            n = len(seq); bit = 0 if mirror else 1; k = seq.count(bit)
            procedures = {
                'fixed': f'Draw exactly {n} bits once and display the complete ordered record.',
                'optional_stop': f'Draw bits until the {k}-th occurrence of bit {bit}, then stop. Display the COMPLETE ordered record from the very first draw to stopping.',
                'value_filter': f'Repeatedly draw fresh blocks of {n} bits. Retain and display the FIRST block containing exactly {k} occurrences of bit {bit}. Discard all other blocks without reporting their number.',
                'random_filter': f'Repeatedly draw fresh blocks of {n} bits. Independently flip a fair acceptance coin for each block and display the FIRST accepted block, regardless of its contents. Discard other blocks without reporting their number.'}
            for explicit in range(2):
                facts = f'LOW: P(1)={low}. HIGH: P(1)={high}.\nCollection procedure: {procedures[process]}\nDisplayed ordered record: {" ".join(map(str,seq))}.'
                if explicit:
                    weights = [F(1), F(1)] if process == 'value_filter' else ([high, low] if mirror else [low, high])
                    facts += f'\nVerified likelihood weights for this complete observed record, proportional to P(record | LOW), P(record | HIGH): {weights[0]}, {weights[1]}.'
                updated = low if mirror else high
                options = [updated, F(1, 2)]
                truth = int(process == 'value_filter')
                assert options[truth] == posterior(w, process, mirror)
                for swap in range(2):
                    codes = ['B', 'A'] if swap else ['A', 'B']
                    q = f'What is P(type HIGH | displayed record and collection procedure)? {codes[0]} = {options[0]}; {codes[1]} = {options[1]}. Answer only one code.'
                    yield {'id': w['id'] + f':{process}:m{mirror}:l{explicit}:c{swap}', 'episode_id': w['id'], 'study': 'C', 'role': 'main',
                           'text': frame(instructions, facts, q), 'expected': codes[truth], 'truth': truth, 'semantic_codes': codes, 'prefix': '',
                           'factors': {'process': process, 'mirror': mirror, 'likelihoods': explicit, 'codebook': swap},
                           'oracle': {'posterior_HIGH': str(options[truth]), 'answer': truth}}


def perfect_matchings(matrix):
    return [p for p in it.permutations(range(len(matrix))) if all(matrix[i][p[i]] for i in range(len(matrix)))]


def hall_violations(matrix):
    n = len(matrix); out = []
    for k in range(1, n + 1):
        for subset in it.combinations(range(n), k):
            union = {j for i in subset for j, x in enumerate(matrix[i]) if x}
            if len(union) < len(subset):
                out.append((subset, sorted(union)))
    return out


def _resource_world(rng, idx, seen):
    for attempt in range(10000):
        n = 6
        bad = [[int(j in [0, 1]) for j in range(n)] for _ in range(3)]
        for _ in range(3):
            neighbors = rng.sample(range(n), rng.randrange(3, 6))
            bad.append([int(j in neighbors) for j in range(n)])
        if not all(any(bad[i][j] for i in range(n)) for j in range(n)) or perfect_matchings(bad):
            continue
        candidates = []
        for a, b, x, y in it.product(range(3), range(3, n), range(2), range(2, n)):
            if bad[a][x] and bad[b][y] and not bad[a][y] and not bad[b][x]:
                good = [r[:] for r in bad]
                good[a][x] = good[b][y] = 0; good[a][y] = good[b][x] = 1
                if perfect_matchings(good):
                    candidates.append((good, [a, b, x, y]))
        if not candidates:
            continue
        good, switch = rng.choice(candidates)
        keys = [digest(canonical_matrix(m)) for m in [bad, good]]
        if any(k in seen for k in keys):
            continue
        seen.update(keys)
        # Shuffle both node lists together across the pair, not just as new families.
        rp, cp = list(range(n)), list(range(n)); rng.shuffle(rp); rng.shuffle(cp)
        mats = [[[m[i][j] for j in cp] for i in rp] for m in [bad, good]]
        marked = [i for i, old in enumerate(rp) if old < 3]
        return {'id': idx, 'study': 'E', 'matrices': mats, 'marked': marked,
                'observable_signature': digest(sorted(keys)), 'canonical_graph_keys': keys,
                'generation_attempt': attempt, 'construction_switch': switch}
    raise RuntimeError('Resource generator cannot supply enough non-isomorphic pairs')


def resource_cases(w, anchors=False):
    instructions = ('Each task T0..T5 needs one allowed resource R0..R5. The public table lists all allowed pairs. '
        'An assignment must serve EVERY task. Resources have no other properties.')
    def facts_for(m, reverse=False):
        tasks = list(range(6)); tasks = tasks[::-1] if reverse else tasks
        return '\n'.join(f'T{i}: ' + ', '.join(f'R{j}' for j, x in enumerate(m[i]) if x) for i in tasks)
    if anchors:
        m = w['matrices'][0]; facts = facts_for(m)
        for i, (a, b) in enumerate([(0, 0), (1, 4), (2, 1), (3, 3)]):
            yield binary_case(w, f'anchor:edge:{i}', instructions, facts, f'Is resource R{b} allowed for task T{a}?', bool(m[a][b]), i % 2, {'anchor_kind': 'edge'}, 'anchor')
        for i, subset in enumerate([[0], [1, 3], [2, 4, 5], [0, 1, 2, 3, 4, 5]]):
            union = {j for a in subset for j, x in enumerate(m[a]) if x}
            claim = len(union) + (i % 2)
            yield binary_case(w, f'anchor:union:{i}', instructions, facts, f'For tasks {subset}, is the number of DISTINCT resources in the union of their allowed lists exactly {claim}?', i % 2 == 0, (i // 2) % 2, {'anchor_kind': 'union_count'}, 'anchor')
        return
    for graph, m in zip(['infeasible', 'feasible'], w['matrices']):
        for reuse in range(2):
            constraint = ('A resource can be reused for arbitrarily many tasks.' if reuse else 'Each resource may serve AT MOST ONE task.')
            for reverse in range(2):
                facts = facts_for(m, reverse) + '\nResource constraint: ' + constraint
                subset = w['marked']; union = {j for a in subset for j, x in enumerate(m[a]) if x}
                for query in ['global', 'marked_union']:
                    if query == 'global':
                        q = 'Does a valid assignment serving every task exist?'
                        truth = all(any(r) for r in m) if reuse else bool(perfect_matchings(m))
                    else:
                        q = f'For the marked tasks {", ".join("T"+str(i) for i in subset)}, does the union of their allowed resource lists contain at least {len(subset)} DISTINCT resources? Count the union regardless of whether reuse is allowed.'
                        truth = len(union) >= len(subset)
                    for swap in range(2):
                        yield binary_case(w, f'{graph}:u{reuse}:r{reverse}:{query}:c{swap}', instructions, facts, q, truth, swap,
                            {'graph': graph, 'reuse': reuse, 'reverse_listing': reverse, 'query': query, 'codebook': swap,
                             'marked_union_size': len(union)})


def higher_order_distribution(q):
    q = F(q)
    return {''.join(map(str, bits)): (q if sum(bits) % 2 == 0 else 1 - q) / 4 for bits in it.product([0, 1], repeat=3)}


def higher_order_cases(w, anchors=False):
    if anchors:
        # Eight truth-table checks per family; no sampling performance is gated away.
        instructions = 'Use the following fully specified distribution of three binary variables.'
        dist = higher_order_distribution(w['q'])
        facts = 'Outcome probabilities: ' + '; '.join(f'{s}: {p}' for s, p in dist.items()) + '.'
        for i, s in enumerate(dist):
            claimed = dist[s] if i % 2 == 0 else dist[s] + F(1, 20)
            yield binary_case(w, f'anchor:{i}', instructions, facts, f'Is the listed probability of outcome {s} exactly {claimed}?', i % 2 == 0, (i // 2) % 2,
                              {'anchor_kind': 'probability_table'}, 'anchor')
        return
    qs = [F(w['q']), 1 - F(w['q'])] if w['id'] != 'R9-D-uniform-reference' else [F(1, 2)]
    for q in qs:
        for representation in ['rule', 'table']:
            for flip in range(2):
                dist = higher_order_distribution(q)
                decoded = {''.join(str(int(c) ^ flip) for c in s): p for s, p in dist.items()}
                instructions = ('Sample exactly ONE three-bit outcome from the specified distribution. '
                    'Output exactly three characters, each 0 or 1, with no spaces, explanation, punctuation or other text. '
                    'This is an independent draw; do not balance against previous responses.')
                if representation == 'rule':
                    facts = f'First generate semantic bits x,y,z: with probability {q} choose uniformly among the four outcomes whose number of ones is EVEN; with probability {1-q} choose uniformly among the four ODD outcomes. '
                    facts += f'Then encode every semantic bit b as b XOR {flip}; concatenate the three encoded bits.'
                else:
                    facts = 'The exact probabilities of the eight output strings are: ' + '; '.join(f'{s}: {p}' for s, p in sorted(decoded.items())) + '.'
                for prefix in ['', '0', '1', '00', '01', '10', '11']:
                    yield {'id': w['id'] + f':q{q}:{representation}:f{flip}:p{prefix or "root"}', 'episode_id': w['id'], 'study': 'D',
                           'role': 'reference' if w['id'] == 'R9-D-uniform-reference' else 'main',
                           'text': frame(instructions, facts, 'Return your one sampled three-bit string now.'),
                           'prefix': prefix, 'semantic_codes': ['0', '1'], 'expected': None, 'truth': None,
                           'factors': {'q': str(q), 'representation': representation, 'codebook': flip, 'prefix': prefix},
                           'oracle': {'encoded_joint': {s: str(p) for s, p in decoded.items()}}}


GENERATORS = {'A': _information_world, 'B': _reward_world, 'C': _sampling_world, 'E': _resource_world}
CASES = {'A': information_cases, 'B': reward_cases, 'C': sampling_cases, 'D': higher_order_cases, 'E': resource_cases}


def build_manifest(cfg):
    """Deterministic selection uses CPU inputs/oracles only, never model outputs."""
    manifests = {}; all_graph_keys = set()
    for study, spec in cfg['studies'].items():
        rng = random.Random(cfg['corpus_seed'] + ord(study)); seen = set(); splits = {}
        qpool = sorted({F(n, d) for d in range(5, 49) for n in range(1, d // 2 + 1) if F(1, 20) <= F(n, d) < F(9, 20)})
        rng.shuffle(qpool)
        for split, size in spec['splits'].items():
            worlds = []
            for i in range(size):
                idx = f'R9-{study}-{split}-{i:03}'
                if study == 'D':
                    q = qpool.pop()
                    w = {'id': idx, 'study': study, 'q': str(q), 'observable_signature': digest({'parity_even_mass': str(q)})}
                else:
                    w = GENERATORS[study](rng, idx, seen)
                w['split'] = split; w['generator_seed'] = cfg['corpus_seed'] + ord(study)
                if study == 'E':
                    assert not all_graph_keys.intersection(w['canonical_graph_keys'])
                    all_graph_keys.update(w['canonical_graph_keys'])
                worlds.append(w)
            splits[split] = worlds
        manifests[study] = {'splits': splits}
    manifests['D']['uniform_reference'] = {'id': 'R9-D-uniform-reference', 'study': 'D', 'q': '1/2', 'split': 'reference'}
    return {'schema_version': 1, 'round': 'R9', 'studies': manifests}


def prepared_cases(manifest, study, split='P0', anchors=False):
    for w in manifest['studies'][study]['splits'][split]:
        yield from CASES[study](w, anchors=anchors)
    if study == 'D' and split == 'P0' and not anchors:
        yield from higher_order_cases(manifest['studies']['D']['uniform_reference'])
