"""Exact CPU checks of proposed R9 protocols, not actual-model evidence."""
from fractions import Fraction as F
from itertools import permutations, product
from pathlib import Path
import hashlib
import json
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matching_count(matrix):
    n = len(matrix)
    return sum(all(matrix[i][p[i]] for i in range(n)) for p in permutations(range(n)))


def degrees(matrix):
    return [sum(row) for row in matrix], [sum(row[j] for row in matrix) for j in range(len(matrix))]


def winning_policy_count(payoff, observation):
    groups = sorted(set(observation))
    return sum(all(payoff[s][policy[groups.index(observation[s])]] for s in range(len(payoff)))
               for policy in product(range(len(payoff[0])), repeat=len(groups)))


def main():
    started = time.perf_counter()
    payoff = [[1, 0], [1, 0], [0, 1], [0, 1]]
    info = {'none': [0, 0, 0, 0], 'full': [0, 1, 2, 3],
            'partial_useful': [0, 0, 1, 1], 'partial_useless': [0, 1, 0, 1]}
    policy_counts = {k: winning_policy_count(payoff, obs) for k, obs in info.items()}
    assert policy_counts == {'none': 0, 'full': 1, 'partial_useful': 1, 'partial_useless': 0}

    bad = [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]
    good = [[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1]]
    assert degrees(bad) == degrees(good) == ([2, 2, 2, 2], [3, 3, 1, 1])
    counts = [matching_count(m) for m in [bad, good]]
    assert counts == [0, 2]
    regular_count = 0
    regular_failures = 0
    for bits in product([0, 1], repeat=16):
        m = [list(bits[i:i+4]) for i in range(0, 16, 4)]
        if degrees(m) != ([2]*4, [2]*4):
            continue
        regular_count += 1
        regular_failures += not matching_count(m)
    assert regular_count == 90 and regular_failures == 0

    prior = F(1, 2)
    p0, p1 = F(1, 4), F(3, 4)
    likelihoods = [p*p*(1-p) for p in [p0, p1]]
    posterior = likelihoods[1]*prior / sum(x*prior for x in likelihoods)
    conditioned = [x / (3*p*p*(1-p)) for x, p in zip(likelihoods, [p0, p1])]
    selected_posterior = conditioned[1] / sum(conditioned)
    assert posterior == F(3, 4) and conditioned == [F(1, 3), F(1, 3)] and selected_posterior == F(1, 2)

    # U has8 elements; nested hypotheses have2 and4 elements. Both visible
    # positives are contained in both hypotheses. IID draws use replacement.
    strong = [F(1, 2)**2, F(1, 4)**2]
    weak_all_trials_visible = [F(1, 8)**2]*2
    weak_positives_only_kept = strong.copy()
    assert strong[0]/strong[1] == 4
    assert weak_all_trials_visible[0]/weak_all_trials_visible[1] == 1
    assert weak_positives_only_kept[0]/weak_positives_only_kept[1] == 4

    elapsed = time.perf_counter() - started
    assert elapsed < 60
    result = {
        'schema_version': 1, 'date_kst': '2026-10-30', 'status': 'PASS',
        'scope': 'Exact synthetic CPU protocol checks, not a novel theorem or an actual12B measurement',
        'new_model_forwards': 0, 'model_revision': 'NOT_APPLICABLE_NO_MODEL_LOADED',
        'information_partition': {'payoff': payoff, 'winning_policy_counts': policy_counts,
                                  'interpretation': 'Same statewise success can differ in implementable common policy; the two partial partitions have equal group counts but different feasibility.'},
        'degree_preserving_matching': {'infeasible': bad, 'feasible': good, 'degree_vectors': degrees(bad),
                                      'matching_counts': counts, 'edge_switch_preserves_each_node_degree': True,
                                      'all_4_by_4_degree2_matrices': regular_count, 'regular_matrices_without_perfect_matching': regular_failures,
                                      'scope_limit': 'Regular-case enumeration is4x4 only; no new general theorem claimed. Do not choose an all-regular family for the contrasting oracle.'},
        'collection_process': {'observed_sequence': '101', 'hypothesis_head_probabilities': ['1/4', '3/4'],
                               'fixed_length_likelihoods': list(map(str, likelihoods)),
                               'stop_on_second1_likelihoods': list(map(str, likelihoods)),
                               'fixed_and_stopped_posterior_higher_bias': str(posterior),
                               'keep_only_3draw_blocks_with_exactly_two1_likelihoods': list(map(str, conditioned)),
                               'selected_posterior_higher_bias': str(selected_posterior)},
        'sampling_instruction_censoring': {'strong_likelihood_ratio_small_over_large': str(strong[0]/strong[1]),
                                           'weak_all_trials_shown_ratio': str(weak_all_trials_visible[0]/weak_all_trials_visible[1]),
                                           'sample_from_U_but_only_positives_shown_ratio': str(weak_positives_only_kept[0]/weak_positives_only_kept[1]),
                                           'interpretation': 'Only-positive retention is not the proposed weak-sampling condition. State whether sampling uses replacement.'},
        'Claude_candidate1_cost_countercheck': {'proposed40worlds_5k_5arms': 1000,
                                               'with_declared3queries_per_cell': 3000,
                                               'k0_first_last_diagnostic_prompts': 'Identical within each diagnostic rule; repeated arms do not create independent cases.'},
        'device': 'CPU', 'dtype': 'exact rational/integer arithmetic', 'seed': 'not applicable; exhaustive enumeration',
        'python': sys.version.split()[0], 'platform': platform.platform(), 'elapsed_seconds': elapsed,
        'peak_CPU_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'source_sha256': sha(Path(__file__)), 'budget_config': 'configs/round9_protocol_audits.yaml',
        'budget_config_sha256': sha(ROOT/'configs/round9_protocol_audits.yaml'),
    }
    (ROOT/'logs/summary/round9_protocol_audits.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': 'PASS', 'new_model_forwards': 0, 'matching_counts': counts,
                      'regular_graphs_checked': regular_count, 'policy_counts': policy_counts,
                      'fixed_and_selected_posteriors': [str(posterior), str(selected_posterior)]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
