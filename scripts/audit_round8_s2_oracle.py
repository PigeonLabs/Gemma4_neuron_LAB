"""Exact synthetic CPU counterexample to an underspecified S2 selection oracle.

No LLM inference, empirical model claim, or refutation of a paper's conditional
generative assumptions. A research-selection diagnostic, not a new model study.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import platform
import resource
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_counts(x):
    counts = {}
    for truth in (0, 1):
        overlap = x if truth == 0 else 4 - x
        for a_error, v_error in itertools.product((0, 1), repeat=2):
            n = (8 + overlap if (a_error, v_error) == (0, 0)
                 else overlap if (a_error, v_error) == (1, 1)
                 else 4 - overlap)
            counts[(truth, truth ^ a_error, truth ^ v_error)] = n
    assert sum(counts.values()) == 32
    return counts


def marginal(counts, axes):
    out = {}
    for values, n in counts.items():
        key = tuple(values[i] for i in axes)
        out[key] = out.get(key, 0) + n
    return out


def entropy(values):
    total = sum(values)
    return -sum((v / total) * math.log2(v / total) for v in values if v)


def conditional_entropy(counts, given):
    groups = {}
    for (truth, a, v), n in counts.items():
        key = tuple((truth, a, v)[i] for i in given)
        groups.setdefault(key, [0, 0])[truth] += n
    return sum(sum(values) / 32 * entropy(values) for values in groups.values())


def oracle_accuracy(counts, given):
    groups = {}
    for values, n in counts.items():
        key = tuple(values[i] for i in given)
        groups.setdefault(key, [0, 0])[values[0]] += n
    return sum(max(values) for values in groups.values()) / 32


def main():
    start = time.perf_counter()
    baseline = make_counts(2)
    rows = []
    for x in range(5):
        c = make_counts(x)
        # Every observable pairwise table, including truth/verifier confusion,
        # is identical. Joint error counts are also identical.
        assert all(marginal(c, axes) == marginal(baseline, axes)
                   for axes in ((0, 1), (0, 2), (1, 2)))
        overlap = sum(n for (t, a, v), n in c.items() if a != t and v != t)
        assert overlap == 4
        h_before = conditional_entropy(c, (1,))
        h_after = conditional_entropy(c, (1, 2))
        rows.append({'triple_interaction_index': x,
                     'table_axes': ['truth', 'existing_verdict', 'new_verdict'],
                     'counts': [{'values': list(k), 'n': n} for k, n in c.items()],
                     'individual_accuracy': .75,
                     'both_wrong_rate': overlap / 32,
                     'all_three_pairwise_tables_equal': True,
                     'conditional_information_bits': h_before - h_after,
                     'existing_only_Bayes_accuracy': oracle_accuracy(c, (1,)),
                     'both_verdicts_Bayes_accuracy': oracle_accuracy(c, (1, 2))})
    assert rows[0]['both_verdicts_Bayes_accuracy'] == .875
    assert rows[2]['both_verdicts_Bayes_accuracy'] == .75
    out = {'schema_version': 1, 'status': 'PASS',
           'scope': 'exact synthetic CPU oracle audit; NOT an actual12B result',
           'new_model_forwards': 0, 'table_population_size': 32,
           'rows': rows,
           'conclusion': 'Equal accuracy, pairwise verdict tables and pairwise error overlap do not determine additional-verifier information gain or Bayes accuracy. S2 needs an explicit joint-distribution oracle.',
           'novelty': 'A basic higher-order probability counterexample; not a novel theorem or empirical LLM finding.',
           'paper_scope': 'Does not refute CAVEAT or SPLIT inside their declared conditional generative assumptions.',
           'budget_config': 'configs/round8_selection.yaml',
           'budget_config_sha256': hashlib.sha256((ROOT / 'configs/round8_selection.yaml').read_bytes()).hexdigest(),
           'python': platform.python_version(), 'seed': 'not applicable; exact enumeration',
           'elapsed_seconds': time.perf_counter() - start,
           'peak_CPU_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
           'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'logs/summary/round8_s2_oracle_audit.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': out['status'], 'new_model_forwards': 0,
                      'same_pairwise_Bayes_accuracies': [row['both_verdicts_Bayes_accuracy'] for row in rows]}))


if __name__ == '__main__':
    main()
