"""Synthetic pre-model falsification of the proposed calculation-reuse P0.

Two different causal readout mechanisms match every proposed natural-donor
observable: stored result reuse versus stored raw-input recomputation. This is
an instrument qualification failure, not a result about Gemma or a new theorem.
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
INPUTS = tuple(''.join(bits) for bits in itertools.product('01', repeat=4))
FUNCTIONS = {
    'reverse': lambda x: x[::-1],
    'not': lambda x: ''.join('1' if c == '0' else '0' for c in x),
    'parity': lambda x: str(sum(map(int, x)) % 2),
    'first_pair': lambda x: x[:2],
}


def common_value_readout(value, base, donor):
    """Constructed four-code head, shared by the two countermodels.

    This extends the existence counterexample to logit-margin observations.
    It is not an empirical Gemma head or a head fitted to model outcomes.
    """
    options = [base]
    if donor != base:
        options.append(donor)
    while len(options) < 4:
        options.append(('unused_synthetic_value', len(options)))
    assert value in options
    return tuple(30 * math.tanh((2.0 if value == choice else -1.0) / 30)
                 for choice in options)


def result_reuse(f, receiver_x, donor_x, structure, query):
    stored_u = f(donor_x)
    independent_y = f(receiver_x)  # g(y)=y, same clean answer by construction
    if query == 'first' or structure in ('explicit_share', 'duplicate'):
        return stored_u
    assert structure == 'coincidental_independent'
    return independent_y


def raw_recompute(f, receiver_x, donor_x, structure, query):
    stored_raw_x = donor_x
    independent_y = f(receiver_x)
    if query == 'first' or structure in ('explicit_share', 'duplicate'):
        # Re-executes the function on every queried result. No stored u is read.
        return f(stored_raw_x)
    assert structure == 'coincidental_independent'
    return independent_y


def main():
    started = time.perf_counter()
    rows = []
    for name, f in FUNCTIONS.items():
        compared = mismatches = logit_mismatches = same_value_distinct_input = 0
        for x, donor_x in itertools.product(INPUTS, repeat=2):
            same_value_distinct_input += int(x != donor_x and f(x) == f(donor_x))
            for structure, query in itertools.product(
                    ('explicit_share', 'duplicate', 'coincidental_independent'),
                    ('first', 'second')):
                a = result_reuse(f, x, donor_x, structure, query)
                b = raw_recompute(f, x, donor_x, structure, query)
                compared += 1
                mismatches += int(a != b)
                logit_mismatches += int(common_value_readout(a, f(x), f(donor_x)) !=
                                        common_value_readout(b, f(x), f(donor_x)))
        assert mismatches == logit_mismatches == 0
        rows.append({'function': name, 'inputs': len(INPUTS),
                     'injective': len({f(x) for x in INPUTS}) == len(INPUTS),
                     'same_value_distinct_input_donors': same_value_distinct_input,
                     'proposed_observables_compared': compared,
                     'countermodel_mismatches': mismatches,
                     'shared_readout_logit_vector_mismatches': logit_mismatches})
    assert rows[0]['same_value_distinct_input_donors'] == 0
    assert rows[1]['same_value_distinct_input_donors'] == 0
    assert rows[2]['same_value_distinct_input_donors'] > 0
    # Audit Claude's cross-review repair: x fixed, function name fixed, table
    # contents changed in the donor. Copying the operator/table rather than a
    # result still passes its input-integrity probe and all stated comparisons.
    permutations = tuple(itertools.permutations(range(4)))
    table_compared = table_mismatches = table_logit_mismatches = 0
    same_value_different_table = changed_value_table = 0
    for receiver_f, donor_f, x in itertools.product(permutations, permutations, range(4)):
        same_value_different_table += int(receiver_f != donor_f and receiver_f[x] == donor_f[x])
        changed_value_table += int(receiver_f[x] != donor_f[x])
        for structure, query in itertools.product(
                ('explicit_share', 'duplicate', 'coincidental_independent'),
                ('first', 'second', 'input_probe')):
            reused_result = donor_f[x]
            copied_operator = donor_f
            if query == 'input_probe':
                cache_output = recomputed_output = x
            elif query == 'first' or structure in ('explicit_share', 'duplicate'):
                cache_output = reused_result
                recomputed_output = copied_operator[x]
            else:
                cache_output = recomputed_output = receiver_f[x]
            table_compared += 1
            table_mismatches += int(cache_output != recomputed_output)
            if query != 'input_probe':
                table_logit_mismatches += int(
                    common_value_readout(cache_output, receiver_f[x], donor_f[x]) !=
                    common_value_readout(recomputed_output, receiver_f[x], donor_f[x]))
    assert table_mismatches == table_logit_mismatches == 0 and same_value_different_table == 480
    clean = common_value_readout(0, 0, 1)
    patched = common_value_readout(1, 0, 1)
    synthetic_theta = (patched[1] - patched[0]) - (clean[1] - clean[0])
    assert synthetic_theta > .1
    out = {'schema_version': 1, 'status': 'PASS',
           'protocol_qualification': 'NOT_IDENTIFIABLE_UNDER_PROPOSED_OBSERVABLES',
           'scope': 'synthetic CPU countermodels; NOT an actual12B experiment',
           'new_model_forwards': 0, 'functions': rows,
           'total_observables_compared': sum(x['proposed_observables_compared'] for x in rows),
           'causal_mechanisms': {
               'reuse': 'prefix site stores u=f(x); first/explicit-share/duplicate read u',
               'recompute': 'prefix site stores raw x; each query re-executes f(x), never reading a cached u'},
           'conclusion': 'A natural donor changes f(x) and x together. Both mechanisms pass the explicit-sharing witness, show excess transfer for duplicate versus independent coincidental computations, and match the same-value control.',
           'injective_donor_issue': 'For fixed reverse/not, same-result different-input donors do not exist. Noninjective functions permit these donors but do not distinguish the two mechanisms here.',
           'same_input_different_table_repair': {
               'receiver_and_donor_bijections': len(permutations),
               'input_domain_size': 4,
               'observables_compared': table_compared,
               'countermodel_mismatches': table_mismatches,
               'shared_readout_logit_vector_mismatches': table_logit_mismatches,
               'same_value_different_table_pairs': same_value_different_table,
               'changed_value_pairs': changed_value_table,
               'input_probe_preserved': True,
               'countermodel': 'Site stores the donor function table, not its evaluated result. Each query recomputes f(x); x probes remain unchanged.',
               'conclusion': 'Fixing raw x and probing x does not rule out copying the changed operator/table. No abstract computation-reuse identification from these observables.'},
           'limitation': 'Not a universal impossibility theorem: other interventions or observables might distinguish mechanisms. The present proposal does not specify them.',
           'readout_extension': {
               'type': 'constructed common value-to-four-code-logit map, softcap30',
               'not_a_Gemma_readout': True,
               'scope': 'The original enumeration checked answer values only. This explicit common head shows a possible equal-logit extension, not empirical equality of actual model logits.',
               'matched_margin_effect_example_nats': synthetic_theta,
               'example': 'Both countermodels pass explicit sharing and show this positive duplicate-versus-independent margin difference; the recomputation model never reads a stored result.',
               'code_permutation': 'Equal vectors remain equal after any common permutation of the four response codes.'},
           'novelty': 'Protocol counterexample, not a novel structural identity or a paper contribution.',
           'budget_config': 'configs/round8_selection.yaml',
           'budget_config_sha256': hashlib.sha256((ROOT / 'configs/round8_selection.yaml').read_bytes()).hexdigest(),
           'python': platform.python_version(), 'seed': 'not applicable; exhaustive enumeration',
           'elapsed_seconds': time.perf_counter() - started,
           'peak_CPU_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
           'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / 'logs/summary/round8_reuse_identifiability_audit.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': out['status'], 'protocol': out['protocol_qualification'],
                      'synthetic_comparisons': out['total_observables_compared'],
                      'actual_model_forwards': 0}))


if __name__ == '__main__':
    main()
