#!/usr/bin/env python3
"""Post hoc reference-model sensitivity; never changes registered B verdict."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    agg=json.loads((ROOT/'logs/summary/round10b_aggregate.json').read_text())
    phase=agg['studies']['B']['P0']; n=phase['outcome']['independent_families']
    rows=[json.loads(s) for s in (ROOT/'runs'/phase['run_id']/'rows.jsonl').read_text().splitlines()]
    groups={}
    for r in rows:
        if r['role']=='main':groups.setdefault(r['episode_id'],[]).append(r)
    assert len(groups)==n
    assert all(all(r['semantic_choice']==r['truth'] for r in rr) for rr in groups.values())
    alpha=.05
    out={'schema_version':1,'kind':'POST_HOC_CPU_SENSITIVITY_NOT_A_REGISTERED_PRIMARY_CI',
        'new_model_forwards':0,'observed_families':n,'observed_worlds_with_any_main_error':0,
        'registered_classification_preserved':phase['outcome']['classification'],
        'registered_bootstrap_result_preserved':phase['outcome']['primary'],
        'reference_model':'If worlds were IID Bernoulli draws for ANY non-invariance/error event, zero events in n worlds gives one-sided exact upper1-alpha^(1/n). This is an explicit reference-model sensitivity, not a proof the actual rejection-sampled finite corpus satisfies IID assumptions.',
        'reference_one_sided95_upper_event_probability':1-alpha**(1/n),
        'counterexample_population':{'assumptions':'IID worlds: fraction0.10 have signed redistribution contrast+1; remaining0.90 have contrast0. Then population mean contrast is+0.10.',
            'population_mean_contrast':.10,'probability_observe_n_zero_contrasts':.9**n},
        'reference_zero_event_n_for95upper_at_most10percent':math.ceil(math.log(alpha)/math.log(.9)),
        'interpretation':'A degenerate empirical bootstrap[0,0] from12zero worlds does not establish a population mean below10pp. B Negative is the registered exploratory sample-level decision; population equivalence and an internal-mechanism absence claim remain unestablished.',
        'no_extra_sampling_or_candidate_change':True,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'logs/summary/round10b_uncertainty_sensitivity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['observed_families','reference_one_sided95_upper_event_probability','counterexample_population','registered_classification_preserved']}))


if __name__=='__main__':main()
