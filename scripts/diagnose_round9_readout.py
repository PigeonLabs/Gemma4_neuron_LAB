#!/usr/bin/env python3
"""Posthoc descriptive input/first-token audit; no selection or new model calls."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text())


def main():
    summaries={k:read(ROOT/f'logs/summary/round9_P0_{k}.json') for k in 'ABCDE'};out={}
    for k,s in summaries.items():
        raw=ROOT/'runs'/s['run_id']/'rows.jsonl';rows=[json.loads(x) for x in raw.read_text().splitlines()]
        groups=defaultdict(list)
        for r in rows:groups[r['role']+'_'+r['factors'].get('anchor_kind','main')].append(r)
        d={}
        for name,rs in groups.items():
            d[name]={'rows':len(rs),'full_top1_texts':dict(Counter(r['full_top1_text'] for r in rs)),
                     'format_valid':sum(r['format_valid'] for r in rs),'full_correct':sum(r['full_correct'] for r in rs) if k!='D' or name.startswith('anchor') else None,
                     'valid_mass_mean':sum(r['valid_probability_mass'] for r in rs)/len(rs),
                     'expected_semantic_option_indices':dict(Counter(str(r['truth']) for r in rs))}
        duplicates=defaultdict(list)
        for r in rows:duplicates[r['authoritative_input_sha256']].append(r)
        repeated=[rs for rs in duplicates.values() if len(rs)>1]
        for rs in repeated:
            assert len({r['episode_id'] for r in rs})==1
            for r in rs[1:]:
                assert r['answer_logits']==rs[0]['answer_logits'] and r['full_top1_id']==rs[0]['full_top1_id']
                assert r['valid_probability_mass']==rs[0]['valid_probability_mass']
        out[k]={'raw_rows_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'groups':d,
                'actual_rows':len(rows),'unique_authoritative_inputs':len(duplicates),
                'repeated_input_groups':len(repeated),'extra_repeated_rows':sum(len(v)-1 for v in repeated),
                'repeat_scope':'within same family/reference only; never independent observations',
                'same_input_same_recorded_readout_exact':True}
        if k=='D':
            trees=defaultdict(list)
            for r in rows:
                if r['role']=='anchor':continue
                f=r['factors'];trees[(r['role'],f['q'],f['representation'],f['codebook'])].append((f['prefix'],r['authoritative_input_sha256']))
            out[k]['prefix_tree_counts']={}
            for role in ['main','reference']:
                for rep in ['rule','table']:
                    tt=[tuple(sorted(v)) for key,v in trees.items() if key[0]==role and key[2]==rep]
                    out[k]['prefix_tree_counts'][role+'_'+rep]={'registered':len(tt),'unique':len(set(tt))}
    result={'schema_version':1,'status':'PASS','scope':'POSTHOC_DESCRIPTIVE_AUDIT_NOT_NEW_SCIENTIFIC_TEST','new_model_forwards':0,
            'classification_changes':False,'registered_sources_inputs_and_gates_changed':False,'studies':out,
            'material_qualifications':[
                'A complex prompts often begin with To; first-token code failure does not establish inability to evaluate strategies.',
                'B anchor cue requests route choice while final query asks YES/NO about a sum. Potential instruction conflict is a design concern, not experimentally proven cause.',
                'C probability anchor mixes format failure and incorrect valid decisions; cannot identify probability reasoning ability from this gate.',
                'C probability anchors always place the correct fraction first (semantic option0), despite counterbalanced A/B codes. Even a passed anchor would not exclude an always-first-option heuristic.',
                'E union_count11/64 valid and11/64 correct: all valid rows correct; format failure is not evidence union-count competence is17%.',
                'D complete tables for q/flip and1-q/opposite-flip can repeat. Their identical inputs/readouts add calls but no independent codebook evidence.',
                'Full-vocab logits were not retained; native recorded top1/mass cannot be independently re-evaluated by this CPU script.']}
    (ROOT/'logs/summary/round9_readout_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','new_forwards':0,'actual_science_rows':sum(v['actual_rows'] for v in out.values()),
                      'D_unique_inputs':out['D']['unique_authoritative_inputs'],'D_extra_repeat_rows':out['D']['extra_repeated_rows']}))


if __name__=='__main__':main()
