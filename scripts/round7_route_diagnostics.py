#!/usr/bin/env python3
"""Post-hoc route-choice descriptions of old R6 discovery, no new inference."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
records=[]
for stage in ('round6_A_discovery','round6x_A_discovery'):
    p=ROOT/f'logs/summary/{stage}.json';s=json.loads(p.read_text());run=ROOT/'runs'/s['run_id'];path=run/'rows.jsonl'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==s['raw_rows_sha256']
    data=json.loads((run/'manifest.json').read_text());eps={e['id']:e for e in data['discovery']};rows=[json.loads(x) for x in path.read_text().splitlines()]
    for arm in ('readiness_conflict_direct','readiness_conflict_compose'):
        selected=[r for r in rows if r['arm']==arm];counts=Counter();worlds=defaultdict(Counter);pairs=defaultdict(list)
        for r in selected:
            ep=eps[r['episode_id']];case=next(c for c in ep['cases'] if c['id']==r['case_id']);w=ep['worlds']['receiver'];x=case['query']
            a,b=w['h'][x],w['g'][w['f'][x]];z=r['codebook'][r['prediction']]
            label='direct_table_answer' if z==a else 'composition_answer' if z==b else 'other_answer'
            counts[label]+=1;worlds[ep['id']][label]+=1;pairs[ep['id'],r['query_index']].append(z)
        records.append({'stage':stage,'arm':arm,'rows':len(selected),'independent_worlds':len(worlds),'semantic_choice_counts':dict(counts),
                        'per_world':[{'id':eid,**dict(c)} for eid,c in sorted(worlds.items())],
                        'query_pairs_with_codebook_dependent_semantic_choice':sum(len(set(v))>1 for v in pairs.values()),
                        'query_pairs':len(pairs),'raw_sha256':s['raw_rows_sha256']})
result={'schema_version':1,'status':'PASS','scope':'post-hoc description on old R6 discovery only; not new independent validation, no causal mechanism inferred',
        'new_model_forwards':0,'records':records,'limit':'different prompt conditions do not isolate table presence, format, length, or computation; matched fresh factorial intervention required'}
(ROOT/'logs/summary/round7_route_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'PASS','records':[{k:v for k,v in r.items() if k not in ('per_world','raw_sha256')} for r in records],'new_model_forwards':0}))
