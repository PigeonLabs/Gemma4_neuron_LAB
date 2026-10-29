"""R7-A: a diagnostic example's logical footprint versus Hamming proximity."""
from __future__ import annotations
from collections import defaultdict
import hashlib,json,random
import numpy as np
from .round6_engine import margin,pair_score,bootstrap
from .runtime import save_json


def rule(x,spec):return int(x[spec['index']])^spec['invert']
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def make_world(seed,eid):
    rng=random.Random(seed);indices=rng.sample(range(4),2);rules=[{'index':i,'invert':rng.randrange(2)} for i in indices]
    universe=[f'{i:04b}' for i in range(16)]
    disagree=[x for x in universe if rule(x,rules[0])!=rule(x,rules[1])]
    evidence=rng.choice(disagree)
    initial=[]
    def distance(x):return sum(a!=b for a,b in zip(x,evidence))
    queries=[]
    for dependent,n in ((True,2),(False,1)):
        for proximity,dist in (('near',1),('far',3)):
            pool=[x for x in universe if x!=evidence and x not in initial and (rule(x,rules[0])!=rule(x,rules[1]))==dependent and distance(x)==dist]
            assert len(pool)>=n,(seed,dependent,proximity,pool)
            for x in rng.sample(pool,n):queries.append({'x':x,'dependent':dependent,'proximity':proximity,'h0':rule(x,rules[0]),'h1':rule(x,rules[1]),'distance':dist})
    query_values={q['x'] for q in queries}
    for val in (0,1):
        initial.append(rng.choice([x for x in universe if x not in query_values and rule(x,rules[0])==rule(x,rules[1])==val]))
    return {'id':eid,'seed':seed,'rules':rules,'initial':initial,'evidence':evidence,'queries':queries,
            'signature':canonical({'rules':rules,'initial':initial,'evidence':evidence,'queries':queries})}


def case(w,qi,book_index,evidence_rule=0,namespace='TARGET',anchor=None,code_only=None):
    q=w['queries'][qi];rules=w['rules'];book={'A':book_index,'B':1-book_index};code=lambda y:next(k for k,v in book.items() if v==y)
    lines=[]
    for i,spec in enumerate(rules):
        phrase=f"bit {spec['index']+1}"+(' flipped (0 becomes 1, 1 becomes 0)' if spec['invert'] else ' unchanged')
        lines.append(f'Rule R{i}: return {phrase}.')
    instruction='Bits are indexed from left to right starting at 1. Exactly one of the two rules governs system TARGET. Use examples tagged TARGET to infer its rule. OTHER is an unrelated system and its examples give no evidence about TARGET.'
    if anchor is not None:instruction+=f' For this question the governing TARGET rule is explicitly R{anchor}.'
    facts='\n'.join(lines)+'\n'+'\n'.join(f'TARGET example: {x} -> {rule(x,rules[0])}.' for x in w['initial'])
    evidence_label=rule(w['evidence'],rules[evidence_rule])
    if code_only is None:facts+=f"\nNew example from {namespace}: {w['evidence']} -> {evidence_label}."
    else:facts+=f'\nThe answer value is explicitly {code_only}; no rule calculation is needed.'
    correct=code_only if code_only is not None else rule(q['x'],rules[anchor if anchor is not None else evidence_rule])
    text=instruction+'\nCUE_BEGIN:\nAnswer the TARGET question.\nFACTS_BEGIN:\n'+facts+'\nEND_FACTS\nNEW_QUERY:\n'+f"Input {q['x']}. Response codes: A={book['A']}; B={book['B']}. C and D are unused. Reply with only A or B."
    return {'id':f"{w['id']}-q{qi}-b{book_index}-e{evidence_rule}-{namespace}-a{anchor}-v{code_only}",'episode_id':w['id'],'study':'A','text':text,'expected':code(correct),
            'factors':{'query_index':qi,'book_index':book_index,'evidence_rule':evidence_rule,'namespace':namespace,'anchor':anchor,'code_only':code_only,'dependent':q['dependent'],'proximity':q['proximity']},
            'has_unique_target_answer':namespace=='TARGET' or not q['dependent'] or anchor is not None or code_only is not None,
            'oracle':{'query':q['x'],'evidence':w['evidence'],'h0':q['h0'],'h1':q['h1'],'sign':q['h1']-q['h0'],'book':book,'reference_value':correct,'rules':rules}}


def build_manifest(cfg):
    splits={};seen=set()
    for si,(split,n) in enumerate(cfg['splits'].items()):
        splits[split]=[]
        for i in range(n):
            for attempt in range(1000):
                w=make_world(cfg['corpus_seed']+si*1000000+i*1000+attempt,f'r7A-{split}-{i:03}')
                if w['signature'] not in seen:break
            else:raise ValueError('No unique oracle world within deterministic generation cap')
            seen.add(w['signature']);splits[split].append(w)
    signatures=[w['signature'] for ww in splits.values() for w in ww]
    assert len(signatures)==len(set(signatures))
    return {'schema_version':1,'study':'R7_A_logical_footprint','splits':splits,'seed':cfg['corpus_seed']}


def run(j,worlds,cfg):
    quality=defaultdict(list)
    for w in worlds:
        for r in (0,1):
            for qi in (0,2):
                row,_,_=j.call(case(w,qi,w['seed']%2,anchor=r),arm=f'anchor_rule_{r}',phase='P0_quality');quality[f'rule_{r}_{w["queries"][qi]["proximity"]}'].append(row['correct'])
        for bi in (0,1):
            row,_,_=j.call(case(w,0,bi,code_only=w['seed']%2),arm='code_anchor',phase='P0_quality');quality['code_anchor'].append(row['correct'])
    readiness={k:{'correct':int(sum(v)),'rows':len(v),'accuracy':float(np.mean(v))} for k,v in quality.items()}
    eligible=all(v['accuracy']>=(cfg['code_accuracy_min'] if k=='code_anchor' else cfg['anchor_accuracy_min']) for k,v in readiness.items())
    save_json(j.p.run.path/'quality.json',readiness)
    print(json.dumps({'event':'R7_A_quality','readiness':readiness,'eligible':eligible,'calls':j.p.calls}),flush=True)
    if not eligible:return {'gate':'STOPPED_AT_RULE_OR_CODE_ANCHOR','classification':'Inconclusive','scientific_status':'NOT_IDENTIFIABLE','readiness':readiness,'P1':'NOT_EVALUATED','P2':'NOT_EVALUATED'}
    episodes=[]
    for w in worlds:
        effects=defaultdict(list);preservation=[];accuracy=[]
        for qi,q in enumerate(w['queries']):
            for bi in (0,1):
                by={}
                for namespace in ('TARGET','OTHER'):
                    for r in (0,1):
                        material=case(w,qi,bi,r,namespace)
                        row,_,_=j.call(material,arm=f'evidence_{namespace}_R{r}',phase='P0')
                        book=material['oracle']['book'];one=next(k for k,v in book.items() if v==1);zero=next(k for k,v in book.items() if v==0)
                        by[namespace,r]=pair_score(row,one,zero)
                        if namespace=='TARGET':accuracy.append(int(row['correct']))
                target=by['TARGET',1]-by['TARGET',0];sham=by['OTHER',1]-by['OTHER',0]
                if q['dependent']:effects[q['proximity']].append((q['h1']-q['h0'])*(target-sham))
                else:preservation.append({'target_absolute':abs(target),'sham_absolute':abs(sham),'excess_absolute':abs(target)-abs(sham)})
        episodes.append({'id':w['id'],'near':float(np.mean(effects['near'])),'far':float(np.mean(effects['far'])),
                         'primary':float(np.mean(effects['near']+effects['far'])),'agreement_target_absolute':float(np.mean([r['target_absolute'] for r in preservation])),
                         'agreement_sham_absolute':float(np.mean([r['sham_absolute'] for r in preservation])),
                         'agreement_excess_absolute':float(np.mean([r['excess_absolute'] for r in preservation])),
                         'TARGET_accuracy':float(np.mean(accuracy))})
        print(json.dumps({'event':'R7_A_episode','episode':w['id'],'calls':j.p.calls}),flush=True)
    stats={k:bootstrap([e[k] for e in episodes],cfg) for k in ('primary','near','far','agreement_excess_absolute')}
    guard=float(np.mean([e['agreement_target_absolute'] for e in episodes]))<=cfg['agreement_absolute_margin_change_max']
    point_positive=stats['primary']['ci95'][0]>cfg['practical_margin_nats'] and stats['far']['ci95'][0]>cfg['practical_margin_nats'] and guard
    negative=stats['primary']['ci95'][1]<cfg['practical_margin_nats'] or stats['far']['ci95'][1]<cfg['practical_margin_nats']
    classification='Positive' if point_positive else 'Negative' if negative else 'Inconclusive'
    return {'gate':'P0_COMPLETED','classification':classification,'scientific_status':'EXPLORATORY_P0_ONLY','readiness':readiness,'statistics':stats,
            'agreement_preservation_passed':guard,'episodes':episodes,'inference':'counterfactual example/namespace effects in declared rule tasks; not an internal mechanism or a proven novel contribution',
            'P1':'REQUIRES_FRESH_REGISTRATION' if classification=='Positive' else 'NOT_EVALUATED','P2':'NOT_EVALUATED'}


def prepared_cases(manifest,cfg):
    for w in manifest['splits']['P0']:
        for r in (0,1):
            for qi in (0,2):yield case(w,qi,w['seed']%2,anchor=r)
        for bi in (0,1):yield case(w,0,bi,code_only=w['seed']%2)
        for qi in range(6):
            for bi in (0,1):
                for namespace in ('TARGET','OTHER'):
                    for r in (0,1):yield case(w,qi,bi,r,namespace)
