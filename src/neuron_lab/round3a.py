"""Numerical qualification and fixed-semantic-axis contrasts; no GPU inference."""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def option_metrics(logits, case, labels):
    row=logits[0].double()
    a,b=(labels[x] for x in case['code'])
    margin=float(case['c']*(row[a]-row[b]))
    correct=labels[case['correct_label']]
    return {'semantic_margin':margin,'correctness_margin':case['r']*margin,
            'forced_choice_correct':case['r']*margin>0,'top1_id':int(row.argmax()),
            'raw_top1_correct':int(row.argmax())==correct,
            'label_logits':{x:float(row[labels[x]]) for x in case['code']}}


def ranks(x):
    x=np.asarray(x,float)
    result=np.empty(len(x),float)
    for v in np.unique(x):
        selected=np.flatnonzero(x==v)
        result[selected]=np.count_nonzero(x<v)+(len(selected)-1)/2
    return result


def correlation(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None


def signed_table(rows):
    groups=defaultdict(dict)
    for row in rows:
        if row['arm']=='candidate':
            groups[(row['dtype'],row['candidate_id'],row['case_id'])][row['sign']]=row
    result=[]
    for (dtype,cid,case_id),g in sorted(groups.items()):
        if set(g)!={-1,1}: raise ValueError('Incomplete signs')
        a,b=g[1],g[-1]
        result.append({'dtype':dtype,'candidate_id':cid,'case_id':case_id,
            'family_id':a['family_id'],'task':a['task'],'difficulty':a['difficulty'],
            'code':a['code'],'r':a['r'],'c':a['c'],
            'd':(a['semantic_margin']-b['semantic_margin'])/2,
            'actual_nonzero_both_signs':a['actual_delta']!=0 and b['actual_delta']!=0})
    return result


def qualify(rows,cfg,technical_passed):
    table=signed_table(rows)
    q=cfg['qualification']
    comparisons=[]
    for candidate in cfg['candidates']:
        cid=candidate['id']
        series={dtype:{r['case_id']:r for r in table if r['candidate_id']==cid and r['dtype']==dtype}
                for dtype in ('bfloat16','float32')}
        if set(series['bfloat16'])!=set(series['float32']) or len(series['float32'])!=q['families']*q['variants_per_family']:
            raise ValueError('Incomplete paired dtype table')
        ids=sorted(series['float32'])
        a,b=(np.array([series[d][i]['d'] for i in ids]) for d in ('bfloat16','float32'))
        mask=abs(b)>=q['nontrivial_fp32_effect_floor_nats']
        count=int(mask.sum())
        agreement=float(np.mean(np.sign(a[mask])==np.sign(b[mask]))) if count else None
        spearman=correlation(ranks(a),ranks(b))
        error=float(np.mean(abs(a-b)))
        bound=max(q['maximum_mean_absolute_error_floor_nats'],q['maximum_mean_absolute_error_relative']*float(np.mean(abs(b))))
        fractions={dtype:float(np.mean([r['actual_delta']!=0 for r in rows if r['arm']=='candidate' and r['candidate_id']==cid and r['dtype']==dtype])) for dtype in series}
        checks={'technical':technical_passed,
                'nonzero_delivery':all(v>=q['actual_nonzero_fraction_minimum'] for v in fractions.values()),
                'sufficient_nontrivial_cases':count>=q['minimum_nontrivial_signed_cases'],
                'sign_agreement':agreement is not None and agreement>=q['minimum_sign_agreement'],
                'rank_stability':spearman is not None and spearman>=q['minimum_spearman'],
                'mean_absolute_error':error<=bound}
        comparisons.append({**candidate,'eligible':all(checks.values()),'checks':checks,
            'n_signed_cases':len(ids),'n_nontrivial_fp32':count,'nonzero_delivery_fraction':fractions,
            'sign_agreement_nontrivial':agreement,'spearman_signed_effects':spearman,
            'mean_absolute_error_nats':error,'mean_absolute_error_bound_nats':bound,
            'absolute_error_99th_percentile_nats':float(np.quantile(abs(a-b),.99)),
            'mean_abs_signed_effect_BF16':float(np.mean(abs(a))),
            'mean_abs_signed_effect_FP32':float(np.mean(abs(b))),
            'paired_effects':[{'case_id':i,'family_id':series['float32'][i]['family_id'],'BF16':float(a[j]),'FP32':float(b[j])} for j,i in enumerate(ids)]})
    eligible=[r['id'] for r in comparisons if r['eligible']]
    return {'status':'PASS' if technical_passed and eligible else 'STOPPED_AT_NUMERICAL_QUALIFICATION',
            'technical_checks_passed':technical_passed,'eligible_candidates':eligible,'candidates':comparisons,
            'scope':'precision qualification at the registered dose/site/tasks, not semantic function or absence of function',
            'future_discovery':'AUTHORIZED_BY_GATE' if eligible and technical_passed else 'NOT_EVALUATED',
            'signed_effects':table}


def factorial(rows, beta=0.0):
    if len(rows)!=4 or {(r['r'],r['c']) for r in rows}!={(1,1),(1,-1),(-1,1),(-1,-1)}:
        raise ValueError('Need one complete rule x mapping factorial')
    d=np.array([r['d']-beta*r.get('clean_semantic_margin',0.) for r in rows])
    r=np.array([x['r'] for x in rows]);c=np.array([x['c'] for x in rows])
    return dict(zip(('C','G','L','J'),map(float,[d.mean(),(r*d).mean(),(c*d).mean(),(r*c*d).mean()])))
