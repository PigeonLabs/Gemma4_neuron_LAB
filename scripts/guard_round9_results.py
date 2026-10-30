#!/usr/bin/env python3
"""Keep invalid estimands null in a separate layer; never edit R9 measurements."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reg=json.loads((ROOT/'logs/summary/round9_P0_registration.json').read_text())
    frozen=dict(reg['operative_source_sha256'])
    paths=[f'logs/summary/round9_P0_{k}.json' for k in 'ABCDE']+[
        'logs/summary/round9_P0_technical.json','logs/summary/round9_P0_model_load.json','logs/summary/round9_P0_registration.json']
    frozen.update({p:sha(ROOT/p) for p in paths})
    for p,h in {**frozen,**reg['protected_prior_sha256']}.items():assert sha(ROOT/p)==h,p
    freeze={'schema_version':1,'status':'FROZEN','scope':'original measured R9 science plus its pre-output operative sources; reports and posthoc interpretation are separate',
            'protected_measured_sha256':frozen,'prior_protected_file_count':len(reg['protected_prior_sha256']),
            'private_manifest_sha256':reg['manifest_sha256'],'private_encoded_inputs_sha256':reg['encoded_inputs_sha256']}
    fp=ROOT/'logs/summary/round9_measurement_freeze.json'
    if fp.exists():assert json.loads(fp.read_text())==freeze,'A frozen measurement changed'
    else:fp.write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n')
    guarded={}
    for k in 'ABCDE':
        p=ROOT/f'logs/summary/round9_P0_{k}.json';s=json.loads(p.read_text());o=s['outcome'];valid=o['gate']=='P0_MEASURED'
        guarded[k]={'original_summary_sha256':sha(p),'original_gate':o['gate'],'original_classification':o['classification'],
            'estimand_valid':valid,'primary_estimate':None,'primary_CI':None,
            'interpretation_status':'EVALUATED_EXPLORATORY_FINITE_GRID' if valid else 'NOT_EVALUATED' if o['gate']=='STOPPED_AT_BASIC_ANCHORS' else 'NOT_IDENTIFIABLE',
            'diagnostic_numeric_fields_are_behavior_evidence':valid,
            'main_rows_are_evidence_of_task_absence':False}
        if k=='D':guarded[k].update(primary_estimate={'selective_families':o['selective_q_families'],'families':o['q_families']},
            conditional_high_order_error_given_normal_low_order='NOT_IDENTIFIABLE_NO_ELIGIBLE_LOW_ORDER_NORMAL_CONDITIONS',
            claim_scope='Registered joint selective pattern absent here. The conditional question among low-order-normal conditions has no eligible observations; no absence of higher-order errors, free-generation failure or population theorem is inferred.')
    assert all(not guarded[k]['estimand_valid'] and guarded[k]['primary_estimate'] is None for k in 'ABCE')
    assert guarded['D']['estimand_valid'] and guarded['D']['primary_estimate']=={'selective_families':0,'families':8}
    out={'schema_version':1,'status':'PASS','new_model_forwards':0,'original_measured_classifications_changed':False,
         'purpose':'Consumer-facing validity guard; A numeric0/CI0 from invalid first-token coding must not be reused as observed NO or zero effect.',
         'verdict_precedence':['Failed basic anchors => Inconclusive/NOT_IDENTIFIABLE, main NOT_EVALUATED',
                               'Measured main fails full-vocab format => Inconclusive/NOT_IDENTIFIABLE',
                               'Only then evaluate study-local Positive/Negative criteria'],
         'C_yaml_qualification':'The local P0_negative prose mentions anchor failure; the registered global anchor_failure policy and actual executable gate take precedence. Original YAML and classifications are preserved.',
         'studies':guarded,'measurement_freeze_sha256':sha(fp),'script_sha256':sha(Path(__file__))}
    (ROOT/'logs/summary/round9_interpretation_guard.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','invalid_primary_estimands_null':['A','B','C','E'],'new_model_forwards':0,'frozen_R9_paths':len(frozen)}))


if __name__=='__main__':main()
