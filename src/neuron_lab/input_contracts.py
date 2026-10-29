"""Pre-output input validity contracts for future studies, not a research result.

Scientific generators supply observable semantic signatures. Irrelevant episode
IDs or unrendered latent fields must not be part of those signatures.
"""
from __future__ import annotations
import hashlib,json

class InputContractError(ValueError):pass

def token_fingerprint(input_ids):
    if not input_ids or any(type(i) is not int or i<0 for i in input_ids):raise InputContractError('Invalid authoritative token sequence')
    return hashlib.sha256(json.dumps(input_ids,separators=(',',':')).encode()).hexdigest()

def validate_splits(splits):
    """Require unique observed worlds and disjoint principal model inputs.

    Each episode has id, observable_signature and principal_input_sha256.
    Repeated intervention arms belong to one episode and may repeat its input.
    Generic instrument anchors are not principal inputs and must be declared
    separately by the study. This audit never sees model outputs.
    """
    ids=set();semantics={};tokens={};counts={}
    for split,episodes in splits.items():
        n=0
        for e in episodes:
            eid=e['id'];world=e['observable_signature'];inputs=set(e['principal_input_sha256'])
            if eid in ids or not world or not inputs:raise InputContractError('Repeated ID or empty observed contract')
            ids.add(eid)
            if world in semantics:raise InputContractError(f'Observed world repeats: {semantics[world]} and {(split,eid)}')
            semantics[world]=(split,eid)
            for h in inputs:
                if len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise InputContractError('Invalid input digest')
                if h in tokens and tokens[h]!=(split,eid):raise InputContractError(f'Model input repeats: {tokens[h]} and {(split,eid)}')
                tokens[h]=(split,eid)
            n+=1
        counts[split]=n
    return {'status':'PASS','episodes_by_split':counts,'unique_observable_worlds':len(semantics),'unique_principal_inputs':len(tokens),'model_outputs_seen':0}

def validate_deterministic_rule_anchor(spec,examples):
    """Reject a declared bit rule paired with a contradictory labelled example."""
    for x,label in examples:
        if not isinstance(x,str) or not set(x)<={'0','1'} or label not in (0,1):raise InputContractError('Invalid bit-rule example')
        i=spec['index'];flip=spec['invert']
        if not 0<=i<len(x) or flip not in (0,1):raise InputContractError('Invalid explicit rule')
        if int(x[i])^flip!=label:raise InputContractError('Explicit deterministic rule contradicts its labelled examples')
    return {'status':'PASS','examples':len(examples)}
