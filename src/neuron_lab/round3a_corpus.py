"""New, paired source families for R3-A; no factual background is required."""
from __future__ import annotations

from collections import Counter
import random
import string


def build_manifest(cfg):
    rng = random.Random(cfg['corpus_seed'])
    used, cases = set(), []
    colors = ['blue', 'green', 'red', 'gold', 'pink', 'white', 'black', 'brown', 'purple', 'orange']
    def word(n):
        while True:
            value = ''.join(rng.choice(string.ascii_lowercase) for _ in range(n))
            if value not in used:
                used.add(value)
                return value
    for split in ('qualification', 'discovery', 'calibration', 'confirmation'):
        for index in range(cfg[split]['families']):
            task = cfg['tasks'][index % 2]
            difficulty = cfg['difficulties'][(index // 2) % 2]
            family = f'r3a-{split}-{index:03d}'
            if task == 'reverse_vs_sort':
                source = word(5 if difficulty == 'easy' else 12)
                u, v = source[::-1], ''.join(sorted(source))
                while u == v:
                    source = word(5 if difficulty == 'easy' else 12)
                    u, v = source[::-1], ''.join(sorted(source))
                description = f'Input string: {source}.'
                actions = {1:'reverse the character order', -1:'sort the characters alphabetically'}
            else:
                count = 3 if difficulty == 'easy' else 8
                keys = [word(5 if difficulty == 'easy' else 7) for _ in range(count)]
                values = rng.sample(colors, count)
                query = rng.randrange(count)
                u, v = values[query], keys[query]
                description = 'Supplied dictionary: ' + '; '.join(f'{k}={x}' for k,x in zip(keys,values)) + f'. Query: {keys[query]}.'
                actions = {1:'translate the query using the supplied dictionary', -1:'copy the query exactly'}
            for pair in cfg[split]['label_pairs']:
                for r in (1,-1):
                    for c in (1,-1):
                        first, second = (u,v) if c==1 else (v,u)
                        action = actions[r]
                        prefix = {'qualification':'Select the exact result.', 'discovery':'Follow the stated operation using only the supplied input.',
                                  'calibration':'Compute the requested output from this input.', 'confirmation':'Use this instance to carry out the named operation.'}[split]
                        text = f'{prefix}\n{description}\nOperation: {action}.\n{pair[0]}: {first}\n{pair[1]}: {second}\nReply with only {pair[0]} or {pair[1]}.'
                        cases.append({'id':f'{family}-{pair}-r{r}-c{c}', 'family_id':family, 'split':split,
                            'task':task,'difficulty':difficulty,'code':pair,'r':r,'c':c,'u':u,'v':v,'text':text,
                            'correct_label':pair[0] if r*c==1 else pair[1]})
    validate_manifest(cases,cfg)
    return {'schema_version':1,'seed':cfg['corpus_seed'],'cases':cases,
            'unit':'all rule/mapping/code derivatives of one source stay in one split',
            'code_scope':'AB and CD both calibrated; no held-out-token claim'}


def validate_manifest(cases,cfg):
    if len({c['id'] for c in cases}) != len(cases) or len({c['text'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case or prompt')
    groups = {}
    source_split = {}
    for row in cases:
        if row['u']==row['v'] or row['r'] not in (-1,1) or row['c'] not in (-1,1):
            raise ValueError('Invalid oracle')
        if row['correct_label'] != row['code'][0 if row['r']*row['c']==1 else 1]:
            raise ValueError('Wrong label map')
        key = (row['task'],row['u'],row['v'])
        if source_split.setdefault(key,row['split']) != row['split']:
            raise ValueError('Source leakage across splits')
        groups.setdefault(row['family_id'],[]).append(row)
    for rows in groups.values():
        if len({r['split'] for r in rows})!=1:
            raise ValueError('Family leakage')
        split=rows[0]['split']
        expected={(p,r,c) for p in cfg[split]['label_pairs'] for r in (1,-1) for c in (1,-1)}
        if {(r['code'],r['r'],r['c']) for r in rows}!=expected or len(rows)!=len(expected):
            raise ValueError('Incomplete factorial family')
    counts=Counter(c['split'] for c in cases)
    for split in ('qualification','discovery','calibration','confirmation'):
        if counts[split]!=cfg[split]['families']*cfg[split]['variants_per_family']:
            raise ValueError('Unexpected split size')
