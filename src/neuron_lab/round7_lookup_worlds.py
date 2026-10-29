"""Fresh, oracle-only lookup worlds; no outcome-dependent input selection."""
from __future__ import annotations
import hashlib,json,random

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
X=('Mia','Leo','Ava','Ian');Y=('red','blue','green','gold');Z=('north','south','east','west')
def code(book,value):return next(k for k,v in book.items() if v==value)
def make(seed,eid):
    rng=random.Random(seed);ys=list(Y);zs=list(Z);rng.shuffle(ys);rng.shuffle(zs)
    f=dict(zip(X,ys));g=dict(zip(Y,zs));order=list(X);rng.shuffle(order);yg=list(Y);rng.shuffle(yg)
    books=[]
    for bi in range(2):
        zz=list(Z);rng.shuffle(zz);books.append(dict(zip('ABCD',zz)))
    return {'id':eid,'seed':seed,'f':f,'g':g,'order':order,'g_order':yg,'queries':list(X[:2]),'books':books,
            'signature':digest([f,g,order,yg,books])}
def manifest(cfg):
    splits={};seen=set()
    for si,(split,n) in enumerate(cfg['splits'].items()):
        out=[]
        for i in range(n):
            w=make(cfg['corpus_seed']+si*1000000+i,f'r7{cfg["study_id"].split("_")[1]}-{split}-{i:03}');assert w['signature'] not in seen
            seen.add(w['signature']);out.append(w)
        splits[split]=out
    return {'schema_version':1,'study':cfg['study_id'],'splits':splits,'seed':cfg['corpus_seed']}
def code_case(w,bi,study,value=None):
    value=value or Z[w['seed']%4];book=w['books'][bi]
    text='Return only the response code for the explicitly stated answer.\nCUE_BEGIN:\nUse the codebook literally.\nFACTS_BEGIN:\nThe answer value is '+value+'.\nEND_FACTS\nNEW_QUERY:\nCodebook: '+'; '.join(f'{k}={v}' for k,v in book.items())+'. Reply with one A/B/C/D.'
    return {'id':f'{w["id"]}-code-{bi}','episode_id':w['id'],'study':study,'text':text,'expected':code(book,value),'factors':{'codebook_index':bi},'oracle':{'codebook':book,'correct':value}}
def encode_quality(rows):
    return {k:{'correct':sum(bool(r['correct']) for r in vv),'rows':len(vv),'accuracy':sum(bool(r['correct']) for r in vv)/len(vv)} for k,vv in rows.items()}
