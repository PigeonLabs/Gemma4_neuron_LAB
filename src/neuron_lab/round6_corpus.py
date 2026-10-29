"""Fresh symbolic worlds and counterfactual oracles for the three R6 studies."""
from __future__ import annotations
import copy
import hashlib
import json
import random


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def _world(rng,xs,ys,zs,first=None):
    yy=list(ys);rng.shuffle(yy)
    f=dict(zip(xs,yy)) if first is None else dict(first)
    w={"f":f,"g":{y:rng.choice(zs) for y in ys},"h":{x:rng.choice(zs) for x in xs},"order":{}}
    for k in ("f","g","h"):
        w["order"][k]=list(w[k]);rng.shuffle(w["order"][k])
    return w


def _codes(rng,zs):
    z=list(zs);rng.shuffle(z)
    return dict(zip("ABCD",z))


def code_for(book,z):
    return next(k for k,v in book.items() if v==z)


def make_episode(seed,eid,study,split):
    rng=random.Random(seed)
    xs=[f"x{i:02}" for i in rng.sample(range(100),8)]
    ys=[f"y{i:02}" for i in rng.sample(range(100),8)]
    zs=[f"z{i}" for i in rng.sample(range(10),4)]
    rng.shuffle(xs);rng.shuffle(ys)
    ep={"id":eid,"seed":seed,"study":study,"split":split,"xs":xs,"ys":ys,"zs":zs,
        "cue_mapping":rng.randrange(2),"worlds":{},"cases":[]}
    phase_index={"technical":0,"discovery":0,"calibration":1,"confirmation":2}[split]
    ep["names"]=list([("f","g","h"),("u","v","w"),("p","q","r")][phase_index])
    ep["cue_label"]=["SELECTOR","ROUTE_KEY","SOURCE_KEY"][phase_index]
    if study=="A":
        labels=("receiver","donor","sham","rescue")
        ep["worlds"]={k:_world(rng,xs,ys,zs) for k in labels}
        for i,x in enumerate(xs):
            a,b,c,d=(zs[(i+j)%4] for j in range(4))
            for key,h,g in (("receiver",a,b),("donor",c,d),("sham",a,b),("rescue",c,d)):
                w=ep["worlds"][key];w["h"][x]=h;w["g"][w["f"][x]]=g
        for qi,x in enumerate(xs[:2]):
            a,b,c,d=(zs[(qi+j)%4] for j in range(4))
            for ci in range(2):
                book=_codes(rng,zs)
                remaining=[k for k in "ABCD" if book[k] not in {a,b}]
                rng.shuffle(remaining)
                db={remaining[0]:c,remaining[1]:d}
                restkeys=[k for k in "ABCD" if k not in db];restvals=[a,b];rng.shuffle(restvals)
                db.update(zip(restkeys,restvals));db={k:db[k] for k in "ABCD"}
                for cue in (0,1):
                    ep["cases"].append({"id":f"{eid}-q{qi}-c{ci}-s{cue}","query":x,"query_index":qi,
                        "codebook_index":ci,"selector":cue,"book":book,"donor_book":db,
                        "original":a if cue==0 else b,"target":b if cue==0 else a,
                        "donor_selected":c if cue==0 else d,"donor_opposite":d if cue==0 else c})
    elif study=="B":
        w=_world(rng,xs,ys,zs)
        for i,x in enumerate(xs):
            w["h"][x]=zs[i%4];w["g"][w["f"][x]]=zs[i%4]
        ep["worlds"]={"receiver":w}
        for qi,x in enumerate(xs[:2]):
            for ci in range(2):
                book=_codes(rng,zs)
                ep["cases"].append({"id":f"{eid}-q{qi}-c{ci}","query":x,"query_index":qi,
                    "codebook_index":ci,"selector":0,"book":book,"donor_book":book,
                    "original":zs[qi%4],"wrong1":zs[(qi+1)%4],"wrong2":zs[(qi+2)%4]})
    elif study=="C":
        labels=("receiver0","receiver1","donor","otherg","sham0","sham1","rescue_correct","rescue_wrong")
        ep["worlds"]={k:_world(rng,xs,ys,zs) for k in labels}
        # Each query uses two disjoint named intermediates. Complete each f to
        # a permutation before assigning g, so no oracle-changing collisions.
        for key,w in ep["worlds"].items():
            desired={xs[i]:ys[2*i+(1 if key in {"donor","otherg","rescue_wrong"} else 0)] for i in range(2)}
            free=[y for y in ys if y not in desired.values()];rng.shuffle(free)
            desired.update(zip(xs[2:],free));w["f"]=desired
        for qi in range(2):
            x,y0,y1=xs[qi],ys[2*qi],ys[2*qi+1]
            a,b,c,d=(zs[(qi+j)%4] for j in range(4))
            for k in ("receiver0","sham0"):
                ep["worlds"][k]["g"].update({y0:a,y1:b})
            for k in ("receiver1","sham1"):
                ep["worlds"][k]["g"].update({y0:a,y1:d})
            for k in ("donor","otherg","rescue_correct","rescue_wrong"):
                ep["worlds"][k]["g"].update({y0:c,y1:c})
            for ci in range(2):
                book=_codes(rng,zs)
                for function,target in ((0,b),(1,d)):
                    allowed=[k for k in "ABCD" if book[k] not in {a,target}]
                    letter=rng.choice(allowed);others=[z for z in zs if z!=c];rng.shuffle(others)
                    db=dict(zip([k for k in "ABCD" if k!=letter],others));db[letter]=c;db={k:db[k] for k in "ABCD"}
                    ep["cases"].append({"id":f"{eid}-q{qi}-c{ci}-g{function}","query":x,"query_index":qi,
                        "codebook_index":ci,"function":function,"selector":1,"book":book,"donor_book":db,
                        "original":a,"target":target,"donor_selected":c,"intermediate":y1,"original_intermediate":y0})
    else:
        raise ValueError(study)
    validate_episode(ep)
    return ep


def validate_episode(ep):
    for w in ep["worlds"].values():
        assert len(w["f"])==8 and len(set(w["f"].values()))==8
        assert set(w["f"].values())==set(w["g"])
    for c in ep["cases"]:
        assert set(c["book"])==set(c["donor_book"])==set("ABCD")
        if ep["study"]=="A":
            r,d=ep["worlds"]["receiver"],ep["worlds"]["donor"];x=c["query"]
            rvals={r["h"][x],r["g"][r["f"][x]]};dvals={d["h"][x],d["g"][d["f"][x]]}
            assert len(rvals|dvals)==4 and not rvals&dvals
            assert not {code_for(c["book"],z) for z in rvals}&{code_for(c["donor_book"],z) for z in dvals}
            assert answer(ep,c,"recipient")==c["original"] and answer(ep,c,"donor_opposite")==c["donor_opposite"]
        elif ep["study"]=="C":
            w=ep["worlds"][f"receiver{c['function']}"]
            assert w["g"][c["intermediate"]]==c["target"]
            assert len({c["original"],c["target"],c["donor_selected"]})==3
            assert code_for(c["donor_book"],c["donor_selected"]) not in {code_for(c["book"],c["original"]),code_for(c["book"],c["target"])}
            assert answer(ep,c,"recipient")==c["original"]


def resolve(ep,c,role,context="conflict"):
    study=ep["study"];cue=c["selector"];book=c["book"]
    if study=="A":
        key={"recipient":"receiver","donor_same":"donor","donor_opposite":"donor","semantic_sham":"sham",
             "independent_rescue_same":"rescue","independent_rescue_opposite":"rescue"}[role]
        if role.endswith("opposite"):cue=1-cue
        if role not in {"recipient","semantic_sham"}:book=c["donor_book"]
        w=copy.deepcopy(ep["worlds"][key])
        if context=="agreement":w["g"][w["f"][c["query"]]]=w["h"][c["query"]]
    elif study=="B":
        w=copy.deepcopy(ep["worlds"]["receiver"])
        if context in {"conflict1","conflict2"}:w["g"][w["f"][c["query"]]]=c["wrong1" if context=="conflict1" else "wrong2"]
    else:
        key={"recipient":f"receiver{c['function']}","intermediate_donor":"donor","same_intermediate_other_g_donor":"otherg",
             "semantic_sham":f"sham{c['function']}","independent_correct_intermediate_rescue":"rescue_correct",
             "independent_wrong_intermediate_rescue":"rescue_wrong"}[role]
        w=copy.deepcopy(ep["worlds"][key]);cue=1
        if role not in {"recipient","semantic_sham"}:book=c["donor_book"]
    return w,cue,book


def answer(ep,c,role="recipient",context="conflict",witness=None):
    w,cue,_=resolve(ep,c,role,context)
    if witness=="direct_only":cue=0
    if witness=="composition_only":cue=1
    return w["h"][c["query"]] if cue==0 else w["g"][w["f"][c["query"]]]


def render(ep,c,role="recipient",context="conflict",witness=None,explicit_z=None,fact_edit=None):
    w,cue,book=resolve(ep,c,role,context)
    f,g,h=ep["names"]
    if fact_edit:
        if fact_edit=="direct":w["h"][c["query"]]=c["donor_selected"]
        else:w["g"][w["f"][c["query"]]]=c["donor_selected"]
    if witness=="direct_only":cue=0
    if witness=="composition_only" or ep["study"]=="C":cue=1
    if explicit_z is not None:
        instruction="Use the explicit Z value below. Convert it with the response codebook."
    elif ep["study"]=="C" or witness=="composition_only":
        instruction=f"Use {f} from X to Y, then {g} from Y to Z."
    elif witness=="direct_only":
        instruction=f"Use {h} directly from X to Z."
    else:
        direct_key=ep["cue_mapping"];compose_key=1-direct_key
        instruction=f"Use only the supplied facts. Key {direct_key} means use {h} directly from X to Z. Key {compose_key} means use {f} from X to Y then {g} from Y to Z. The tables can disagree: obey the key."
    key=str(ep["cue_mapping"] if cue==0 else 1-ep["cue_mapping"])
    text=instruction+f"\nCUE_BEGIN:\n{ep['cue_label']}={key}\nFACTS_BEGIN:\n"
    if explicit_z is not None:
        text+=f"Z={explicit_z}"
    else:
        keys=("f","g") if ep["study"]=="C" or witness=="composition_only" else ("h",) if witness=="direct_only" else ("f","g","h")
        names={"f":f,"g":g,"h":h}
        text+="\n".join(names[k]+": "+",".join(f"{a}>{w[k][a]}" for a in w["order"][k]) for k in keys)
    text+="\nEND_FACTS\nNEW_QUERY:\n"+f"X={c['query']}; "+" ".join(f"{a}:{b}" for a,b in book.items())+". Reply with only one code A/B/C/D."
    z=(explicit_z if explicit_z is not None else w["h"][c["query"]] if cue==0 else w["g"][w["f"][c["query"]]])
    return {"text":text,"expected":code_for(book,z),"semantic_answer":z,"codebook":book}


def build_manifest(cfg):
    manifest={"schema_version":1,"seed":cfg["corpus_seed"],"unit":"episode contains all related worlds, queries, donors and codebooks","studies":{}}
    for si,study in enumerate("ABC"):
        splits={}
        for sj,(split,n) in enumerate([("technical",2),*cfg["splits"].items()]):
            splits[split]=[make_episode(cfg["corpus_seed"]+si*1000000+sj*10000+i,f"r6-{study}-{split}-{i:03}",study,split) for i in range(n)]
        manifest["studies"][study]=splits
    signatures=[digest(ep["worlds"]) for splits in manifest["studies"].values() for eps in splits.values() for ep in eps]
    assert len(signatures)==len(set(signatures))
    return manifest
