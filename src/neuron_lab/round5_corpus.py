"""Frozen, fresh R5 task instances; no model inference or answer-based filtering."""
from __future__ import annotations

import hashlib
import json
from .round5_design import make_episode, validate_episode, reference_record


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def render(episode, case, witness="dual_route", statuses=None):
    if witness not in {"direct_only", "composition_only", "dual_route", "code_map_only"}:
        raise ValueError("Unknown operational witness")
    def pairs(key):
        return ",".join(f"{a}>{episode[key][a]}" for a in episode["orders"][key])
    head = "Use only the supplied information to find the Z value for the queried X. "
    if witness == "direct_only":
        head += "Use the direct table h from X to Z.\nh: " + pairs("direct")
    elif witness == "composition_only":
        head += "First use f from X to Y, then g from Y to Z.\nf: " + pairs("first") + "\ng: " + pairs("second")
    elif witness == "dual_route":
        head += ("f maps X to Y, g maps Y to Z, and h maps X to Z. Both routes agree.\n"
                 + "f: " + pairs("first") + "\ng: " + pairs("second") + "\nh: " + pairs("direct"))
    else:
        head = "Convert the explicitly supplied Z value into its response code."
    if statuses is not None:
        if witness != "dual_route" or len(statuses) > len(episode["train"]):
            raise ValueError("Invalid history scope")
        head += ("\nEvaluator records: Reference is the evaluator's correct answer, not your previous output. "
                 "diagnostic-pass 0=incorrect,1=correct,2=unreported.\n")
        head += "\n".join(reference_record(c,s) for c,s in zip(episode["train"],statuses))
    codes = " ".join(f"{a}:{b}" for a,b in case["codebook"].items())
    query = (f"Z={case['answer']}" if witness == "code_map_only" else f"X={case['query']}")
    return head + f"\nNEW_QUERY:\n{query}; {codes}. Find the Z value and reply with only its code A/B/C/D."


def build_manifest(cfg):
    result = {"schema_version":1,"study_id":cfg["study_id"],"seed":cfg["corpus_seed"],
              "unit":"all queries and witness derivatives of an episode stay in one split", "splits":{}}
    for split, offset, count in (("technical", 0, cfg["technical_prompts"]),
            ("discovery",100000,cfg["discovery"]["episodes"]),
            ("calibration",200000,cfg["calibration"]["episodes"]),
            ("confirmation",300000,cfg["confirmation"]["episodes"])):
        result["splits"][split] = [make_episode(cfg["corpus_seed"]+offset+i, f"r5-{split}-{i:03}") for i in range(count)]
    signatures = set()
    for episodes in result["splits"].values():
        for e in episodes:
            validate_episode(e)
            signature=digest({k:e[k] for k in ("first","second","direct","orders")})
            if signature in signatures:
                raise ValueError("Repeated source episode between splits")
            signatures.add(signature)
    return result


def discovery_cases(manifest,cfg):
    rows=[]
    for e in manifest["splits"]["discovery"]:
        for c in e["test"][:cfg["discovery"]["probe_queries_per_episode"]]:
            for w in cfg["discovery"]["witnesses"]:
                rows.append({"id":c["id"]+"-"+w,"episode_id":e["id"],"split":"discovery",
                    "witness":w,"case":c,"text":render(e,c,w),"features":{
                    "direct_row_fraction":e["orders"]["direct"].index(c["query"])/15,
                    "first_row_fraction":e["orders"]["first"].index(c["query"])/15,
                    "second_row_fraction":e["orders"]["second"].index(e["first"][c["query"]])/15}})
    if len(rows)!=cfg["budget_explanation"]["discovery_screen_cases"]:
        raise ValueError("Discovery cell count does not match registered budget")
    return rows
