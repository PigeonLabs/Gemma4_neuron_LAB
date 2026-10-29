#!/usr/bin/env python3
"""Post-hoc R5-discovery margin diagnostics for R6 planning, CPU only.

No selection/gate or R5 file is changed. These selected descriptive effects are
not confirmation, semantic mediation, or evidence of behavioral recovery.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"logs/summary/round6_margin_diagnostics.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze():
    source=ROOT/"logs/summary/round5_discovery.json"
    r5=json.loads(source.read_text())
    raw=ROOT/"runs"/r5["run_id"]/"rows.jsonl"
    assert sha(raw)==r5["raw_rows_sha256"]
    rows=[json.loads(x) for x in raw.read_text().splitlines()]
    healthy={r["case_id"]:r for r in rows if r["arm"]=="healthy"}
    cells=defaultdict(list)
    max_margin_error=0.
    for r in rows:
        four=r["post_softcap_logits"]
        ix="ABCD".index(r["correct_code"])
        other=[x for i,x in enumerate(four) if i!=ix]
        maximum=max(other)
        margin=four[ix]-maximum-math.log(sum(math.exp(x-maximum) for x in other))
        max_margin_error=max(max_margin_error,abs(margin-r["margin"]))
        if r["arm"]!="healthy":
            b=healthy[r["case_id"]]
            assert r["witness"]==b["witness"] and r["episode_id"]==b["episode_id"]
            cells[r["arm"],r["witness"]].append({"episode_id":r["episode_id"],
                "margin_damage":b["margin"]-r["margin"],
                "correctness_damage":int(b["correct"])-int(r["correct"])})
    assert max_margin_error<1e-10
    result=[]
    for (arm,witness),values in sorted(cells.items()):
        episodes=sorted(set(x["episode_id"] for x in values))
        assert len(values)==24 and len(episodes)==8
        mean=lambda xs:sum(xs)/len(xs)
        episode_values=[{"episode_id":e,"mean_margin_damage":mean([x["margin_damage"] for x in values if x["episode_id"]==e])}
                        for e in episodes]
        result.append({"arm":arm,"witness":witness,"queries":24,"independent_episodes":8,
            "mean_margin_damage_nats":mean([x["margin_damage"] for x in values]),
            "mean_absolute_margin_change_nats":mean([abs(x["margin_damage"]) for x in values]),
            "maximum_absolute_margin_change_nats":max(abs(x["margin_damage"]) for x in values),
            "accuracy_damage":mean([x["correctness_damage"] for x in values]),"episode_means":episode_values})
    assert len(result)==64
    largest={w:max([r for r in result if r["witness"]==w],key=lambda r:r["mean_absolute_margin_change_nats"])["arm"]
             for w in r5["analysis"]["healthy_accuracy"]}
    healthy_margins={}
    for witness in r5["analysis"]["healthy_accuracy"]:
        margins=[r["margin"] for r in healthy.values() if r["witness"]==witness]
        healthy_margins[witness]={"minimum":min(margins),"median":statistics.median(margins),"maximum":max(margins),"queries":len(margins)}
    return {"schema_version":1,"status":"PASS","study_id":"R6_planning",
        "scope":"post-hoc descriptive reanalysis of R5 discovery only; not a new 12B run or preregistered R5 endpoint",
        "new_GPU_forwards":0,"R5_gate_unchanged":r5["analysis"]["gate_outcome"],
        "not_claimed":["zero_causal_effect","behavioral_recovery","semantic_bottleneck","independent_confirmation"],
        "metric":"correct code post-softcap logit minus logsumexp of other three codes",
        "damage_sign":"healthy minus ablated; positive is margin decrease",
        "aggregation":"all 24 queries, three per independent episode; no correctness-conditioned subset",
        "uncertainty":"descriptive only; no p-values or selected-candidate confirmatory intervals",
        "largest_absolute_mean_selection":"post-hoc maximum among all 16 groups, for diagnostic orientation only",
        "largest_absolute_mean_arm_by_witness":largest,"cells":result,
        "healthy_margin_nats_by_witness":healthy_margins,
        "maximum_independent_margin_replay_error":max_margin_error,
        "source_run_id":r5["run_id"],"source_result_sha256":sha(source),"source_raw_rows_sha256":sha(raw),
        "analysis_code_sha256":sha(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--check",action="store_true");a=p.parse_args()
    result=analyze()
    if a.check:
        assert json.loads(DEST.read_text())==result,"Post-hoc diagnostic receipt is stale"
    else:
        DEST.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"status":"PASS","source":"R5 discovery","new_GPU_forwards":0,"cells":len(result["cells"]),
                      "max_margin_replay_error":result["maximum_independent_margin_replay_error"]}))


if __name__=="__main__":
    main()
