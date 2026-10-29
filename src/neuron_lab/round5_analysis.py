"""Discovery-only R5 selection. Episode-resampled uncertainty is descriptive here."""
from __future__ import annotations
import itertools
import numpy as np


def auc(scores, errors):
    scores, errors = np.asarray(scores), np.asarray(errors, dtype=bool)
    if not errors.any() or errors.all():
        return None
    delta = scores[errors][:, None] - scores[~errors][None, :]
    return float(((delta > 0) + .5 * (delta == 0)).mean())


def paired_interval(values, resamples=10000, seed=51991):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite, nonempty episode contrasts required")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(axis=1)
    return {"point": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(),
            "episodes": len(values), "unit": "episode", "scope": "descriptive discovery selection"}


def summarize(rows, groups, cfg):
    dc = cfg["discovery"]
    clean = {r["case_id"]: r for r in rows if r["arm"] == "healthy"}
    arms = {g["id"]: [r for r in rows if r["arm"] == g["id"]] for g in groups}
    healthy = {w: float(np.mean([r["correct"] for r in clean.values() if r["witness"] == w]))
               for w in dc["witnesses"]}
    task_ok = (healthy["dual_route"] >= dc["healthy_dual_accuracy_min"] and
               healthy["code_map_only"] >= dc["healthy_code_map_accuracy_min"] and
               min(healthy["direct_only"], healthy["composition_only"]) >= dc["healthy_single_route_accuracy_min"])
    stats = []
    for g in groups:
        arm = arms[g["id"]]
        if not arm:
            continue
        if set(r["case_id"] for r in arm) != set(clean):
            raise ValueError("Incomplete paired discovery arm")
        damage = {}
        accuracy = {}
        for w in dc["witnesses"]:
            selected = [r for r in arm if r["witness"] == w]
            accuracy[w] = float(np.mean([r["correct"] for r in selected]))
            episodes = sorted(set(r["episode_id"] for r in selected))
            contrasts = [np.mean([int(clean[r["case_id"]]["correct"])-int(r["correct"])
                                  for r in selected if r["episode_id"] == e]) for e in episodes]
            damage[w] = paired_interval(contrasts, dc["bootstrap_resamples"], dc["bootstrap_seed"])
        dual = [r for r in arm if r["witness"] == "dual_route"]
        features = []
        for name in dc["error_features"]:
            score = auc([r["features"][name] for r in dual], [not r["correct"] for r in dual])
            if score is not None:
                features.append({"feature": name, "orientation": 1 if score >= .5 else -1,
                                 "auc": max(score, 1-score)})
        feature = max(features, key=lambda f:f["auc"]) if features else None
        route_difference = damage["composition_only"]["point"]-damage["direct_only"]["point"]
        selectivity = route_difference if g["orientation"] == "composition" else -route_difference
        eligible = (damage["dual_route"]["point"] >= dc["pair_point_damage_min"] and
                    selectivity >= dc["pair_route_damage_difference_min"] and
                    damage["code_map_only"]["point"] <= dc["pair_code_map_damage_max"] and
                    feature is not None and feature["auc"] >= dc["pair_error_auc_min"])
        stats.append({"id":g["id"], "layer":g["layer"], "size":len(g["coordinates"]),
                      "orientation":g["orientation"],"accuracy":accuracy,"damage":damage,
                      "route_selectivity":float(selectivity),"error_feature":feature,"eligible":bool(eligible)})
    pairs=[]
    by_id={g["id"]:g for g in groups}
    for a,b in itertools.product([s for s in stats if s["eligible"] and s["orientation"]=="composition"],
                                 [s for s in stats if s["eligible"] and s["orientation"]=="direct"]):
        ga,gb=by_id[a["id"]],by_id[b["id"]]
        if ga["layer"]==gb["layer"] and set(ga["coordinates"]) & set(gb["coordinates"]):
            continue
        if abs(a["damage"]["dual_route"]["point"]-b["damage"]["dual_route"]["point"])>dc["pair_total_damage_difference_max"]:
            continue
        pairs.append({"A":a["id"],"B":b["id"],"rank":[min(a["route_selectivity"],b["route_selectivity"]),
                      min(a["damage"]["dual_route"]["point"],b["damage"]["dual_route"]["point"]),
                      min(a["error_feature"]["auc"],b["error_feature"]["auc"])]})
    pairs.sort(key=lambda p:(*[-v for v in p["rank"]],p["A"],p["B"]))
    return {"healthy_accuracy":healthy,"task_readiness_passed":bool(task_ok),"candidates":stats,
            "eligible_pair_count":len(pairs),"selected_pair":pairs[0] if pairs else None,
            "selection_scope":"discovery only; requires independent calibration",
            "gate_outcome":("STOPPED_AT_TASK_READINESS" if not task_ok else
                            "DISCOVERY_PAIR_SELECTED" if pairs else "STOPPED_AT_DISCOVERY_LESION_GATE")}
