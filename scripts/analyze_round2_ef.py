#!/usr/bin/env python3
"""Frozen family-level analysis for the real R2-E/F experiments (CPU only)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neuron_lab.runtime import save_json
from neuron_lab.round2_ef import family_E_table, F_contrasts, interval, sign_p, ridge_predict


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted, previous = [1.] * len(values), 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(values) - rank) * values[index]))
        adjusted[index] = previous
    return adjusted


def safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def score(y, prediction, calibration_mean):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    mse = float(np.mean((y - prediction) ** 2))
    denominator = float(np.mean((y - y.mean()) ** 2))
    deployment_denominator = float(np.mean((y - calibration_mean) ** 2))
    return {"mse": mse, "mae": float(np.mean(abs(y - prediction))),
            "r2": 1 - mse / denominator if denominator > 1e-12 else None,
            "predictive_r2_relative_to_calibration_mean": 1 - mse / deployment_denominator if deployment_denominator > 1e-12 else None}


def prediction_contrast(y, p0, p1, seed, resamples):
    y, p0, p1 = (np.asarray(a, float) for a in (y, p0, p1))
    squared_reduction = (y - p0) ** 2 - (y - p1) ** 2
    absolute_reduction = abs(y - p0) - abs(y - p1)
    variance = float(np.mean((y - y.mean()) ** 2))
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(y), (resamples, len(y)))
    yy = y[ids]
    var = ((yy - yy.mean(1, keepdims=True)) ** 2).mean(1)
    valid = var > 1e-12
    boot = squared_reduction[ids].mean(1)[valid] / var[valid]
    low, high = np.quantile(boot, [.025, .975]) if len(boot) else (None, None)
    return {"n_families": len(y), "delta_r2": float(squared_reduction.mean() / variance) if variance > 1e-12 else None,
            "delta_r2_ci_low": float(low) if low is not None else None,
            "delta_r2_ci_high": float(high) if high is not None else None,
            "invalid_bootstrap_fraction": float(1 - valid.mean()),
            "mse_reduction": interval(squared_reduction, seed + 1, resamples),
            "mae_reduction": interval(absolute_reduction, seed + 2, resamples),
            "two_sided_sign_randomization_p": sign_p(squared_reduction, seed + 3),
            "resampling_scope": "confirmation families; fitted calibration parameters held fixed"}


def E_analysis(table, models, cfg):
    b = np.array([r["baseline_features"] for r in table])
    y, s = np.array([r["y"] for r in table]), np.array([r["s"] for r in table])
    predictions = {"baseline_only": ridge_predict(models["baseline_only"], b[:, :4]),
        "baseline_geometry": ridge_predict(models["baseline_geometry"], b),
        "baseline_geometry_controls": ridge_predict(models["baseline_geometry_controls"], np.column_stack([b, s])),
        "baseline_controls": ridge_predict(models["baseline_controls"], np.column_stack([b[:, :4], s]))}
    predicted_s = ridge_predict(models["control_from_baseline"], b[:, :4])
    predictions["zero_probe_chained"] = ridge_predict(models["baseline_controls"], np.column_stack([b[:, :4], predicted_s]))
    for key, model in models["control_realizations"].items():
        predictions[key] = ridge_predict(model, np.column_stack([b, [r[key] for r in table]]))
    seed, count = cfg["bootstrap_seed"], cfg["bootstrap_resamples"]
    primary = prediction_contrast(y, predictions["baseline_geometry"], predictions["baseline_geometry_controls"], seed, count)
    ordered = np.argsort([-r["magnitude"] for r in table], kind="stable")
    robustness = []
    for fraction in cfg["E"]["robustness_delete_top_fractions"]:
        delete = int(np.floor(len(y) * fraction))
        ids = ordered[delete:]
        item = prediction_contrast(y[ids], predictions["baseline_geometry"][ids], predictions["baseline_geometry_controls"][ids], seed + delete + 10, count)
        robustness.append({"deleted_top_fraction": fraction, "deleted_families": delete, **item})
    mags = np.array([r["magnitude"] for r in table])
    total = float(mags.sum())
    concentration = {str(k): float(mags[ordered[:k]].sum() / total) if total else None for k in (1, 3, 5)}
    enriched = [{**row, "predictions": {key: float(p[i]) for key, p in predictions.items()}} for i, row in enumerate(table)]
    return {"n_families": len(y), "primary": primary, "models": {key: score(y, p, models["calibration_target_mean"]) for key, p in predictions.items()},
        "robustness": robustness, "concentration_top_family_shares": concentration,
        "candidate_control_magnitude_correlation": safe_corr(y, s),
        "direction_realization_correlation": safe_corr([r["s_direction_0"] for r in table], [r["s_direction_1"] for r in table]),
        "candidate_response_zero_families": int(np.count_nonzero(mags == 0)),
        "total_top1_flip_condition_rows": sum(r["all_flip_count"] for r in table),
        "top1_flip_source_families": sum(r["all_flip_count"] > 0 for r in table),
        "measurement_scope": "control-probe-assisted prediction conditional on measured candidate geometry; no confirmation refit; not a zero-probe predictor",
        "zero_probe_scope": "separate chained model uses only baseline features; candidate/control intervention measurements are not its inputs",
        "family_predictions": enriched}


def ratio_interval(numerator, denominator, seed, count):
    a, b = np.asarray(numerator, float), np.asarray(denominator, float)
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(a), (count, len(a)))
    den = b[ids].mean(1)
    if np.any(den <= 0) or b.mean() <= 0:
        return {"status": "NOT_IDENTIFIABLE", "estimate": None, "reason": "bootstrap denominator crosses zero"}
    values = a[ids].mean(1) / den
    lo, hi = np.quantile(values, [.025, .975])
    return {"status": "PASS", "estimate": float(a.mean() / b.mean()), "ci_low": float(lo), "ci_high": float(hi), "definition": "ratio of paired family means"}


def F_analysis(table, cfg):
    result = {"n_source_families": len({r["family_id"] for r in table}), "groups": {}}
    for mode in cfg["F"]["modes"]:
        chosen = [r for r in table if r["mode"] == mode]
        stats = {key: interval([r[key] for r in chosen], cfg["bootstrap_seed"] + i + 100, cfg["bootstrap_resamples"])
                 for i, key in enumerate(chosen[0]) if key not in {"family_id", "mode"}}
        for i, key in enumerate(("interaction", "specificity", "direction_specificity", "first_stage_H003")):
            stats[key]["two_sided_sign_randomization_p"] = sign_p([r[key] for r in chosen], cfg["bootstrap_seed"] + 200 + i)
        damage_gate = stats["damage"]["ci_low"] > cfg["F"]["practical_damage_floor_nats"]
        ratios = {"raw_recovery_fraction": ratio_interval([r["raw_recovery"] for r in chosen], [r["damage"] for r in chosen], cfg["bootstrap_seed"] + 300, cfg["bootstrap_resamples"]),
                  "conditional_recovery_fraction": ratio_interval([r["interaction"] for r in chosen], [r["damage"] for r in chosen], cfg["bootstrap_seed"] + 301, cfg["bootstrap_resamples"])} if damage_gate else {
                  key: {"status": "NOT_IDENTIFIABLE", "estimate": None, "reason": "preregistered damage gate failed"} for key in ("raw_recovery_fraction", "conditional_recovery_fraction")}
        result["groups"][mode] = {"stats": stats, "damage_gate_passed": damage_gate, **ratios}
    result["family_contrasts"] = table
    return result


def quality_summary(E_rows, F_rows):
    baseline = [r for r in E_rows + F_rows if r["arm"] == "clean"]
    groups = defaultdict(list)
    for row in baseline:
        groups[row["study"], row["mode"], row["task"]].append(row)
    accuracies = [{"study": key[0], "mode": key[1], "task": key[2], "cases": len(rows),
        "forced_choice_correct": sum(r["forced_choice_correct"] for r in rows),
        "raw_top1_correct": sum(r["raw_top1_correct"] for r in rows),
        "top1_valid_AB": sum(r["top1_label_valid"] for r in rows)} for key, rows in sorted(groups.items())]
    controls = [r for r in E_rows if r["arm"] in {"random_neuron", "random_direction"}]
    match = {}
    for arm in ("random_neuron", "random_direction"):
        rows = [r for r in controls if r["arm"] == arm]
        errors = np.array([r["norm_relative_error"] for r in rows])
        match[arm] = {"rows": len(rows), "median_norm_relative_error": float(np.median(errors)),
            "max_norm_relative_error": float(errors.max()), "fraction_within_10_percent": float(np.mean(errors <= .1)),
            "zero_delivered_postnorm_rows": sum(r["postnorm_l2"] == 0 for r in rows)}
    f_bycase = defaultdict(dict)
    for r in F_rows:
        f_bycase[r["case_id"]][r["arm"]] = r
    ratios = []
    for arms in f_bycase.values():
        denom = arms["rescue"]["downstream_local_postnorm_l2"]
        if denom > 0:
            ratios.append(arms["random_rescue"]["downstream_local_postnorm_l2"] / denom)
    return {"baseline_accuracy": accuracies, "E_norm_matching": match,
        "E_candidate_outside_calibration_support_fraction": float(np.mean([r["outside_calibration_support"] for r in E_rows if r["arm"] == "candidate"])),
        "F_native_random_to_candidate_rescue_norm": {"defined_cases": len(ratios), "median": float(np.median(ratios)) if ratios else None,
            "fraction_within_half_to_twice": float(np.mean((np.array(ratios) >= .5) & (np.array(ratios) <= 2))) if ratios else None},
        "F_technical_restore_max_abs_logit_delta": max(r["max_abs_logit_restore_error"] for r in F_rows if r["arm"] == "technical_restore"),
        "metric": "signed correct-vs-incorrect A/B logit margin; forced-choice correctness and unrestricted next-token correctness reported separately"}


def analyze(phase):
    cfg = yaml.safe_load((ROOT / "configs/round2_ef.yaml").read_text())
    frozen = json.loads((ROOT / cfg["frozen_parameters"]).read_text())
    manifest = json.loads((ROOT / cfg["manifest"]).read_text())
    if sha(ROOT / cfg["manifest"]) != frozen["registration"]["manifest_sha256"]:
        raise ValueError("Manifest changed since calibration")
    for path, digest in frozen["registration"]["implementation_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Frozen analysis/implementation changed: {path}")
    phases = ["calibration", "confirmation"] if phase == "confirmation" else ["calibration"]
    result = {"schema_version": 1, "study_id": "round2_ef", "status": "PASS", "analysis_phase": phase,
        "model_revision": cfg["model_revision"], "manifest_sha256": sha(ROOT / cfg["manifest"]),
        "frozen_parameters_sha256": sha(ROOT / cfg["frozen_parameters"]), "source_runs": {}, "phases": {},
        "all_measurements": "actual Gemma 4 12B BF16 CUDA inference; synthetic tests are separate",
        "scientific_scope": "new study motivated by exposed Round 1 results; generated instances conditional on four authored task templates",
        "uncertainty_scope": "family resampling conditional on frozen calibration, model and control identities; not uncertainty over all possible neurons/models/templates"}
    for p in phases:
        summary = json.loads((ROOT / f"logs/summary/round2_ef_{p}.json").read_text())
        if summary["status"] != "PASS":
            raise ValueError("Cannot analyze incomplete/failed inference phase")
        run = ROOT / "runs" / summary["run_id"]
        for name, digest in summary["raw_sha256"].items():
            if sha(run / name) != digest:
                raise ValueError(f"Raw integrity mismatch: {name}")
        for name, digest in summary["protected_round1_sha256"].items():
            if sha(ROOT / name) != digest:
                raise ValueError("Round 1 registered artifacts changed")
        E_rows, F_rows = jsonl(run / "E_interventions.jsonl"), jsonl(run / "F_interventions.jsonl")
        cases = [c for c in manifest["cases"] if c["split"] == p]
        expected_cases = Counter(c["study"] for c in cases)
        if len(E_rows) != expected_cases["E"] * 33 or len(F_rows) != expected_cases["F"] * 14:
            raise ValueError("Incomplete registered condition matrix")
        if len({(r["case_id"], r["arm"], r.get("candidate_id"), r.get("sign"), r.get("seed")) for r in E_rows}) != len(E_rows):
            raise ValueError("Duplicate E condition rows")
        if len({(r["case_id"], r["arm"]) for r in F_rows}) != len(F_rows):
            raise ValueError("Duplicate F condition rows")
        e = E_analysis(family_E_table(E_rows, cases), frozen["E_models"], cfg)
        f = F_analysis(F_contrasts(F_rows, cases), cfg)
        q = quality_summary(E_rows, F_rows)
        result["phases"][p] = {"E": e, "F": f, "quality": q, "in_sample": p == "calibration"}
        result["source_runs"][p] = {k: summary[k] for k in ("run_id", "elapsed_seconds", "forward_calls", "combined_forward_calls", "E_rows", "F_rows", "peak_allocated_bytes", "peak_reserved_bytes", "raw_sha256", "registration")}
    if phase == "confirmation":
        conf = result["phases"]["confirmation"]
        e, f = conf["E"], conf["F"]["groups"]["translate"]
        e_p = e["primary"]["two_sided_sign_randomization_p"]
        f_p = f["stats"]["interaction"]["two_sided_sign_randomization_p"]
        ep, fp = holm([e_p, f_p])
        e["primary"]["holm_two_primary_p"] = ep
        f["stats"]["interaction"]["holm_two_primary_p"] = fp
        e_success = (e["primary"]["delta_r2"] is not None and e["primary"]["delta_r2"] >= cfg["E"]["success_minimum_delta_r2"]
            and e["primary"]["delta_r2_ci_low"] > 0 and e["primary"]["mae_reduction"]["mean"] > 0 and ep < .05
            and all(r["delta_r2"] is not None and r["delta_r2"] > 0 for r in e["robustness"]))
        e["hypothesis_status"] = "SUPPORTED_WITHIN_REGISTERED_SCOPE" if e_success else "NOT_SUPPORTED"
        if e["candidate_response_zero_families"] == e["n_families"] or e["primary"]["delta_r2"] is None:
            e["hypothesis_status"] = "NOT_IDENTIFIABLE"
        s = f["stats"]
        specific = s["specificity"]["ci_low"] > 0 and s["direction_specificity"]["ci_low"] > 0
        if not f["damage_gate_passed"]:
            f["hypothesis_status"] = "NOT_IDENTIFIABLE"
            f["reason"] = "The lower 95% CI for natural-counterfactual damage did not exceed the registered 0.10-nat floor"
        elif s["interaction"]["ci_low"] > cfg["F"]["practical_interaction_floor_nats"] and fp < .05 and specific and s["probability_interaction"]["mean"] > 0:
            f["hypothesis_status"] = "CONDITIONAL_COMPENSATION_EVIDENCE_REQUIRES_CONTROL_GEOMETRY_REVIEW"
        else:
            f["hypothesis_status"] = "NOT_SUPPORTED"
        result["co_primary_holm"] = {"E": ep, "F": fp, "number_of_tests": 2}
        result["total_condition_rows"] = sum(r["E_rows"] + r["F_rows"] for r in result["source_runs"].values())
        result["total_forward_calls"] = result["source_runs"]["confirmation"]["combined_forward_calls"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["calibration", "confirmation"], default="confirmation")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = analyze(args.phase)
    name = "round2_ef_analysis.json" if args.phase == "confirmation" else "round2_ef_calibration_analysis.json"
    path = ROOT / "logs/summary" / name
    if args.check:
        if json.loads(path.read_text()) != result:
            raise SystemExit("Saved analysis differs from frozen recomputation")
    else:
        save_json(path, result)
    text = {"status": "PASS", "analysis_phase": args.phase, "summary": str(path)}
    for phase, item in result["phases"].items():
        text[phase] = {"E_delta_r2": item["E"]["primary"]["delta_r2"],
            "F_damage": item["F"]["groups"]["translate"]["stats"]["damage"],
            "F_interaction": item["F"]["groups"]["translate"]["stats"]["interaction"]}
    print(json.dumps(text))


if __name__ == "__main__":
    main()
