#!/usr/bin/env python3
"""Post-hoc R2 descriptive decompositions; no new confirmatory tests or fits."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def compute():
    source = ROOT / "logs/summary/round2_ef_analysis.json"
    s = json.loads(source.read_text())
    data = s["phases"]["confirmation"]["E"]["family_predictions"]
    groups = defaultdict(list)
    for row in data:
        groups[row["task"]].append(row)
    denominator = sum((r["y"] - r["predictions"]["baseline_geometry"]) ** 2 -
                      (r["y"] - r["predictions"]["baseline_geometry_controls"]) ** 2 for r in data)
    tasks, centered_y, centered_s = [], [], []
    between, within = 0., 0.
    for task, rows in sorted(groups.items()):
        y, ss, p0, p1 = (np.array([r[k] if k in r else r["predictions"][k] for r in rows]) for k in ("y", "s", "baseline_geometry", "baseline_geometry_controls"))
        delta = (y - p0) ** 2 - (y - p1) ** 2
        residual0, residual1 = y - p0, y - p1
        between += len(rows) * (residual0.mean() ** 2 - residual1.mean() ** 2)
        within += len(rows) * (residual0.var() - residual1.var())
        centered_y.extend(y - y.mean())
        centered_s.extend(ss - ss.mean())
        tasks.append({"task": task, "n_families": len(rows), "mse_reduction": float(delta.mean()),
            "share_of_total_squared_error_reduction": float(delta.sum() / denominator),
            "candidate_log_response_variance": float(y.var()),
            "candidate_control_log_response_correlation": float(np.corrcoef(y, ss)[0, 1])})
    run = ROOT / "runs" / s["source_runs"]["confirmation"]["run_id"]
    rows = [json.loads(line) for line in (run / "E_interventions.jsonl").read_text().splitlines()]
    clean = {r["case_id"]: r["y"] for r in rows if r["arm"] == "clean"}
    magnitudes = defaultdict(list)
    for r in rows:
        if r["arm"] == "candidate":
            magnitudes[r["candidate_id"]].append(abs(r["y"] - clean[r["case_id"]]))
    means = {cid: float(np.mean(values)) for cid, values in magnitudes.items()}
    return {"schema_version": 1, "analysis_type": "POST_HOC_DESCRIPTIVE_NOT_PREREGISTERED; no new hypothesis tests or confirmation refits",
        "source": "logs/summary/round2_ef_analysis.json", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "raw_E_sha256": hashlib.sha256((run / "E_interventions.jsonl").read_bytes()).hexdigest(),
        "tasks": tasks, "task_centered_candidate_control_correlation": float(np.corrcoef(centered_y, centered_s)[0, 1]),
        "squared_error_reduction_decomposition": {"total_sum": float(denominator), "task_mean_error_component": float(between),
            "within_task_error_variance_component": float(within), "identity_error": float(abs(denominator - between - within))},
        "candidate_mean_absolute_semantic_changes": means,
        "candidate_share_of_aggregated_absolute_change": {cid: value / sum(means.values()) for cid, value in means.items()},
        "limitations": "descriptive checks on exposed confirmation; cannot promote task/candidate effects to new confirmatory claims"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = compute()
    path = ROOT / "logs/summary/round2_ef_diagnostics.json"
    if args.check:
        if json.loads(path.read_text()) != result:
            raise SystemExit("R2 diagnostics are stale")
    else:
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))
