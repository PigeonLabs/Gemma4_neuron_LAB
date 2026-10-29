#!/usr/bin/env python3
"""Post-hoc, CPU-only concentration diagnostics; never changes Round 1 results.

The raw run files remain local. --check recomputes and compares the small public
summary. Repeated conditions/seeds are not treated as independent prompts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neuron_lab.corpus import build_corpus, validate_corpus

OUTPUT = ROOT / "logs/summary/round2_diagnostics.json"
STAGES = ("steering", "validation", "expanded")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(rows, field):
    return dict(sorted(Counter(row[field] for row in rows).items()))


def pearson(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    denominator = math.sqrt(sum(x*x for x in dx) * sum(y*y for y in dy))
    return sum(x*y for x, y in zip(dx, dy)) / denominator if denominator else None


def analyze():
    corpus = build_corpus()
    validate_corpus(corpus)
    prompts = {row["id"]: row for row in corpus}
    round1 = json.loads((ROOT / "logs/summary/analysis.json").read_text())
    exact_checks = []
    for candidate in round1["candidates"]:
        families = candidate["counterexamples"]["families"]
        positive = [f for f in families if f["expected_activation_group"] == "high"]
        negative = [f for f in families if f["expected_activation_group"] == "low"]
        if len(positive) != 4 or len(negative) != 4:
            raise ValueError("Round 1 counterexample family size changed")
        scores = [f["prompt_mean"] for f in positive + negative]

        def auc(indices):
            selected = set(indices)
            return sum((scores[i] > scores[j]) + 0.5 * (scores[i] == scores[j])
                       for i in selected for j in range(8) if j not in selected) / 16

        observed = auc(range(4))
        extreme = sum(abs(auc(ix) - 0.5) >= abs(observed - 0.5) - 1e-12
                      for ix in combinations(range(8), 4))
        exact_checks.append({"id": candidate["id"], "auc": observed, "permutations": 70,
                             "two_sided_extreme_permutations": extreme, "p_exact_two_sided": extreme / 70,
                             "registered_monte_carlo_p_two_sided": candidate["counterexamples"]["family_level"]["permutation_p_two_sided"]})
    maximum = 0
    for rank, item in enumerate(sorted(exact_checks, key=lambda d: d["p_exact_two_sided"])):
        maximum = max(maximum, min(1, (4 - rank) * item["p_exact_two_sided"]))
        item["p_exact_holm"] = maximum
    result = {
        "status": "PASS",
        "analysis_type": "post_hoc_exploratory; no new model execution",
        "code_sha256": digest(Path(__file__)),
        "corpus_code_sha256": digest(ROOT / "src/neuron_lab/corpus.py"),
        "registered_round1_analysis_sha256": digest(ROOT / "logs/summary/analysis.json"),
        "unit": "condition row for row counts; unique authored prompt for prompt counts",
        "counterexample_exact_sensitivity_analysis": {
            "label": "post-hoc exhaustive check; does not replace registered Monte Carlo results",
            "statistic": "abs(AUC-0.5); all choose(8,4)=70 label allocations; family means retained",
            "ideal_no_tie_4v4_min_two_sided_p": 2 / math.comb(8, 4),
            "ideal_no_tie_4v4_min_first_step_holm_p_four_tests": 8 / math.comb(8, 4),
            "ideal_no_tie_5v5_min_first_step_holm_p_four_tests": 8 / math.comb(10, 5),
            "candidates": exact_checks,
            "limit": "The exact-test floor is not a strict bound on a finite Monte Carlo estimate; five versus five fixes granularity only, not power.",
        },
        "limits": [
            "No answer margin or semantic correctness of top1 changes was saved.",
            "Concentration suggests a hypothesis; it does not identify a mechanism.",
            "Pooled KL weights the original heterogeneous arm/seed schedule; splits are not comparable treatment mixtures.",
            "Candidate/control Pearson correlations are descriptive on the same 45 prompts, not independent replications.",
            "Round 1 test has now informed hypotheses; Round 2 confirmation requires entirely new source families.",
        ],
        "splits": {},
    }
    for stage in STAGES:
        summary_path = ROOT / f"logs/summary/{stage}.json"
        summary = json.loads(summary_path.read_text())
        raw_path = ROOT / "runs" / summary["run_id"] / "interventions.jsonl"
        raw_digest = digest(raw_path)
        if raw_digest != summary["raw_sha256"]:
            raise ValueError(f"Raw hash mismatch: {stage}")
        all_rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
        if len(all_rows) != summary["intervention_count"]:
            raise ValueError(f"Row count mismatch: {stage}")
        rows = [r for r in all_rows if r["role"] == "corpus"]
        keys = [(r["candidate_id"], r["prompt_id"], r["condition"], r["seed"]) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate condition keys: {stage}")
        for row in rows:
            prompt = prompts[row["prompt_id"]]
            if any(row[k] != prompt[k] for k in ("source_id", "category", "split")):
                raise ValueError(f"Prompt provenance mismatch: {stage}")
            if not math.isfinite(row["kl_base_to_changed"]) or row["kl_base_to_changed"] < 0:
                raise ValueError(f"Invalid KL: {stage}")
        flips = [r for r in rows if r["top1_changed"]]
        baseline = {(r["candidate_id"], r["prompt_id"]): r["generation"]["token_ids"]
                    for r in rows if r["condition"] == "baseline" and "generation" in r}
        generated = [r for r in rows if r["condition"] != "baseline" and "generation" in r]
        for row in generated:
            changed = row["generation"]["token_ids"] != baseline[row["candidate_id"], row["prompt_id"]]
            if changed != row["generation"]["changed_from_baseline"]:
                raise ValueError(f"Generation flag mismatch: {stage}")
        changed_generated = [r for r in generated if r["generation"]["changed_from_baseline"]]
        kl_by_prompt = defaultdict(float)
        for row in rows:
            if row["condition"] not in {"baseline", "activation_restoration"}:
                kl_by_prompt[row["prompt_id"]] += row["kl_base_to_changed"]
        total_kl = sum(kl_by_prompt.values())
        shares = [{"prompt_id": pid, "source_id": prompts[pid]["source_id"],
                   "pooled_kl_sum": value, "share": value / total_kl if total_kl else None}
                  for pid, value in sorted(kl_by_prompt.items(), key=lambda kv: (-kv[1], kv[0]))]
        correlations = {}
        for candidate_id in sorted({r["candidate_id"] for r in rows}):
            candidate = {r["prompt_id"]: r["kl_base_to_changed"] for r in rows
                         if r["candidate_id"] == candidate_id and r["condition"] == "add_+1sd"}
            controls = defaultdict(list)
            for row in rows:
                if row["candidate_id"] == candidate_id and row["condition"] == "random_direction_projected":
                    controls[row["prompt_id"]].append(row["kl_base_to_changed"])
            if set(candidate) != set(controls):
                raise ValueError(f"Unpaired controls: {stage}/{candidate_id}")
            pids = sorted(candidate)
            correlations[candidate_id] = {
                "n_prompts": len(pids), "control_seed_counts": sorted({len(controls[p]) for p in pids}),
                "pearson_r_candidate_kl_vs_control_seed_mean_kl": pearson(
                    [candidate[p] for p in pids], [statistics.mean(controls[p]) for p in pids]),
            }
        result["splits"][summary["split"]] = {
            "run_id": summary["run_id"], "model_revision": summary["model_revision"],
            "summary_path": summary_path.relative_to(ROOT).as_posix(),
            "summary_sha256": digest(summary_path),
            "raw_path": raw_path.relative_to(ROOT).as_posix(), "raw_sha256": raw_digest,
            "all_rows": len(all_rows), "corpus_rows": len(rows),
            "corpus_prompts": len({r["prompt_id"] for r in rows}),
            "condition_row_counts": counts(rows, "condition"),
            "top1_change": {
                "condition_rows": len(flips), "unique_prompts": len({r["prompt_id"] for r in flips}),
                "by_prompt": counts(flips, "prompt_id"), "by_candidate": counts(flips, "candidate_id"),
                "by_condition": counts(flips, "condition"),
                "prompt_details": [prompts[p] for p in sorted({r["prompt_id"] for r in flips})],
            },
            "generation_change": {
                "nonbaseline_comparisons": len(generated), "changed_comparisons": len(changed_generated),
                "unique_evaluated_prompts": len({r["prompt_id"] for r in generated}),
                "unique_changed_prompts": len({r["prompt_id"] for r in changed_generated}),
                "by_prompt": counts(changed_generated, "prompt_id"),
            },
            "pooled_kl_concentration": {"total": total_kl, "top3_share": sum(s["share"] for s in shares[:3]),
                                        "prompt_shares": shares},
            "candidate_control_kl_correlations": correlations,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = analyze()
    if args.check:
        if not OUTPUT.exists() or json.loads(OUTPUT.read_text()) != result:
            raise SystemExit("Round 2 exploratory summary is missing or stale")
    else:
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "output": str(OUTPUT), "check_only": args.check,
                      "top1_change_rows": {s: d["top1_change"]["condition_rows"] for s, d in result["splits"].items()},
                      "top1_changed_prompts": {s: d["top1_change"]["unique_prompts"] for s, d in result["splits"].items()}}))
