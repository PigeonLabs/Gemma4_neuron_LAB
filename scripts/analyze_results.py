#!/usr/bin/env python3
"""Recompute registered diagnostics from measured, hash-verified experiment runs.

Run with ``.venv/bin/python scripts/analyze_results.py`` to write the public
analysis and conservative neuron catalog. ``--check`` validates sources and,
when outputs already exist, checks freshness without changing any files.
Missing experimental stages remain NOT_EVALUATED. No model/GPU is loaded.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neuron_lab.corpus import build_corpus, validate_corpus
from neuron_lab.evaluation import permutation_auc_p, roc_auc

SEED = 20261029
RESAMPLES = 2000
STAGES = {"steering": "discovery", "validation": "validation", "expanded": "test"}
ROLES = ("positive_control", "negative_control", "near_miss", "paraphrase", "out_of_distribution")
HIGH_ROLES = {"positive_control", "paraphrase", "out_of_distribution"}
OUTPUTS = ("logs/summary/analysis.json", "catalog/public/neurons.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def loads(value: str):
    return json.loads(value, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)


def load(path: Path):
    return loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name: str, nonnegative=False) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"Invalid finite number: {name}")
    require(not nonnegative or value >= 0, f"Negative value: {name}")
    return float(value)


def not_evaluated(reason: str) -> dict:
    return {"status": "NOT_EVALUATED", "reason": reason}


def descriptive(values) -> dict:
    """Descriptive summaries only; seeds do not count as prompt replicates."""
    a = np.asarray(list(values), dtype=np.float64)
    if not len(a):
        return {"status": "NOT_EVALUATED", "n": 0, "mean": None, "min": None, "median": None, "max": None}
    require(np.isfinite(a).all(), "Nonfinite descriptive input")
    return {"status": "PASS", "n": len(a), "mean": float(a.mean()), "min": float(a.min()),
            "median": float(np.median(a)), "max": float(a.max())}


def holm_adjust(p_values: dict[str, float], family_size: int = 4) -> dict[str, float]:
    """Holm FWER correction; absent hypotheses still count in the fixed family."""
    require(len(p_values) <= family_size, "More p-values than registered hypotheses")
    adjusted, previous = {}, 0.0
    for rank, (key, p) in enumerate(sorted(p_values.items(), key=lambda pair: (pair[1], pair[0]))):
        number(p, "p-value", nonnegative=True)
        require(p <= 1, "p-value exceeds one")
        previous = max(previous, min(1.0, (family_size - rank) * p))
        adjusted[key] = previous
    return adjusted


def auc_summary(positive: list[float], negative: list[float]) -> dict:
    if not positive or not negative:
        return not_evaluated("Both positive and negative prompt/source groups are required")
    return {"status": "PASS", "n_positive": len(positive), "n_negative": len(negative),
            "auc": roc_auc(positive, negative), "positive_mean": float(np.mean(positive)),
            "negative_mean": float(np.mean(negative)),
            "permutation_p_two_sided": permutation_auc_p(positive, negative, seed=SEED, resamples=RESAMPLES)}


def stratified_interaction(positive: list[float], negative: list[float]) -> dict:
    """Prompt bootstrap within fixed target/other strata, never over seeds."""
    if not positive or not negative:
        return not_evaluated("Both target and other corpus prompts are required")
    p, n = np.asarray(positive, dtype=np.float64), np.asarray(negative, dtype=np.float64)
    require(np.isfinite(p).all() and np.isfinite(n).all(), "Nonfinite causal signature values")
    rng = np.random.default_rng(SEED)
    sampled_p = p[rng.integers(0, len(p), size=(RESAMPLES, len(p)))].mean(axis=1)
    sampled_n = n[rng.integers(0, len(n), size=(RESAMPLES, len(n)))].mean(axis=1)
    low, high = np.percentile(sampled_p - sampled_n, [2.5, 97.5])
    return {"status": "PASS", "n_positive": len(p), "n_negative": len(n),
            "target_delta_mean": float(p.mean()), "other_delta_mean": float(n.mean()),
            "interaction_mean": float(p.mean() - n.mean()), "ci_low": float(low), "ci_high": float(high),
            "positive_lower_bound": bool(low > 0), "seed": SEED, "resamples": RESAMPLES,
            "ci_method": "95% percentile stratified bootstrap of independent corpus prompts; random seeds averaged within prompt",
            "units": "nats", "semantic_causal_claim": False}


def load_inputs(root: Path):
    frozen_path = root / "configs/hypotheses.json"
    cases_path = root / "configs/validation_cases.json"
    frozen, cases = load(frozen_path), load(cases_path)
    frozen_hash = sha256(frozen_path)
    require(isinstance(frozen.get("candidates"), list) and len(frozen["candidates"]) == 4, "Exactly four frozen hypotheses are required")
    candidates = {c["id"]: c for c in frozen["candidates"]}
    require(len(candidates) == 4, "Duplicate hypothesis ID")
    coordinates = set()
    for c in candidates.values():
        require(isinstance(c.get("layer"), int) and c["layer"] >= 0, "Invalid candidate layer")
        require(isinstance(c.get("neuron"), int) and c["neuron"] >= 0, "Invalid candidate neuron")
        require((c["layer"], c["neuron"]) not in coordinates, "Duplicate candidate coordinate")
        coordinates.add((c["layer"], c["neuron"]))
        number(c.get("std"), "discovery std", nonnegative=True)
        require(isinstance(c.get("target_category"), str), "Missing fixed target category")
    require(cases.get("hypotheses_sha256") == frozen_hash, "Counterexamples do not match frozen hypotheses hash")
    require(isinstance(cases.get("prompts"), list), "Missing counterexample prompts")
    corpus = build_corpus()
    validate_corpus(corpus + cases["prompts"])
    for case in cases["prompts"]:
        require(case.get("candidate_id") in candidates and case.get("role") in ROLES and case.get("split") == "validation", "Invalid registered counterexample")
        expected = "high" if case["role"] in HIGH_ROLES else "low"
        require(case.get("expected_activation_group") == expected, "Counterexample role/expectation mismatch")
    sources = {"configs/hypotheses.json": frozen_hash, "configs/validation_cases.json": sha256(cases_path)}
    survey_path = root / "logs/summary/survey.json"
    survey = None
    if survey_path.exists():
        sources["logs/summary/survey.json"] = sha256(survey_path)
        require(sources["logs/summary/survey.json"] == frozen.get("discovery_summary_sha256"), "Discovery summary changed after hypotheses were frozen")
        survey = load(survey_path)
        require(survey.get("model_revision") == frozen.get("model_revision") and survey.get("status") == "PASS", "Discovery model/status mismatch")
    return frozen, cases, corpus, survey, sources


def load_stage(root: Path, stage: str, split: str, frozen: dict, cases: dict, corpus: list, sources: dict):
    relative = f"logs/summary/{stage}.json"
    summary_path = root / relative
    if not summary_path.exists():
        return None, {"stage": stage, "split": split, **not_evaluated("Stage summary is not available")}
    summary = load(summary_path)
    sources[relative] = sha256(summary_path)
    if summary.get("status") != "PASS":
        return None, {"stage": stage, "split": split, "run_id": summary.get("run_id"), **not_evaluated("Stage did not finish with PASS")}
    run_id = summary.get("run_id", "")
    require(isinstance(run_id, str) and re.fullmatch(r"[0-9]{8}T[0-9]{6}_" + re.escape(stage), run_id), f"Unsafe or mismatched run_id for {stage}")
    require(summary.get("stage") == stage and summary.get("split") == split, f"Stage/split mismatch for {stage}")
    require(summary.get("model_revision") == frozen.get("model_revision"), f"Model revision mismatch for {stage}")
    require(summary.get("hypotheses_sha256") == sources["configs/hypotheses.json"], f"Frozen hypotheses hash mismatch for {stage}")
    run_dir = root / "runs" / run_id
    raw_path, obs_path = run_dir / "interventions.jsonl", run_dir / "activation_observations.json"
    require(raw_path.is_file(), f"Missing raw interventions for {stage}")
    raw_hash = sha256(raw_path)
    require(raw_hash == summary.get("raw_sha256"), f"Raw SHA256 mismatch for {stage}")
    sources[str(raw_path.relative_to(root))] = raw_hash
    require(obs_path.is_file(), f"Missing local activation observations for {stage}")
    observations = load(obs_path)
    require(observations == summary.get("activation_observations"), f"Public/local activation observations disagree for {stage}")
    sources[str(obs_path.relative_to(root))] = sha256(obs_path)
    metadata_path = run_dir / "metadata.json"
    metadata = load(metadata_path)
    require(metadata.get("run_id") == run_id and metadata.get("status") == "PASS", f"Local run metadata is not PASS for {stage}")
    cfg = metadata.get("config", {})
    require(cfg.get("hypotheses_sha256") == sources["configs/hypotheses.json"], f"Metadata hypotheses hash mismatch for {stage}")
    seeds = cfg.get("control_seeds", [])
    require(len(seeds) > 0 and len(seeds) == len(set(seeds)) and all(isinstance(s, int) for s in seeds), f"Invalid control seeds for {stage}")
    sources[str(metadata_path.relative_to(root))] = sha256(metadata_path)
    selected_candidates = [c for c in frozen["candidates"] if stage != "steering" or c["group"] == "primary"]
    by_candidate = {c["id"]: c for c in selected_candidates}
    expected_corpus = [p for p in corpus if p["split"] == split]
    if stage == "steering":
        seen = set()
        expected_corpus = [p for p in expected_corpus if p["category"] not in seen and not seen.add(p["category"])]
    expected_cases = cases["prompts"] if stage == "validation" else []
    prompts = {p["id"]: p for p in expected_corpus + expected_cases}
    require(summary.get("corpus_prompt_count") == len(expected_corpus), f"Corpus count mismatch for {stage}")
    require(summary.get("counterexample_prompt_count") == len(expected_cases), f"Case count mismatch for {stage}")
    require(summary.get("prompt_count") == len(prompts) and summary.get("candidate_count") == len(by_candidate), f"Run prompt/candidate count mismatch for {stage}")
    rows = [loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(rows) == summary.get("intervention_count"), f"Raw intervention count mismatch for {stage}")
    seen_rows = set()
    for row in rows:
        cid, pid = row.get("candidate_id"), row.get("prompt_id")
        require(cid in by_candidate and pid in prompts, f"Unknown raw candidate/prompt in {stage}")
        candidate, prompt = by_candidate[cid], prompts[pid]
        require((row.get("layer"), row.get("neuron")) == (candidate["layer"], candidate["neuron"]), f"Intervention coordinate mismatch for {cid}")
        require(not prompt.get("candidate_id") or prompt["candidate_id"] == cid, f"Counterexample assigned to the wrong candidate in {stage}")
        for field in ("source_id", "category", "split"):
            require(row.get(field) == prompt[field], f"Raw {field} mismatch for {pid}")
        require(row.get("role") == prompt.get("role", "corpus"), f"Raw role mismatch for {pid}")
        key = (cid, pid, row.get("condition"), row.get("seed"))
        require(key not in seen_rows, f"Duplicate intervention row {key}")
        seen_rows.add(key)
        for field in ("kl_base_to_changed", "js_divergence", "total_variation", "max_abs_logit_delta", "activation_l2", "projected_l2"):
            number(row.get(field), field, nonnegative=True)
        require(isinstance(row.get("top1_changed"), bool), "top1_changed must be boolean")
        if row.get("condition", "").startswith("random_"):
            require(row.get("seed") in seeds, "Unregistered random-control seed")
    strengths = cfg.get("strengths_sigma") if stage == "steering" else [-1.0, 1.0]
    expected_conditions = {("baseline", None), ("zero_ablation", None), ("activation_restoration", None)}
    expected_conditions |= {(f"add_{v:+g}sd", None) for v in strengths}
    expected_conditions |= {(f"{kind}_{space}", seed) for kind in ("random_neuron", "random_direction") for space in ("activation", "projected") for seed in seeds}
    expected_rows = {(c["id"], p["id"], condition, seed) for c in selected_candidates for p in prompts.values()
                     if not p.get("candidate_id") or p["candidate_id"] == c["id"] for condition, seed in expected_conditions}
    require(seen_rows == expected_rows, f"Incomplete or unexpected condition matrix for {stage}")
    require(summary.get("restoration_logit_exact") is True and all(r["max_abs_logit_delta"] == 0 for r in rows if r["condition"] == "activation_restoration"), f"Restoration parity mismatch for {stage}")
    obs_keys = set()
    for row in observations:
        cid, pid = row.get("candidate_id"), row.get("prompt_id")
        require(cid in by_candidate and pid in prompts, f"Unknown activation observation in {stage}")
        prompt = prompts[pid]
        for field in ("source_id", "category", "split"):
            require(row.get(field) == prompt[field], f"Observation {field} mismatch for {pid}")
        require(row.get("role") == prompt.get("role", "corpus"), f"Observation role mismatch for {pid}")
        require((cid, pid) not in obs_keys, f"Duplicate activation observation in {stage}")
        obs_keys.add((cid, pid))
        number(row.get("prompt_mean"), "prompt_mean")
        number(row.get("last_activation"), "last_activation")
    # The capture pass measures all candidates on every prompt. Interpretation
    # below additionally joins candidate-specific cases to their owning ID.
    require(obs_keys == {(cid, pid) for cid in by_candidate for pid in prompts}, f"Incomplete activation observation matrix for {stage}")
    data = {"summary": summary, "rows": rows, "observations": observations, "seeds": seeds, "run_id": run_id}
    receipt = {"stage": stage, "split": split, "status": "PASS", "run_id": run_id, "summary_sha256": sources[relative],
               "raw_sha256": raw_hash, "raw_hash_verified": True, "schema_verified": True, "n_interventions": len(rows)}
    return data, receipt


def norm_matching(candidate: dict, data: dict) -> dict:
    rows = [r for r in data["rows"] if r["candidate_id"] == candidate["id"] and r["role"] == "corpus"]
    if not rows:
        return not_evaluated("Candidate was not included in this stage")
    target = {r["prompt_id"]: r for r in rows if r["condition"] == "add_+1sd"}
    controls = [r for r in rows if r["condition"] == "random_direction_projected"]
    ratios, prompt_ratios = [], defaultdict(list)
    zero_denominators = 0
    for control in controls:
        reference = target[control["prompt_id"]]
        if control["projected_l2"] == 0:
            zero_denominators += 1
        else:
            ratio = reference["projected_l2"] / control["projected_l2"]
            ratios.append(ratio)
            prompt_ratios[control["prompt_id"]].append(ratio)
    return {"status": "PASS", "definition": "actual candidate add_+1sd projected_l2 / actual random_direction_projected projected_l2, after BF16 rounding",
            "n_prompts": len(target), "n_control_seeds": len(data["seeds"]), "control_seeds": data["seeds"],
            "n_zero_control_denominators": zero_denominators, "pair_ratios": descriptive(ratios),
            "prompt_mean_ratios": descriptive(np.mean(v) for v in prompt_ratios.values()),
            "candidate_projected_l2": descriptive(r["projected_l2"] for r in target.values()),
            "control_projected_l2": descriptive(r["projected_l2"] for r in controls),
            "scope": "Descriptive norm-match audit; random seeds are not independent prompt replicates"}


def causal_signature(candidate: dict, data: dict) -> dict:
    rows = [r for r in data["rows"] if r["candidate_id"] == candidate["id"] and r["role"] == "corpus"]
    if not rows:
        return not_evaluated("Candidate was not included in this stage")
    groups = defaultdict(list)
    for row in rows:
        groups[row["prompt_id"]].append(row)
    per_prompt = []
    for pid, group in sorted(groups.items()):
        additive = [r for r in group if r["condition"] == "add_+1sd"]
        controls = [r for r in group if r["condition"] == "random_direction_projected"]
        require(len(additive) == 1 and len(controls) == len(data["seeds"]), f"Unpaired causal comparison for {candidate['id']}/{pid}")
        control_mean = float(np.mean([r["kl_base_to_changed"] for r in controls]))
        per_prompt.append({"prompt_id": pid, "source_id": additive[0]["source_id"], "category": additive[0]["category"],
                           "candidate_kl": additive[0]["kl_base_to_changed"], "control_seed_mean_kl": control_mean,
                           "delta": additive[0]["kl_base_to_changed"] - control_mean})
    positive = [r["delta"] for r in per_prompt if r["category"] == candidate["target_category"]]
    negative = [r["delta"] for r in per_prompt if r["category"] != candidate["target_category"]]
    result = stratified_interaction(positive, negative)
    result.update({"run_id": data["run_id"], "definition": "mean_target[KL(add_+1sd)-mean_seed KL(random_direction_projected)] - mean_other[same paired difference]",
                   "position_scope": data["summary"]["position_policy"], "per_prompt": per_prompt})
    return result


def counterexamples(candidate: dict, data: dict, cases: dict) -> dict:
    owned = {p["id"]: p for p in cases["prompts"] if p["candidate_id"] == candidate["id"]}
    measured = {r["prompt_id"]: r for r in data["observations"]
                if r["candidate_id"] == candidate["id"] and r["prompt_id"] in owned}
    require(set(measured) == set(owned), f"Missing candidate-matched counterexamples for {candidate['id']}")
    roles, families = {}, defaultdict(list)
    evidence = []
    for pid, case in owned.items():
        row = measured[pid]
        families[case["source_id"]].append((case, row))
        evidence.append({"prompt_id": pid, "source_id": case["source_id"], "role": case["role"],
                         "expected_activation_group": case["expected_activation_group"],
                         "prompt_mean": row["prompt_mean"], "last_activation": row["last_activation"]})
    for role in ROLES:
        values = [measured[pid]["prompt_mean"] for pid, case in owned.items() if case["role"] == role]
        roles[role] = descriptive(values)
    family_values = []
    for source, group in sorted(families.items()):
        expected = {case["expected_activation_group"] for case, _ in group}
        require(len(expected) == 1, "Counterexample family has inconsistent high/low expectations")
        family_values.append({"source_id": source, "expected_activation_group": expected.pop(),
                              "prompt_mean": float(np.mean([row["prompt_mean"] for _, row in group])),
                              "n_prompts": len(group), "prompt_ids": [case["id"] for case, _ in group]})
    positive = [f["prompt_mean"] for f in family_values if f["expected_activation_group"] == "high"]
    negative = [f["prompt_mean"] for f in family_values if f["expected_activation_group"] == "low"]
    require(len(positive) == 4 and len(negative) == 4, "Expected four high and four low independent counterexample families")
    family_auc = auc_summary(positive, negative)
    family_auc.update({"unit": "source_id family mean; positive originals and their paraphrases share one family",
                       "n_pairwise_inversions": sum(p < n for p in positive for n in negative),
                       "n_pairwise_ties": sum(p == n for p in positive for n in negative)})
    return {"status": "PASS", "run_id": data["run_id"], "n_prompts": len(measured), "roles": roles,
            "family_level": family_auc, "families": family_values, "evidence": evidence,
            "scope": "Only observations matching the case's frozen candidate ID; activation discrimination is not behavioral causal validation"}


def activation_statistics(candidate: dict, survey: dict | None) -> dict:
    match = None if survey is None else next((r for r in survey.get("top_candidates", []) if (r["layer"], r["neuron"]) == (candidate["layer"], candidate["neuron"])), None)
    if match is None:
        return {"status": "NOT_EVALUATED", "reason": "No matching public discovery-statistics record", "frozen_discovery_std": candidate["std"]}
    fields = ("mean", "std", "min", "max", "cross_prompt_std", "quantiles", "quantile_sample_count", "category_means", "selectivity_z", "selectivity_gap")
    return {"status": "PASS", "source": "logs/summary/survey.json", "run_id": survey["run_id"],
            "n_discovery_prompts": survey["prompts"], "token_scope": survey["token_scope"],
            "quantile_method": survey.get("quantile_method"), **{key: match[key] for key in fields if key in match}}


def compute(root: Path = ROOT):
    frozen, cases, corpus, survey, sources = load_inputs(root)
    stage_data, receipts = {}, []
    for stage, split in STAGES.items():
        data, receipt = load_stage(root, stage, split, frozen, cases, corpus, sources)
        stage_data[split] = data
        receipts.append(receipt)
    records = []
    for candidate in frozen["candidates"]:
        cid = candidate["id"]
        record = {"id": cid, "layer": candidate["layer"], "neuron": candidate["neuron"],
                  "target_category": candidate["target_category"], "correlation": {}, "causal_signature": {}, "norm_matching": {}}
        for split in ("validation", "test"):
            data = stage_data[split]
            if data is None:
                record["correlation"][split] = not_evaluated(f"{split} run is unavailable")
                record["causal_signature"][split] = not_evaluated(f"{split} run is unavailable")
                continue
            obs = [r for r in data["observations"] if r["candidate_id"] == cid and r["role"] == "corpus"]
            positive = [r["prompt_mean"] for r in obs if r["category"] == candidate["target_category"]]
            negative = [r["prompt_mean"] for r in obs if r["category"] != candidate["target_category"]]
            require(len(positive) == 3 and len(negative) == 42, f"Held-out corpus stratum counts changed for {cid}/{split}")
            record["correlation"][split] = {**auc_summary(positive, negative), "run_id": data["run_id"], "unit": "independent corpus prompt; user-content mean activation"}
            record["causal_signature"][split] = causal_signature(candidate, data)
        for split, data in stage_data.items():
            record["norm_matching"][split] = norm_matching(candidate, data) if data else not_evaluated(f"{split} run is unavailable")
        record["counterexamples"] = counterexamples(candidate, stage_data["validation"], cases) if stage_data["validation"] else not_evaluated("validation run is unavailable")
        records.append(record)
    # Each split is a separate, pre-specified family of four fixed hypotheses.
    for split in ("validation", "test"):
        adjusted = holm_adjust({r["id"]: r["correlation"][split]["permutation_p_two_sided"] for r in records if r["correlation"][split]["status"] == "PASS"})
        for record in records:
            if record["id"] in adjusted:
                record["correlation"][split].update({"permutation_p_holm": adjusted[record["id"]], "holm_family_size": 4})
    adjusted_cases = holm_adjust({r["id"]: r["counterexamples"]["family_level"]["permutation_p_two_sided"] for r in records if r["counterexamples"]["status"] == "PASS"})
    for record in records:
        if record["id"] in adjusted_cases:
            record["counterexamples"]["family_level"].update({"permutation_p_holm": adjusted_cases[record["id"]], "holm_family_size": 4,
                                                               "multiplicity_scope": "Separate four-hypothesis counterexample-family analysis"})
    analysis = {"schema_version": 1, "status": "PASS", "model_revision": frozen["model_revision"],
                "seed": SEED, "resamples": RESAMPLES, "stage_sources": receipts, "source_sha256": sources,
                "all_stages_available": all(r["status"] == "PASS" for r in receipts), "candidates": records,
                "registered_checks": frozen.get("registered_checks", {}),
                "interpretation": {"semantic_function": "NOT_EVALUATED", "catalog_label_policy": "All four entries remain ambiguous pending independent review",
                                   "bootstrap_unit": "independent corpus prompt; counterexamples use source_id family means",
                                   "p_value_scope": "Two-sided label permutations, 2000 resamples, plus-one correction; Holm across four fixed hypotheses per held-out split",
                                   "selection_caveat": "Discovery screened 737280 coordinates; no discovery significance claim. Four hypotheses were frozen before held-out measurements.",
                                   "causal_signature_scope": "Category-selective next-token distribution effect at the last chat-template token only; no semantic behavior claim"}}
    catalog = []
    by_id = {r["id"]: r for r in records}
    for candidate in frozen["candidates"]:
        record = by_id[candidate["id"]]
        run_ids = [candidate["discovery_evidence_run"]]
        run_ids += [data["run_id"] for data in stage_data.values() if data and any(row["candidate_id"] == candidate["id"] for row in data["rows"])]
        catalog.append({"id": candidate["id"], "model_revision": frozen["model_revision"], "layer": candidate["layer"], "neuron": candidate["neuron"],
                        "hypothesis": candidate["hypothesis"], "target_category": candidate["target_category"], "status": "ambiguous",
                        "activation_statistics": activation_statistics(candidate, survey),
                        "evidence": {"discovery_top_contexts": candidate.get("top_contexts", []), "discovery_low_contexts": candidate.get("low_contexts", []),
                                     "analysis": "logs/summary/analysis.json", "run_ids": sorted(set(run_ids))}, "run_ids": sorted(set(run_ids)),
                        "correlation": record["correlation"], "causal_signature": record["causal_signature"], "norm_matching": record["norm_matching"],
                        "counterexamples": record["counterexamples"], "competing_hypotheses": candidate.get("competing_hypotheses", []),
                        "uncertainty": ["Only three positive corpus prompts per held-out split; confidence intervals can be unstable.",
                                        "Frozen candidate selection followed a 737280-coordinate discovery screen.",
                                        "AUC is activation correlation; KL measures distribution change at a chat-template wrapper token.",
                                        "Nominal norm matching is affected by BF16 rounding; actual projected norms are reported.",
                                        "Semantic task correctness, representative general performance, and independent model replicates remain NOT_EVALUATED.",
                                        "Counterexample paraphrases share source_id with their originals and are grouped before AUC."]})
    return analysis, catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate hashes/schema and freshness; never write outputs")
    args = parser.parse_args()
    analysis, catalog = compute()
    outputs = (analysis, catalog)
    if args.check:
        for relative, expected in zip(OUTPUTS, outputs):
            path = ROOT / relative
            if path.exists():
                require(load(path) == expected, f"Generated output is stale: {relative}; rerun without --check")
    else:
        for relative, value in zip(OUTPUTS, outputs):
            path = ROOT / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(path)
    print(json.dumps({"status": "PASS", "mode": "check" if args.check else "write", "all_stages_available": analysis["all_stages_available"],
                      "stages": [{"stage": r["stage"], "status": r["status"]} for r in analysis["stage_sources"]], "catalog_entries": len(catalog)}))


if __name__ == "__main__":
    main()
