#!/usr/bin/env python3
"""Run a separately frozen post-validation exact-answer utility check.

``--check`` validates the config and strict scorer on CPU, without loading a
model or touching the GPU. Default execution owns one ModelRuntime/GPU lock,
runs the existing smoke gate, and records only aggregate public results.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neuron_lab.corpus import build_corpus, normalize_text
from neuron_lab.evaluation import generated_quality, prompt_bootstrap

CONDITIONS = ["baseline", "add_+1sd", "add_-1sd", "zero_ablation", "random_direction_projected"]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_normalize(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Exact-answer scorer requires a string")
    return text.strip().casefold()


def exact_score(text: str, accepted_answers: list[str]) -> int:
    """No punctuation removal, extraction, Unicode folding, or extra parsing."""
    return int(strict_normalize(text) in {strict_normalize(answer) for answer in accepted_answers})


def read_config(path: Path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    hypotheses_path = ROOT / "configs/hypotheses.json"
    frozen = json.loads(hypotheses_path.read_text())
    if cfg.get("hypotheses_sha256") != file_hash(hypotheses_path) or cfg.get("model_revision") != frozen["model_revision"]:
        raise ValueError("Utility config does not match frozen model/hypotheses")
    if cfg.get("design_after_validation") is not True or cfg.get("scoring") != "exact_strip_casefold":
        raise ValueError("This utility check must disclose its post-validation design and strict scorer")
    if cfg.get("conditions") != CONDITIONS or cfg.get("position") != "last" or cfg.get("control_seed") != 11:
        raise ValueError("Unexpected frozen intervention settings")
    for field, limit in [("max_seconds", 600), ("max_new_tokens", 12), ("max_run_disk_gb", 1), ("max_gpu_memory_gb", 80)]:
        if not isinstance(cfg.get(field), (int, float)) or not 0 < cfg[field] <= limit:
            raise ValueError(f"Invalid or excessive utility budget: {field}")
    if cfg.get("min_gpu_headroom_gb", 0) < 4 or cfg.get("bootstrap_resamples") != 2000:
        raise ValueError("Utility requires 4GB GPU headroom and 2000 prompt bootstrap resamples")
    if not isinstance(cfg.get("max_new_tokens"), int) or isinstance(cfg["max_new_tokens"], bool):
        raise ValueError("max_new_tokens must be an integer")
    if not isinstance(cfg.get("seed"), int) or isinstance(cfg["seed"], bool) or cfg["seed"] < 0:
        raise ValueError("Utility seed must be a nonnegative integer")
    frozen_time = datetime.fromisoformat(cfg["frozen_kst"])
    if frozen_time.tzinfo is None or frozen_time > datetime.now(ZoneInfo("Asia/Seoul")):
        raise ValueError("Invalid utility freeze timestamp")
    cases = cfg.get("cases", [])
    if len(cases) != 16 or Counter(c.get("group") for c in cases) != {"general": 8, "translation": 8}:
        raise ValueError("Expected exactly eight general and eight translation prompts")
    original = build_corpus()
    cases_path = ROOT / "configs/validation_cases.json"
    if cases_path.exists():
        original += json.loads(cases_path.read_text())["prompts"]
    original_texts = {normalize_text(p["text"]) for p in original}
    original_ids, original_sources = {p["id"] for p in original}, {p["source_id"] for p in original}
    seen_ids, seen_sources, seen_texts = set(), set(), set()
    for case in cases:
        for field in ("id", "source_id", "group", "task", "text"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"Missing utility case field: {field}")
        if case["id"] in original_ids | seen_ids or case["source_id"] in original_sources | seen_sources:
            raise ValueError("Utility case identifiers overlap an original or another utility source")
        normalized = normalize_text(case["text"])
        if normalized in original_texts | seen_texts:
            raise ValueError("Utility prompt duplicates an original or another utility prompt")
        accepted = case.get("accepted_answers")
        if not isinstance(accepted, list) or not accepted or any(not isinstance(a, str) or not a.strip() for a in accepted):
            raise ValueError("Accepted exact-answer variants must be explicit nonempty strings")
        if len({strict_normalize(a) for a in accepted}) != len(accepted):
            raise ValueError("Duplicate normalized accepted answer")
        seen_ids.add(case["id"])
        seen_sources.add(case["source_id"])
        seen_texts.add(normalized)
    candidates = frozen.get("candidates", [])
    if len(candidates) != 4 or len({c["id"] for c in candidates}) != 4:
        raise ValueError("Expected four distinct frozen neuron candidates")
    return cfg, candidates


def score_record(case: dict, value: dict, baseline: dict) -> dict:
    accuracy = exact_score(value["text"], case["accepted_answers"])
    baseline_accuracy = exact_score(baseline["text"], case["accepted_answers"])
    return {"case_id": case["id"], "source_id": case["source_id"], "group": case["group"], "task": case["task"],
            "accuracy": accuracy, "baseline_accuracy": baseline_accuracy, "paired_accuracy_delta": accuracy - baseline_accuracy,
            "changed_from_baseline": value["token_ids"] != baseline["token_ids"],
            "gained": bool(accuracy and not baseline_accuracy), "lost": bool(baseline_accuracy and not accuracy),
            "quality_proxy": generated_quality(value["token_ids"], value["text"]),
            "text": value["text"], "token_ids": value["token_ids"]}


def summarize(rows: list[dict], candidates: list[dict], cfg: dict) -> list[dict]:
    baseline_rows = {r["case_id"]: r for r in rows if r["condition"] == "baseline"}
    result = []
    for candidate in candidates:
        for condition in CONDITIONS:
            selected = list(baseline_rows.values()) if condition == "baseline" else [r for r in rows if r["candidate_id"] == candidate["id"] and r["condition"] == condition]
            for group in ("all", "general", "translation"):
                records = [r for r in selected if group == "all" or r["group"] == group]
                expected = 16 if group == "all" else 8
                if len(records) != expected or len({r["source_id"] for r in records}) != expected:
                    raise ValueError("Incomplete or dependent utility comparison group")
                result.append({"candidate_id": candidate["id"], "condition": condition, "group": group, "n_prompts": expected,
                               "correct": sum(r["accuracy"] for r in records), "accuracy": sum(r["accuracy"] for r in records) / expected,
                               "baseline_correct": sum(r["baseline_accuracy"] for r in records),
                               "baseline_accuracy": sum(r["baseline_accuracy"] for r in records) / expected,
                               "paired_accuracy_delta": prompt_bootstrap([r["paired_accuracy_delta"] for r in records], seed=cfg["seed"], resamples=cfg["bootstrap_resamples"]),
                               "gained_count": sum(r["gained"] for r in records), "lost_count": sum(r["lost"] for r in records),
                               "outputs_changed_count": sum(r["changed_from_baseline"] for r in records),
                               "collapse_count": sum(r["quality_proxy"]["collapsed"] for r in records),
                               "empty_count": sum(r["quality_proxy"]["empty"] for r in records),
                               "baseline_shared_across_candidates": True})
    return result


def execute(runtime, cfg: dict, candidates: list[dict], config_path: Path, smoke_result: dict) -> dict:
    import torch
    from neuron_lab.hooks import ActivationHook
    from neuron_lab.runtime import Run, save_json
    from neuron_lab.steering import make_spec

    if not runtime.smoke_passed or smoke_result.get("status") != "PASS":
        raise RuntimeError("Utility study requires a passing BF16 smoke test")
    if runtime.cfg["revision"] != cfg["model_revision"]:
        raise RuntimeError("Runtime revision differs from frozen utility revision")
    config_hash = file_hash(config_path)
    if yaml.safe_load(config_path.read_text(encoding="utf-8")) != cfg:
        raise RuntimeError("Utility configuration changed after CPU validation")
    runtime.verify_sources()
    run = Run("utility", cfg)
    rows = []
    raw_path = run.path / "utility_generations.jsonl"
    weights = {}

    def check_budget():
        run.check_budget()
        if torch.cuda.max_memory_reserved() > cfg["max_gpu_memory_gb"] * 1e9:
            raise RuntimeError("Utility GPU reserved-memory cap exceeded")
        if torch.cuda.mem_get_info()[0] < cfg["min_gpu_headroom_gb"] * 1e9:
            raise RuntimeError("Utility GPU headroom below four GB; leave other jobs untouched")

    def clean_hooks():
        if any(module._forward_pre_hooks for module in runtime.modules.values()):
            raise AssertionError("An intervention hook remained registered")

    try:
        clean_hooks()
        with torch.inference_mode():
            weights = {c["layer"]: runtime.modules[c["layer"]].weight.float() for c in candidates}
            with raw_path.open("w", encoding="utf-8") as output:
                def append(record):
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    rows.append(record)

                for index, case in enumerate(cfg["cases"]):
                    check_budget()
                    inputs = runtime.inputs(case["text"])
                    natural = {}
                    with ExitStack() as stack:
                        for candidate in candidates:
                            layer = candidate["layer"]
                            stack.enter_context(ActivationHook(runtime.modules[layer], capture=lambda x, layer=layer: natural.update({layer: x})))
                        runtime.logits(inputs)
                    clean_hooks()
                    baseline = runtime.generate(inputs, cfg["max_new_tokens"])
                    append({"candidate_id": "shared", "condition": "baseline", "seed": None,
                            **score_record(case, baseline, baseline)})
                    for candidate in candidates:
                        module = runtime.modules[candidate["layer"]]
                        condition_specs = [
                            {"name": "add_+1sd", "kind": "add", "sigma": 1.0},
                            {"name": "add_-1sd", "kind": "add", "sigma": -1.0},
                            {"name": "zero_ablation", "kind": "zero", "sigma": None},
                            {"name": "random_direction_projected", "kind": "random_direction", "space": "projected", "seed": cfg["control_seed"], "sigma": 1.0},
                        ]
                        for condition in condition_specs:
                            check_budget()
                            spec = make_spec(condition, candidate, module, natural[candidate["layer"]], weights[candidate["layer"]])
                            with ActivationHook(module, spec):
                                changed = runtime.generate(inputs, cfg["max_new_tokens"])
                            clean_hooks()
                            append({"candidate_id": candidate["id"], "layer": candidate["layer"], "neuron": candidate["neuron"],
                                    "condition": condition["name"], "seed": condition.get("seed"), "strength_sigma": condition["sigma"],
                                    "nominal_activation_strength": float(spec.strength), **score_record(case, changed, baseline)})
                    save_json(run.path / "progress.json", {"completed_cases": index + 1, "total_cases": len(cfg["cases"]), "generations": len(rows)})
                    print(json.dumps({"event": "utility_progress", "completed_cases": index + 1, "total_cases": len(cfg["cases"]), "generations": len(rows)}), flush=True)
            check_budget()
            clean_hooks()
            runtime.verify_sources()
            if file_hash(config_path) != config_hash or file_hash(ROOT / "configs/hypotheses.json") != cfg["hypotheses_sha256"]:
                raise RuntimeError("Frozen utility configuration/hypotheses changed during execution")
            expected_generations = len(cfg["cases"]) * (1 + len(candidates) * (len(CONDITIONS) - 1))
            if len(rows) != expected_generations:
                raise AssertionError("Unexpected actual generation count")
            result = {"model_revision": cfg["model_revision"], "design_after_validation": True, "frozen_kst": cfg["frozen_kst"],
                      "config_sha256": config_hash, "hypotheses_sha256": cfg["hypotheses_sha256"], "smoke_run_id": smoke_result["run_id"],
                      "prompt_count": len(cfg["cases"]), "group_prompt_counts": dict(Counter(c["group"] for c in cfg["cases"])),
                      "generation_count": len(rows), "baseline_generation_count": len(cfg["cases"]),
                      "intervention_generation_count": len(rows) - len(cfg["cases"]), "max_new_tokens": cfg["max_new_tokens"],
                      "conditions": CONDITIONS, "candidate_count": len(candidates), "aggregates": summarize(rows, candidates, cfg),
                      "source_ids": [c["source_id"] for c in cfg["cases"]], "case_ids": [c["id"] for c in cfg["cases"]],
                      "raw_sha256": file_hash(raw_path), "all_hook_handles_removed": True,
                      "scope": cfg["scope"], "control_matching": cfg["matching_scope"],
                      "scoring": "Exact match after strip().casefold() only; explicit accepted variants frozen before utility execution. No answer extraction or punctuation removal.",
                      "confidence_interval": "95% percentile prompt bootstrap of paired accuracy difference; n=8 per group or n=16 overall; shared baseline is not an additional replicate.",
                      "limitations": ["Additional authored cases designed after validation; separate from the original discovery/validation/test corpus.",
                                      "Eight general and eight translation cases do not establish general performance preservation.",
                                      "A 12-token limit can truncate an otherwise correct answer; formatting differences count as exact-answer failures.",
                                      "AUC/KL hypotheses are not promoted to semantic causal claims by this small utility check."]}
            return run.finish(result)
    except Exception as exc:
        run.finish({"model_revision": cfg["model_revision"], "error": str(exc), "completed_generations": len(rows),
                    "config_sha256": config_hash, "scope": cfg["scope"]}, "FAIL")
        raise
    finally:
        weights.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="CPU-only config/scorer validation; do not create a model runtime")
    args = parser.parse_args()
    config_path = ROOT / "configs/utility.yaml"
    cfg, candidates = read_config(config_path)
    if args.check:
        assert exact_score(" Au \n", ["Au"]) == 1
        assert exact_score("Au.", ["Au"]) == 0
        assert exact_score("The answer is 42", ["42"]) == 0
        assert exact_score("42\nexplanation", ["42"]) == 0
        print(json.dumps({"status": "PASS", "mode": "CPU check", "cases": len(cfg["cases"]), "groups": dict(Counter(c["group"] for c in cfg["cases"])),
                          "expected_actual_generations": 272, "config_sha256": file_hash(config_path)}))
        return
    from neuron_lab.runtime import ModelRuntime, save_json
    from neuron_lab.smoke import smoke
    import torch

    runtime = None
    try:
        runtime = ModelRuntime()
        smoke_result = smoke(runtime)
        result = execute(runtime, cfg, candidates, config_path, smoke_result)
        print(json.dumps({"status": result["status"], "run_id": result["run_id"], "generation_count": result["generation_count"]}), flush=True)
    except Exception as exc:
        if runtime is None or not getattr(runtime, "smoke_passed", False):
            save_json(ROOT / "runs/utility_startup_failure.json", {"status": "FAIL", "error": str(exc), "model_revision": cfg["model_revision"],
                                                                   "config_sha256": file_hash(config_path), "failed_before_utility_run": True})
        raise
    finally:
        if runtime is not None:
            # Release only this process's model and lock; other GPU jobs remain
            # untouched. Process exit is the final CUDA-context cleanup boundary.
            lock_file = runtime.lock_file
            try:
                runtime.modules.clear()
                del runtime.model
                torch.cuda.empty_cache()
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()


if __name__ == "__main__":
    main()
