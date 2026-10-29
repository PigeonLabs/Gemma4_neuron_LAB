#!/usr/bin/env python3
"""Prove CPU analysis replay with only public files and measured run inputs.

The isolated temporary checkout excludes weights and the original full discovery
candidate artifact. Derived analysis/catalog files are removed before compute(),
so they cannot supply data circularly. Completed measured causal stages alone
provide metadata, interventions, and activation observations. No inference runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = "logs/summary/reproduction.json"
RUN_INPUTS = ("metadata.json", "interventions.jsonl", "activation_observations.json")
EXCLUDED_DERIVED = ("logs/summary/analysis.json", "catalog/public/neurons.json", RECEIPT)
RELEVANT_SOURCES = ("configs/hypotheses.json", "configs/validation_cases.json", "logs/summary/survey.json",
                    "logs/summary/steering.json", "logs/summary/validation.json", "logs/summary/expanded.json",
                    "catalog/public/frozen_evidence.json")


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_sha(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scientific(value):
    """Paths/hashes of source receipts can differ; scientific fields cannot."""
    if isinstance(value, dict):
        return {key: scientific(item) for key, item in value.items() if key not in {"source", "source_sha256"}}
    if isinstance(value, list):
        return [scientific(item) for item in value]
    return value


def source_snapshot(root: Path) -> dict:
    return {relative: sha256(root / relative) if (root / relative).is_file() else None for relative in RELEVANT_SOURCES}


def replay(root: Path = ROOT) -> dict:
    analyzer = module(root / "scripts/analyze_results.py", "_lab_replay_analyzer")
    publisher = module(root / "scripts/check_publication.py", "_lab_replay_publication")
    require((root / analyzer.FROZEN_EVIDENCE).is_file(), "Export catalog/public/frozen_evidence.json before the replay check")
    before = source_snapshot(root)
    entries = publisher.manifest()
    baseline_analysis, baseline_catalog = analyzer.compute(root=root)
    require(source_snapshot(root) == before, "A measured stage changed during source capture; rerun the reproduction check")
    frozen = analyzer.load(root / "configs/hypotheses.json")
    survey_run = analyzer.load(root / "logs/summary/survey.json")["run_id"]
    fixed_ids = [candidate["id"] for candidate in frozen["candidates"]]
    require(len(fixed_ids) == len(set(fixed_ids)) == 4, "The replay must compare exactly four frozen candidates")
    completed = [row for row in baseline_analysis["stage_sources"] if row["status"] == "PASS"]
    manifest_receipt = [{"path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"]}
                        for row in entries if row["path"] not in EXCLUDED_DERIVED]
    copied_runs, comparisons = [], {}
    with tempfile.TemporaryDirectory(prefix="gemma4-public-replay-") as temporary:
        isolated = Path(temporary)
        # Write the exact scanned UTF-8 snapshot, rather than following live paths.
        for entry in entries:
            relative = entry["path"]
            require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Unsafe publication path")
            destination = isolated / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(entry["content"], encoding="utf-8")
            require(sha256(destination) == entry["sha256"], "Isolated public file changed during copying")
        # Removing these strengthens the proof that no derived output is input.
        for relative in EXCLUDED_DERIVED:
            path = isolated / relative
            if path.exists():
                path.unlink()
        for stage in analyzer.STAGES:
            relative = f"logs/summary/{stage}.json"
            expected_hash = before[relative]
            copied = isolated / relative
            require((sha256(copied) if copied.is_file() else None) == expected_hash,
                    "Publication snapshot and measured stage capture disagree; rerun after the stage finishes")
        for stage in completed:
            run_receipt = {"stage": stage["stage"], "run_id": stage["run_id"], "files": {}}
            for filename in RUN_INPUTS:
                relative = f"runs/{stage['run_id']}/{filename}"
                source = root / relative
                require(source.is_file(), f"Completed run input is missing: {relative}")
                expected = baseline_analysis["source_sha256"].get(relative)
                require(expected is not None and sha256(source) == expected, f"Measured run input hash changed: {relative}")
                target = isolated / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                require(sha256(target) == expected, f"Isolated raw input differs: {relative}")
                run_receipt["files"][filename] = {"sha256": expected, "bytes": target.stat().st_size}
            copied_runs.append(run_receipt)
        private_candidate_path = isolated / "runs" / survey_run / "candidates.json"
        require(not private_candidate_path.exists(), "The proof must exclude the private full discovery candidate artifact")
        require(not (isolated / "data").exists() and not (isolated / ".venv").exists(), "Weights or environment directories entered the replay copy")
        require(not any(isolated.rglob("*.safetensors")), "A model weight file entered the replay copy")
        for relative in EXCLUDED_DERIVED:
            require(not (isolated / relative).exists(), "A derived analysis/catalog artifact entered the replay inputs")
        isolated_analysis, isolated_catalog = analyzer.compute(root=isolated)
        isolated_by_id = {row["id"]: row for row in isolated_analysis["candidates"]}
        baseline_by_id = {row["id"]: row for row in baseline_analysis["candidates"]}
        isolated_catalog_by_id = {row["id"]: row for row in isolated_catalog}
        baseline_catalog_by_id = {row["id"]: row for row in baseline_catalog}
        require(set(isolated_by_id) == set(baseline_by_id) == set(fixed_ids), "Frozen candidate population changed in the replay")
        for cid in fixed_ids:
            require(isolated_catalog_by_id[cid]["activation_statistics"]["source"] == analyzer.FROZEN_EVIDENCE,
                    "Public fallback was not actually used for every frozen candidate")
            candidate_checks = {}
            for key in ("correlation", "causal_signature", "counterexamples", "norm_matching"):
                require(scientific(isolated_by_id[cid][key]) == scientific(baseline_by_id[cid][key]),
                        f"Scientific replay mismatch: {cid}/{key}")
                candidate_checks[key] = "PASS"
            require(scientific(isolated_catalog_by_id[cid]["activation_statistics"]) == scientific(baseline_catalog_by_id[cid]["activation_statistics"]),
                    f"Measured frozen discovery profile differs: {cid}")
            candidate_checks["discovery_activation_profile"] = "PASS"
            comparisons[cid] = candidate_checks
        require(isolated_analysis["category_profile_relationships"]["source"] == analyzer.FROZEN_EVIDENCE,
                "Category-profile relationships did not use the public frozen evidence")
        require(scientific(isolated_analysis["category_profile_relationships"]) == scientific(baseline_analysis["category_profile_relationships"]),
                "Category-profile relationships changed in public replay")
        baseline_payload = scientific({"analysis": baseline_analysis, "catalog": baseline_catalog})
        isolated_payload = scientific({"analysis": isolated_analysis, "catalog": isolated_catalog})
        require(baseline_payload == isolated_payload, "Non-provenance analysis or catalog content changed in public replay")
        require(not any(key.endswith(f"{survey_run}/candidates.json") for key in isolated_analysis["source_sha256"]),
                "Isolated analysis recorded a forbidden private discovery source")
        require(source_snapshot(root) == before, "Original registered inputs changed during replay; no receipt written")
        proof = {"schema_version": 1, "status": "PASS", "model_revision": frozen["model_revision"],
            "hypotheses_sha256": before["configs/hypotheses.json"],
            "registered_discovery_summary_sha256": before["logs/summary/survey.json"],
            "public_frozen_evidence_sha256": before[analyzer.FROZEN_EVIDENCE],
            "scope": "CPU analysis replay from an isolated publication-allowlist copy plus only the three required raw files of completed causal runs; this is not GPU/model inference replication",
            "private_full_discovery_artifact_absent": True, "public_fallback_used_for_all_four_candidates": True,
            "weights_copied": False, "gpu_inference_executed": False,
            "derived_analysis_and_catalog_removed_before_compute": list(EXCLUDED_DERIVED),
            "publication_allowlist_file_count": len(entries),
            "copied_public_input_manifest_sha256": payload_sha(manifest_receipt),
            "completed_run_inputs": copied_runs,
            "stages": isolated_analysis["stage_sources"],
            "all_stages_available": isolated_analysis["all_stages_available"],
            "candidate_checks": comparisons, "category_profile_relationships_equal": True,
            "entire_scientific_payload_equal": True, "scientific_payload_sha256": payload_sha(isolated_payload),
            "comparison_exclusions": ["source path labels", "source_sha256 receipt maps"],
            "isolated_discovery_source": analyzer.FROZEN_EVIDENCE,
            "original_registered_inputs_preserved": True,
            "temporary_copy_removed_after_check": True,
            "generated_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat()}
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true", help="Perform the proof without writing the root reproduction receipt")
    args = parser.parse_args()
    receipt = replay()
    if not args.no_write:
        output = ROOT / RECEIPT
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)
    print(json.dumps({"status": receipt["status"], "all_stages_available": receipt["all_stages_available"],
                      "candidate_checks": receipt["candidate_checks"],
                      "receipt": None if args.no_write else RECEIPT}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(f"Isolated public replay failed: {exc}", file=sys.stderr)
        sys.exit(1)
