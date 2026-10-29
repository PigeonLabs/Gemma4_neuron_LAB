#!/usr/bin/env python3
"""Read-only CPU verification of recorded artifacts and their public references.

Uses only the standard library in this process. Existing analysis/build scripts
run in read-only modes in subprocesses with CUDA hidden. No inference, weights,
network service, Git operation, or scientific-output repair is performed.

Default: print a concise receipt, with missing work as NOT_EVALUATED.
--final: require all recorded stages and final deliverables to pass.
--write: also write logs/summary/verification.json; never hashes itself.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unicodedata
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs/summary/verification.json"
SCRIPT = Path(__file__).resolve()
RAW_STAGES = ("smoke", "pilot", "survey", "steering", "validation", "expanded", "utility")
FINAL_SUMMARIES = ("model_load", *RAW_STAGES, "analysis", "synthesis", "explorer_qa", "live", "reproduction")
FINAL_FILES = (
    "README.md", "AGENTS.md", "requirements.lock.txt", "configs/hypotheses.json",
    "configs/validation_cases.json", "configs/utility.yaml", "catalog/public/candidates.json",
    "catalog/public/frozen_evidence.json", "catalog/public/neurons.json",
    "docs/ARCHITECTURE.md", "docs/METHODOLOGY.md", "docs/REFERENCE_AUDIT.md",
    "docs/RESEARCH_LOG.md", "docs/REPRODUCTION.md", "docs/EXPLORER_QA.md",
    "reports/INITIAL_FINDINGS_KO.md", "reports/RESEARCH_SUMMARY_KO.md",
    "explorer/app.py", "explorer/README.md",
)
PRIVATE_PARTS = {"data", "runs", "checkpoints", ".venv", ".git", ".aws", ".codex", ".agents"}
WEIGHT_SUFFIXES = {".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt", ".onnx"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, f"Duplicate JSON object key: {key}")
        value[key] = item
    return value


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def finite_tree(value, location="$"):
    if isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite JSON number at {location}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f"{location}[{index}]")


def loads(text: str):
    value = json.loads(text, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    finite_tree(value)
    return value


def load(path: Path):
    return loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    require(path.resolve() != SCRIPT, "Verifier self-hashing is excluded")
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot load {relative(path)}")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def check(name: str, operation):
    started = time.monotonic()
    try:
        result = operation()
        if not isinstance(result, dict):
            result = {"details": result}
        return {"name": name, "status": result.pop("status", "PASS"),
                "elapsed_seconds": round(time.monotonic() - started, 3), **result}
    except Exception as exc:
        return {"name": name, "status": "FAIL", "elapsed_seconds": round(time.monotonic() - started, 3),
                "errors": [f"{type(exc).__name__}: {exc}"]}


def not_evaluated(reason: str, **extra):
    return {"status": "NOT_EVALUATED", "reason": reason, **extra}


def publication_inputs():
    publisher = module(ROOT / "scripts/check_publication.py", "_artifact_publication_scan")
    entries = publisher.manifest()
    for entry in entries:
        path = Path(entry["path"])
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe publication path")
        require(not (set(path.parts) & PRIVATE_PARTS), f"Private path in publication: {path}")
        require(path.suffix.lower() not in WEIGHT_SUFFIXES, f"Model/binary artifact in publication: {path}")
    return entries


def json_and_svg(entries):
    json_count = svg_count = 0
    failures = []
    for entry in entries:
        path = entry["path"]
        try:
            if path.endswith(".json"):
                loads(entry["content"])
                json_count += 1
            elif path.endswith(".svg"):
                svg = ET.fromstring(entry["content"])
                require(svg.tag.rsplit("}", 1)[-1] == "svg", "XML root is not SVG")
                svg_count += 1
        except (ValueError, ET.ParseError) as exc:
            failures.append(f"{path}: {exc}")
    return {"status": "FAIL" if failures else "PASS", "json_files": json_count,
            "svg_files": svg_count, **({"errors": failures} if failures else {})}


def without_code(text: str) -> str:
    # Markdown code examples do not define document links.
    lines, fence = [], None
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            lines.append("")
        else:
            lines.append("" if fence else line)
    return re.sub(r"(`+).*?\1", "", "\n".join(lines), flags=re.S)


def link_targets(text: str) -> list[str]:
    text = without_code(text)
    targets = []
    # Inline links/images, including angle-wrapped paths and one balanced pair.
    pattern = r"!?\[(?:\\.|[^\]\\])*\]\(\s*(<[^>]*>|(?:\\.|[^()\s]|\([^()\n]*\))+)(?:\s+[\"'][^\n]*?[\"'])?\s*\)"
    targets += [match.group(1) for match in re.finditer(pattern, text)]
    # Reference definitions also cover full/collapsed/shortcut reference links.
    targets += re.findall(r"(?m)^\s{0,3}\[[^\]\n]+\]:\s*(<[^>]*>|\S+)", text)
    targets += [match.group(2) for match in re.finditer(
        r"<(?:a|img)\b[^>]*(?:href|src)\s*=\s*([\"'])(.*?)\1", text, flags=re.I | re.S)]
    return targets


def markdown_links(entries):
    targets_checked, documents, failures = 0, 0, []
    for entry in entries:
        name = entry["path"]
        if not (name == "README.md" or name.startswith(("docs/", "reports/", "catalog/", "explorer/"))):
            continue
        if not name.endswith(".md"):
            continue
        documents += 1
        source = ROOT / name
        for raw_target in link_targets(entry["content"]):
            target = re.sub(r"\\([\\()\[\] ])", r"\1", raw_target.strip().strip("<>"))
            try:
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue  # http(s), mailto, other external schemes, #anchors
                decoded = unquote(parsed.path)
                path = (ROOT / decoded.lstrip("/")) if decoded.startswith("/") else source.parent / decoded
                resolved = path.resolve()
                require(resolved.is_relative_to(ROOT), "Path escapes the workspace")
                require(resolved.exists(), "Target does not exist")
                require(not (set(resolved.relative_to(ROOT).parts) & PRIVATE_PARTS), "Target is a private unpublished path")
                require(resolved.suffix.lower() not in WEIGHT_SUFFIXES, "Target is a model/binary artifact")
                targets_checked += 1
            except (OSError, ValueError) as exc:
                failures.append(f"{name}: {raw_target} ({exc})")
    return {"status": "FAIL" if failures else "PASS", "documents": documents,
            "local_targets": targets_checked, **({"errors": failures} if failures else {})}


def run_directory(run_id) -> Path:
    require(isinstance(run_id, str) and RUN_ID.fullmatch(run_id), "Missing or unsafe run ID")
    path = ROOT / "runs" / run_id
    require(path.resolve().is_relative_to((ROOT / "runs").resolve()), "Run path escapes runs/")
    return path


def verify_hash(path: Path, expected) -> bool:
    require(isinstance(expected, str) and SHA256.fullmatch(expected), f"Invalid registered hash for {relative(path)}")
    if not path.is_file():
        return False
    require(not path.is_symlink(), f"Symlink artifact refused: {relative(path)}")
    require(path.suffix.lower() not in WEIGHT_SUFFIXES, "Weight reads are forbidden")
    require(digest(path) == expected, f"SHA256 mismatch: {relative(path)}")
    return True


def registered_hashes():
    hypotheses_path = ROOT / "configs/hypotheses.json"
    cases_path = ROOT / "configs/validation_cases.json"
    if not hypotheses_path.is_file() or not cases_path.is_file():
        return not_evaluated("Frozen hypotheses or validation cases are missing")
    frozen, cases = load(hypotheses_path), load(cases_path)
    frozen_hash, cases_hash = digest(hypotheses_path), digest(cases_path)
    require(cases.get("hypotheses_sha256") == frozen_hash, "Validation cases no longer match frozen hypotheses")
    survey = ROOT / "logs/summary/survey.json"
    if not survey.is_file():
        return not_evaluated("Registered discovery summary is missing")
    require(digest(survey) == frozen.get("discovery_summary_sha256"), "Discovery summary changed after registration")
    ids = [row.get("id") for row in frozen.get("candidates", [])]
    require(len(ids) == len(set(ids)) == 4 and all(isinstance(x, str) and x for x in ids), "Expected four distinct frozen hypotheses")
    receipts = 0
    for stage in ("steering", "validation", "expanded", "utility"):
        path = ROOT / f"logs/summary/{stage}.json"
        if not path.is_file():
            continue
        summary = load(path)
        if summary.get("status") != "PASS":
            continue
        require(summary.get("hypotheses_sha256") == frozen_hash, f"Hypothesis hash changed in {stage}")
        metadata = run_directory(summary.get("run_id")) / "metadata.json"
        if not metadata.is_file():
            continue
        recorded = load(metadata).get("source_sha256", {})
        for rel, expected in (("configs/hypotheses.json", frozen_hash), ("configs/validation_cases.json", cases_hash)):
            if rel in recorded:
                require(recorded[rel] == expected, f"Registered input changed since {stage}: {rel}")
        receipts += 1
    return {"hypotheses_sha256": frozen_hash, "validation_cases_sha256": cases_hash,
            "discovery_summary_sha256": digest(survey), "run_receipts_checked": receipts}


def normalized(text: str) -> str:
    return re.sub(r"\W+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def corpus_partitions():
    # This local module imports only re/unicodedata/collections, never a model.
    corpus = module(ROOT / "src/neuron_lab/corpus.py", "_artifact_authored_corpus")
    original = corpus.build_corpus()
    require(len(original) == 150, "Expected exactly 150 authored original prompts")
    require(Counter(row["split"] for row in original) == {"discovery": 60, "validation": 45, "test": 45}, "Original split counts changed")
    require(len({row["source_id"] for row in original}) == 150, "Original source groups are not independent")
    categories = Counter(row["category"] for row in original)
    require(len(categories) == 15 and set(categories.values()) == {10}, "Original category coverage changed")
    survey_path = ROOT / "logs/summary/survey.json"
    if survey_path.is_file() and load(survey_path).get("status") == "PASS":
        expected = load(survey_path).get("corpus_sha256")
        actual = hashlib.sha256(json.dumps(original, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        require(actual == expected, "Original corpus changed after discovery measurement")
    cases_path = ROOT / "configs/validation_cases.json"
    if not cases_path.is_file():
        return not_evaluated("Counterexample cases are missing", original_prompts=len(original))
    cases = load(cases_path).get("prompts", [])
    require(len(cases) == 40 and all(row.get("split") == "validation" for row in cases), "Expected 40 within-validation counterexample prompts")
    corpus.validate_corpus(original + cases)
    require(Counter(row.get("role") for row in cases) == {role: 8 for role in (
        "positive_control", "negative_control", "near_miss", "paraphrase", "out_of_distribution")}, "Counterexample role counts changed")
    counter_ids = {row["candidate_id"] for row in cases}
    require(len(counter_ids) == 4 and all(sum(row["candidate_id"] == cid for row in cases) == 10 for cid in counter_ids), "Counterexample candidate coverage changed")
    require(len({row["source_id"] for row in cases}) == 32, "Expected 32 counterexample source groups after paired paraphrases")
    for row in cases:
        if row["role"] == "paraphrase":
            require(any(other["role"] == "positive_control" and other["candidate_id"] == row["candidate_id"]
                        and other["source_id"] == row["source_id"] for other in cases), "Paraphrase lost its positive-original source group")
    utility_path = ROOT / "logs/summary/utility.json"
    if not utility_path.is_file() or load(utility_path).get("status") != "PASS":
        return not_evaluated("Utility execution not yet complete; its 16 measured cases cannot be checked",
                             original_prompts=150, counterexample_prompts=40, counterexample_source_groups=32)
    utility_summary = load(utility_path)
    meta_path = run_directory(utility_summary.get("run_id")) / "metadata.json"
    if not meta_path.is_file():
        return not_evaluated("Completed utility metadata is not locally available")
    metadata = load(meta_path)
    config = metadata.get("config", {})
    utility = config.get("cases", [])
    config_path = ROOT / "configs/utility.yaml"
    require(verify_hash(config_path, utility_summary.get("config_sha256")), "Registered utility configuration is missing")
    require(config.get("hypotheses_sha256") == digest(ROOT / "configs/hypotheses.json"), "Utility metadata hypotheses hash changed")
    require(len(utility) == 16 and Counter(row.get("group") for row in utility) == {"general": 8, "translation": 8}, "Expected 8 general and 8 translation utility cases")
    require(utility_summary.get("case_ids") == [row["id"] for row in utility], "Utility summary case IDs differ from execution metadata")
    require(utility_summary.get("source_ids") == [row["source_id"] for row in utility], "Utility source IDs differ from execution metadata")
    all_ids, all_texts, source_splits = set(), set(), defaultdict(set)
    for family, rows in (("original", original), ("counter", cases), ("utility", utility)):
        for row in rows:
            for field in ("id", "source_id", "text"):
                require(isinstance(row.get(field), str) and row[field].strip(), f"Missing {family} prompt field: {field}")
            require(row["id"] not in all_ids, "Prompt ID overlap across corpora")
            all_ids.add(row["id"])
            text = normalized(row["text"])
            require(text not in all_texts, "Normalized prompt duplication across corpora")
            all_texts.add(text)
            split = "utility" if family == "utility" else row["split"]
            source_splits[row["source_id"]].add(split)
    require(all(len(splits) == 1 for splits in source_splits.values()), "A source group crosses study partitions")
    original_sources = {row["source_id"] for row in original}
    counter_sources = {row["source_id"] for row in cases}
    utility_sources = {row["source_id"] for row in utility}
    require(not original_sources & counter_sources and not original_sources & utility_sources and not counter_sources & utility_sources,
            "Source groups overlap original/counterexample/utility families")
    require(len(utility_sources) == 16, "Utility sources are dependent")
    return {"original_prompts": 150, "counterexample_prompts": 40, "utility_prompts": 16,
            "total_prompts": len(all_ids), "total_source_groups": len(source_splits),
            "original_split_counts": dict(Counter(row["split"] for row in original)),
            "utility_cases_source": relative(meta_path) + "#config.cases",
            "scope": "Exact normalized duplicates and registered source-group disjointness; semantic leakage is not automatically excluded"}


def raw_artifacts(stage: str):
    path = ROOT / f"logs/summary/{stage}.json"
    if not path.is_file():
        return not_evaluated(f"{stage} summary is missing")
    summary = load(path)
    if summary.get("status") != "PASS":
        return not_evaluated(f"Recorded {stage} status is {summary.get('status', 'UNKNOWN')}")
    directory = run_directory(summary.get("run_id"))
    expected = []
    if "output_sha256" in summary:
        expected.append((directory / "outputs.json", summary["output_sha256"]))
    if "raw_sha256" in summary:
        filename = "utility_generations.jsonl" if stage == "utility" else "interventions.jsonl"
        expected.append((directory / filename, summary["raw_sha256"]))
    hashes = summary.get("artifact_hashes", {})
    require(isinstance(hashes, dict), f"Invalid {stage} artifact hash map")
    for name, sha in hashes.items():
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts and rel.parts[0] == "statistics"
                and rel.suffix == ".parquet", f"Unexpected statistics artifact path in {stage}")
        expected.append((directory / rel, sha))
    require(bool(expected), f"PASS {stage} has no registered artifact hashes")
    checked, missing = [], []
    for artifact, sha in expected:
        (checked if verify_hash(artifact, sha) else missing).append(relative(artifact))
    run_summary, metadata = directory / "summary.json", directory / "metadata.json"
    if run_summary.is_file():
        require(load(run_summary) == summary, f"Public and local run summaries differ: {stage}")
    else:
        missing.append(relative(run_summary))
    if metadata.is_file():
        meta = load(metadata)
        require(meta.get("run_id") == summary.get("run_id") and meta.get("status") == "PASS",
                f"Run metadata identity/status mismatch: {stage}")
    else:
        missing.append(relative(metadata))
    if missing:
        return not_evaluated("Registered local artifacts are unavailable", run_id=summary["run_id"],
                             checked_artifacts=len(checked), missing=missing)
    return {"run_id": summary["run_id"], "checked_artifacts": len(checked)}


def completeness():
    missing, failures = [], []
    for name in FINAL_FILES:
        path = ROOT / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(name)
    for stage in FINAL_SUMMARIES:
        path = ROOT / f"logs/summary/{stage}.json"
        if not path.is_file():
            missing.append(relative(path))
            continue
        value = load(path)
        if value.get("status") != "PASS":
            failures.append(f"{stage}: {value.get('status', 'UNKNOWN')}")
    qa_path = ROOT / "logs/summary/explorer_qa.json"
    if qa_path.is_file():
        qa = load(qa_path)
        qa_checks = qa.get("checks")
        if not isinstance(qa_checks, dict) or not qa_checks or any(value is not True for value in qa_checks.values()):
            failures.append("Explorer QA checks are not all recorded true")
    analysis_path = ROOT / "logs/summary/analysis.json"
    if analysis_path.is_file() and load(analysis_path).get("all_stages_available") is not True:
        missing.append("analysis: all measured causal stages")
    reproduction_path = ROOT / "logs/summary/reproduction.json"
    if reproduction_path.is_file() and load(reproduction_path).get("all_stages_available") is not True:
        missing.append("reproduction: all measured causal stages")
    if failures:
        return {"status": "FAIL", "errors": failures, "missing": missing}
    if missing:
        return not_evaluated("Final deliverables or stages are incomplete", missing=missing)
    return {"required_files": len(FINAL_FILES), "required_summary_stages": len(FINAL_SUMMARIES)}


def readonly_command(script: str, flags: list[str], prerequisites: tuple[str, ...] = ()):
    path = ROOT / script
    if not path.is_file():
        return not_evaluated(f"Read-only checker is unavailable: {script}")
    missing = [name for name in prerequisites if not (ROOT / name).is_file()]
    if missing:
        return not_evaluated("Checker inputs or generated destination are not present", missing=missing)
    # Only options actually documented in that local script are allowed.
    source = path.read_text(encoding="utf-8")
    for flag in flags:
        require(re.search(r"add_argument\(\s*[\"']" + re.escape(flag) + r"[\"']", source),
                f"Read-only option is not documented by {script}: {flag}")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, script, *flags], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=180)
    message = (result.stdout.strip() + ("\n" + result.stderr.strip() if result.stderr.strip() else "")).strip()
    return {"status": "PASS" if result.returncode == 0 else "FAIL", "command": [script, *flags],
            "returncode": result.returncode, "output": message[-3000:]}


def verify(final: bool = False) -> dict:
    started = time.monotonic()
    checks = []
    entries = []
    try:
        entries = publication_inputs()
        checks.append({"name": "publication_allowlist", "status": "PASS", "files": len(entries),
                       "total_bytes": sum(entry["bytes"] for entry in entries)})
    except Exception as exc:
        checks.append({"name": "publication_allowlist", "status": "FAIL", "errors": [str(exc)]})
    if entries:
        checks.append(check("public_json_and_svg", lambda: json_and_svg(entries)))
        checks.append(check("markdown_local_links", lambda: markdown_links(entries)))
    else:
        checks.append({"name": "public_json_and_svg", **not_evaluated("No validated allowlist snapshot")})
        checks.append({"name": "markdown_local_links", **not_evaluated("No validated allowlist snapshot")})
    checks.append(check("registered_inputs", registered_hashes))
    checks.append(check("corpus_partitions", corpus_partitions))
    for stage in RAW_STAGES:
        checks.append(check("raw_" + stage, lambda stage=stage: raw_artifacts(stage)))
    commands = (
        ("analysis_freshness", "scripts/analyze_results.py", ["--check"], ("configs/hypotheses.json", "configs/validation_cases.json")),
        ("readme_freshness", "scripts/build_readme.py", ["--check"], ("README.md",)),
        ("report_freshness", "scripts/build_report.py", ["--check"], ("reports/RESEARCH_SUMMARY_KO.md",)),
        ("public_cpu_replay", "scripts/check_reproduction.py", ["--no-write"], ("catalog/public/frozen_evidence.json",)),
        ("utility_config_cpu", "scripts/run_utility.py", ["--check"], ("configs/utility.yaml", "configs/hypotheses.json")),
    )
    for name, script, flags, prerequisites in commands:
        checks.append(check(name, lambda script=script, flags=flags, prerequisites=prerequisites:
                            readonly_command(script, flags, prerequisites)))
    checks.append(check("final_completeness", completeness))
    # A long CPU replay must not silently verify a changing publication snapshot.
    changed = []
    for entry in entries:
        path = ROOT / entry["path"]
        if path == SCRIPT or path == OUTPUT:
            continue
        if path.suffix in {".json", ".md", ".svg"} and (not path.is_file() or digest(path) != entry["sha256"]):
            changed.append(entry["path"])
    checks.append({"name": "snapshot_unchanged", "status": "FAIL" if changed else "PASS",
                   **({"errors": ["Input changed while checking: " + name for name in changed]} if changed else {})})
    failures = [row["name"] for row in checks if row["status"] == "FAIL"]
    incomplete = [row["name"] for row in checks if row["status"] == "NOT_EVALUATED"]
    status = "FAIL" if failures or (final and incomplete) else "NOT_EVALUATED" if incomplete else "PASS"
    return {"schema_version": 1, "status": status, "final_mode": final,
            "generated_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "checks": checks, "failed_checks": failures, "not_evaluated_checks": incomplete,
            "scope": "Recorded artifact consistency, read-only CPU replay and publication hygiene; no GPU inference or new scientific evidence",
            "weights_read": False, "gpu_jobs_submitted": False, "scientific_outputs_modified": False,
            "verifier_self_hash_included": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", help="Fail on any missing required stage/artifact or incomplete check")
    parser.add_argument("--write", action="store_true", help="Write only logs/summary/verification.json after checking")
    parser.add_argument("--check", action="store_true", help="Explicit read-only mode (the default); cannot be combined with --write")
    args = parser.parse_args()
    if args.check and args.write:
        parser.error("--check and --write cannot be combined")
    receipt = verify(final=args.final)
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        temporary = OUTPUT.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(OUTPUT)
    for row in receipt["checks"]:
        print(f"{row['status']:13} {row['name']}")
        for error in row.get("errors", [])[:8]:
            print("  " + error)
        if row.get("missing"):
            print("  missing: " + ", ".join(row["missing"][:8]))
        if row["status"] == "FAIL" and row.get("output"):
            print("  " + row["output"].replace("\n", "\n  "))
    print(json.dumps({"status": receipt["status"], "final_mode": args.final,
                      "checks": len(receipt["checks"]), "failed": receipt["failed_checks"],
                      "not_evaluated": receipt["not_evaluated_checks"],
                      "receipt": relative(OUTPUT) if args.write else None}, ensure_ascii=False))
    return 1 if receipt["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
