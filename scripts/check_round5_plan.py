#!/usr/bin/env python3
"""Check the reviewed R5 design on CPU, without model inference or registration."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neuron_lab.round5_design import (
    make_episode, validate_episode, table_text, query_text, history_text,
    forward_budget, digest,
)

SOURCES = ["configs/round5_plan.yaml", "src/neuron_lab/round5_design.py",
           "scripts/check_round5_plan.py", "tests/test_round5_design.py"]
DEST = ROOT / "logs/summary/round5_design_verification.json"


def verify() -> dict:
    cfg = yaml.safe_load((ROOT / SOURCES[0]).read_text())
    if cfg["execution_status"] != "NOT_EVALUATED" or cfg["registration_status"] != "NOT_REGISTERED":
        raise ValueError("A design example cannot be called executed or registered")
    if cfg["selection"]["primary_lesions"] is not None or cfg["selection"]["recruited_groups"] is not None:
        raise ValueError("Discovery has not selected any R5 coordinates")
    budget = forward_budget(cfg["stages"])
    if budget["planned_actual_forward_calls"] > cfg["budget"]["max_actual_forward_calls"]:
        raise ValueError("The explicit call budget does not fit the cap")
    # Independent hand calculation checks the actual table, not the helper's sum.
    explicit = 128 + (72*17 + 8*2*8 + 8*18*8 + 8*4*4*8 + 6*4*4*8)
    explicit += 16*3*2*3 + 16*2*8 + 16*18*8 + 16*3*8 + 4*4*16*4
    explicit += 48*2*8 + 48*18*8 + 48*6*5*8 + 48*3*8 + 8*6*6*8
    if explicit != budget["planned_actual_forward_calls"]:
        raise ValueError("The independent cell-count calculation disagrees")
    episodes = [make_episode(cfg["example_manifest_seed"] + i, f"CPU-design-example-{i:03}") for i in range(64)]
    for episode in episodes:
        validate_episode(episode)
    fingerprints = [digest({k: e[k] for k in ("first", "second", "direct", "orders")}) for e in episodes]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Repeated synthetic table assignment")
    from transformers import AutoTokenizer
    local = Path((ROOT / "runs/model_path.txt").read_text().strip())
    if local.name != cfg["model_revision"]:
        raise ValueError("Pinned local revision does not match the plan")
    tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
    for label in cfg["task"]["reply_codes"]:
        if len(tokenizer.encode(label, add_special_tokens=False)) != 1:
            raise ValueError("Non-single-token response code")
    maximum = 0
    status_token_lengths = {}
    for episode in episodes[:8]:
        for statuses in (["0"]*8, ["1"]*8, ["2"]*8, ["0", "1"]*4):
            body = table_text(episode) + "\nRecords (0=incorrect,1=correct,2=unreported):\n" + history_text(episode, statuses)
            for case in episode["test"]:
                text = body + "\nNEW_QUERY:\n" + query_text(case)
                tokens = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                    return_dict=True, add_generation_prompt=True, enable_thinking=False)
                length = len(tokens["input_ids"])
                if length < 100:
                    raise ValueError("Unexpectedly short serialized episode; check tokenizer return shape")
                maximum = max(maximum, length)
        for s in "012":
            status_token_lengths[s] = len(tokenizer.encode("diagnostic-pass=" + s + ".", add_special_tokens=False))
    if maximum > cfg["execution_contract"]["max_input_tokens"]:
        raise ValueError(f"The prepared task exceeds the context cap: {maximum}")
    if len(set(status_token_lengths.values())) != 1:
        raise ValueError("Status encodings have different token counts")
    raw = ROOT / "runs/round5_design_peer_review/cpu_example_episodes.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps({"scope": "CPU design examples; not actual discovery/confirmation data",
                               "episodes": episodes[:8]}, ensure_ascii=False, indent=2) + "\n")
    return {"schema_version": 1, "study_id": cfg["study_id"], "status": "PASS",
            "inference_executed": False, "actual_model_results": "NOT_EVALUATED",
            "registration_status": cfg["registration_status"],
            "synthetic_oracle_episodes_checked": 64, "tokenized_example_episodes": 8,
            "max_example_input_tokens": maximum, "status_encoding_token_lengths": status_token_lengths,
            "four_single_token_codes": True, "disjoint_train_test_X_queries": True,
            "response_and_answer_balance": True, "independent_budget_calculation": explicit,
            "budget": budget, "max_actual_forward_calls": cfg["budget"]["max_actual_forward_calls"],
            "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
            "private_example_manifest_sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    result = verify()
    if args.check:
        if json.loads(DEST.read_text()) != result:
            raise ValueError("R5 CPU design verification is stale")
    else:
        DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "model_inference": False,
                      "max_example_input_tokens": result["max_example_input_tokens"],
                      "planned_forward_calls": result["independent_budget_calculation"]}))


if __name__ == "__main__":
    main()
