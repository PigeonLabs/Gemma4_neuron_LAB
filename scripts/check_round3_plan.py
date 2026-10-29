#!/usr/bin/env python3
"""CPU proposal arithmetic and analytic contrast checks, not model inference."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
started = time.monotonic()
cfg = yaml.safe_load((ROOT / "configs/round3_plan.yaml").read_text())
checks = {}
checks["explicitly_unexecuted"] = cfg["status"] == "DRAFT_NOT_EXECUTED_NOT_FROZEN" and cfg["actual_model_evaluation"] == "NOT_EVALUATED"
q = cfg["qualification"]
checks["P0_arithmetic"] = q["new_families"] * q["variants_per_family"] * q["arms_per_variant_per_dtype"] * q["dtypes"] == q["planned_base_forward_calls"] <= q["max_forward_calls"]
for exp in cfg["experiments"]:
    count = sum(exp[phase]["families"] * exp["variants_per_family"] * exp[phase]["arms_per_variant"] for phase in ("discovery", "calibration", "confirmation"))
    checks[exp["id"] + "_arithmetic"] = count == exp["planned_base_forward_calls"] <= exp["max_forward_calls"]
    checks[exp["id"] + "_scope"] = exp["status"] == "NOT_EVALUATED" and exp["novelty_status"] == "UNVERIFIED"

r = np.repeat([1., -1.], 2)
c = np.tile([1., -1.], 2)
basis = np.stack([np.ones(4), r, c, r*c], axis=1)
coefficients = np.array([2., 3., 5., 7.])
raw_label_effect = c * (basis @ coefficients)
semantic_effect = c * raw_label_effect
checks["analytic_C_G_L_J_recovery"] = bool(np.allclose(basis.T @ semantic_effect / 4, coefficients))
pure_label_effect = c * np.ones(4) * 5
checks["label_bias_does_not_create_rule_gain"] = bool(np.allclose(basis.T @ pure_label_effect / 4, [0, 0, 5, 0]))
easy_clean_margin = 10 * r
checks["confidence_can_mimic_rule_gain_at_ceiling"] = bool(np.mean(r * (.4 * easy_clean_margin)) == 4)
checks["ceiling_confidence_design_is_rank_deficient"] = np.linalg.matrix_rank(np.column_stack([basis, easy_clean_margin])) == 4
extended = np.vstack([basis, basis])
hard_margin = np.r_[easy_clean_margin, [2., -1., 3., -4.]]
checks["additional_difficulty_can_identify_confidence_term"] = np.linalg.matrix_rank(np.column_stack([extended, hard_margin])) == 5

from transformers import AutoTokenizer
local = Path((ROOT / "runs/model_path.txt").read_text().strip())
if local.name != cfg["model_revision"]:
    raise ValueError("Tokenizer revision differs from proposed model")
tok = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
ids = {name: tok.encode(name, add_special_tokens=False) for name in ("A", "B", "C", "D")}
checks["ABCD_distinct_single_tokens"] = all(len(v)==1 for v in ids.values()) and len({v[0] for v in ids.values()}) == 4
checks = {name: bool(passed) for name, passed in checks.items()}
result = {"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL","scope":"CPU analytical toy contrasts, proposal arithmetic and local tokenizer only; not actual model outcomes",
    "new_model_inference_executed":False,"actual_experiment_status":"NOT_EVALUATED","model_revision":cfg["model_revision"],"token_ids":ids,"checks":checks,
    "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("configs/round3_plan.yaml","scripts/check_round3_plan.py")},"elapsed_seconds":time.monotonic()-started}
(ROOT/"logs/summary/round3_plan_checks.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
print(json.dumps(result))
if result["status"] != "PASS":
    raise SystemExit(1)
