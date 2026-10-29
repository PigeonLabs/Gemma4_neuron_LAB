#!/usr/bin/env python3
"""Recompute public family-level R2 statistics without weights or raw tensors.

This validates the published sufficient summaries, not the upstream inference.
Use analyze_round2_ef.py --check for local raw/logit file integrity as well.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_round2_ef import E_analysis, F_analysis, holm


def main():
    s = json.loads((ROOT / "logs/summary/round2_ef_analysis.json").read_text())
    frozen = json.loads((ROOT / "configs/round2_ef_frozen.json").read_text())
    cfg = yaml.safe_load((ROOT / "configs/round2_ef.yaml").read_text())
    checks = []
    for phase, item in s["phases"].items():
        e = E_analysis(item["E"]["family_predictions"], frozen["E_models"], cfg)
        expected = copy.deepcopy(item["E"])
        expected.pop("hypothesis_status", None)
        expected["primary"].pop("holm_two_primary_p", None)
        if e != expected:
            raise ValueError(f"Public E family replay mismatch: {phase}")
        f = F_analysis(item["F"]["family_contrasts"], cfg)
        expected_f = copy.deepcopy(item["F"])
        expected_f["groups"]["translate"].pop("hypothesis_status", None)
        expected_f["groups"]["translate"].pop("reason", None)
        expected_f["groups"]["translate"]["stats"]["interaction"].pop("holm_two_primary_p", None)
        if f != expected_f:
            raise ValueError(f"Public F family replay mismatch: {phase}")
        checks += [f"{phase}_E_frozen_predictions_and_bootstrap", f"{phase}_F_factorial_bootstrap_and_damage_gate"]
    conf = s["phases"]["confirmation"]
    ep, fp = holm([conf["E"]["primary"]["two_sided_sign_randomization_p"], conf["F"]["groups"]["translate"]["stats"]["interaction"]["two_sided_sign_randomization_p"]])
    if s["co_primary_holm"] != {"E": ep, "F": fp, "number_of_tests": 2}:
        raise ValueError("Holm replay mismatch")
    print(json.dumps({"status": "PASS", "scope": "public family summaries only; upstream raw inference not independently revalidated", "checks": checks + ["two_primary_Holm"],
        "E_delta_r2": conf["E"]["primary"]["delta_r2"], "F_damage_gate": conf["F"]["groups"]["translate"]["damage_gate_passed"]}))


if __name__ == "__main__":
    main()
