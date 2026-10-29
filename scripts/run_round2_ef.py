#!/usr/bin/env python3
"""Execute two registered, bounded real-BF16 studies; CPU --prepare/--check.

Calibration writes a frozen parameter file. Confirmation is a separate command
and requires that file to be committed to local Git before inference starts.
Round 1 summaries, candidates and input configurations are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neuron_lab.runtime import ModelRuntime, Run, save_json, source_hashes
from neuron_lab.round2_ef_corpus import build_manifest, validate_manifest
from neuron_lab.round2_ef import (AlignedProbe, cpu_observation, probe_logits, option_metrics,
    geometry, direction, matched_post_vector, frozen_geometry, family_E_table, fit_E_models)
from neuron_lab.evaluation import distribution_metrics

CONFIG = ROOT / "configs/round2_ef.yaml"
PROTECTED = ["logs/summary/analysis.json", "logs/summary/synthesis.json", "logs/summary/model_load.json",
             "logs/summary/smoke.json", "configs/hypotheses.json", "configs/validation_cases.json",
             "configs/utility.yaml", "catalog/public/neurons.json"]
IMPLEMENTATION = ["scripts/run_round2_ef.py", "scripts/analyze_round2_ef.py", "src/neuron_lab/round2_ef.py",
                  "src/neuron_lab/round2_ef_corpus.py", "src/neuron_lab/runtime.py", "src/neuron_lab/hooks.py",
                  "src/neuron_lab/smoke.py", "src/neuron_lab/evaluation.py", "configs/round2_ef.yaml"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    return {p: sha(ROOT / p) for p in IMPLEMENTATION}


def git(*args):
    return subprocess.check_output(["git", "--git-dir=runs/git-metadata", "--work-tree=.", *args], cwd=ROOT, text=True).strip()


class ForwardBudget:
    def __init__(self, runtime, cfg, prior=0, prior_seconds=0):
        self.count, self.prior = 0, prior
        self.started, self.cfg, self.prior_seconds = time.monotonic(), cfg, prior_seconds
        self.handle = runtime.model.register_forward_pre_hook(self.tick)

    def tick(self, *args):
        self.count += 1
        if self.count + self.prior > self.cfg["max_forward_calls"]:
            raise RuntimeError("Registered total forward-call cap reached")
        if time.monotonic() - self.started + self.prior_seconds > self.cfg["max_seconds"]:
            raise TimeoutError("Registered combined GPU execution time cap reached")


def row_base(case, logits, inputs, label_ids):
    return {"study": case["study"], "case_id": case["id"], "family_id": case["family_id"],
            "task": case["task"], "mode": case["mode"], "split": case["split"], "order": case["order"],
            "input_tokens": inputs["input_ids"].shape[-1], "position_index": inputs["input_ids"].shape[-1] - 1,
            "selected_token_id": int(inputs["input_ids"][0, -1]), **option_metrics(logits, case, label_ids)}


def make_row(case, arm, logits, baseline_logits, inputs, label_ids, **kwargs):
    return {**row_base(case, logits, inputs, label_ids), "arm": arm,
            **distribution_metrics(baseline_logits, logits), **kwargs}


def write_rows(path, rows):
    with path.open("a") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


@torch.inference_mode()
def audit_probe(runtime, cases, labels):
    checks, audit = {}, []
    t0 = time.monotonic()
    for i, case in enumerate(cases[:3]):
        inputs = runtime.inputs(case["text"])
        bare = runtime.logits(inputs)
        observed, natural = probe_logits(runtime, inputs)
        checks[f"{i}_read_only_exact"] = torch.equal(bare, observed)
        c = float(natural[15]["act"][9637])
        edited, _ = probe_logits(runtime, inputs, [{"layer": 15, "neuron": 9637, "kind": "technical_restore", "value": c}])
        checks[f"{i}_technical_restore_exact"] = torch.equal(bare, edited)
        checks[f"{i}_removal_exact"] = torch.equal(bare, runtime.logits(inputs))
        edits = [{"layer": 15, "neuron": 9637, "kind": "add", "value": .5},
                 {"layer": 26, "neuron": 719, "kind": "zero"}]
        with AlignedProbe(runtime, edits) as joint_probe:
            changed = runtime.logits(inputs)
            capture = joint_probe.observed
        checks[f"{i}_causal_hook_order"] = joint_probe.event_order == [15, 22, 26, 35]
        expected = natural[15]["act"][9637].clone() + .5
        checks[f"{i}_joint_bf16_formula"] = bool(torch.equal(capture[15]["after_act"][9637], expected) and capture[26]["after_act"][719] == 0)
        delta = capture[15]["post"] - natural[15]["post"]
        single, single_obs = probe_logits(runtime, inputs, edits[:1])
        replay_delta = single_obs[15]["post"].float() - natural[15]["post"].float()
        replay, _ = probe_logits(runtime, inputs, post_edits={15: replay_delta})
        checks[f"{i}_postnorm_replay_exact"] = torch.equal(single, replay)
        checks[f"{i}_finite"] = bool(changed.isfinite().all())
        for layer, obs in capture.items():
            reconstructed = (obs["residual_mid"] + obs["post"]) * obs["layer_scalar"]
            checks[f"{i}_residual_reconstruction_L{layer}"] = torch.equal(reconstructed, obs["layer_output"])
        ids = inputs["input_ids"][0].cpu().tolist()
        audit.append({"case_id": case["id"], "input_ids": ids,
            "selected_index": len(ids) - 1, "selected_token_id": ids[-1],
            "tail_tokens": runtime.tokenizer.convert_ids_to_tokens(ids[-12:]),
            "labels": labels, "selected_is_final_chat_template_token": True})
    checks["all_new_hooks_removed"] = all(not m._forward_pre_hooks and not m._forward_hooks for m in runtime.modules.values())
    if not all(checks.values()):
        raise RuntimeError(f"New aligned-hook audit failed: {checks}")
    return {"status": "PASS", "checks": checks, "token_audit": audit, "seconds": time.monotonic() - t0}


@torch.inference_mode()
def run_E(runtime, run, cases, baselines, params, cfg, labels, budget):
    rows, seeds = [], cfg["E"]["control_seed_calibration" if cases[0]["split"] == "calibration" else "control_seed_confirmation"]
    for j, case in enumerate(cases):
        run.check_budget()
        inputs = runtime.inputs(case["text"])
        saved = baselines[case["id"]]
        base_logits = saved["logits"].to(runtime.cfg["device"])
        base = saved["observed"]
        case_rows = [make_row(case, "clean", base_logits, base_logits, inputs, labels)]
        for cid, candidate in params["candidates"].items():
            layer, n = candidate["layer"], candidate["neuron"]
            control = params["random_neurons"][cid]
            for sign in (-1, 1):
                amount = sign * cfg["E"]["sigma"] * candidate["std"]
                logits, obs = probe_logits(runtime, inputs, [{"layer": layer, "neuron": n, "kind": "add", "value": amount}])
                geom = geometry(base, obs, layer)
                ref = obs[layer]["post"].float().cpu() - base[layer]["post"].float()
                value = float(obs[layer]["after_act"][n])
                case_rows.append(make_row(case, "candidate", logits, base_logits, inputs, labels,
                    candidate_id=cid, layer=layer, neuron=n, sign=sign, nominal_delta=amount,
                    actual_value=value, outside_calibration_support=not candidate["min"] <= value <= candidate["max"], **geom))
                target = geom["postnorm_l2"]
                ca = amount * candidate["write_norm"] / control["write_norm"]
                attempts = []
                for attempt in range(1 + cfg["E"]["random_neuron_norm_corrections_maximum"]):
                    cl, co = probe_logits(runtime, inputs, [{"layer": layer, "neuron": control["neuron"], "kind": "add", "value": ca}])
                    cg = geometry(base, co, layer, ref)
                    mismatch = abs(cg["postnorm_l2"] - target) / target if target else float(cg["postnorm_l2"] != 0)
                    attempts.append({"candidate_id": cid, "sign": sign, "case_id": case["id"], "attempt": attempt,
                                     "delta": ca, "postnorm_l2": cg["postnorm_l2"], "norm_relative_error": mismatch})
                    if mismatch <= cfg["E"]["norm_match_tolerance"] or cg["postnorm_l2"] == 0 or attempt == cfg["E"]["random_neuron_norm_corrections_maximum"]:
                        break
                    ca *= min(4, max(.25, target / cg["postnorm_l2"]))
                write_rows(run.path / "geometry_attempts.jsonl", attempts)
                cvalue = float(co[layer]["after_act"][control["neuron"]])
                case_rows.append(make_row(case, "random_neuron", cl, base_logits, inputs, labels,
                    candidate_id=cid, layer=layer, neuron=control["neuron"], sign=sign, nominal_delta=ca,
                    actual_value=cvalue, outside_calibration_support=not control["min"] <= cvalue <= control["max"],
                    norm_relative_error=mismatch, geometry_attempts=len(attempts), **cg))
                for seed in seeds:
                    v = direction(len(base[layer]["post"]), seed + 1000 * layer) * sign
                    v, error = matched_post_vector(base[layer]["post"], v, target)
                    dl, do = probe_logits(runtime, inputs, post_edits={layer: v})
                    dg = geometry(base, do, layer, ref)
                    actual_error = abs(dg["postnorm_l2"] - target) / target if target else float(dg["postnorm_l2"] != 0)
                    case_rows.append(make_row(case, "random_direction", dl, base_logits, inputs, labels,
                        candidate_id=cid, layer=layer, seed=seed, sign=sign,
                        norm_relative_error=actual_error, planned_rounding_match_error=error, **dg))
        rows.extend(case_rows)
        write_rows(run.path / "E_interventions.jsonl", case_rows)
        if (j + 1) % 8 == 0:
            print(json.dumps({"event": "E_progress", "split": case["split"], "cases": j + 1, "total": len(cases), "forwards": budget.count}), flush=True)
    return rows


@torch.inference_mode()
def run_F(runtime, run, cases, baselines, params, cfg, labels, budget):
    lookup = {(c["family_id"], c["mode"], c["order"]): c for c in cases}
    rows = []
    rn = params["random_neurons"]["H003"]["neuron"]
    for j, case in enumerate(cases):
        run.check_budget()
        inputs = runtime.inputs(case["text"])
        base = baselines[case["id"]]
        b_logits, b_obs = base["logits"].to(runtime.cfg["device"]), base["observed"]
        opposite = "copy" if case["mode"] == "translate" else "translate"
        donor = baselines[lookup[case["family_id"], opposite, case["order"]]["id"]]["observed"]
        d = {"layer": 15, "neuron": 9637, "kind": "replace", "value": float(donor[15]["act"][9637])}
        dz = {"layer": 15, "neuron": 9637, "kind": "zero"}
        rv = params["F_rescue"][case["mode"]]["candidate"]["value"]
        rr = params["F_rescue"][case["mode"]]["random"]["value"]
        r = {"layer": 26, "neuron": 719, "kind": "replace", "value": rv}
        random_r = {"layer": 26, "neuron": rn, "kind": "replace", "value": rr}
        rz = {"layer": 26, "neuron": 719, "kind": "zero"}
        restore = {"layer": 15, "neuron": 9637, "kind": "replace", "value": float(b_obs[15]["act"][9637])}
        dlclean = {"layer": 26, "neuron": 719, "kind": "replace", "value": float(b_obs[26]["act"][719])}
        specs = {"damage_cf": [d], "damage_zero": [dz], "rescue": [r], "damage_cf_rescue": [d, r],
            "damage_zero_rescue": [dz, r], "random_rescue": [random_r], "damage_cf_random_rescue": [d, random_r],
            "technical_restore": [d, restore], "joint_zero": [dz, rz], "downstream_zero": [rz],
            "damage_cf_downstream_clean": [d, dlclean]}
        values = {"clean": (b_logits, b_obs)}
        for arm, edits in specs.items():
            values[arm] = probe_logits(runtime, inputs, edits)
        clean_r_norm = geometry(b_obs, values["rescue"][1], 26)["postnorm_l2"]
        vec, error = matched_post_vector(b_obs[26]["post"], direction(len(b_obs[26]["post"]), 631), clean_r_norm)
        values["direction_rescue"] = probe_logits(runtime, inputs, post_edits={26: vec})
        values["damage_cf_direction_rescue"] = probe_logits(runtime, inputs, [d], post_edits={26: vec})
        case_rows = []
        for arm, (logits, obs) in values.items():
            local_base = cpu_observation(values["damage_cf"][1]) if arm.startswith("damage_cf_") else cpu_observation(values["damage_zero"][1]) if arm == "damage_zero_rescue" else b_obs
            local = geometry(local_base, obs, 26)
            g = geometry(b_obs, obs, 26)
            row = make_row(case, arm, logits, b_logits, inputs, labels, **g,
                upstream_clean_value=float(b_obs[15]["act"][9637]), upstream_after_value=float(obs[15]["after_act"][9637]),
                downstream_clean_value=float(b_obs[26]["act"][719]), downstream_after_value=float(obs[26]["after_act"][719]),
                downstream_local_postnorm_l2=local["postnorm_l2"], downstream_local_layer_output_l2=local["layer_output_l2"],
                rescue_value=rv, random_rescue_neuron=rn, random_rescue_value=rr,
                max_abs_logit_restore_error=float((logits - b_logits).abs().max()) if arm == "technical_restore" else None)
            case_rows.append(row)
        if next(r for r in case_rows if r["arm"] == "technical_restore")["max_abs_logit_restore_error"] != 0:
            raise RuntimeError("Same-forward technical restoration failed")
        rows.extend(case_rows)
        write_rows(run.path / "F_interventions.jsonl", case_rows)
        if (j + 1) % 16 == 0:
            print(json.dumps({"event": "F_progress", "split": case["split"], "cases": j + 1, "total": len(cases), "forwards": budget.count}), flush=True)
    return rows


def check_files(cfg):
    manifest = json.loads((ROOT / cfg["manifest"]).read_text())
    validate_manifest(manifest["cases"], cfg)
    expected = build_manifest(cfg)
    if manifest != expected:
        raise ValueError("Manifest does not reproduce from frozen construction rules")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--phase", choices=["calibration", "confirmation"])
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    manifest_path = ROOT / cfg["manifest"]
    if args.prepare:
        manifest = build_manifest(cfg)
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("Refusing to overwrite a different existing manifest")
        save_json(manifest_path, manifest)
    manifest = check_files(cfg)
    if args.check or args.prepare:
        print(json.dumps({"status": "PASS", "mode": "CPU validation only", "cases": len(manifest["cases"]),
                          "counts": dict(Counter((c["study"] + "/" + c["split"]) for c in manifest["cases"])), "manifest_sha256": sha(manifest_path)}))
        return
    if not args.phase:
        parser.error("Select --phase calibration or confirmation")
    protected = {p: sha(ROOT / p) for p in PROTECTED}
    registered = {"implementation_sha256": fingerprint(), "manifest_sha256": sha(manifest_path), "local_head_before_inference": git("rev-parse", "HEAD")}
    for path, digest in {**fingerprint(), cfg["manifest"]: sha(manifest_path)}.items():
        tracked = subprocess.check_output(["git", "--git-dir=runs/git-metadata", "--work-tree=.", "show", f"HEAD:{path}"], cwd=ROOT)
        if hashlib.sha256(tracked).hexdigest() != digest:
            raise RuntimeError(f"Uncommitted preregistered implementation/input: {path}")
    frozen_path = ROOT / cfg["frozen_parameters"]
    previous = json.loads(frozen_path.read_text()) if args.phase == "confirmation" else None
    if previous:
        if previous["registration"]["implementation_sha256"] != fingerprint() or previous["registration"]["manifest_sha256"] != sha(manifest_path):
            raise RuntimeError("Confirmation implementation/manifest differs from frozen calibration")
        committed = git("show", f"HEAD:{cfg['frozen_parameters']}")
        if json.loads(committed) != previous:
            raise RuntimeError("Frozen parameters must be committed locally before confirmation")
    runtime = ModelRuntime(audit_path=ROOT / f"logs/summary/round2_ef_model_load_{args.phase}.json")
    budget = ForwardBudget(runtime, cfg, prior=previous["calibration_forward_calls"] if previous else 0,
                           prior_seconds=previous["calibration_elapsed_seconds"] if previous else 0)
    from neuron_lab.smoke import smoke
    smoke_result = smoke(runtime, stage=f"round2_ef_smoke_{args.phase}")
    if smoke_result["status"] != "PASS":
        raise RuntimeError("BF16 smoke gate failed")
    cases = [c for c in manifest["cases"] if c["split"] == args.phase]
    labels = {l: runtime.tokenizer.encode(l, add_special_tokens=False) for l in ("A", "B")}
    if any(len(v) != 1 for v in labels.values()) or labels["A"] == labels["B"]:
        raise RuntimeError("Option labels must be different single tokens")
    labels = {l: v[0] for l, v in labels.items()}
    audit = audit_probe(runtime, cases, labels)
    run = Run(f"round2_ef_{args.phase}", cfg)
    try:
        save_json(run.path / "registration.json", registered)
        save_json(run.path / "hook_audit.json", audit)
        baselines, first_fifty = {}, time.monotonic()
        forward_before = budget.count
        for i, case in enumerate(cases):
            run.check_budget()
            inputs = runtime.inputs(case["text"])
            logits, obs = probe_logits(runtime, inputs)
            baselines[case["id"]] = {"logits": logits.cpu(), "observed": cpu_observation(obs),
                                     "metrics": row_base(case, logits, inputs, labels)}
            if i + 1 == 50:
                print(json.dumps({"event": "timing_50", "seconds": time.monotonic() - first_fifty, "forwards": budget.count - forward_before}), flush=True)
        torch.save(baselines, run.path / "baseline_cache.pt")
        if previous:
            params = previous["geometry"]
        else:
            params = frozen_geometry(cases, {k: v["observed"] for k, v in baselines.items()}, cfg,
                                     {l: runtime.modules[l].weight.detach() for l in (15, 22, 26, 35)})
        save_json(run.path / "geometry.json", params)
        E_rows = run_E(runtime, run, [c for c in cases if c["study"] == "E"], baselines, params, cfg, labels, budget)
        F_rows = run_F(runtime, run, [c for c in cases if c["study"] == "F"], baselines, params, cfg, labels, budget)
        paths = ["E_interventions.jsonl", "F_interventions.jsonl", "baseline_cache.pt", "geometry.json", "registration.json", "hook_audit.json", "geometry_attempts.jsonl"]
        if any(sha(ROOT / p) != h for p, h in protected.items()):
            raise RuntimeError("A protected Round 1 artifact changed")
        summary = run.finish({"model_revision": runtime.cfg["revision"], "phase": args.phase,
            "registration": registered, "frozen_parameters_sha256": sha(frozen_path) if previous else None,
            "n_cases": len(cases), "n_families": len({c["family_id"] for c in cases}),
            "E_rows": len(E_rows), "F_rows": len(F_rows), "forward_calls": budget.count,
            "combined_forward_calls": budget.count + budget.prior,
            "label_token_ids": labels, "protected_round1_sha256": protected,
            "raw_sha256": {p: sha(run.path / p) for p in paths}, "new_hook_audit": audit["checks"],
            "scientific_hypotheses_status": "requires CPU analysis; execution PASS is not hypothesis support"})
        if not previous:
            frozen = {"status": "FROZEN_AFTER_CALIBRATION_BEFORE_CONFIRMATION", "model_revision": cfg["model_revision"],
                "frozen_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(), "registration": registered,
                "geometry": params, "E_models": fit_E_models(family_E_table(E_rows, cases)),
                "calibration_run_id": run.id, "calibration_forward_calls": budget.count,
                "calibration_elapsed_seconds": time.monotonic() - budget.started,
                "calibration_raw_sha256": summary["raw_sha256"], "confirmation_inference_started": False}
            save_json(frozen_path, frozen)
        print(json.dumps({"event": "round2_phase_complete", "phase": args.phase, "run_id": run.id,
                          "status": summary["status"], "forwards": budget.count, "frozen_parameters": str(frozen_path)}), flush=True)
    except Exception as exc:
        run.finish({"error": str(exc), "forward_calls": budget.count, "model_revision": runtime.cfg["revision"]}, "FAIL")
        raise
    finally:
        budget.handle.remove()


if __name__ == "__main__":
    main()
