"""R2-E/F aligned-site probes, geometry matching and frozen CPU estimators."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import torch

LAYERS = (15, 22, 26, 35)


class AlignedProbe:
    """Capture/modify one final prefill position, with deterministic hook order.

    Edits are native down_proj-input scalars. Residual vectors are applied at
    the post-MLP RMSNorm output. Multiple native layers can be edited in the
    same forward; the model's forward order supplies causal ordering.
    """
    def __init__(self, runtime, edits=(), post_edits=None):
        self.runtime, self.edits = runtime, list(edits)
        self.post_edits = post_edits or {}
        self.observed, self.handles, self.event_order = {}, [], []

    def __enter__(self):
        modules = dict(self.runtime.model.named_modules())
        for layer in LAYERS:
            down = self.runtime.modules[layer]
            decoder_name = self.runtime.module_names[layer].removesuffix(".mlp.down_proj")
            decoder = modules[decoder_name]
            self.observed[layer] = {}

            def pre(module, args, layer=layer):
                self.event_order.append(layer)
                x = args[0]
                self.observed[layer]["act"] = x[0, -1].detach().clone()
                changes = [e for e in self.edits if e["layer"] == layer]
                y = x.clone() if changes else x
                for e in changes:
                    n, value = e["neuron"], float(e.get("value", 0))
                    if e["kind"] == "add":
                        y[:, -1, n] += value
                    elif e["kind"] == "zero":
                        y[:, -1, n] = 0
                    elif e["kind"] == "replace":
                        y[:, -1, n] = value
                    elif e["kind"] == "technical_restore":
                        y[:, -1, n] = 0
                        y[:, -1, n] = value
                    else:
                        raise ValueError("Unsupported aligned edit")
                self.observed[layer]["after_act"] = y[0, -1].detach().clone()
                return (y, *args[1:]) if changes else None

            def down_out(module, args, output, layer=layer):
                self.observed[layer]["pre_norm"] = output[0, -1].detach().clone()

            def norm_out(module, args, output, layer=layer, decoder=decoder):
                self.observed[layer]["natural_post"] = output[0, -1].detach().clone()
                changed = output
                if layer in self.post_edits:
                    changed = output.clone()
                    changed[0, -1] = (output[0, -1].float() + self.post_edits[layer].to(output.device).float()).to(output.dtype)
                self.observed[layer]["post"] = changed[0, -1].detach().clone()
                self.observed[layer]["layer_scalar"] = float(decoder.layer_scalar) if layer == decoder.layer_idx else 1.0
                return changed if layer in self.post_edits else None

            def layer_out(module, args, output, layer=layer):
                self.observed[layer]["layer_output"] = output[0, -1].detach().clone()

            def mid_in(module, args, layer=layer):
                self.observed[layer]["residual_mid"] = args[0][0, -1].detach().clone()

            self.handles.append(down.register_forward_pre_hook(pre))
            self.handles.append(down.register_forward_hook(down_out))
            self.handles.append(decoder.post_feedforward_layernorm.register_forward_hook(norm_out))
            self.handles.append(decoder.pre_feedforward_layernorm.register_forward_pre_hook(mid_in))
            self.handles.append(decoder.register_forward_hook(layer_out))
        return self

    def __exit__(self, *args):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


def cpu_observation(observed):
    return {layer: {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in values.items()}
            for layer, values in observed.items()}


def probe_logits(runtime, inputs, edits=(), post_edits=None):
    with AlignedProbe(runtime, edits, post_edits) as probe:
        logits = runtime.logits(inputs)
    return logits, probe.observed


def option_metrics(logits, case, label_ids):
    row = logits[0]
    correct = label_ids[case["correct_label"]]
    wrong = label_ids["B" if case["correct_label"] == "A" else "A"]
    semantic1 = label_ids[case["semantic1_label"]]
    semantic0 = label_ids["B" if case["semantic1_label"] == "A" else "A"]
    logp = row.double().log_softmax(-1)
    top_values, top_ids = row.topk(2)
    top = int(top_ids[0])
    return {"y": float(row[correct] - row[wrong]),
            "semantic_axis_margin": float(row[semantic1] - row[semantic0]),
            "top1_margin": float(top_values[0] - top_values[1]),
            "entropy": float(-(logp.exp() * logp).sum()), "top1_id": top,
            "top1_label_valid": top in label_ids.values(), "raw_top1_correct": top == correct,
            "forced_choice_correct": bool(row[correct] > row[wrong]),
            "correct_label_logprob": float(logp[correct])}


def geometry(base, changed, layer, vector=None):
    b = base[layer]
    p0, p1 = b["post"].float(), changed[layer]["post"].float().cpu()
    change = p1 - p0
    raw = changed[layer]["pre_norm"].float().cpu() - b["pre_norm"].float()
    acts = changed[layer]["after_act"].float().cpu() - b["after_act"].float()
    norm = float(change.norm())
    reference = p0 if vector is None else vector.float().cpu()
    denom = float(reference.norm()) * norm
    return {"postnorm_l2": norm, "pre_norm_mlp_l2": float(raw.norm()),
            "layer_output_l2": float((changed[layer]["layer_output"].float().cpu() - b["layer_output"].float()).norm()),
            "actual_activation_l2": float(acts.norm()),
            "postnorm_cosine_to_reference": float(change.dot(reference) / denom) if denom else None,
            "layer_scalar": b["layer_scalar"],
            "postnorm_branch_l2_after_layer_scalar": norm * abs(b["layer_scalar"])}


def direction(width, seed):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    v = torch.randn(width, dtype=torch.float64, generator=generator)
    return (v / v.norm()).float()


def matched_post_vector(base, vector, target_norm):
    """Match delivered BF16 delta, including rounding, without model outcomes."""
    b = base.detach().cpu()
    vector = vector.detach().cpu().float()
    if target_norm <= 0:
        return torch.zeros_like(vector), 0.0
    lo, hi = 0.0, max(target_norm, 1e-8)
    best, best_error = None, float("inf")
    for _ in range(12):
        actual = (b.float() + hi * vector).to(b.dtype).float() - b.float()
        if float(actual.norm()) >= target_norm:
            break
        hi *= 2
    for _ in range(24):
        scale = (lo + hi) / 2
        actual = (b.float() + scale * vector).to(b.dtype).float() - b.float()
        norm = float(actual.norm())
        if abs(norm - target_norm) < best_error:
            best, best_error = scale * vector, abs(norm - target_norm)
        if norm > target_norm:
            hi = scale
        else:
            lo = scale
    return best, best_error / target_norm


def frozen_geometry(cases, observations, cfg, weights):
    """Select random coordinates by calibration variance/write norm only."""
    result = {"candidates": {}, "random_neurons": {}, "F_rescue": {}}
    e_cases = [c for c in cases if c["study"] == "E"]
    for cand in cfg["candidates"]:
        layer, n = cand["layer"], cand["neuron"]
        values = torch.stack([observations[c["id"]][layer]["act"].float() for c in e_cases])
        std = values.std(0, unbiased=True)
        norms = weights[layer].float().norm(dim=0).cpu()
        target_std, target_norm = float(std[n]), float(norms[n])
        if target_std <= 0 or target_norm <= 0:
            raise ValueError("No calibration variation for a frozen candidate")
        result["candidates"][cand["id"]] = {**cand, "std": target_std,
            "min": float(values[:, n].min()), "max": float(values[:, n].max()),
            "write_norm": target_norm}
        gen = np.random.default_rng(cfg["E"]["control_seed_calibration"][0] + layer)
        pool = gen.choice(values.shape[1], cfg["E"]["random_coordinate_pool"], replace=False)
        pool = [int(i) for i in pool if int(i) != n and float(std[i]) > 0 and float(norms[i]) > 0]
        matched = min(pool, key=lambda i: abs(math.log(float(std[i]) / target_std)) + abs(math.log(float(norms[i]) / target_norm)))
        result["random_neurons"][cand["id"]] = {"layer": layer, "neuron": matched,
             "std": float(std[matched]), "write_norm": float(norms[matched]),
             "min": float(values[:, matched].min()), "max": float(values[:, matched].max())}
    random_n = result["random_neurons"]["H003"]["neuron"]
    for mode in cfg["F"]["modes"]:
        f_cases = [c for c in cases if c["study"] == "F" and c["mode"] == mode]
        result["F_rescue"][mode] = {}
        for key, n in [("candidate", 719), ("random", random_n)]:
            vals = [float(observations[c["id"]][26]["act"][n]) for c in f_cases]
            result["F_rescue"][mode][key] = {"value": float(np.median(vals)),
                "min": min(vals), "max": max(vals), "n_cases": len(vals), "selection": "natural conditional median; no behavioral search"}
    return result


def family_E_table(rows, cases, epsilon=1e-6):
    """One row per E source family; signs, controls, choice orders stay paired."""
    selected = {c["id"]: c for c in cases if c["study"] == "E"}
    by_case, families = defaultdict(list), defaultdict(list)
    for row in rows:
        if row["study"] == "E":
            by_case[row["case_id"]].append(row)
    for case_id, items in by_case.items():
        case = selected[case_id]
        baseline = next(r for r in items if r["arm"] == "clean")
        candidate_rows = [r for r in items if r["arm"] == "candidate"]
        controls = [r for r in items if r["arm"] in {"random_neuron", "random_direction"}]
        if len(candidate_rows) != 8 or len(controls) != 24:
            raise ValueError("Incomplete E intervention matrix")
        m = np.mean([abs(r["y"] - baseline["y"]) for r in candidate_rows])
        control_m = np.median([abs(r["y"] - baseline["y"]) for r in controls])
        seeds = sorted({r["seed"] for r in controls if r["arm"] == "random_direction"})
        subcontrols = {"native": [r for r in controls if r["arm"] == "random_neuron"],
            **{f"direction_{i}": [r for r in controls if r.get("seed") == seed] for i, seed in enumerate(seeds)}}
        separated = {f"control_{k}": float(np.median([abs(r["y"] - baseline["y"]) for r in values])) for k, values in subcontrols.items()}
        dose = np.mean([r["postnorm_l2"] for r in candidate_rows])
        output_dose = np.mean([r["layer_output_l2"] for r in candidate_rows])
        families[case["family_id"]].append({"magnitude": float(m), "control_magnitude": float(control_m),
            **separated, "postnorm_dose": float(dose), "baseline_features": [abs(baseline["y"]), baseline["top1_margin"],
            baseline["entropy"], math.log(baseline["input_tokens"]), math.log(dose + epsilon), math.log(output_dose + epsilon)],
            "candidate_kl": float(np.mean([r["kl_base_to_changed"] for r in candidate_rows])),
            "candidate_flip_count": sum(r["top1_changed"] for r in candidate_rows),
            "all_flip_count": sum(r.get("top1_changed", False) for r in items),
            "task": case["task"]})
    table = []
    for family_id, items in sorted(families.items()):
        table.append({"family_id": family_id, "task": items[0]["task"],
            "y": float(math.log(np.mean([r["magnitude"] for r in items]) + epsilon)),
            "s": float(math.log(np.mean([r["control_magnitude"] for r in items]) + epsilon)),
            **{f"s_{k}": float(math.log(np.mean([r[f"control_{k}"] for r in items]) + epsilon)) for k in ("native", "direction_0", "direction_1")},
            "magnitude": float(np.mean([r["magnitude"] for r in items])),
            "baseline_features": np.mean([r["baseline_features"] for r in items], axis=0).tolist(),
            "candidate_kl": float(np.mean([r["candidate_kl"] for r in items])),
            "all_flip_count": sum(r["all_flip_count"] for r in items),
            "candidate_flip_count": sum(r["candidate_flip_count"] for r in items)})
    return table


def ridge_fit(x, y, penalty=1.0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    p = np.eye(design.shape[1]) * penalty
    p[0, 0] = 0
    beta = np.linalg.solve(design.T @ design + p, design.T @ y)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "beta": beta.tolist(), "penalty": penalty}


def ridge_predict(model, x):
    x = np.asarray(x, float)
    design = np.column_stack([np.ones(len(x)), (x - model["mean"]) / model["scale"]])
    return design @ np.asarray(model["beta"])


def fit_E_models(table):
    b = np.array([r["baseline_features"] for r in table])
    y, s = np.array([r["y"] for r in table]), np.array([r["s"] for r in table])
    return {"baseline_only": ridge_fit(b[:, :4], y), "baseline_geometry": ridge_fit(b, y),
            "baseline_controls": ridge_fit(np.column_stack([b[:, :4], s]), y),
            "baseline_geometry_controls": ridge_fit(np.column_stack([b, s]), y),
            "control_from_baseline": ridge_fit(b[:, :4], s),
            "control_realizations": {k: ridge_fit(np.column_stack([b, [r[k] for r in table]]), y) for k in ("s_native", "s_direction_0", "s_direction_1")},
            "calibration_target_mean": float(y.mean()),
            "training_families": len(table), "fit_source": "calibration only"}


def interval(values, seed, resamples=5000):
    a = np.asarray(values, float)
    if len(a) == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), (resamples, len(a)))].mean(1)
    lo, hi = np.quantile(means, [.025, .975])
    return {"n": len(a), "mean": float(a.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def sign_p(values, seed, resamples=10000):
    a = np.asarray(values, float)
    if not len(a) or np.all(a == 0):
        return 1.0
    if len(a) <= 16:
        masks = np.arange(2 ** len(a), dtype=np.uint32)[:, None]
        signs = 2 * ((masks >> np.arange(len(a))) & 1).astype(float) - 1
        null = (signs * a).mean(1)
        return float(np.count_nonzero(np.abs(null) >= abs(a.mean()) - 1e-12) / len(null))
    rng = np.random.default_rng(seed)
    null = (rng.choice([-1., 1.], (resamples, len(a))) * a).mean(1)
    return float((np.count_nonzero(np.abs(null) >= abs(a.mean()) - 1e-12) + 1) / (resamples + 1))


def F_contrasts(rows, cases):
    grouped = defaultdict(dict)
    lookup = {c["id"]: c for c in cases if c["study"] == "F"}
    for row in rows:
        if row["study"] == "F":
            grouped[row["case_id"]][row["arm"]] = row
    family = defaultdict(list)
    expected = {"clean", "damage_cf", "rescue", "damage_cf_rescue", "random_rescue", "damage_cf_random_rescue", "direction_rescue", "damage_cf_direction_rescue", "damage_zero", "damage_zero_rescue", "technical_restore", "joint_zero", "downstream_zero", "damage_cf_downstream_clean"}
    for case_id, arms in grouped.items():
        if set(arms) != expected:
            raise ValueError("Incomplete F factorial conditions")
        y = {k: v["y"] for k, v in arms.items()}
        baseline, damaged = y["clean"], y["damage_cf"]
        i = (y["damage_cf_rescue"] - damaged) - (y["rescue"] - baseline)
        ir = (y["damage_cf_random_rescue"] - damaged) - (y["random_rescue"] - baseline)
        iv = (y["damage_cf_direction_rescue"] - damaged) - (y["direction_rescue"] - baseline)
        case = lookup[case_id]
        prob = {k: float(1 / (1 + np.exp(-np.clip(v, -700, 700)))) for k, v in y.items()}
        family[case["family_id"], case["mode"]].append({"damage": baseline - damaged,
            "interaction": i, "random_interaction": ir, "direction_interaction": iv,
            "specificity": i - ir, "direction_specificity": i - iv,
            "direct_steering": y["rescue"] - baseline,
            "raw_recovery": y["damage_cf_rescue"] - damaged,
            "probability_interaction": (prob["damage_cf_rescue"] - prob["damage_cf"]) - (prob["rescue"] - prob["clean"]),
            "first_stage_H003": arms["damage_cf"]["downstream_after_value"] - arms["clean"]["downstream_after_value"],
            "H001_natural_contrast": arms["damage_cf"]["upstream_after_value"] - arms["clean"]["upstream_after_value"],
            "oracle_block_recovery": y["damage_cf_downstream_clean"] - damaged,
            "oracle_direct_effect": y["damage_cf_downstream_clean"] - baseline,
            "downstream_zero_damage": baseline - y["downstream_zero"],
            "joint_zero_damage": baseline - y["joint_zero"],
            "joint_zero_nonadditivity": (baseline - y["joint_zero"]) - (baseline - y["damage_zero"]) - (baseline - y["downstream_zero"]),
            "zero_damage": baseline - y["damage_zero"],
            "zero_interaction": (y["damage_zero_rescue"] - y["damage_zero"]) - (y["rescue"] - baseline),
            "technical_restore_error": abs(y["technical_restore"] - baseline),
            "baseline_correct": float(arms["clean"]["forced_choice_correct"]),
            "damage_correct": float(arms["damage_cf"]["forced_choice_correct"]),
            "rescued_correct": float(arms["damage_cf_rescue"]["forced_choice_correct"])})
    return [{"family_id": fid, "mode": mode, **{k: float(np.mean([v[k] for v in vals])) for k in vals[0]}}
            for (fid, mode), vals in sorted(family.items())]
