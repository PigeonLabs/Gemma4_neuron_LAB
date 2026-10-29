"""Discovery-frozen dose sweeps and held-out correlation/causal diagnostics."""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
import math

import numpy as np
import torch
import yaml

from .corpus import build_corpus, split_corpus, validate_corpus
from .evaluation import distribution_metrics, generated_quality, permutation_auc_p, prompt_bootstrap, roc_auc
from .hooks import ActivationHook, Intervention
from .runtime import ROOT, Run, save_json
from .survey import content_mask


def conditions(candidate: dict, cfg: dict, stage: str):
    values = cfg["strengths_sigma"] if stage == "steering" else [-1.0, 1.0]
    result = [{"name": "baseline", "kind": "baseline", "sigma": 0.0}]
    result += [{"name": f"add_{v:+g}sd", "kind": "add", "sigma": v} for v in values]
    result += [{"name": "zero_ablation", "kind": "zero", "sigma": None},
               {"name": "activation_restoration", "kind": "restore", "sigma": 0.0}]
    for kind in ["random_neuron", "random_direction"]:
        for space in ["activation", "projected"]:
            for seed in cfg["control_seeds"]:
                result.append({"name": kind+"_"+space, "kind": kind, "space": space, "seed": seed, "sigma": 1.0})
    return result


def make_spec(condition, candidate, module, natural, weight):
    kind, n = condition["kind"], candidate["neuron"]
    delta = float(candidate["std"])
    if kind == "baseline":
        return None
    if kind in {"add", "zero"}:
        return Intervention(kind, neuron=n, strength=(condition["sigma"] or 0)*delta, position="last")
    if kind == "restore":
        return Intervention("restore", neuron=n, replacement=natural, position="last")
    strength = delta
    if condition["space"] == "projected":
        target_norm = torch.linalg.vector_norm(weight[:,n])
        generator = torch.Generator(device="cpu").manual_seed(condition["seed"])
        if kind == "random_neuron":
            index = int(torch.randint(module.in_features, (), generator=generator))
            control_norm = torch.linalg.vector_norm(weight[:,index])
        else:
            direction = torch.randn(module.in_features, dtype=torch.float64, generator=generator)
            direction /= torch.linalg.vector_norm(direction)
            control_norm = torch.linalg.vector_norm(weight @ direction.to(device=weight.device,dtype=weight.dtype))
        strength *= float(target_norm / control_norm.clamp_min(1e-12))
    return Intervention(kind, strength=strength, position="last", seed=condition["seed"])


@torch.inference_mode()
def intervene_logits(runtime, inputs, candidate, condition, natural, weight):
    module = runtime.modules[candidate["layer"]]
    spec = make_spec(condition, candidate, module, natural, weight)
    if spec is None:
        return runtime.logits(inputs), {"activation_l2": 0.0, "projected_l2": 0.0, "resolved_neuron": None}
    observed = {}
    damage_handle = None
    if condition["kind"] == "restore":
        def damage(m, args):
            x = args[0].clone()
            x[:,-1:,candidate["neuron"]] = 0
            return (x,*args[1:])
        damage_handle = module.register_forward_pre_hook(damage)
    try:
        with ActivationHook(module, spec, capture=lambda x: observed.update(before=x[:,-1,:].float())) as hook:
            observer = module.register_forward_pre_hook(lambda m,args: observed.update(after=args[0][:,-1,:].detach().float()))
            try:
                logits = runtime.logits(inputs)
            finally:
                observer.remove()
            resolved = hook.resolved_neuron
    finally:
        if damage_handle is not None:
            damage_handle.remove()
    # Restoration's causal perturbation is measured relative to the clean input.
    before = observed["before"]
    if condition["kind"] == "restore":
        before = before.clone()
        before[:,candidate["neuron"]] = natural[:,-1,candidate["neuron"]].float()
    change = observed["after"] - before
    return logits, {"activation_l2": float(torch.linalg.vector_norm(change)),
                    "projected_l2": float(torch.linalg.vector_norm(change @ weight.T)),
                    "resolved_neuron": resolved,
                    "nominal_strength": float(spec.strength)}


@torch.inference_mode()
def generate_condition(runtime, inputs, candidate, condition, cfg, baseline_generation, natural, weight):
    if condition["kind"] in {"baseline", "restore"}:
        # Restoration is measured with a same-input aligned clean activation in
        # logits experiments; do not mislabel baseline generation as restoration.
        if condition["kind"] == "restore":
            return None
        value = baseline_generation
    else:
        spec = make_spec(condition, candidate, runtime.modules[candidate["layer"]], natural, weight)
        with ActivationHook(runtime.modules[candidate["layer"]], spec):
            value = runtime.generate(inputs, cfg["max_new_tokens"])
    return {"changed_from_baseline": value["token_ids"] != baseline_generation["token_ids"],
            "quality_proxy": generated_quality(value["token_ids"], value["text"]),
            "text": value["text"], "token_ids": value["token_ids"]}


def summarize(rows, activation_rows, candidates, cfg):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["candidate_id"], row["condition"])].append(row)
    aggregates = []
    for (cid,condition), group in groups.items():
        per_prompt = defaultdict(list)
        for row in group:
            per_prompt[row["prompt_id"]].append(row)
        metrics = {}
        for metric in ["kl_base_to_changed", "js_divergence", "total_variation", "top1_changed", "activation_l2", "projected_l2"]:
            values = [float(np.mean([r[metric] for r in prompt_rows])) for prompt_rows in per_prompt.values()]
            metrics[metric] = prompt_bootstrap(values, seed=cfg["seed"], resamples=cfg["bootstrap_resamples"])
        gens = [r["generation"] for r in group if r.get("generation") is not None]
        aggregates.append({"candidate_id": cid, "condition": condition, "n_prompts": len(per_prompt),
                           "n_interventions": len(group), "metrics": metrics,
                           "generation_n": len(gens),
                           "generation_changed_fraction": float(np.mean([g["changed_from_baseline"] for g in gens])) if gens else None,
                           "collapse_proxy_fraction": float(np.mean([g["quality_proxy"]["collapsed"] for g in gens])) if gens else None})
    correlation = []
    for candidate in candidates:
        values = [r for r in activation_rows if r["candidate_id"] == candidate["id"] and r.get("role","corpus") == "corpus"]
        pos = [r["prompt_mean"] for r in values if r["category"] == candidate["target_category"]]
        neg = [r["prompt_mean"] for r in values if r["category"] != candidate["target_category"]]
        if pos and neg:
            correlation.append({"candidate_id": candidate["id"], "target_category": candidate["target_category"],
                                "n_positive": len(pos), "n_negative": len(neg), "auc": roc_auc(pos,neg),
                                "positive_mean": float(np.mean(pos)), "negative_mean": float(np.mean(neg)),
                                "permutation_p_two_sided": permutation_auc_p(pos,neg,seed=cfg["seed"],resamples=2000)})
    return aggregates, correlation


@torch.inference_mode()
def execute(runtime, stage):
    cfg = yaml.safe_load((ROOT / "configs/steering.yaml").read_text())
    freeze_path = ROOT / "configs/hypotheses.json"
    frozen = json.loads(freeze_path.read_text())
    if frozen["model_revision"] != runtime.cfg["revision"]:
        raise ValueError("Hypotheses model revision mismatch")
    candidates = frozen["candidates"]
    if stage == "steering":
        candidates = [c for c in candidates if c["group"] == "primary"]
    split = {"steering":"discovery", "validation":"validation", "expanded":"test"}[stage]
    corpus = build_corpus()
    validate_corpus(corpus)
    prompts = split_corpus(corpus,split)
    if stage == "steering":
        seen = set()
        prompts = [p for p in prompts if p["category"] not in seen and not seen.add(p["category"])]
    if stage == "validation" and (ROOT / "configs/validation_cases.json").exists():
        cases = json.loads((ROOT / "configs/validation_cases.json").read_text())
        if cases["hypotheses_sha256"] != hashlib.sha256(freeze_path.read_bytes()).hexdigest():
            raise ValueError("Validation cases do not match frozen hypotheses")
        prompts += cases["prompts"]
        validate_corpus(corpus+cases["prompts"])
    run = Run(stage,{**cfg, "split":split, "hypotheses_sha256":hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
                     "hypotheses":frozen})
    weights = {c["layer"]: runtime.modules[c["layer"]].weight.float() for c in candidates}
    results, activation_rows = [], []
    target_categories = {c["target_category"] for c in candidates}
    # Generation sample is chosen by frozen categories, before any causal metric.
    generation_ids = []
    for category in sorted(target_categories | {"general_knowledge", "programming", "irrelevant_control"}):
        match = next((p["id"] for p in prompts if p["category"] == category),None)
        if match is not None:
            generation_ids.append(match)
    generation_ids = set(generation_ids[:cfg["max_generation_prompts"]])
    try:
        with (run.path / "interventions.jsonl").open("w") as output:
            for i,prompt in enumerate(prompts):
                run.check_budget()
                inputs = runtime.inputs(prompt["text"])
                mask = content_mask(runtime,prompt["text"],inputs)
                natural = {}
                def capture(layer,x):
                    for c in candidates:
                        if c["layer"] == layer:
                            natural[c["id"]] = x
                            activation_rows.append({"candidate_id":c["id"], "prompt_id":prompt["id"],
                                "category":prompt["category"], "role":prompt.get("role","corpus"),
                                "split":split, "prompt_mean":float(x[...,c["neuron"]][mask].float().mean()),
                                "last_activation":float(x[0,-1,c["neuron"]])})
                with ExitStack() as stack:
                    for layer in sorted({c["layer"] for c in candidates}):
                        stack.enter_context(ActivationHook(runtime.modules[layer],capture=lambda x,l=layer:capture(l,x)))
                    baseline = runtime.logits(inputs)
                baseline_generation = runtime.generate(inputs,cfg["max_new_tokens"]) if prompt["id"] in generation_ids else None
                for candidate in candidates:
                    if prompt.get("candidate_id") and prompt["candidate_id"] != candidate["id"]:
                        continue
                    for condition in conditions(candidate,cfg,stage):
                        run.check_budget()
                        if condition["kind"] == "baseline":
                            changed, norms = baseline, {"activation_l2":0.,"projected_l2":0.,"resolved_neuron":None}
                        else:
                            changed,norms = intervene_logits(runtime,inputs,candidate,condition,natural[candidate["id"]],weights[candidate["layer"]])
                        row = {"candidate_id":candidate["id"], "layer":candidate["layer"],"neuron":candidate["neuron"],
                               "prompt_id":prompt["id"],"category":prompt["category"],"role":prompt.get("role","corpus"),
                               "split":split,"condition":condition["name"],"strength_sigma":condition["sigma"],
                               "seed":condition.get("seed"), **distribution_metrics(baseline,changed),**norms}
                        do_gen = baseline_generation is not None and (condition["kind"] in {"baseline","zero"} or
                             condition["kind"] == "add" and abs(condition["sigma"]) in {1.0,2.0} or
                             condition.get("space") == "projected" and condition.get("seed") == cfg["control_seeds"][0])
                        if do_gen:
                            row["generation"] = generate_condition(runtime,inputs,candidate,condition,cfg,baseline_generation,natural[candidate["id"]],weights[candidate["layer"]])
                        output.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
                        output.flush()
                        results.append(row)
                print(json.dumps({"event":"steering_progress","stage":stage,"completed":i+1,"total":len(prompts),"interventions":len(results)}),flush=True)
                save_json(run.path / "progress.json",{"completed_prompts":i+1,"total_prompts":len(prompts),"interventions":len(results)})
        save_json(run.path / "activation_observations.json",activation_rows)
        aggregate, correlation = summarize(results,activation_rows,candidates,cfg)
        restoration = [r for r in results if r["condition"] == "activation_restoration"]
        if any(r["max_abs_logit_delta"] != 0 for r in restoration):
            raise AssertionError("Aligned technical restoration failed exact logit parity")
        public = {"model_revision":runtime.cfg["revision"],"split":split,"prompt_count":len(prompts),
                  "candidate_count":len(candidates),"intervention_count":len(results),
                  "hypotheses_sha256":hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
                  "aggregates":aggregate,"correlation":correlation,
                  "activation_observations":activation_rows,
                  "restoration_logit_exact":True,"position_policy":"last input token for next-token logits; final token of each forward for greedy generation",
                  "control_matching":"Nominal activation L2 and nominal down-projection L2; actual BF16 activation/projected norms measured per prompt",
                  "confidence_interval":"95% percentile bootstrap over prompts; random control seeds averaged within prompt",
                  "quality_scope":"Empty/repeated token trigram proxies only; no semantic correctness or model performance claim",
                  "raw_sha256":hashlib.sha256((run.path/"interventions.jsonl").read_bytes()).hexdigest()}
        return run.finish(public)
    except Exception as exc:
        run.finish({"error":repr(exc),"completed_interventions":len(results)},"FAIL")
        raise


@torch.inference_mode()
def live(runtime, job):
    """Bounded local-only interactive steering, outside the research test set."""
    sigma = float(job["sigma"])
    maximum = int(job.get("max_new_tokens",24))
    if not math.isfinite(sigma) or abs(sigma)>2 or not 1<=maximum<=24:
        raise ValueError("Live settings exceed the validated +/-2SD / 24-token envelope")
    if not isinstance(job["prompt"],str) or not job["prompt"].strip():
        raise ValueError("A nonempty prompt is required")
    frozen = json.loads((ROOT/"configs/hypotheses.json").read_text())
    candidate = next(c for c in frozen["candidates"] if c["id"]==job["candidate_id"])
    run = Run("live",{"max_seconds":300,"max_run_disk_gb":1,"candidate_id":candidate["id"],"sigma":sigma})
    try:
        inputs=runtime.inputs(job["prompt"])
        baseline=runtime.generate(inputs,maximum)
        run.check_budget()
        with ActivationHook(runtime.modules[candidate["layer"]],Intervention("add",neuron=candidate["neuron"],strength=sigma*candidate["std"],position="last")):
            changed=runtime.generate(inputs,maximum)
        summary=run.finish({"candidate_id":candidate["id"],"sigma":sigma,"changed":baseline["token_ids"]!=changed["token_ids"],"scope":"interactive exploration; excluded from validation/test evidence"})
        save_json(run.path/"outputs.json",{"prompt":job["prompt"],"baseline":baseline,"intervention":changed})
        return {**summary,"baseline":baseline["text"],"intervention":changed["text"]}
    except Exception as exc:
        run.finish({"error":repr(exc)},"FAIL")
        raise
