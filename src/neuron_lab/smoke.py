from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import torch
import yaml

from .hooks import ActivationHook, Intervention
from .runtime import ROOT, Run, save_json


@torch.inference_mode()
def smoke(runtime) -> dict:
    runtime.smoke_passed = False
    cfg = yaml.safe_load((ROOT / "configs/smoke.yaml").read_text())
    run = Run("smoke", cfg)
    checks = {}
    raw = []
    try:
        for i, prompt in enumerate(["Reply with only the number: 7 + 8 =", "한국의 수도를 한 단어로 답하세요.", "Return a JSON object with the key color and value blue."]):
            run.check_budget()
            inputs = runtime.inputs(prompt)
            baseline = runtime.logits(inputs)
            generated = runtime.generate(inputs, cfg["max_new_tokens"])
            shapes = {}
            captured = {}
            def capture(layer, value):
                shapes[str(layer)] = {"shape": list(value.shape), "dtype": str(value.dtype), "finite": bool(value.isfinite().all())}
                if layer == 23:
                    captured["natural"] = value
            with ExitStack() as stack:
                for layer, module in runtime.modules.items():
                    stack.enter_context(ActivationHook(module, capture=lambda x, layer=layer: capture(layer, x)))
                observed = runtime.logits(inputs)
            checks[f"prompt_{i}_read_only_exact"] = torch.equal(baseline, observed)
            checks[f"prompt_{i}_all_shapes"] = len(shapes) == len(runtime.modules) and all(x["shape"][-1] == 15360 and x["finite"] for x in shapes.values())
            with ActivationHook(runtime.modules[23]):
                observed_gen = runtime.generate(inputs, cfg["max_new_tokens"])
            checks[f"prompt_{i}_generation_exact"] = generated["token_ids"] == observed_gen["token_ids"]
            after = runtime.logits(inputs)
            repeated = runtime.generate(inputs, cfg["max_new_tokens"])
            checks[f"prompt_{i}_removal_exact"] = torch.equal(baseline, after)
            checks[f"prompt_{i}_repeat_exact"] = generated["token_ids"] == repeated["token_ids"]
            natural = captured["natural"]
            changed_input = {}
            with ActivationHook(runtime.modules[23], Intervention("add", neuron=0, strength=0.25, position="last")):
                handle = runtime.modules[23].register_forward_pre_hook(lambda m, a: changed_input.update(value=a[0].detach().clone()))
                try:
                    changed = runtime.logits(inputs)
                finally:
                    handle.remove()
            expected = natural.clone()
            expected[:, -1:, 0] += 0.25
            checks[f"prompt_{i}_bf16_add_formula"] = torch.equal(changed_input["value"], expected)
            # First damage the coordinate, then restore its aligned clean value.
            def damage(module, args):
                value = args[0].clone()
                value[:, -1:, 0] = 0
                return (value, *args[1:])
            damage_handle = runtime.modules[23].register_forward_pre_hook(damage)
            try:
                with ActivationHook(runtime.modules[23], Intervention("restore", neuron=0, replacement=natural, position="last")):
                    restored = runtime.logits(inputs)
            finally:
                damage_handle.remove()
            checks[f"prompt_{i}_zero_then_restore_exact"] = torch.equal(baseline, restored)
            with ActivationHook(runtime.modules[23], Intervention("zero", neuron=0, position="last")):
                ablated = runtime.logits(inputs)
            checks[f"prompt_{i}_ablation_finite"] = bool(ablated.isfinite().all())
            checks[f"prompt_{i}_intervention_removal_exact"] = torch.equal(baseline, runtime.logits(inputs))
            raw.append({"prompt": prompt, "generation": generated, "shapes": shapes,
                        "add_logit_max_abs_delta": float((changed - baseline).abs().max()),
                        "ablation_logit_max_abs_delta": float((ablated - baseline).abs().max())})
        checks["all_hook_handles_removed"] = all(len(m._forward_pre_hooks) == 0 for m in runtime.modules.values())
        save_json(run.path / "outputs.json", raw)
        status = "PASS" if all(checks.values()) else "FAIL"
        summary = run.finish({"checks": checks, "passed": sum(checks.values()), "total": len(checks),
                              "prompts": len(raw), "model_revision": runtime.cfg["revision"],
                              "all_layer_capture": 48, "output_sha256": hashlib.sha256((run.path / "outputs.json").read_bytes()).hexdigest(),
                              "max_read_only_logit_difference": 0 if all(v for k,v in checks.items() if "read_only" in k) else None,
                              "restoration_scope": "same-forward zero ablation followed by aligned clean-coordinate replacement; technical recovery, not proof of a semantic mechanism"}, status)
        runtime.smoke_passed = status == "PASS"
        return summary
    except Exception as exc:
        run.finish({"error": str(exc), "checks": checks}, "FAIL")
        raise
