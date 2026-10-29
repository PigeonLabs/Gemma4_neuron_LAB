"""Synthetic CPU invariants only; these are not real-model findings."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from neuron_lab.runtime import ROOT
from neuron_lab.round2_ef_corpus import build_manifest, validate_manifest, choices
from neuron_lab.round2_ef import AlignedProbe, matched_post_vector, direction, option_metrics, ridge_fit, ridge_predict, sign_p, F_contrasts


def test_generated_oracles_keep_family_and_choice_orders_paired():
    cfg = yaml.safe_load((ROOT / "configs/round2_ef.yaml").read_text())
    manifest = build_manifest(cfg)
    assert len(manifest["cases"]) == 512
    for a, b in zip(manifest["cases"][::2], manifest["cases"][1::2]):
        assert a["family_id"] == b["family_id"]
        assert a["correct_label"] != b["correct_label"]
        assert a["semantic1_label"] != b["semantic1_label"]
        assert a["option1"] == b["option1"] and a["option0"] == b["option0"]
    assert build_manifest(cfg) == manifest
    bad = [dict(c) for c in manifest["cases"]]
    bad[1]["split"] = "confirmation"
    with pytest.raises(ValueError, match="Family leakage"):
        validate_manifest(bad, cfg)


def test_signed_metric_ignores_arbitrary_answer_labels():
    _, correct, semantic = choices("toy", "true", "false", True, 0)
    m1 = option_metrics(torch.tensor([[0., 5., 2.]]), {"correct_label": correct, "semantic1_label": semantic}, {"A": 1, "B": 2})
    _, correct, semantic = choices("toy", "true", "false", True, 1)
    m2 = option_metrics(torch.tensor([[0., 2., 5.]]), {"correct_label": correct, "semantic1_label": semantic}, {"A": 1, "B": 2})
    assert m1["y"] == m2["y"] == 3
    assert m1["semantic_axis_margin"] == m2["semantic_axis_margin"] == 3


def test_bf16_direction_matching_measures_delivered_not_requested_norm():
    base = torch.linspace(-20, 20, 1024).to(torch.bfloat16)
    v, error = matched_post_vector(base, direction(1024, 7), .7)
    actual = (base.float() + v).to(torch.bfloat16).float() - base.float()
    assert abs(actual.norm().item() - .7) / .7 == pytest.approx(error)
    assert error < .03
    assert torch.equal(direction(1024, 7), direction(1024, 7))
    assert not torch.equal(direction(1024, 7), direction(1024, 8))


def test_frozen_ridge_extrapolates_without_refitting_confirmation():
    x = np.arange(12, dtype=float).reshape(-1, 1)
    fit = ridge_fit(x, 2 * x[:, 0] + 3, penalty=0)
    before = repr(fit)
    predicted = ridge_predict(fit, [[100], [-50]])
    np.testing.assert_allclose(predicted, [203, -97], atol=1e-10)
    assert repr(fit) == before


def test_sign_randomization_exact_small_sample_resolution():
    assert sign_p([1] * 4, 3) == 2 / 16
    assert sign_p([0] * 4, 3) == 1
    assert sign_p([-1] * 4, 3) == 2 / 16


def test_factorial_contrast_removes_standalone_steering():
    arms = {"clean": 10, "damage_cf": 6, "damage_zero": 5, "rescue": 12,
        "damage_cf_rescue": 8, "damage_zero_rescue": 7, "random_rescue": 11,
        "damage_cf_random_rescue": 7, "direction_rescue": 9, "damage_cf_direction_rescue": 5,
        "technical_restore": 10, "joint_zero": 3, "downstream_zero": 8, "damage_cf_downstream_clean": 7}
    rows = [{"study": "F", "case_id": "toy", "arm": arm, "y": y, "forced_choice_correct": True,
        "downstream_after_value": 2 if arm == "damage_cf" else 3, "upstream_after_value": 1} for arm, y in arms.items()]
    result = F_contrasts(rows, [{"id": "toy", "study": "F", "family_id": "f", "mode": "translate"}])[0]
    assert result["raw_recovery"] == 2
    assert result["interaction"] == 0  # additive steering is not damage-specific rescue
    assert result["damage"] == 4
    assert result["first_stage_H003"] == -1
    assert result["oracle_block_recovery"] == 1


class ToyLayer(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.layer_idx, self.layer_scalar = index, 1.
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(4, 4, bias=False).to(torch.bfloat16)
        self.pre_feedforward_layernorm = torch.nn.Identity()
        self.post_feedforward_layernorm = torch.nn.Identity()

    def forward(self, x):
        return x + self.post_feedforward_layernorm(self.mlp.down_proj(self.pre_feedforward_layernorm(x)))


def test_hooks_change_only_final_position_and_cleanup_after_exception():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleDict({str(i): ToyLayer(i) for i in (15, 22, 26, 35)})
    runtime = SimpleNamespace(model=model, modules={i: model.layers[str(i)].mlp.down_proj for i in (15, 22, 26, 35)},
        module_names={i: f"layers.{i}.mlp.down_proj" for i in (15, 22, 26, 35)})
    x = torch.ones(1, 3, 4, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="deliberate"):
        with AlignedProbe(runtime, [{"layer": 15, "neuron": 1, "kind": "zero"}]) as probe:
            for layer in model.layers.values():
                x = layer(x)
            assert probe.event_order == [15, 22, 26, 35]
            assert probe.observed[15]["after_act"][1] == 0
            assert probe.observed[15]["act"][1] == 1
            raise RuntimeError("deliberate")
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
