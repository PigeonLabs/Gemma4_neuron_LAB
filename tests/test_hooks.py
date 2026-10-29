"""Independent coordinate, parity, RNG, and cleanup checks on small CPU modules."""

import pytest
import torch
from torch import nn

from neuron_lab.hooks import ActivationHook, Intervention


def identity(width=5, dtype=torch.float64):
    module = nn.Linear(width, width, bias=False, dtype=dtype)
    with torch.no_grad():
        module.weight.copy_(torch.eye(width, dtype=dtype))
    return module


def activation(dtype=torch.float64):
    return torch.arange(1, 31, dtype=dtype).reshape(2, 3, 5) / 7


def test_readonly_capture_is_exact_and_cannot_mutate_model_input():
    module = identity()
    x = activation()
    original = x.clone()
    baseline = module(x)
    captures = []

    def capture(value):
        captures.append(value.clone())
        assert not value.requires_grad
        value.fill_(-999)

    with ActivationHook(module, capture=capture) as hook:
        observed = module(x)
    assert hook.calls == 1
    assert torch.equal(observed, baseline)
    assert torch.equal(x, original)
    assert torch.equal(captures[0], original)
    assert len(module._forward_pre_hooks) == 0
    assert torch.equal(module(x), baseline)


def test_hook_targets_post_gating_mlp_coordinate():
    class GatedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(4, 7, bias=False, dtype=torch.float64)
            self.up_proj = nn.Linear(4, 7, bias=False, dtype=torch.float64)
            self.down_proj = nn.Linear(7, 4, bias=False, dtype=torch.float64)

        def forward(self, value):
            intermediate = torch.nn.functional.gelu(self.gate_proj(value), approximate="tanh") * self.up_proj(value)
            return self.down_proj(intermediate)

    module = GatedMLP()
    x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 13
    natural = torch.nn.functional.gelu(module.gate_proj(x), approximate="tanh") * module.up_proj(x)
    expected_intermediate = natural.clone()
    expected_intermediate[..., 4] += 0.75
    expected_output = module.down_proj(expected_intermediate)
    baseline = module(x)
    captured = []
    with ActivationHook(module.down_proj, Intervention("add", neuron=4, strength=0.75), capture=captured.append):
        actual = module(x)
    assert torch.equal(captured[0], natural)
    assert torch.equal(actual, expected_output)
    assert torch.equal(module(x), baseline)


@pytest.mark.parametrize("mode,strength", [("add", -2.5), ("multiply", -0.25), ("zero", 1.0)])
@pytest.mark.parametrize("position", ["all", "last"])
def test_formula_changes_only_selected_coordinate(mode, strength, position):
    x = activation()
    original = x.clone()
    expected = x.clone()
    target = expected[..., 2] if position == "all" else expected[:, -1:, 2]
    if mode == "add":
        target.add_(strength)
    elif mode == "multiply":
        target.mul_(strength)
    else:
        target.zero_()
    with ActivationHook(identity(), Intervention(mode, neuron=2, strength=strength, position=position)) as hook:
        actual = hook.module(x)
    assert torch.equal(actual, expected)
    assert torch.equal(x, original)
    assert actual.dtype == x.dtype
    assert actual.device == x.device


@pytest.mark.parametrize("mode,strength", [("add", 0), ("multiply", 1), ("random_neuron", 0), ("random_direction", 0)])
def test_noop_preserves_input_tensor_identity(mode, strength):
    class Echo(nn.Module):
        def forward(self, value):
            return value
    module = Echo()
    neuron = None if mode.startswith("random_") else 1
    x = activation()
    with ActivationHook(module, Intervention(mode, neuron=neuron, strength=strength)):
        assert module(x) is x


@pytest.mark.parametrize("position", ["all", "last"])
def test_restore_explicit_baseline_after_ablation(position):
    x = activation()
    baseline = x.clone()
    damaged = x.clone()
    if position == "all":
        damaged[..., 3] = 0
    else:
        damaged[:, -1:, 3] = 0
    with ActivationHook(identity(), Intervention("restore", neuron=3, replacement=baseline, position=position)) as hook:
        restored = hook.module(damaged)
    assert torch.equal(restored, baseline)
    assert torch.count_nonzero(damaged[..., 3]) < torch.count_nonzero(baseline[..., 3])


@pytest.mark.parametrize("replacement", [7.0, torch.tensor(7.0), torch.full((2, 3), 7.0)])
def test_replace_scalar_and_coordinate_tensor(replacement):
    x = activation()
    expected = x.clone()
    expected[..., 1] = 7
    with ActivationHook(identity(), Intervention("replace", neuron=1, replacement=replacement)) as hook:
        assert torch.equal(hook.module(x), expected)


def test_capture_records_natural_input_before_intervention():
    x = activation()
    captured = []
    with ActivationHook(identity(), Intervention("zero", neuron=1), capture=captured.append) as hook:
        changed = hook.module(x)
    assert torch.equal(captured[0], x)
    assert torch.count_nonzero(changed[..., 1]) == 0


@pytest.mark.parametrize("mode", ["random_neuron", "random_direction"])
def test_random_controls_reproducible_and_preserve_global_rng(mode):
    x = torch.zeros((2, 3, 127), dtype=torch.float64)
    module = identity(width=127)
    torch.manual_seed(918)
    before = torch.random.get_rng_state().clone()
    with ActivationHook(module, Intervention(mode, strength=-2.5, seed=61)) as hook:
        first = module(x)
        second = module(x)
        resolved = hook.resolved_neuron
    after = torch.random.get_rng_state()
    with ActivationHook(module, Intervention(mode, strength=-2.5, seed=61)):
        repeated = module(x)
    assert torch.equal(first, second)
    assert torch.equal(first, repeated)
    assert torch.equal(before, after)
    assert torch.allclose(torch.linalg.vector_norm(first, dim=-1), torch.full((2, 3), 2.5, dtype=x.dtype), atol=1e-12, rtol=0)
    if mode == "random_neuron":
        assert resolved is not None
        assert torch.count_nonzero(first[0, 0]) == 1
        assert first[0, 0, resolved] == -2.5
    else:
        assert torch.count_nonzero(first[0, 0]) == 127


def test_last_position_random_direction_tracks_each_forward_call():
    module = identity()
    spec = Intervention("random_direction", strength=2, seed=3, position="last")
    with ActivationHook(module, spec):
        prefill = module(torch.zeros((2, 4, 5), dtype=torch.float64))
        decode = module(torch.zeros((2, 1, 5), dtype=torch.float64))
    assert torch.count_nonzero(prefill[:, :-1]) == 0
    assert torch.equal(prefill[:, -1:], decode)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dtype_preserved(dtype):
    x = activation(dtype=dtype)
    with ActivationHook(identity(dtype=dtype), Intervention("add", neuron=4, strength=0.5)) as hook:
        actual = hook.module(x)
    assert actual.dtype == dtype
    expected = x.clone()
    expected[..., 4] += 0.5
    assert torch.equal(actual, expected)


def test_nested_hooks_fail_without_losing_outer_hook():
    module = identity()
    outer = ActivationHook(module, Intervention("zero", neuron=2))
    with outer:
        with pytest.raises(RuntimeError, match="already active"):
            with outer:
                pass
        with pytest.raises(RuntimeError, match="already active"):
            with ActivationHook(module):
                pass
        assert len(module._forward_pre_hooks) == 1
        assert torch.count_nonzero(module(activation())[..., 2]) == 0
    assert len(module._forward_pre_hooks) == 0
    outer.remove()


def test_callback_exception_removes_only_lab_hook():
    module = identity()
    external_calls = []
    external = module.register_forward_pre_hook(lambda _module, _args: external_calls.append(True))

    def failing_capture(_value):
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        with ActivationHook(module, capture=failing_capture):
            module(activation())
    assert len(module._forward_pre_hooks) == 1
    assert external_calls == [True]
    assert torch.equal(module(activation()), activation())
    external.remove()


def test_invalid_coordinate_and_shapes_fail_and_cleanup():
    module = identity()
    with pytest.raises(IndexError, match="out of range"):
        with ActivationHook(module, Intervention("zero", neuron=5)):
            module(activation())
    with pytest.raises(ValueError, match="Replacement shape"):
        with ActivationHook(module, Intervention("replace", neuron=1, replacement=torch.ones((2, 2)))):
            module(activation())
    with pytest.raises(ValueError, match="nonempty activation"):
        with ActivationHook(module):
            module(torch.ones((2, 5), dtype=torch.float64))
    assert len(module._forward_pre_hooks) == 0


@pytest.mark.parametrize("replacement", [float("nan"), float("inf"), torch.full((2, 3), float("nan"))])
def test_nonfinite_replacement_rejected(replacement):
    module = identity()
    with pytest.raises(ValueError, match="finite"):
        with ActivationHook(module, Intervention("replace", neuron=1, replacement=replacement)):
            module(activation())
    assert len(module._forward_pre_hooks) == 0


@pytest.mark.parametrize("kwargs", [
    {"mode": "invalid", "neuron": 0},
    {"mode": "zero", "neuron": -1},
    {"mode": "zero", "neuron": True},
    {"mode": "zero", "neuron": 1.5},
    {"mode": "add", "neuron": 0, "strength": float("nan")},
    {"mode": "multiply", "neuron": 0, "strength": float("inf")},
    {"mode": "replace", "neuron": 0},
    {"mode": "restore", "neuron": 0},
    {"mode": "random_neuron", "neuron": 1},
    {"mode": "zero", "neuron": 0, "position": "first"},
])
def test_invalid_specs_rejected(kwargs):
    with pytest.raises(ValueError):
        Intervention(**kwargs)
