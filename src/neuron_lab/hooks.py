"""Validated MLP-intermediate hooks for text-only neuron experiments.

Attach ``ActivationHook`` to an MLP's ``down_proj``. Its positional input is
``act(gate_proj(hidden)) * up_proj(hidden)`` for Gemma 4 Unified. Intervention
coordinates therefore refer to the post-gating intermediate dimension.

``position='last'`` means the final token of *each forward call*: the last
prefill token followed by the sole/current final decode token when KV caching
is enabled. This is deliberately different from an absolute sequence index.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import RLock
from typing import Callable, Literal
from weakref import WeakKeyDictionary

import torch
from torch import Tensor, nn

Mode = Literal[
    "add", "multiply", "zero", "replace", "restore", "random_neuron", "random_direction"
]
Position = Literal["all", "last"]


@dataclass(frozen=True)
class Intervention:
    """One intervention, expressed in natural activation units.

    ``add`` adds ``strength`` to one coordinate, ``multiply`` scales that
    coordinate by ``strength``, and ``zero`` sets it to zero. ``replace`` and
    ``restore`` use an explicitly supplied scalar, coordinate tensor, or full
    activation tensor. Restoration does not infer a baseline from the modified
    run; callers must supply the matched baseline activation.

    ``random_neuron`` adds the same strength to a seed-selected coordinate.
    ``random_direction`` adds a seed-selected dense vector of nominal L2 norm
    ``abs(strength)`` to every selected token. Its sign follows ``strength``;
    the vector is fixed across calls. Casting to the activation dtype can
    slightly change the actual norm, which experiments should measure.
    """

    mode: Mode
    neuron: int | None = None
    strength: float = 1.0
    position: Position = "all"
    replacement: Tensor | float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        supported = {
            "add", "multiply", "zero", "replace", "restore", "random_neuron", "random_direction"
        }
        if self.mode not in supported:
            raise ValueError(f"Unsupported intervention mode: {self.mode!r}")
        if self.position not in {"all", "last"}:
            raise ValueError("position must be 'all' or 'last'")
        if not isinstance(self.strength, (int, float)) or not math.isfinite(self.strength):
            raise ValueError("strength must be a finite real number")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if self.mode.startswith("random_"):
            if self.neuron is not None:
                raise ValueError("Random controls select their own coordinates; neuron must be None")
        elif not isinstance(self.neuron, int) or isinstance(self.neuron, bool) or self.neuron < 0:
            raise ValueError("neuron must be a nonnegative integer")
        if self.mode in {"replace", "restore"}:
            if self.replacement is None:
                raise ValueError("replace/restore requires an explicit matched replacement")
        elif self.replacement is not None:
            raise ValueError("replacement is only valid for replace/restore")


_active_modules: WeakKeyDictionary = WeakKeyDictionary()
_active_lock = RLock()


class ActivationHook:
    """A context-managed down-projection input hook.

    ``capture`` receives a detached *clone* of the unmodified input, so a
    callback cannot change the model output by writing into its argument.
    A read-only hook returns ``None`` and preserves the original tensor object.
    Changes are out-of-place and preserve input dtype/device.

    Only one lab hook may be active on a module at a time. This makes hook
    ordering explicit and catches accidental nested/double registration.
    Existing hooks owned by other code are not removed or changed.
    """

    def __init__(
        self,
        module: nn.Module,
        intervention: Intervention | None = None,
        capture: Callable[[Tensor], None] | None = None,
    ) -> None:
        self.module = module
        self.intervention = intervention
        self.capture = capture
        self._handle = None
        self._directions: dict[tuple, Tensor] = {}
        self.resolved_neuron: int | None = None
        self.calls = 0

    def __enter__(self) -> "ActivationHook":
        with _active_lock:
            if self._handle is not None or self.module in _active_modules:
                raise RuntimeError("An ActivationHook is already active on this module")
            self._handle = self.module.register_forward_pre_hook(self._pre_hook)
            _active_modules[self.module] = self
        return self

    def remove(self) -> None:
        """Remove this hook; safe to call repeatedly or after an exception."""
        with _active_lock:
            if self._handle is not None:
                self._handle.remove()
                self._handle = None
            if _active_modules.get(self.module) is self:
                del _active_modules[self.module]

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.remove()
        return False

    @staticmethod
    def _replacement_value(replacement: Tensor | float, x: Tensor, index: int, position: Position) -> Tensor:
        value = torch.as_tensor(replacement, device=x.device, dtype=x.dtype).detach()
        selected = x if position == "all" else x[:, -1:, :]
        target = selected[..., index]
        if value.ndim == 0:
            if not torch.isfinite(value):
                raise ValueError("Replacement must contain only finite values")
            return value
        if value.shape == x.shape:
            value = value[..., index]
            if position == "last":
                value = value[:, -1:]
        elif value.shape == selected.shape:
            value = value[..., index]
        elif value.shape == x.shape[:2]:
            if position == "last":
                value = value[:, -1:]
        if value.shape != target.shape:
            raise ValueError(
                f"Replacement shape {tuple(value.shape)} must match selected coordinate "
                f"{tuple(target.shape)}, full activation {tuple(x.shape)}, or be scalar"
            )
        if not torch.isfinite(value).all():
            raise ValueError("Replacement must contain only finite values")
        return value

    def _direction(self, x: Tensor, width: int, spec: Intervention) -> Tensor:
        key = (x.device, x.dtype, width)
        if key not in self._directions:
            generator = torch.Generator(device="cpu").manual_seed(spec.seed)
            direction = torch.randn(width, dtype=torch.float64, generator=generator)
            norm = torch.linalg.vector_norm(direction)
            if norm == 0:
                raise RuntimeError("A zero random direction was sampled")
            direction = direction / norm * float(spec.strength)
            self._directions[key] = direction.to(device=x.device, dtype=x.dtype)
        return self._directions[key]

    def _pre_hook(self, module: nn.Module, args: tuple) -> tuple | None:
        if not args or not isinstance(args[0], Tensor):
            raise ValueError("down_proj must receive its activation tensor as the first positional input")
        x = args[0]
        if x.ndim != 3 or any(size == 0 for size in x.shape):
            raise ValueError(f"Expected nonempty activation [batch, sequence, intermediate], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise ValueError("MLP activation must have a floating-point dtype")
        self.calls += 1
        if self.capture is not None:
            self.capture(x.detach().clone())
        spec = self.intervention
        if spec is None:
            return None
        width = x.shape[-1]
        if spec.mode == "random_neuron":
            generator = torch.Generator(device="cpu").manual_seed(spec.seed)
            index = int(torch.randint(width, (), generator=generator).item())
        else:
            index = spec.neuron
        if index is not None and index >= width:
            raise IndexError(f"Neuron {index} out of range for intermediate dimension {width}")
        self.resolved_neuron = index
        if spec.mode in {"add", "random_neuron", "random_direction"} and spec.strength == 0:
            return None
        if spec.mode == "multiply" and spec.strength == 1:
            return None
        changed = x.clone()
        selected = changed if spec.position == "all" else changed[:, -1:, :]
        if spec.mode == "random_direction":
            selected.add_(self._direction(x, width, spec))
        elif spec.mode in {"add", "random_neuron"}:
            selected[..., index] += spec.strength
        elif spec.mode == "multiply":
            selected[..., index] *= spec.strength
        elif spec.mode == "zero":
            selected[..., index] = 0
        else:
            selected[..., index] = self._replacement_value(spec.replacement, x, index, spec.position)
        return (changed, *args[1:])
