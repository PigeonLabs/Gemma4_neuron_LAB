"""Bounded, per-layer activation summaries; no full activation cache.

Token moments describe the explicitly supplied token mask. Category means and
selectivity give *each prompt equal weight*, irrespective of token length.
Selectivity is descriptive; it is neither a causal effect nor a hypothesis test.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch


class LayerStatistics:
    """Streaming neuron statistics for one down-projection input.

    One update is one complete prompt. Batch members, when present, belong to
    that same prompt/source. ``mask`` is boolean [B,S]; token IDs are optional
    [B,S]. Top/low buffers rank the *prompt extrema*, retaining prompt ID, flat
    token position, and token ID. ``position`` is the original position within
    its sequence (not the rank after masking).

    Quantiles use at most ``reservoir_size`` token rows, chosen with independent
    deterministic hash priorities. This is a uniform priority reservoir across
    observed tokens, not an exact quantile or independent per-neuron reservoir.
    FP32 parallel Welford moments avoid cancellation from sum(x*x)-sum(x)**2.
    """

    def __init__(self, layer: int, reservoir_size: int = 64, top_k: int = 3,
                 seed: int = 0, frequency_threshold: float = 1e-6,
                 moment_dtype: torch.dtype = torch.float32) -> None:
        if reservoir_size < 1 or top_k < 1:
            raise ValueError("reservoir_size and top_k must be positive")
        if not math.isfinite(frequency_threshold) or frequency_threshold < 0:
            raise ValueError("frequency_threshold must be finite and nonnegative")
        if moment_dtype not in (torch.float32, torch.float64):
            raise ValueError("moment_dtype must be float32 or float64")
        self.layer = int(layer)
        self.reservoir_size, self.top_k, self.seed = reservoir_size, top_k, seed
        self.frequency_threshold, self.moment_dtype = frequency_threshold, moment_dtype
        self.count = 0
        self.prompt_ids: list[str] = []
        self.categories: dict[str, dict[str, Any]] = {}
        self._sample_keys: list[tuple[int, str, int]] = []
        self._sample: torch.Tensor | None = None
        self.width: int | None = None

    @torch.no_grad()
    def update(self, tensor: torch.Tensor, prompt_id: str, category: str,
               token_ids: torch.Tensor | None = None,
               mask: torch.Tensor | None = None) -> None:
        if tensor.ndim != 3 or min(tensor.shape) < 1:
            raise ValueError("activation must have nonempty shape [B,S,I]")
        if prompt_id in self.prompt_ids:
            raise ValueError("each prompt may be accumulated only once per layer")
        if not prompt_id or not category:
            raise ValueError("prompt_id and category must be nonempty")
        if self.width is not None and tensor.shape[-1] != self.width:
            raise ValueError("neuron width changed within a layer")
        if self.width is not None and tensor.device != self.mean.device:
            raise ValueError("activation device changed within a layer")
        if mask is not None and (mask.shape != tensor.shape[:2] or mask.dtype != torch.bool):
            raise ValueError("mask must be boolean [B,S]")
        if token_ids is not None and token_ids.shape != tensor.shape[:2]:
            raise ValueError("token_ids must have shape [B,S]")
        keep = torch.ones(tensor.shape[:2], dtype=torch.bool, device=tensor.device)
        if mask is not None:
            keep = mask.to(tensor.device)
        positions = torch.arange(tensor.shape[0] * tensor.shape[1], device=tensor.device)[keep.flatten()]
        x = tensor.detach().reshape(-1, tensor.shape[-1])[keep.flatten()].to(self.moment_dtype)
        if not len(x):
            raise ValueError("token mask selects no tokens")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("nonfinite activation: run must not publish valid statistics")
        token_count, width = x.shape
        chunk_mean = x.mean(dim=0)
        chunk_m2 = x.var(dim=0, correction=0) * token_count
        chunk_min, local_min = x.min(dim=0)
        chunk_max, local_max = x.max(dim=0)
        freq = (x.abs() > self.frequency_threshold).sum(dim=0)
        pindex = len(self.prompt_ids)

        if self.width is None:
            self.width = width
            self.mean = chunk_mean
            self.m2 = chunk_m2
            self.minimum, self.maximum = chunk_min, chunk_max
            self.active_count = freq
            self.prompt_mean = chunk_mean.clone()
            self.prompt_m2 = torch.zeros_like(chunk_mean)
            self._top_values = torch.empty((0, width), dtype=self.moment_dtype, device=x.device)
            self._low_values = torch.empty_like(self._top_values)
            self._top_prompt = torch.empty((0, width), dtype=torch.int64, device=x.device)
            self._low_prompt = torch.empty_like(self._top_prompt)
            self._top_position = torch.empty_like(self._top_prompt)
            self._low_position = torch.empty_like(self._top_prompt)
            self._top_token = torch.empty_like(self._top_prompt)
            self._low_token = torch.empty_like(self._top_prompt)
        else:
            total = self.count + token_count
            delta = chunk_mean - self.mean
            self.mean = self.mean + delta * (token_count / total)
            self.m2 = self.m2 + chunk_m2 + delta.square() * (self.count * token_count / total)
            self.minimum = torch.minimum(self.minimum, chunk_min)
            self.maximum = torch.maximum(self.maximum, chunk_max)
            self.active_count += freq
            delta_prompt = chunk_mean - self.prompt_mean
            self.prompt_mean = self.prompt_mean + delta_prompt / (pindex + 1)
            self.prompt_m2 += delta_prompt * (chunk_mean - self.prompt_mean)

        self.count += token_count
        self.prompt_ids.append(prompt_id)
        if category not in self.categories:
            self.categories[category] = {"count": 0, "sum": torch.zeros_like(chunk_mean)}
        self.categories[category]["count"] += 1
        self.categories[category]["sum"] += chunk_mean

        flat_ids = None if token_ids is None else token_ids.to(x.device).reshape(-1)
        for kind, values, local in (("top", chunk_max, local_max), ("low", chunk_min, local_min)):
            flat_positions = positions[local]
            original_positions = flat_positions % tensor.shape[1]
            ids = torch.full_like(flat_positions, -1) if flat_ids is None else flat_ids[flat_positions]
            existing_values = getattr(self, f"_{kind}_values")
            merged_values = torch.cat((existing_values, values.unsqueeze(0)), dim=0)
            n_keep = min(self.top_k, merged_values.shape[0])
            selected_values, order = torch.topk(merged_values, n_keep, dim=0, largest=kind == "top")
            setattr(self, f"_{kind}_values", selected_values)
            for field, row in (("prompt", torch.full_like(flat_positions, pindex)),
                               ("position", original_positions), ("token", ids)):
                merged = torch.cat((getattr(self, f"_{kind}_{field}"), row.unsqueeze(0)), dim=0)
                setattr(self, f"_{kind}_{field}", merged.gather(0, order))

        # Hash keys do not consume a process-global RNG or depend on update order.
        flat_positions_cpu = positions.cpu().tolist()
        new_keys = [
            (int.from_bytes(hashlib.blake2b(
                f"{self.seed}\0{prompt_id}\0{pos}".encode(), digest_size=8).digest(), "big"),
             prompt_id, pos)
            for pos in flat_positions_cpu
        ]
        combined_keys = self._sample_keys + new_keys
        retained = sorted(range(len(combined_keys)), key=combined_keys.__getitem__)[:self.reservoir_size]
        old_n = len(self._sample_keys)
        # Gather only selected new rows, so no complete prompt is cached.
        rows = [self._sample[i] if i < old_n else x[i - old_n] for i in retained]
        self._sample = torch.stack(rows)
        self._sample_keys = [combined_keys[i] for i in retained]

    @torch.no_grad()
    def finalize(self) -> dict[str, Any]:
        if not self.count:
            raise ValueError("cannot finalize empty statistics")
        pcount = len(self.prompt_ids)
        names = sorted(self.categories)
        cat_means = torch.stack([self.categories[c]["sum"] / self.categories[c]["count"] for c in names])
        if len(names) < 2:
            best_index = torch.zeros(self.width, device=self.mean.device, dtype=torch.long)
            gap = torch.zeros_like(self.mean)
        else:
            # Rest is category-balanced, not token-weighted or prompt-count-weighted.
            gaps = cat_means - (cat_means.sum(dim=0, keepdim=True) - cat_means) / (len(names) - 1)
            gap, best_index = gaps.max(dim=0)
        prompt_std = (self.prompt_m2.clamp_min(0) / pcount).sqrt()
        z = gap / prompt_std.clamp_min(1e-8)
        quantiles = torch.quantile(self._sample.float(), torch.tensor([.05, .5, .95], device=self.mean.device), dim=0)
        columns = {
            "neuron": list(range(self.width)),
            "mean": self.mean.cpu().tolist(),
            "std": (self.m2.clamp_min(0) / self.count).sqrt().cpu().tolist(),
            "min": self.minimum.cpu().tolist(), "max": self.maximum.cpu().tolist(),
            "frequency": (self.active_count / self.count).cpu().tolist(),
            "q05": quantiles[0].cpu().tolist(), "q50": quantiles[1].cpu().tolist(),
            "q95": quantiles[2].cpu().tolist(),
            "prompt_mean": self.prompt_mean.cpu().tolist(),
            "cross_prompt_std": prompt_std.cpu().tolist(),
            "best_category": [names[i] for i in best_index.cpu().tolist()],
            "selectivity_gap": gap.cpu().tolist(), "selectivity_z": z.cpu().tolist(),
        }
        contexts = {}
        for kind in ("top", "low"):
            contexts[f"{kind}_contexts"] = {
                field: getattr(self, f"_{kind}_{field}").cpu().tolist()
                for field in ("values", "prompt", "position", "token")
            }
        return {
            "layer": self.layer, "count": self.count, "prompt_count": pcount,
            "width": self.width, "frequency_threshold": self.frequency_threshold,
            "std_convention": "population", "moment_dtype": str(self.moment_dtype),
            "quantile_method": "uniform_hash_priority_token_reservoir",
            "quantile_sample_count": len(self._sample_keys),
            "sample_keys": [[str(key), prompt, pos] for key, prompt, pos in self._sample_keys],
            "category_prompt_counts": {c: self.categories[c]["count"] for c in names},
            "category_means": {c: cat_means[i].cpu().tolist() for i, c in enumerate(names)},
            "prompt_ids": list(self.prompt_ids), "columns": columns, **contexts,
            "limitations": [
                "Selectivity is discovery-only descriptive evidence, not a causal label.",
                "Categories may differ in length, script, tokens, syntax, and instructions.",
                "Quantiles share a bounded sample of token positions and are approximate.",
                "Top/low buffers retain prompt extrema, not all top tokens globally.",
            ],
        }


def rank_candidates(finalized_layers: list[dict[str, Any]], top_n: int = 20,
                    min_prompt_std: float = 1e-6) -> list[dict[str, Any]]:
    """Rank descriptive category separation; validation data must stay separate."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    rows = []
    for summary in finalized_layers:
        c = summary["columns"]
        if len(summary["category_means"]) < 2:
            continue
        for i, neuron in enumerate(c["neuron"]):
            if c["cross_prompt_std"][i] < min_prompt_std:
                continue
            rows.append({"layer": summary["layer"], "neuron": neuron,
                         "category": c["best_category"][i],
                         "selectivity_z": c["selectivity_z"][i],
                         "selectivity_gap": c["selectivity_gap"][i],
                         "mean": c["mean"][i], "std": c["std"][i],
                         "min": c["min"][i], "max": c["max"][i],
                         "status": "candidate"})
    return sorted(rows, key=lambda r: (-r["selectivity_z"], r["layer"], r["neuron"]))[:top_n]
