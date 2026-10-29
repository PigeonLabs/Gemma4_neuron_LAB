"""Descriptive, reproducible evaluation primitives for neuron interventions.

Distribution changes are not semantic behavior or quality measurements.
Generation repetition and empty-output flags are only failure-screening proxies.
Bootstrap observations must each represent an independent prompt/source group;
passing token observations would give an inappropriate sampling unit.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


def _logit_tensor(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.ndim < 1 or value.shape[-1] == 0 or value.numel() == 0:
        raise ValueError(f"{name} must contain at least one nonempty vocabulary distribution")
    if not value.is_floating_point():
        raise ValueError(f"{name} must have a real floating-point dtype")
    converted = value.detach().to(dtype=torch.float64)
    if not torch.isfinite(converted).all():
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _stable_kl(log_p: Tensor, log_q: Tensor) -> Tensor:
    """KL(p||q), avoiding first-order cancellation for tiny perturbations.

    Normalization implies sum(p * (exp(d)-1)) = 0 for d=log(q/p).
    Hence KL = sum(p * (exp(d)-1-d)), a sum of nonnegative terms.
    A Taylor expansion resolves small d; q-p-p*d avoids exp(d) overflow
    for large likelihood ratios, including underflowed p probabilities.
    """
    delta = log_q - log_p
    p, q = log_p.exp(), log_q.exp()
    # exp(d)-1-d = d²/2 * (1+d/3+d²/12+d³/60+d⁴/360+...).
    # Replacing non-small entries first also avoids an unused large-delta
    # polynomial overflowing inside torch.where's eagerly computed branches.
    small = delta.abs() <= 1e-3
    d = torch.where(small, delta, torch.zeros_like(delta))
    taylor = d.square() * (0.5 + d * (1 / 6 + d * (1 / 24 + d * (1 / 120 + d / 720))))
    contributions = torch.where(small, p * taylor, q - p - p * delta)
    return contributions.clamp_min(0).sum(dim=-1)


def distribution_metrics(baseline_logits: Tensor, changed_logits: Tensor) -> dict[str, float | bool]:
    """Compare aligned logits using FP64 and natural-log divergence units.

    Shapes must match and be ``[..., vocabulary]``. Divergences and total
    variation are averaged over leading rows; ``top1_changed`` is true if any
    row changes argmax. ``max_abs_logit_delta`` is the global maximum. Passing
    one next-token distribution per prompt preserves prompt-level inference.
    """
    baseline = _logit_tensor(baseline_logits, "baseline_logits")
    changed = _logit_tensor(changed_logits, "changed_logits")
    if baseline.shape != changed.shape:
        raise ValueError("Logits must have exactly matching shapes")
    if baseline.device != changed.device:
        changed = changed.to(baseline.device)
    log_p = torch.log_softmax(baseline, dim=-1)
    log_q = torch.log_softmax(changed, dim=-1)
    if not torch.isfinite(log_p).all() or not torch.isfinite(log_q).all():
        raise ValueError("Logit magnitude exceeds finite FP64 normalization range")
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    # Preserve exact equality instead of introducing logaddexp/log(2) rounding
    # noise into a genuinely unchanged distribution.
    log_m = torch.where(log_p == log_q, log_p, log_m)
    kl = _stable_kl(log_p, log_q).mean()
    js = ((_stable_kl(log_p, log_m) + _stable_kl(log_q, log_m)) / 2).mean()
    tv = ((log_p.exp() - log_q.exp()).abs().sum(dim=-1) / 2).mean()
    max_delta = (baseline - changed).abs().max()
    if not all(torch.isfinite(value) for value in (kl, js, tv, max_delta)):
        raise ValueError("Distribution metric exceeded finite FP64 range")
    return {
        "kl_base_to_changed": float(kl.item()),
        "js_divergence": float(js.item()),
        "total_variation": float(tv.clamp(0, 1).item()),
        "top1_changed": bool((baseline.argmax(dim=-1) != changed.argmax(dim=-1)).any().item()),
        "max_abs_logit_delta": float(max_delta.item()),
    }


def _values(values: Sequence[float], name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite nonempty one-dimensional sequence") from exc
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty one-dimensional sequence")
    return array


def _rng(seed: int, resamples: int) -> np.random.Generator:
    if not isinstance(resamples, Integral) or isinstance(resamples, (bool, np.bool_)) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return np.random.default_rng(int(seed))


def prompt_bootstrap(values: list[float], seed: int = 0, resamples: int = 2000) -> dict[str, float | int]:
    """Percentile 95% confidence interval for the mean, resampling prompts."""
    array = _values(values, "values")
    generator = _rng(seed, resamples)
    n = array.size
    # Dividing before summing also avoids unnecessary intermediate overflow.
    mean = float(np.sum(array / n))
    bootstrap_means = np.empty(int(resamples), dtype=np.float64)
    for start in range(0, int(resamples), 128):
        size = min(128, int(resamples) - start)
        sampled = generator.integers(0, n, size=(size, n))
        bootstrap_means[start:start + size] = np.sum(array[sampled] / n, axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    if not np.isfinite([mean, low, high]).all():
        raise ValueError("Bootstrap result exceeded finite FP64 range")
    return {"mean": mean, "ci_low": float(low), "ci_high": float(high), "n": int(n)}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    boundaries = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1], True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        # Positions are zero-based; Mann-Whitney ranks start at one.
        ranks[order[start:end]] = (start + end + 1) / 2
    return ranks


def _auc_from_positive_ranks(ranks: np.ndarray, n_positive: int, n_negative: int) -> float:
    u = ranks.sum() - n_positive * (n_positive + 1) / 2
    return float(u / (n_positive * n_negative))


def roc_auc(positive_values: Sequence[float], negative_values: Sequence[float]) -> float:
    """P(positive > negative) + 0.5 P(tie), without fitted thresholds."""
    positive = _values(positive_values, "positive_values")
    negative = _values(negative_values, "negative_values")
    ranks = _average_ranks(np.concatenate((positive, negative)))
    return _auc_from_positive_ranks(ranks[:positive.size], positive.size, negative.size)


def permutation_auc_p(
    positive_values: Sequence[float], negative_values: Sequence[float], seed: int = 0, resamples: int = 2000
) -> float:
    """Two-sided Monte Carlo label-permutation p-value around AUC=0.5.

    The plus-one correction includes the observed labeling. This unadjusted
    p-value is not a multiple-testing correction or a replication claim.
    """
    positive = _values(positive_values, "positive_values")
    negative = _values(negative_values, "negative_values")
    generator = _rng(seed, resamples)
    n_positive, n_negative = positive.size, negative.size
    ranks = _average_ranks(np.concatenate((positive, negative)))
    observed = abs(_auc_from_positive_ranks(ranks[:n_positive], n_positive, n_negative) - 0.5)
    if observed == 0:
        return 1.0
    exceedances = 0
    for _ in range(int(resamples)):
        selected = generator.permutation(ranks.size)[:n_positive]
        statistic = abs(_auc_from_positive_ranks(ranks[selected], n_positive, n_negative) - 0.5)
        # Rank-derived AUCs are rational; tolerate only floating rounding at a tie.
        exceedances += statistic >= observed - 8 * np.finfo(np.float64).eps
    return float((exceedances + 1) / (int(resamples) + 1))


def generated_quality(ids: Sequence[int] | Tensor, text: str) -> dict[str, int | float | bool]:
    """Screen outputs with empty and repeated-token-trigram proxies.

    Repetition fraction counts duplicate trigram positions after their first
    occurrence, divided by all trigram positions. ``collapsed`` requires at
    least twelve tokens and a repetition fraction of at least 0.5. These flags
    do not measure correctness, fluency, safety, or the requested behavior.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if isinstance(ids, Tensor):
        if ids.ndim != 1:
            raise ValueError("ids must be a one-dimensional token sequence")
        ids = ids.detach().cpu().tolist()
    try:
        tokens = list(ids)
    except TypeError as exc:
        raise ValueError("ids must be a one-dimensional token sequence") from exc
    if any(not isinstance(token, Integral) or isinstance(token, (bool, np.bool_)) or token < 0 for token in tokens):
        raise ValueError("ids must contain nonnegative integer token IDs")
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    repetition = (len(trigrams) - len(set(trigrams))) / len(trigrams) if trigrams else 0.0
    return {
        "tokens": len(tokens),
        "empty": len(tokens) == 0 or not text.strip(),
        "repetition_fraction": float(repetition),
        "collapsed": len(tokens) >= 12 and repetition >= 0.5,
    }
