"""Distribution and inference checks against independent analytic examples."""

import math

import numpy as np
import pytest
import torch

from neuron_lab.evaluation import (
    distribution_metrics,
    generated_quality,
    permutation_auc_p,
    prompt_bootstrap,
    roc_auc,
)


def test_distribution_metrics_known_binary_distributions():
    p = torch.tensor([0.25, 0.75], dtype=torch.float64)
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    observed = distribution_metrics(p.log(), q.log())
    expected_kl = 0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5)
    m = [0.375, 0.625]
    expected_js = sum(0.5 * value * math.log(value / m[i]) for values in [p.tolist(), q.tolist()] for i, value in enumerate(values))
    assert observed["kl_base_to_changed"] == pytest.approx(expected_kl, abs=1e-15)
    assert observed["js_divergence"] == pytest.approx(expected_js, abs=1e-15)
    assert observed["total_variation"] == pytest.approx(0.25, abs=1e-15)
    assert observed["top1_changed"] is True
    assert observed["max_abs_logit_delta"] == pytest.approx(math.log(2), abs=1e-15)


def test_identical_logits_give_exact_zero():
    logits = torch.tensor([3.0, -2.0, 1.0], dtype=torch.float64)
    observed = distribution_metrics(logits, logits.clone())
    for key in ["kl_base_to_changed", "js_divergence", "total_variation", "max_abs_logit_delta"]:
        assert observed[key] == 0
    assert observed["top1_changed"] is False


def test_tiny_perturbation_kl_remains_positive_without_cancellation():
    delta = 1e-9
    baseline = torch.tensor([0.0, 0.0], dtype=torch.float64)
    changed = torch.tensor([delta, -delta], dtype=torch.float64)
    observed = distribution_metrics(baseline, changed)
    # KL(uniform || softmax([d,-d])) = log(cosh(d)) ~ d²/2.
    assert observed["kl_base_to_changed"] > 0
    assert observed["kl_base_to_changed"] == pytest.approx(delta**2 / 2, rel=1e-6)
    assert observed["js_divergence"] == pytest.approx(delta**2 / 8, rel=1e-6)
    assert observed["total_variation"] == pytest.approx(delta / 2, rel=1e-6)


def test_extreme_likelihood_ratios_remain_finite():
    observed = distribution_metrics(torch.tensor([0.0, -10000.0]), torch.tensor([-10000.0, 0.0]))
    assert observed["kl_base_to_changed"] == pytest.approx(10000)
    assert observed["js_divergence"] == pytest.approx(math.log(2))
    assert observed["total_variation"] == pytest.approx(1)
    assert observed["top1_changed"] is True


def test_distribution_leading_rows_are_averaged_and_argmax_changes_aggregated():
    baseline = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    changed = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    one = distribution_metrics(baseline[1], changed[1])
    both = distribution_metrics(baseline, changed)
    assert both["kl_base_to_changed"] == pytest.approx(one["kl_base_to_changed"] / 2)
    assert both["js_divergence"] == pytest.approx(one["js_divergence"] / 2)
    assert both["total_variation"] == pytest.approx(one["total_variation"] / 2)
    assert both["top1_changed"] is True


@pytest.mark.parametrize("baseline,changed", [
    (torch.tensor([]), torch.tensor([])),
    (torch.tensor(1.0), torch.tensor(1.0)),
    (torch.tensor([float("nan")]), torch.tensor([0.0])),
    (torch.tensor([0.0]), torch.tensor([float("inf")])),
    (torch.ones(2), torch.ones(3)),
    (torch.ones((0, 2)), torch.ones((0, 2))),
    (torch.tensor([1, 2]), torch.tensor([2, 1])),
])
def test_invalid_distributions_rejected(baseline, changed):
    with pytest.raises(ValueError):
        distribution_metrics(baseline, changed)


def test_prompt_bootstrap_constant_and_single_prompt():
    assert prompt_bootstrap([4.0], resamples=25) == {"mean": 4.0, "ci_low": 4.0, "ci_high": 4.0, "n": 1}
    constant = prompt_bootstrap([7.0] * 9, resamples=50)
    assert constant["mean"] == pytest.approx(7)
    assert constant["ci_low"] == pytest.approx(7)
    assert constant["ci_high"] == pytest.approx(7)
    assert constant["n"] == 9


def test_bootstrap_reproducible_prompt_unit_and_local_rng():
    before = np.random.get_state()
    first = prompt_bootstrap([1.0, 2.0, 5.0, 8.0], seed=123, resamples=800)
    second = prompt_bootstrap([1.0, 2.0, 5.0, 8.0], seed=123, resamples=800)
    after = np.random.get_state()
    assert first == second
    assert first["mean"] == 4
    assert first["n"] == 4
    assert first["ci_low"] <= 4 <= first["ci_high"]
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("positive,negative,expected", [
    ([3, 4], [1, 2], 1),
    ([1, 2], [3, 4], 0),
    ([5, 5], [5, 5], 0.5),
    ([1, 2], [1, 2], 0.5),
    ([1, 2], [0, 1], 0.875),
    ([1], [1, 2], 0.25),
])
def test_auc_known_values_and_ties(positive, negative, expected):
    assert roc_auc(positive, negative) == expected


def test_auc_agrees_with_direct_pairwise_probability():
    positive = [0.1, 0.3, 0.3, 0.8, 1]
    negative = [0.3, 0.4, 0.5, 0.9]
    pairs = [1 if p > n else 0.5 if p == n else 0 for p in positive for n in negative]
    assert roc_auc(positive, negative) == sum(pairs) / len(pairs)


def test_permutation_all_equal_is_one_and_repeatable():
    assert permutation_auc_p([1, 1], [1, 1], seed=3, resamples=25) == 1
    first = permutation_auc_p([4, 5, 6], [1, 2, 3], seed=8, resamples=2000)
    second = permutation_auc_p([4, 5, 6], [1, 2, 3], seed=8, resamples=2000)
    assert first == second
    # Complete separation has exact two-sided p = 2 / C(6,3) = 0.1.
    assert first == pytest.approx(0.1, abs=0.025)


def test_permutation_detects_both_directions():
    high = list(range(10, 20))
    low = list(range(10))
    assert permutation_auc_p(high, low, seed=0, resamples=500) <= 0.01
    assert permutation_auc_p(low, high, seed=0, resamples=500) <= 0.01


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [[1, 2]], None])
def test_invalid_value_lists_rejected(values):
    with pytest.raises(ValueError):
        prompt_bootstrap(values)
    with pytest.raises(ValueError):
        roc_auc(values, [1])
    with pytest.raises(ValueError):
        roc_auc([1], values)
    with pytest.raises(ValueError):
        permutation_auc_p(values, [1])


@pytest.mark.parametrize("kwargs", [{"seed": -1}, {"seed": True}, {"resamples": 0}, {"resamples": 1.5}, {"resamples": True}])
def test_invalid_resampling_arguments_rejected(kwargs):
    with pytest.raises(ValueError):
        prompt_bootstrap([1, 2], **kwargs)
    with pytest.raises(ValueError):
        permutation_auc_p([1, 2], [3, 4], **kwargs)


def test_generated_quality_empty_and_nonrepeated():
    assert generated_quality([], "") == {"tokens": 0, "empty": True, "repetition_fraction": 0, "collapsed": False}
    observed = generated_quality(list(range(15)), "nonempty")
    assert observed == {"tokens": 15, "empty": False, "repetition_fraction": 0, "collapsed": False}
    assert generated_quality([1, 2], "  ")["empty"] is True


def test_generated_quality_repeated_trigrams_and_minimum_length():
    observed = generated_quality([1, 2, 3] * 4, "repeat")
    assert observed["tokens"] == 12
    assert observed["repetition_fraction"] == 0.7
    assert observed["collapsed"] is True
    assert generated_quality([1] * 11, "short")["collapsed"] is False
    assert generated_quality(torch.tensor([1] * 12), "long")["collapsed"] is True


@pytest.mark.parametrize("ids,text", [(None, "a"), ([1.0], "a"), ([-1], "a"), ([True], "a"), ([[1, 2]], "a"), ([1], None), (torch.ones((1, 2), dtype=torch.int64), "a")])
def test_generated_quality_invalid_inputs_rejected(ids, text):
    with pytest.raises(ValueError):
        generated_quality(ids, text)
