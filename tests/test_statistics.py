import pytest
import torch

from neuron_lab.statistics import LayerStatistics, rank_candidates


def test_streaming_matches_full_population_statistics():
    x = torch.tensor([[[1., -2., 0.], [3., 2., 4.]]])
    y = torch.tensor([[[7., 0., -4.], [5., 4., 2.], [9., -6., 8.]]])
    stats = LayerStatistics(2, reservoir_size=64, moment_dtype=torch.float64)
    stats.update(x, "a", "first")
    stats.update(y, "b", "second")
    out = stats.finalize()
    all_x = torch.cat([x.reshape(-1, 3), y.reshape(-1, 3)]).double()
    for key, expected in (("mean", all_x.mean(0)), ("std", all_x.std(0, correction=0)),
                          ("min", all_x.min(0).values), ("max", all_x.max(0).values),
                          ("frequency", (all_x.abs() > 1e-6).double().mean(0))):
        torch.testing.assert_close(torch.tensor(out["columns"][key], dtype=torch.float64), expected)
    for q, key in ((.05, "q05"), (.5, "q50"), (.95, "q95")):
        torch.testing.assert_close(torch.tensor(out["columns"][key]), torch.quantile(all_x.float(), q, dim=0))
    assert out["count"] == 5 and out["prompt_count"] == 2


def test_prompt_balancing_does_not_weight_long_inputs_more():
    stats = LayerStatistics(0)
    stats.update(torch.tensor([[[0.]]]), "short", "a")
    stats.update(torch.full((1, 9, 1), 10.), "long", "a")
    stats.update(torch.tensor([[[2.]]]), "control", "b")
    out = stats.finalize()
    assert out["category_means"]["a"] == [5.]
    assert out["columns"]["prompt_mean"] == [4.]
    assert out["columns"]["mean"][0] == pytest.approx(92 / 11)
    assert out["columns"]["selectivity_gap"] == [3.]


def test_mask_preserves_original_positions_and_token_ids():
    stats = LayerStatistics(0)
    x = torch.tensor([[[100., 100.], [2., -5.], [9., -8.], [200., 200.]]])
    stats.update(x, "masked", "a", token_ids=torch.tensor([[11, 22, 33, 44]]),
                 mask=torch.tensor([[False, True, True, False]]))
    out = stats.finalize()
    assert out["count"] == 2
    assert out["top_contexts"]["position"] == [[2, 1]]
    assert out["top_contexts"]["token"] == [[33, 22]]
    assert out["low_contexts"]["position"] == [[1, 2]]
    assert out["columns"]["max"] == [9., -5.]


def test_reservoir_is_bounded_deterministic_and_update_order_independent():
    x = torch.arange(90.).reshape(1, 30, 3)
    y = x * 2
    a, b = LayerStatistics(0, reservoir_size=7), LayerStatistics(0, reservoir_size=7)
    a.update(x, "x", "cat")
    a.update(y, "y", "cat")
    b.update(y, "y", "cat")
    b.update(x, "x", "cat")
    oa, ob = a.finalize(), b.finalize()
    assert len(a._sample) == len(b._sample) == 7
    assert oa["sample_keys"] == ob["sample_keys"]
    for key in ("q05", "q50", "q95"):
        assert oa["columns"][key] == ob["columns"][key]
    assert all(lo <= q <= hi for lo, q, hi in zip(oa["columns"]["min"], oa["columns"]["q50"], oa["columns"]["max"]))


def test_top_low_contexts_rank_distinct_prompt_extrema():
    stats = LayerStatistics(0, top_k=2)
    for i, value in enumerate([2., -3., 7., 1.]):
        stats.update(torch.tensor([[[value]]]), f"p{i}", "a")
    out = stats.finalize()
    assert out["top_contexts"]["values"] == [[7.], [2.]]
    assert out["low_contexts"]["values"] == [[-3.], [1.]]
    assert out["top_contexts"]["prompt"] == [[2], [0]]


def test_selectivity_requires_variation_and_multiple_categories():
    stats = LayerStatistics(3)
    stats.update(torch.tensor([[[4., 2.]]]), "p1", "code")
    stats.update(torch.tensor([[[0., 2.]]]), "p2", "math")
    ranked = rank_candidates([stats.finalize()])
    assert len(ranked) == 1
    assert ranked[0]["neuron"] == 0 and ranked[0]["category"] == "code"
    assert ranked[0]["status"] == "candidate"
    one_cat = LayerStatistics(0)
    one_cat.update(torch.ones((1, 1, 1)), "p", "a")
    assert rank_candidates([one_cat.finalize()]) == []


@pytest.mark.parametrize("bad", [torch.zeros((2, 3)), torch.full((1, 2, 3), float("nan"))])
def test_invalid_activations_fail_closed(bad):
    with pytest.raises(ValueError):
        LayerStatistics(0).update(bad, "x", "a")


def test_duplicate_prompt_and_empty_mask_are_rejected():
    stats = LayerStatistics(0)
    stats.update(torch.ones((1, 2, 3)), "x", "a")
    with pytest.raises(ValueError, match="only once"):
        stats.update(torch.ones((1, 2, 3)), "x", "a")
    with pytest.raises(ValueError, match="no tokens"):
        stats.update(torch.ones((1, 2, 3)), "y", "a", mask=torch.zeros((1, 2), dtype=torch.bool))
