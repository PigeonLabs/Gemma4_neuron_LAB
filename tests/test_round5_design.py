"""Synthetic CPU checks only; none are Gemma-4 experimental results."""
from collections import Counter
import copy
import pytest

from neuron_lab.round5_design import (
    make_episode, validate_episode, composed_answer, direct_answer, history_text,
    permute_statuses, paired_behavior, paired_recruitment, forward_budget,
)


def test_independent_oracles_and_unseen_queries():
    assert composed_answer({"x": "y"}, {"y": "z"}, "x") == "z"
    assert direct_answer({"x": "z"}, "x") == "z"
    for seed in range(64):
        episode = make_episode(seed, f"cpu-{seed}")
        assert {r["query"] for r in episode["train"]}.isdisjoint(r["query"] for r in episode["test"])
        for phase in ("train", "test"):
            assert Counter(r["correct_code"] for r in episode[phase]) == Counter(A=2, B=2, C=2, D=2)
    bad = copy.deepcopy(episode)
    bad["direct"][episode["train"][0]["query"]] = "wrong"
    with pytest.raises(ValueError, match="disagree"):
        validate_episode(bad)


def test_corrected_history_changes_only_status_content():
    episode = make_episode(2, "cpu-history")
    first = history_text(episode, ["0", "1"] * 4)
    second = history_text(episode, ["1", "0"] * 4)
    strip = lambda s: s.replace("diagnostic-pass=0", "diagnostic-pass=2").replace("diagnostic-pass=1", "diagnostic-pass=2")
    assert strip(first) == strip(second)
    assert "Reference=" in first
    assert "previously produced" not in first


def test_shuffling_keeps_time_block_counts_and_identity_is_not_removed():
    status = ["0", "0", "1", "1", "1", "0", "0", "0"]
    shuffled = permute_statuses(status, ["early"] * 4 + ["late"] * 4, 71)
    assert Counter(shuffled[:4]) == Counter(status[:4])
    assert Counter(shuffled[4:]) == Counter(status[4:])
    assert permute_statuses(["0"] * 8, ["all"] * 8, 31) == ["0"] * 8


def test_crossover_does_not_make_mismatched_harm_a_recovery():
    scores = {(l, h): .8 for l in "AB" for h in ("A", "B", "demo", "Aperm", "Bperm")}
    scores["A", "B"] = scores["B", "A"] = .4
    result = paired_behavior(scores)
    assert result["lesion_history_crossover"] == pytest.approx(.4)
    assert result["recovery_vs_demo"] == 0
    assert result["truthful_vs_permuted"] == 0


def test_generic_demonstration_gain_is_not_lesion_specific():
    scores = {(l, h): .8 for l in "AB" for h in ("A", "B", "demo", "Aperm", "Bperm")}
    for l in "AB":
        scores[l, "demo"] = .6
    result = paired_behavior(scores)
    assert result["recovery_vs_demo"] == pytest.approx(.2)
    assert result["lesion_history_crossover"] == 0


def test_nonspecific_second_ablation_is_removed_by_random_panel():
    controls = {l: ["G1", "G2", "G3"] for l in "AB"}
    damage = {(l, h, g): (.4 if h == l else .1) for l in "AB" for h in (l, "demo")
              for g in ("RA", "RB", "G1", "G2", "G3")}
    assert paired_recruitment(damage, controls) == 0
    for l in "AB":
        damage[l, l, "R" + l] += .2
    assert paired_recruitment(damage, controls) == pytest.approx(.2)


def test_budget_is_calls_not_queries_and_is_never_model_evidence():
    result = forward_budget([{"id": "synthetic", "components": {
        "source_history": 16, "behavior": 112, "secondary_masks": 240, "norm_replay": 288}}])
    assert result["planned_actual_forward_calls"] == 656
    assert result["measurement_status"] == "NOT_EVALUATED"
    with pytest.raises(ValueError):
        forward_budget([{"id": "bad", "components": {"calls": 1.5}}])
