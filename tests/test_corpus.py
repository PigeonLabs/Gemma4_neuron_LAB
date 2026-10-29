from collections import Counter

import pytest

from neuron_lab.corpus import build_corpus, corpus_summary, split_corpus, validate_corpus


def test_pilot_corpus_has_exact_source_disjoint_stratified_partitions():
    rows = build_corpus()
    assert len(rows) == 150
    assert len({r["category"] for r in rows}) == 15
    assert Counter(r["split"] for r in rows) == {"discovery": 60, "validation": 45, "test": 45}
    assert len({r["source_id"] for r in rows}) == 150
    for split, count in (("discovery", 4), ("validation", 3), ("test", 3)):
        assert set(Counter(r["category"] for r in split_corpus(rows, split)).values()) == {count}
    assert corpus_summary(rows)["total"] == 150


def test_normalized_duplicates_are_detected_across_splits():
    rows = build_corpus()[:2]
    rows[1]["text"] = "  " + rows[0]["text"].upper() + "  "
    rows[1]["split"] = "test"
    with pytest.raises(ValueError, match="normalized duplicate"):
        validate_corpus(rows)


def test_source_group_must_not_cross_split_even_for_distinct_text():
    rows = build_corpus()[:2]
    rows[1]["source_id"] = rows[0]["source_id"]
    rows[1]["split"] = "test"
    with pytest.raises(ValueError, match="source group"):
        validate_corpus(rows)


def test_variants_can_share_source_within_same_split():
    rows = build_corpus()[:2]
    rows[1]["source_id"] = rows[0]["source_id"]
    validate_corpus(rows)


def test_bad_schema_fails_before_running_model():
    rows = build_corpus()[:1]
    rows[0]["source_id"] = ""
    with pytest.raises(ValueError):
        validate_corpus(rows)
