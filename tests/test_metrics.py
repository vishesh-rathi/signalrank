"""Tests for ranker.metrics — NDCG/MAP/P@k against hand-computed values."""

import math

from ranker.metrics import average_precision, composite, dcg, ndcg_at_k, precision_at_k


def test_dcg_known_value():
    # rels [3, 2] -> (2^3 - 1)/log2(2) + (2^2 - 1)/log2(3)
    expected = 7 / 1.0 + 3 / math.log2(3)
    assert abs(dcg([3, 2]) - expected) < 1e-9


def test_ndcg_perfect_ordering_is_one():
    rels = [3, 2, 1, 0]
    assert abs(ndcg_at_k(rels, rels, 4) - 1.0) < 1e-9


def test_ndcg_reversed_ordering_is_low():
    predicted = [0, 1, 2, 3]
    ideal_pool = [3, 2, 1, 0]
    assert ndcg_at_k(predicted, ideal_pool, 4) < 0.7


def test_precision_at_k_uses_tier_threshold():
    predicted = [5, 0, 3, 2]  # tiers >= 3 count as relevant
    assert precision_at_k(predicted, 4, threshold=3) == 0.5


def test_average_precision_binary():
    # Relevant (tier >= 1) at positions 1 and 3: AP = (1/1 + 2/3) / 2
    predicted = [1, 0, 1, 0]
    assert abs(average_precision(predicted, threshold=1) - (1.0 + 2 / 3) / 2) < 1e-9


def test_composite_bounds_and_perfect_score():
    perfect = [5, 4, 3, 2, 1]
    score = composite(perfect, perfect)
    assert 0.0 <= score <= 1.0
    assert score > 0.9  # perfect ordering nearly maxes every term


def test_ndcg_k_larger_than_lists_is_one_when_perfectly_ordered():
    rels = [3, 1]
    assert ndcg_at_k(rels, rels, 10) == 1.0


def test_ndcg_all_zero_pool_is_zero_not_div_by_zero():
    assert ndcg_at_k([0, 0, 0], [0, 0, 0], 3) == 0.0


def test_precision_at_k_zero_k_is_zero():
    assert precision_at_k([5, 4, 3], 0) == 0.0


def test_average_precision_empty_and_none_relevant():
    assert average_precision([]) == 0.0
    assert average_precision([0, 0, 0], threshold=1) == 0.0
