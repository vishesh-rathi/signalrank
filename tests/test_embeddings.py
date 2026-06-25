"""Tests for ranker.embeddings — pure-numpy cosine scoring (no torch at import)."""

import sys

import numpy as np

from ranker.embeddings import cosine_topk, semantic_fit_scores


def test_identical_vectors_score_one():
    vector = np.array([1.0, 0.0, 0.0])
    probes = np.array([[1.0, 0.0, 0.0]])
    assert abs(cosine_topk(vector, probes, k=1) - 1.0) < 1e-6


def test_orthogonal_and_opposite_clamp_to_zero():
    vector = np.array([1.0, 0.0])
    probes = np.array([[0.0, 1.0], [-1.0, 0.0]])  # cosines 0 and -1
    assert cosine_topk(vector, probes, k=1) == 0.0


def test_topk_averages_only_the_best_k():
    vector = np.array([1.0, 0.0])
    probes = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])  # sims 1, 0, 1
    assert abs(cosine_topk(vector, probes, k=2) - 1.0) < 1e-6


def test_importing_module_does_not_pull_in_torch():
    # The ranking path must stay numpy-only; torch loads lazily inside load_model.
    assert "torch" not in sys.modules


def test_semantic_fit_scores_are_pool_percentiles():
    # Three candidates with increasing alignment to the single probe must map to
    # the percentile-rank endpoints 0.0, 0.5, 1.0 — a uniform recall axis, not the
    # raw cosine's lopsided ~[0.6, 0.69] clump.
    probes = np.array([[1.0, 0.0]], dtype=np.float16)
    embeddings = np.array(
        [[0.2, 1.0], [0.7, 1.0], [1.0, 0.0]],
        dtype=np.float16,  # worst, middle, best
    )
    scores = semantic_fit_scores(embeddings, probes)
    assert scores.tolist() == [0.0, 0.5, 1.0]


def test_semantic_fit_scores_degenerate_pool_does_not_divide_by_zero():
    probes = np.array([[1.0, 0.0]], dtype=np.float16)
    assert semantic_fit_scores(np.array([[1.0, 0.0]], dtype=np.float16), probes).tolist() == [0.0]
    assert semantic_fit_scores(np.empty((0, 2), dtype=np.float16), probes).tolist() == []
