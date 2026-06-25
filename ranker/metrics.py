"""Ranking-quality metrics matching the hackathon's scoring formula.

The organizers score submissions as
``0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`` against hidden
relevance tiers (0-5). We reimplement the same metrics so the offline
tuning harness optimizes exactly what the judges measure.

All functions take relevance tiers in *predicted rank order* (index 0 is the
candidate our system ranked first). ``ndcg_at_k`` additionally needs the full
pool of tiers so the ideal ordering is computed over everything available,
not just what we happened to rank highly.
"""

import math


def dcg(relevances: list[int]) -> float:
    """Discounted cumulative gain with exponential gain (2^rel - 1)."""
    return sum((2**rel - 1) / math.log2(pos + 2) for pos, rel in enumerate(relevances))


def ndcg_at_k(predicted: list[int], pool: list[int], k: int) -> float:
    """NDCG@k: DCG of our top-k, normalized by the best possible top-k DCG.

    ``pool`` is every labeled relevance available — the ideal DCG must be
    allowed to pick the globally best k items, otherwise NDCG inflates.
    """
    ideal = dcg(sorted(pool, reverse=True)[:k])
    return dcg(predicted[:k]) / ideal if ideal > 0 else 0.0


def precision_at_k(predicted: list[int], k: int, threshold: int = 3) -> float:
    """Fraction of the top-k with tier >= ``threshold`` (the spec uses 3)."""
    if k <= 0:
        return 0.0
    return sum(1 for rel in predicted[:k] if rel >= threshold) / k


def average_precision(predicted: list[int], threshold: int = 1) -> float:
    """Mean of precision@i over every relevant position i (binary at ``threshold``).

    Normalized by the relevant items *found in ``predicted``*, not by the total
    relevant in the pool. These coincide — and the value equals textbook AP —
    only when ``predicted`` is the full ranked candidate set. That is exactly how
    ``eval/tune.py`` calls it (it ranks the entire labeled dev set); a truncated
    ``predicted`` would inflate the result.
    """
    hits = 0
    precision_sum = 0.0
    for pos, rel in enumerate(predicted, start=1):
        if rel >= threshold:
            hits += 1
            precision_sum += hits / pos
    return precision_sum / hits if hits else 0.0


def composite(predicted: list[int], pool: list[int]) -> float:
    """The hackathon's weighted composite over a predicted ranking."""
    return (
        0.50 * ndcg_at_k(predicted, pool, 10)
        + 0.30 * ndcg_at_k(predicted, pool, 50)
        + 0.15 * average_precision(predicted, threshold=1)
        + 0.05 * precision_at_k(predicted, 10, threshold=3)
    )
