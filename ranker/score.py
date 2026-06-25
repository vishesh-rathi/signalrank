"""Score composition and top-N ranking.

The whole model in one line: ``score = fit × behavioral_multiplier``, with
honeypots forced to zero before they can reach the top-100 (ranking >10% of
them there disqualifies the submission). Sorting uses ``(-score, candidate_id)``
— the ascending-id tie-break is required verbatim by the contest validator.
"""

from datetime import date

import numpy as np

from ranker.behavioral import behavioral_multiplier
from ranker.embeddings import semantic_fit_scores
from ranker.features import weighted_fit
from ranker.honeypot import is_honeypot
from ranker.util import build_narrative


def score_candidate(
    candidate: dict,
    semantic: float | None,
    today: date,
) -> tuple[float, dict]:
    """Score one candidate; the trace records every component for explainability.

    ``semantic`` is the candidate's pool-normalized JD similarity in [0, 1]
    (from ``semantic_fit_scores``), or None for lexical-only scoring.
    """
    if is_honeypot(candidate, today):
        return 0.0, {"honeypot": True, "score": 0.0}
    text = build_narrative(candidate).lower()
    fit, feature_trace = weighted_fit(candidate, text, semantic)
    multiplier, behavioral_trace = behavioral_multiplier(candidate, today)
    score = fit * multiplier
    trace = {**feature_trace, **behavioral_trace, "honeypot": False, "score": score}
    return score, trace


def rank_candidates(
    candidates: list[dict],
    embeddings: np.ndarray | None,
    id_index: dict[str, int],
    probe_matrix: np.ndarray | None,
    today: date,
    top_n: int = 100,
) -> list[dict]:
    """Score every candidate and return the top ``top_n`` with 1-based ranks.

    A candidate missing from ``id_index`` (or a missing embeddings artifact
    altogether) degrades gracefully to lexical-only scoring rather than failing
    the run — robustness over completeness for a single record.
    """
    row_count = embeddings.shape[0] if embeddings is not None else 0
    # Pool-normalized semantic similarity is computed once over the whole matrix
    # (it is a percentile rank — inherently a pool-level quantity, not derivable
    # per candidate) and then looked up per row during scoring.
    semantic_scores = (
        semantic_fit_scores(embeddings, probe_matrix)
        if embeddings is not None and probe_matrix is not None
        else None
    )
    scored: list[tuple[float, str, dict, dict]] = []
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id") or ""
        semantic = None
        if semantic_scores is not None:
            row = id_index.get(candidate_id)
            if row is not None and 0 <= row < row_count:
                semantic = float(semantic_scores[row])
        score, trace = score_candidate(candidate, semantic, today)
        # Rank on the SAME six-decimal value the CSV will carry. Candidates whose
        # raw scores differ only beyond 6 dp are equal once written, and the
        # contest validator then demands the ascending candidate_id tie-break;
        # sorting on the raw score would violate it.
        scored.append((round(score, 6), candidate_id, candidate, trace))

    scored.sort(key=lambda row: (-row[0], row[1]))
    return [
        {
            "candidate_id": candidate_id,
            "rank": position,
            "score": score,
            "candidate": candidate,
            "trace": trace,
        }
        for position, (score, candidate_id, candidate, trace) in enumerate(scored[:top_n], start=1)
    ]
