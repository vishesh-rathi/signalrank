"""Tests for ranker.score — honeypot zeroing, composition, ordering, tie-break."""

from datetime import date

from ranker.score import rank_candidates, score_candidate

TODAY = date(2026, 6, 9)


def make_strong_candidate(candidate_id: str = "CAND_0000002") -> dict:
    """A senior NLP/ranking builder in Pune with healthy engagement signals."""
    return {
        "candidate_id": candidate_id,
        "profile": {
            "years_of_experience": 7.0,
            "current_title": "ML Engineer",
            "current_industry": "Fintech",
            "location": "Pune, MH",
            "country": "India",
            "summary": "Built a ranking system and semantic search in production.",
        },
        "career_history": [
            {
                "company": "Swiggy",
                "title": "ML Engineer",
                "description": "Led production ranking systems end-to-end.",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 77,
            }
        ],
        "education": [{"tier": "tier_1", "start_year": 2013, "end_year": 2017}],
        "skills": [],
        "redrob_signals": {
            "open_to_work_flag": True,
            "notice_period_days": 20,
            "recruiter_response_rate": 0.85,
            "last_active_date": "2026-05-20",
            "github_activity_score": 55,
            "verified_email": True,
        },
    }


def make_honeypot_candidate() -> dict:
    candidate = make_strong_candidate("CAND_0000003")
    candidate["career_history"][0]["duration_months"] = 400  # impossible vs ~77-month span
    return candidate


def test_honeypot_scores_zero_with_flagged_trace():
    score, trace = score_candidate(make_honeypot_candidate(), None, TODAY)
    assert score == 0.0
    assert trace == {"honeypot": True, "score": 0.0}


def test_strong_candidate_scores_meaningfully_positive():
    score, trace = score_candidate(make_strong_candidate(), None, TODAY)
    assert score > 0.3
    assert trace["honeypot"] is False
    assert trace["score"] == score
    assert "fit" in trace and "mult" in trace  # both halves of the formula traced


def test_rank_orders_best_first_and_scores_non_increasing():
    ranked = rank_candidates(
        [make_honeypot_candidate(), make_strong_candidate()],
        embeddings=None,
        id_index={},
        probe_matrix=None,
        today=TODAY,
        top_n=2,
    )
    assert [row["rank"] for row in ranked] == [1, 2]
    assert ranked[0]["candidate_id"] == "CAND_0000002"
    assert ranked[0]["score"] >= ranked[1]["score"]


def test_equal_scores_break_ties_by_candidate_id_ascending():
    twin_a = make_strong_candidate("CAND_0000010")
    twin_b = make_strong_candidate("CAND_0000009")  # identical profile, lower id
    ranked = rank_candidates(
        [twin_a, twin_b],
        embeddings=None,
        id_index={},
        probe_matrix=None,
        today=TODAY,
        top_n=2,
    )
    assert ranked[0]["score"] == ranked[1]["score"]
    assert ranked[0]["candidate_id"] == "CAND_0000009"  # ascending id wins the tie


def test_emitted_scores_are_rounded_to_six_decimals():
    # The CSV carries 6 dp; ranking on the same rounded value is what keeps the
    # validator's "equal score -> candidate_id ascending" rule satisfiable.
    ranked = rank_candidates(
        [make_strong_candidate("CAND_0000007")], None, {}, None, TODAY, top_n=1
    )
    score = ranked[0]["score"]
    assert score == round(score, 6)


def test_out_of_range_embedding_index_falls_back_not_crashes():
    import numpy as np

    candidate = make_strong_candidate("CAND_0000002")
    embeddings = np.zeros((1, 4), dtype=np.float16)
    stale_index = {"CAND_0000002": 5}  # points past the (1-row) matrix
    probe_matrix = np.ones((2, 4), dtype=np.float16)
    ranked = rank_candidates([candidate], embeddings, stale_index, probe_matrix, TODAY, top_n=1)
    assert ranked[0]["candidate_id"] == "CAND_0000002"  # scored lexical-only, no IndexError
    assert ranked[0]["trace"]["semantic"] is None
