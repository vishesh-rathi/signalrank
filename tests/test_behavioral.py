"""Tests for ranker.behavioral — the availability/engagement multiplier."""

from datetime import date

from ranker.behavioral import _notice_factor, availability_score, behavioral_multiplier

TODAY = date(2026, 6, 9)


def test_notice_factor_full_credit_within_buyout_then_convex_decay():
    # JD: "love sub-30-day notice ... can buy out up to 30 days." Anything inside
    # the buy-out window earns full credit.
    assert _notice_factor(0) == 1.0
    assert _notice_factor(30) == 1.0
    # Beyond it, convex decay to zero at the 180-day ceiling: monotonically
    # decreasing and strictly steeper than the old linear (180 - d) / 180 curve
    # for long notices (the point of the fix — "the bar gets higher").
    assert _notice_factor(60) > _notice_factor(90) > _notice_factor(120) > _notice_factor(180)
    assert _notice_factor(180) == 0.0
    assert _notice_factor(90) < (180 - 90) / 180  # 0.36 < 0.50
    assert _notice_factor(120) < (180 - 120) / 180  # 0.16 < 0.33


def test_availability_rewards_short_notice_over_long():
    short = availability_score({"open_to_work_flag": True, "notice_period_days": 15})
    long = availability_score({"open_to_work_flag": True, "notice_period_days": 120})
    assert short > long
    # A 15-day notice with open-to-work is maximally available on these two axes.
    assert short == 1.0 * 0.5 + 0.4 * 1.0


def make_ideal_candidate() -> dict:
    """Open to work, instant responder, active last week, verified + active GitHub."""
    return {
        "skills": [],
        "redrob_signals": {
            "open_to_work_flag": True,
            "notice_period_days": 15,
            "willing_to_relocate": True,
            "recruiter_response_rate": 0.9,
            "avg_response_time_hours": 10,
            "last_active_date": "2026-06-02",
            "github_activity_score": 60,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
            "skill_assessment_scores": {},
        },
    }


def make_ghost_candidate() -> dict:
    """Closed to work, never responds, inactive 9+ months, nothing verified."""
    return {
        "skills": [],
        "redrob_signals": {
            "open_to_work_flag": False,
            "notice_period_days": 180,
            "willing_to_relocate": False,
            "recruiter_response_rate": 0.02,
            "avg_response_time_hours": 280,
            "last_active_date": "2025-09-01",
            "github_activity_score": -1,
            "verified_email": False,
            "verified_phone": False,
            "linkedin_connected": False,
            "skill_assessment_scores": {},
        },
    }


def test_ideal_candidate_multiplier_near_one():
    multiplier, _ = behavioral_multiplier(make_ideal_candidate(), TODAY)
    assert multiplier > 0.85


def test_ghost_candidate_multiplier_near_floor():
    multiplier, _ = behavioral_multiplier(make_ghost_candidate(), TODAY)
    assert 0.30 <= multiplier <= 0.45


def test_multiplier_never_exceeds_one_or_dips_below_floor():
    for candidate in (make_ideal_candidate(), make_ghost_candidate()):
        multiplier, _ = behavioral_multiplier(candidate, TODAY)
        assert 0.30 <= multiplier <= 1.0


def test_inflated_skill_claims_lower_credibility():
    honest = make_ideal_candidate()
    inflated = make_ideal_candidate()
    inflated["skills"] = [
        {"name": "RAG", "proficiency": "expert"},
        {"name": "NLP", "proficiency": "expert"},
    ]
    inflated["redrob_signals"]["skill_assessment_scores"] = {"RAG": 20, "NLP": 25}
    assert behavioral_multiplier(inflated, TODAY)[0] < behavioral_multiplier(honest, TODAY)[0]


def test_skill_duration_inflation_lowers_credibility():
    # Claiming a skill for more months than the whole career has existed is
    # impossible self-reporting (Design 4.5) and must cost credibility.
    honest = make_ideal_candidate()
    honest["career_history"] = [{"company": "A", "duration_months": 60}]
    inflated = make_ideal_candidate()
    inflated["career_history"] = [{"company": "A", "duration_months": 60}]
    inflated["skills"] = [
        {"name": "RAG", "proficiency": "expert", "duration_months": 200},
        {"name": "NLP", "proficiency": "expert", "duration_months": 180},
    ]
    assert behavioral_multiplier(inflated, TODAY)[0] < behavioral_multiplier(honest, TODAY)[0]


def test_trace_exposes_all_subscores():
    _, trace = behavioral_multiplier(make_ideal_candidate(), TODAY)
    assert set(trace) == {"availability", "responsiveness", "recency", "credibility", "mult"}


def test_multiplier_with_no_signals_block_stays_in_range():
    multiplier, trace = behavioral_multiplier({}, TODAY)
    assert 0.30 <= multiplier <= 1.0
    assert set(trace) == {"availability", "responsiveness", "recency", "credibility", "mult"}


def test_future_last_active_does_not_overflow_multiplier():
    candidate = make_ideal_candidate()
    candidate["redrob_signals"]["last_active_date"] = "2027-01-01"  # after the snapshot date
    multiplier, _ = behavioral_multiplier(candidate, TODAY)
    assert multiplier <= 1.0
