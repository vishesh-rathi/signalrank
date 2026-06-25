"""Tests for ranker.honeypot — deterministic impossible-profile detection."""

from datetime import date

from ranker.honeypot import honeypot_flags, is_honeypot

TODAY = date(2026, 6, 9)


def make_clean_candidate() -> dict:
    """A consistent profile: 6 yrs experience, two jobs whose months match their dates."""
    return {
        "profile": {"years_of_experience": 6.0},
        "career_history": [
            {"start_date": "2022-01-01", "end_date": "2024-01-01", "duration_months": 24},
            {
                "start_date": "2024-02-01",
                "end_date": None,
                "duration_months": 28,
                "is_current": True,
            },
        ],
        "skills": [{"name": "NLP", "proficiency": "advanced", "duration_months": 20}],
        "education": [{"start_year": 2014, "end_year": 2018}],
    }


def test_clean_profile_is_not_honeypot():
    assert is_honeypot(make_clean_candidate(), TODAY) is False


def test_impossible_tenure_is_flagged():
    candidate = make_clean_candidate()
    # Claims 166 months at a job whose dates span ~28 months.
    candidate["career_history"][1]["duration_months"] = 166
    assert "tenure_impossible" in honeypot_flags(candidate, TODAY)


def test_phantom_expertise_is_flagged():
    candidate = make_clean_candidate()
    candidate["skills"] = [
        {"name": name, "proficiency": "expert", "duration_months": 0}
        for name in ("RAG", "NLP", "Milvus")
    ]
    assert "phantom_expertise" in honeypot_flags(candidate, TODAY)


def test_career_exceeding_stated_experience_is_flagged():
    candidate = make_clean_candidate()
    candidate["profile"]["years_of_experience"] = 2.0  # 24mo + slack 24 < 52mo total
    assert "career_exceeds_life" in honeypot_flags(candidate, TODAY)


def test_experience_exceeding_career_span_is_flagged():
    # The spec's first honeypot example: "8 years of experience at a company
    # founded 3 years ago". Career starts 2023-01 (~41 months to the snapshot)
    # but the profile claims 15 years -> impossible vs. the observable span.
    candidate = make_clean_candidate()
    candidate["profile"]["years_of_experience"] = 15.0
    candidate["career_history"] = [
        {"start_date": "2023-01-01", "end_date": None, "duration_months": 41, "is_current": True}
    ]
    assert "experience_exceeds_career_span" in honeypot_flags(candidate, TODAY)


def test_omitted_early_roles_do_not_trip_the_span_check():
    # An honest senior who lists only recent roles: 9 years stated, earliest
    # listed role starts ~6 years ago. The 48-month slack absorbs the gap so a
    # real candidate who simply pruned old jobs is not flagged.
    candidate = make_clean_candidate()
    candidate["profile"]["years_of_experience"] = 9.0
    candidate["career_history"] = [
        {"start_date": "2020-06-01", "end_date": None, "duration_months": 72, "is_current": True}
    ]
    assert "experience_exceeds_career_span" not in honeypot_flags(candidate, TODAY)


def test_null_profile_does_not_silently_skip_yoe_checks():
    # A record with an explicit null profile must not raise (and be swallowed),
    # bypassing the years-of-experience consistency checks. No yoe -> simply no
    # yoe-based flag, but the call must stay defensive and not crash.
    candidate = make_clean_candidate()
    candidate["profile"] = None
    assert isinstance(honeypot_flags(candidate, TODAY), list)  # no TypeError


def test_education_year_inversion_is_flagged():
    candidate = make_clean_candidate()
    candidate["education"] = [{"start_year": 2020, "end_year": 2016}]
    assert "edu_inversion" in honeypot_flags(candidate, TODAY)


def test_float_duration_still_flags_impossible_tenure():
    candidate = make_clean_candidate()
    # A current role (null end_date) measured to the snapshot spans ~28 months;
    # claiming 166.0 is impossible regardless of the value being a float.
    candidate["career_history"][1] = {
        "start_date": "2024-02-01",
        "end_date": None,
        "duration_months": 166.0,
        "is_current": True,
    }
    assert "tenure_impossible" in honeypot_flags(candidate, TODAY)


def test_past_job_with_null_end_date_is_not_flagged():
    # A *non-current* role with a missing end date has an unknowable span, so the
    # tenure check must skip it rather than substitute the snapshot date — doing
    # the latter would inflate the span and falsely flag an honest candidate.
    candidate = make_clean_candidate()
    candidate["career_history"] = [
        {
            "start_date": "2018-01-01",
            "end_date": None,  # past role, no end date, not current
            "duration_months": 24,
        }
    ]
    assert "tenure_impossible" not in honeypot_flags(candidate, TODAY)


def test_empty_or_missing_sections_are_not_honeypots():
    assert is_honeypot({}, TODAY) is False
    assert is_honeypot({"career_history": [], "skills": [], "education": []}, TODAY) is False


def test_current_job_consistent_to_today_is_clean():
    # ~28 months from 2024-02-01 to the 2026-06-09 snapshot matches the claim.
    assert "tenure_impossible" not in honeypot_flags(make_clean_candidate(), TODAY)
