"""Deterministic honeypot detection via internal-consistency checks.

The candidate pool plants ~80 profiles with impossible facts (tenure longer
than the job's date span, "expert" skills used for zero months, more career
months than stated years of experience). The hidden ground truth pins them to
relevance tier 0, and a submission ranking too many of them is disqualified —
so any flagged candidate is forced to score 0 before ranking.

Every check is a cheap arithmetic comparison with a conservative threshold
(real honeypots overshoot by ~100+ months); legitimate candidates with small
data wobbles must never be flagged.
"""

from datetime import date

from ranker import config
from ranker.util import months_between, pdate


def honeypot_flags(candidate: dict, today: date) -> list[str]:
    """Return the list of impossibility flags raised by this profile."""
    flags: list[str] = []
    career = candidate.get("career_history") or []

    # Claimed tenure vs. the actual span between start and end dates.
    for entry in career:
        start = pdate(entry.get("start_date"))
        claimed = entry.get("duration_months")
        if start is None or not isinstance(claimed, int | float):
            continue
        end = pdate(entry.get("end_date"))
        if end is None:
            # A missing end date is measurable only for a *current* role (it runs
            # to the snapshot date). For a past role we cannot know the span, so
            # we skip rather than substitute `today` — that would fabricate an
            # inflated span and wrongly flag an honest candidate as a honeypot.
            if not entry.get("is_current"):
                continue
            end = today
        if abs(claimed - months_between(start, end)) > config.TENURE_DELTA_MAX:
            flags.append("tenure_impossible")
            break

    # "Expert/advanced" proficiency in several skills never actually used.
    phantom = sum(
        1
        for skill in candidate.get("skills") or []
        if skill.get("proficiency") in ("expert", "advanced") and skill.get("duration_months") == 0
    )
    if phantom >= config.PHANTOM_EXPERT_MIN:
        flags.append("phantom_expertise")

    # Total career months cannot meaningfully exceed stated years of experience.
    # ``profile`` may be null on a malformed record; the subscript inside the
    # try/except must not assume it is a dict (every other check uses the
    # ``.get(...) or {}`` idiom — match it so a null profile cannot silently skip
    # the check via an uncaught TypeError path).
    try:
        stated_months = float((candidate.get("profile") or {})["years_of_experience"]) * 12
    except (KeyError, TypeError, ValueError):
        stated_months = None
    if stated_months is not None:
        total = sum(
            months
            for entry in career
            if isinstance(months := entry.get("duration_months"), int | float)
        )
        if total > stated_months + config.CAREER_OVER_LIFE_SLACK:
            flags.append("career_exceeds_life")

        # Stated experience vs. the calendar span the listed career actually
        # covers (spec example: "8 years of experience at a company founded 3
        # years ago"). The dataset has no company-founded field, so this class
        # surfaces as years_of_experience far exceeding earliest-start -> snapshot.
        # earliest-start -> today is the widest defensible span (most lenient
        # denominator), so an honest candidate who simply omitted early roles is
        # not flagged.
        starts = [
            start for entry in career if (start := pdate(entry.get("start_date"))) is not None
        ]
        if starts:
            span = months_between(min(starts), today)
            if stated_months > span + config.CAREER_SPAN_SLACK:
                flags.append("experience_exceeds_career_span")

    # A degree cannot end before it starts.
    for entry in candidate.get("education") or []:
        start_year, end_year = entry.get("start_year"), entry.get("end_year")
        if isinstance(start_year, int) and isinstance(end_year, int) and end_year < start_year:
            flags.append("edu_inversion")
            break

    return flags


def is_honeypot(candidate: dict, today: date) -> bool:
    """True if any impossibility flag fires (one is enough to exclude)."""
    return bool(honeypot_flags(candidate, today))
