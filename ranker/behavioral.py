"""Multiplicative behavioral modifier: is this candidate actually reachable?

The JD is explicit that a perfect-on-paper candidate who is unresponsive and
months-inactive is "not actually available". Fit is therefore scaled by a
multiplier in [config.MULT_FLOOR, 1.0] built from four sub-scores. Because it
multiplies (never adds), poor engagement sinks a profile but good engagement
cannot lift a weak fit above an available strong one. The sub-scores also
separate "behavioral twin" candidates whose profiles are otherwise identical.

``today`` is always injected (the dataset snapshot date): using the wall clock
would make scores non-reproducible across reruns, which the contest forbids.
"""

from datetime import date

from ranker import config
from ranker.util import pdate


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _notice_factor(notice_days: float) -> float:
    """Notice-period credit in [0, 1], grounded in the JD's own framing.

    "We'd love sub-30-day notice. We can buy out up to 30 days. 30+ day notice
    candidates are still in scope but the bar gets higher." A notice within the
    30-day buy-out window earns full credit; beyond it the factor decays convexly
    (squared remaining-fraction) to zero at the 180-day ceiling, so longer
    notices stay in scope but the bar rises progressively — far steeper than a
    linear decay (90d: 0.36 vs 0.50; 120d: 0.16 vs 0.33) without excluding anyone.
    """
    if notice_days <= config.NOTICE_BUYOUT_DAYS:
        return 1.0
    remaining = (config.NOTICE_MAX_DAYS - notice_days) / (
        config.NOTICE_MAX_DAYS - config.NOTICE_BUYOUT_DAYS
    )
    return _clamp01(remaining) ** 2


def availability_score(signals: dict) -> float:
    """Open-to-work flag (half the weight), short notice period, relocation intent."""
    score = 0.5 if signals.get("open_to_work_flag") else 0.0
    notice_days = signals.get("notice_period_days")
    if isinstance(notice_days, int | float):
        score += 0.4 * _notice_factor(notice_days)
    if signals.get("willing_to_relocate"):
        score += 0.1
    return _clamp01(score)


def responsiveness_score(signals: dict) -> float:
    """Mostly the recruiter response rate, plus how fast they reply (2-280 h range)."""
    response_rate = signals.get("recruiter_response_rate") or 0.0
    response_hours = signals.get("avg_response_time_hours")
    speed = (
        _clamp01((280 - response_hours) / 280)
        if isinstance(response_hours, int | float)
        else 0.5  # unknown speed is neutral, not damning
    )
    return _clamp01(0.7 * response_rate + 0.3 * speed)


def recency_score(signals: dict, today: date) -> float:
    """Linear decay from "active today" (1.0) to "gone 180+ days" (0.0)."""
    last_active = pdate(signals.get("last_active_date"))
    if last_active is None:
        return 0.3  # unknown: below-neutral, since recency is scarce in this pool
    return _clamp01((180 - (today - last_active).days) / 180)


def credibility_score(candidate: dict) -> float:
    """External validation (GitHub, verifications) minus an inflation penalty.

    The JD prizes open-source/external validation; 64.6% of the pool has no
    GitHub at all, so a positive activity score is a real differentiator.
    Two inflation patterns cost credibility (Design 4.5): claiming expert/advanced
    proficiency in a skill the platform assessed below 40/100, and claiming a
    skill for more months than the candidate's entire career has lasted.
    """
    signals = candidate.get("redrob_signals") or {}
    score = 0.2  # base: absence of validation is common, not disqualifying
    github_activity = signals.get("github_activity_score")
    if isinstance(github_activity, int | float) and github_activity > 0:
        score += 0.4 * _clamp01(github_activity / 60)
    if signals.get("verified_email"):
        score += 0.2
    if signals.get("verified_phone"):
        score += 0.1
    if signals.get("linkedin_connected"):
        score += 0.1
    claimed_proficiency = {
        (skill.get("name") or "").lower(): skill.get("proficiency")
        for skill in candidate.get("skills") or []
    }
    assessments = signals.get("skill_assessment_scores") or {}
    inflated_claims = sum(
        1
        for skill_name, assessed in assessments.items()
        if claimed_proficiency.get((skill_name or "").lower()) in ("expert", "advanced")
        and isinstance(assessed, int | float)
        and assessed < 40
    )
    score -= 0.1 * min(inflated_claims, 2)  # cap: two bad claims is already the message

    # Skill-duration inflation: a skill used for more months than the candidate's
    # whole career has existed is impossible self-reporting (Design 4.5, the half
    # the original implementation dropped). Penalized symmetrically to the
    # assessment-based half and only when there is a positive career span to
    # compare against.
    career_months = sum(
        months
        for entry in candidate.get("career_history") or []
        if isinstance(months := entry.get("duration_months"), int | float) and months > 0
    )
    if career_months > 0:
        duration_inflated = sum(
            1
            for skill in candidate.get("skills") or []
            if isinstance(used := skill.get("duration_months"), int | float)
            and used > career_months
        )
        score -= 0.1 * min(duration_inflated, 2)
    return _clamp01(score)


def behavioral_multiplier(candidate: dict, today: date) -> tuple[float, dict]:
    """Blend the four sub-scores and map into [MULT_FLOOR, 1.0]; return (mult, trace)."""
    signals = candidate.get("redrob_signals") or {}
    availability = availability_score(signals)
    responsiveness = responsiveness_score(signals)
    recency = recency_score(signals, today)
    credibility = credibility_score(candidate)
    weights = config.MULT_WEIGHTS
    blend = (
        weights["availability"] * availability
        + weights["responsiveness"] * responsiveness
        + weights["recency"] * recency
        + weights["credibility"] * credibility
    )
    multiplier = config.MULT_FLOOR + (1 - config.MULT_FLOOR) * blend
    trace = {
        "availability": availability,
        "responsiveness": responsiveness,
        "recency": recency,
        "credibility": credibility,
        "mult": multiplier,
    }
    return multiplier, trace
