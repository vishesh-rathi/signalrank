"""Interpretable fit features: how well a profile matches the JD, and why.

Each feature maps a candidate to [0, 1] and is designed to defeat a specific
trap in the dataset (see the module-level rules in config.py). ``weighted_fit``
blends them with config.WEIGHTS and returns a trace dict — the trace is what
makes every ranking decision explainable and powers the reasoning column.

Contract: every ``text``/``narrative`` parameter is ALREADY lowercased by the
caller (one .lower() per candidate, done in score.py). ``weighted_fit`` derives
and lowercases the two evidence source texts itself (see
``util.build_evidence_texts``) — they are a different field selection from the
narrative, not a re-spelling of it.
"""

import re

from ranker import config
from ranker.util import build_evidence_texts, compile_lexicon

# Compile each lexicon once into a single alternation regex (see compile_lexicon).
# Scoring 100K narratives against ~150 terms makes lexical matching the hot path,
# so feature functions search these precompiled patterns directly rather than
# re-deriving them per call.
_MED = compile_lexicon(tuple(config.MED_PHRASES))
_DOMAIN_POSITIVE = compile_lexicon(tuple(config.DOMAIN_POSITIVE))
_DOMAIN_NEGATIVE = compile_lexicon(tuple(config.DOMAIN_NEGATIVE))
_RESEARCH = compile_lexicon(tuple(config.RESEARCH_WORDS))
_PRODUCTION = compile_lexicon(tuple(config.PRODUCTION_WORDS))
_SCOPE = compile_lexicon(tuple(config.SCOPE_WORDS))
_TARGET_CITIES = compile_lexicon(tuple(config.TARGET_CITIES))
_SECONDARY_CITIES = compile_lexicon(tuple(config.SECONDARY_CITIES))
# One pattern per STRONG concept so evidence can count distinct concepts (depth),
# not merely detect presence (see config.STRONG_CONCEPTS).
_STRONG_CONCEPTS = {
    name: compile_lexicon(tuple(terms)) for name, terms in config.STRONG_CONCEPTS.items()
}


# A keyword inside a sentence about what the candidate WANTS, is merely
# LEARNING about, or directly after a negating/comparative word, is not
# built-evidence ("looking to grow into a role closer to ... modern ranking
# systems"; "lighter weight than ranking systems at FAANG"; "taking online
# courses on RAG and vector databases"). Three suppression rules:
#   - negation: a cue at most two words before the phrase suppresses it; the
#     tight window keeps "scaled ranking systems to more than 50M queries"
#     countable (its cue sits >2 words away or after the phrase).
#   - aspiration: a desire cue earlier in the SAME sentence suppresses every
#     match after it — that sentence describes a goal, not work performed.
#   - education: a course/exploration cue earlier in the same sentence means
#     the candidate is LEARNING about the topic, not that they built systems.
#     The non-tech candidate template (Content Writer, HR Manager, Graphic
#     Designer…) uses this pattern: "taking online courses on RAG and vector
#     databases, experimenting with LangChain". Without suppressing these,
#     4,164 non-tech trap candidates get evidence from course mentions.
_NEGATION_BEFORE = re.compile(
    r"(?:\bthan|\bnot|\bno|\bnever|\bwithout|\binstead of)\s+(?:[\w'/-]+\s+){0,2}$"
)
_ASPIRATION_CUE = re.compile(
    r"\b(?:looking (?:for|to)|seeking|open to|interested in|aspir\w*|hoping to|hope(?:s)? to|"
    r"aiming (?:for|to)|keen (?:on|to)|closer to|grow into|want(?:s)? to|plan(?:s|ning)? to|"
    r"transition(?:ing)? (?:to|into)|would love to|"
    r"taking (?:online )?(?:courses?|classes?)|enrolled in|experimenting|"
    r"exploring|learning about|studying|side project|excited about)\b"
)
_NEGATION_WINDOW = 40


# Self-disclaimers that indicate the candidate explicitly says they lack depth.
# The GENERIC summary template says "lighter on the deep-learning side" or
# "lighter weight than ranking systems at FAANG" — this is the candidate's own
# assessment that they are NOT a deep builder. Their 'recommendation' evidence is
# real (they did build recommendation-style features with classical methods), but
# it should grade lower than a candidate who claims no such disclaimer.
_DISCLAIMER_CUE = re.compile(
    r"\b(?:lighter on the deep.?learning|lighter weight than|not my strongest|"
    r"still building depth|building depth on)\b"
)

# Non-tech title keywords. A career composed entirely of non-tech titles
# (Content Writer, Graphic Designer, HR Manager, etc.) with zero engineering
# roles should not get full domain-gate credit even if narrative text contains
# ML vocabulary from courses or tool usage.
# Note: bare "engineer" is too broad (Mechanical/Civil/Chemical Engineer) — we
# match tech-qualified engineering or standalone tech role terms.
_TECH_TITLE_SIGNAL = re.compile(
    r"(?i)(?:"
    r"(?:(?:ml|ai|software|data|nlp|search|backend|full.?stack|platform|"
    r"infrastructure|devops|sre|cloud|site.reliability|applied|research|"
    r"deep.learning|machine.learning|computer.vision)\s+(?:engineer|scientist|"
    r"architect|developer|analyst))|"
    r"(?:(?:senior|staff|principal|lead|junior|intern)?\s*(?:ml|ai)\s+engineer)|"
    r"(?:data\s+(?:engineer|scientist|analyst))|"
    r"(?:software\s+(?:engineer|developer))|"
    r"developer|scientist|"
    r"machine learning|deep learning|nlp|"
    r"applied scientist|research scientist"
    r")",
)

# The JD explicitly rejects CV/speech/robotics-primary candidates unless they
# have significant NLP/IR exposure. Generic "NLP" or aspirational retrieval
# language is not enough; require actual built-systems concepts to rescue them.
_REJECTED_DOMAIN_TITLE = re.compile(
    r"(?i)\b(?:computer\s+vision|cv\s+engineer|speech|asr|robotics?|"
    r"perception|image\s+classification|object\s+detection)\b"
)


def _suppressed(text: str, start: int) -> bool:
    """True when the match at ``start`` is negated or merely aspirational."""
    if _NEGATION_BEFORE.search(text[max(0, start - _NEGATION_WINDOW) : start]):
        return True
    sentence_start = max(text.rfind(stop, 0, start) for stop in ".!?\n") + 1
    return bool(_ASPIRATION_CUE.search(text, sentence_start, start))


def evidence_concepts(text: str) -> list[str]:
    """Distinct STRONG built-systems concepts demonstrated in ``text``.

    A concept counts only if it has at least one non-suppressed match (negated
    and aspirational mentions are not evidence). Returned in config order so the
    list is deterministic and reads naturally in the reasoning column
    ("built ranking and retrieval systems").
    """
    return [
        name
        for name, pattern in _STRONG_CONCEPTS.items()
        if any(not _suppressed(text, match.start()) for match in pattern.finditer(text))
    ]


def _grade_evidence(
    primary: list[str],
    corroboration: list[str],
    text: str,
    has_disclaimer: bool = False,
) -> float:
    """Grade built-evidence by concept *depth* in the PRIMARY sources.

    Distinct primary (summary/title) concepts: >=3 -> 1.0 (deep builder, rare),
    2 -> 0.85, 1 -> 0.7 (real but shallow). A flat 1.0 for any single match
    saturates the whole top cohort and lets the behavioral multiplier — not
    technical depth — decide NDCG@10. Concepts appearing ONLY in role
    descriptions (recycled boilerplate in this pool) corroborate at 0.6: above
    generic ML prose (0.5) but below any claim the candidate makes in their own
    summary. Nothing is 0.0 except a narrative with no ML signal at all.

    When ``has_disclaimer`` is True the candidate's summary explicitly says they
    are NOT a deep builder ("lighter on the deep-learning side"). Their
    recommendation evidence is real but grades no higher than 0.55 — above MED
    prose (0.5) but below corroboration (0.6), since the candidate themselves
    says the depth isn't there.
    """
    distinct = len(primary)
    if distinct >= 3:
        score = 1.0
    elif distinct == 2:
        score = 0.85
    elif distinct == 1:
        score = 0.7
    elif corroboration:
        score = 0.6
    elif _MED.search(text):
        score = 0.5
    else:
        return 0.0
    # A self-disclaiming summary caps evidence: the candidate says they are
    # lighter / still building depth. Their real but shallow built-evidence
    # should not compete with non-disclaiming builders at the same concept count.
    if has_disclaimer:
        score = min(score, 0.55)
    return score


def evidence_score(text: str) -> float:
    """Graded lexical built-evidence in [0, 1] for a single primary text."""
    return _grade_evidence(evidence_concepts(text), [], text)


def domain_gate(text: str) -> float:
    """Gate technical fit by domain: the JD wants NLP/IR, rejects CV/speech/
    robotics and research-without-production.

    1.0 = positive-domain language present; 0.2 = negative domain or pure
    research; 0.6 = neutral (no domain signal either way).

    Positive evidence takes precedence over negative: a profile carrying BOTH an
    NLP/IR term and a CV/robotics term keeps the gate open at 1.0. This is
    deliberate — a genuine retrieval engineer who also once touched vision is the
    JD's target, not the CV-only specialist the gate exists to sink.
    """
    if _DOMAIN_POSITIVE.search(text):
        return 1.0
    is_rejected_domain = bool(_DOMAIN_NEGATIVE.search(text)) or (
        bool(_RESEARCH.search(text)) and not _PRODUCTION.search(text)
    )
    return 0.2 if is_rejected_domain else 0.6


def _title_coherence_gate(candidate: dict) -> float:
    """Check that the candidate's career titles include at least one tech role.

    The dataset plants non-tech professionals (Content Writers, HR Managers,
    Graphic Designers, Accountants, Civil Engineers) whose summaries mention AI
    keywords from courses — 4,164 such candidates carry 'vector databases' in
    their summaries. A career with zero tech titles means the narrative evidence
    is from education/exploration, not professional engineering work.

    Returns 1.0 for any career containing a tech title signal, 0.2 for a career
    composed entirely of non-tech roles. Checked on titles from all roles (not
    just current) — a former ML Engineer who transitioned to management is fine.
    """
    titles = []
    title = (candidate.get("profile") or {}).get("current_title") or ""
    if title:
        titles.append(title)
    for entry in candidate.get("career_history") or []:
        role_title = entry.get("title") or ""
        if role_title:
            titles.append(role_title)
    if not titles:
        return 0.6  # no titles available — neutral, not punishing
    if any(_TECH_TITLE_SIGNAL.search(t) for t in titles):
        return 1.0
    return 0.2  # zero tech titles across entire career


def _rejected_domain_title_gate(candidate: dict, concepts: list[str]) -> float:
    """Clamp CV/speech/robotics-primary titles without proven NLP/IR build depth.

    Positive domain words still rescue genuine cross-domain builders, but only
    when the candidate demonstrates at least one strong JD concept in their own
    summary/title evidence. This keeps "Computer Vision Engineer ... want to get
    serious about retrieval" below candidates who actually built search or
    recommendation systems.
    """
    if concepts:
        return 1.0
    profile = candidate.get("profile") or {}
    current_title = profile.get("current_title") or ""
    if _REJECTED_DOMAIN_TITLE.search(current_title):
        return 0.2
    titles = [
        entry.get("title") or ""
        for entry in candidate.get("career_history") or []
        if entry.get("title")
    ]
    if titles and all(_REJECTED_DOMAIN_TITLE.search(title) for title in titles):
        return 0.2
    return 1.0


def technical_fit(
    candidate: dict,
    evidence_text: str,
    support_text: str,
    narrative: str,
    semantic: float | None,
) -> tuple[float, dict]:
    """Domain-gated blend of lexical evidence and semantic JD similarity.

    Built-evidence is graded from ``evidence_text`` (summary + role titles) with
    ``support_text`` (role descriptions) corroborating only — see
    ``util.build_evidence_texts`` for why the sources are split. The domain gate,
    the title coherence gate, and the generic-ML floor read the full ``narrative``.

    Lexical evidence is precision (exact built-systems phrasing); the embedding
    similarity is recall (plain-language fits the lexicon misses). ``semantic``
    is the candidate's POOL-NORMALIZED similarity in [0, 1] (computed once over
    the whole pool by ``ranker.embeddings.semantic_fit_scores`` — the raw cosine
    occupies only ~0.09 of the nominal range, so an un-normalized value is inert).
    When no embedding is available (tests, lexical-only fallback) it is None and
    evidence stands alone.
    """
    has_disclaimer = bool(_DISCLAIMER_CUE.search(narrative))
    concepts = evidence_concepts(evidence_text)
    support = evidence_concepts(support_text) if support_text else []
    evidence = _grade_evidence(concepts, support, narrative, has_disclaimer)
    gate = domain_gate(narrative)
    title_gate = _title_coherence_gate(candidate)
    domain_title_gate = _rejected_domain_title_gate(candidate, concepts)
    effective_gate = min(gate, title_gate, domain_title_gate)
    if semantic is not None:
        share = config.TECH_SEMANTIC_SHARE
        blended = (1 - share) * evidence + share * semantic
    else:
        blended = evidence
    technical = blended * effective_gate
    return technical, {
        "evidence": evidence,
        "evidence_concepts": concepts,
        "domain_gate": gate,
        "title_gate": title_gate,
        "domain_title_gate": domain_title_gate,
        "semantic": semantic,
        "has_disclaimer": has_disclaimer,
    }


def seniority_score(candidate: dict, text: str) -> float:
    """Seniority from years of experience plus scope language, not job title.

    The JD's sweet spot is 5-9 years; the bands taper on both sides (a 13+
    year "architect who no longer codes" is an explicit JD concern). A
    junior/intern title halves the score; ownership language nudges it up.
    """
    try:
        years = float(candidate["profile"]["years_of_experience"])
    except (KeyError, TypeError, ValueError):
        years = 0.0
    if 5 <= years <= 9:
        band = 1.0
    elif 4 <= years < 5 or 9 < years <= 10:
        band = 0.8
    elif 3 <= years < 4 or 10 < years <= 12:
        band = 0.5
    elif years > 12:
        band = 0.35
    else:
        band = 0.25
    title = (candidate.get("profile", {}).get("current_title") or "").lower()
    if "junior" in title or "intern" in title:
        band *= 0.5
    if _SCOPE.search(text):
        band = min(1.0, band + 0.1)
    return band


def stability_score(candidate: dict) -> float:
    """Career tenure stability — the JD's anti-"title-chaser" signal.

    The JD explicitly rejects candidates "optimizing for Senior -> Staff ->
    Principal titles by switching companies every 1.5 years" and wants someone who
    "plans to be here for 3+ years". So a pattern of many short stints is
    down-weighted and longer average tenure is rewarded.

    Only roles with a positive recorded ``duration_months`` count. A one- or
    two-role history carries too little signal to judge and stays neutral rather
    than being punished (an early career, or two genuine startup failures, must
    not read as job-hopping). A recently started current role does not by itself
    look unstable: averaged against a longer prior role, the mean stays high.
    """
    tenures = [
        months
        for entry in candidate.get("career_history") or []
        if isinstance(months := entry.get("duration_months"), int | float) and months > 0
    ]
    if len(tenures) < 2:
        return 0.7  # too little history to judge; neither reward nor punish
    average = sum(tenures) / len(tenures)
    if average >= 30:
        band = 1.0
    elif average >= 24:
        band = 0.85
    elif average >= 18:
        band = 0.6
    elif average >= 12:
        band = 0.4
    else:
        band = 0.25
    # Four-plus short stints in a row is the clearest title-chaser pattern.
    if len(tenures) >= 4 and average < 18:
        band = min(band, 0.3)
    return band


def product_score(candidate: dict) -> float:
    """Product-company experience over a services/consulting-only career.

    The JD is explicit that prior product experience counts: "If you're
    currently at one of these [consulting] companies but have prior
    product-company experience, that's fine." So the signal is the BEST industry
    across the whole career (current role included), not the current employer
    alone. A career spent entirely inside the named services firms remains the
    strongest down-weight, regardless of industry labels.

    Company names use substring containment on a short controlled list
    (handles "Tata Consultancy Services Ltd"); this is a deliberate exception
    to the token-matching rule, which exists for free-prose narrative text.
    """
    companies = [
        (entry.get("company") or "").lower() for entry in candidate.get("career_history") or []
    ]
    services_only = bool(companies) and all(
        any(firm in company for firm in config.SERVICES_FIRMS) for company in companies
    )
    if services_only:
        return 0.2
    industries = {
        entry.get("industry")
        for entry in candidate.get("career_history") or []
        if entry.get("industry")
    }
    current = candidate.get("profile", {}).get("current_industry")
    if current:
        industries.add(current)
    if industries & config.PRODUCT_INDUSTRIES:
        return 1.0
    if industries and industries <= config.SERVICES_INDUSTRIES:
        return 0.4
    return 0.6


def location_score(candidate: dict) -> float:
    """Proximity to the JD's hiring cities; relocation intent rescues abroad.

    JD-named cities get full credit; Bangalore/Bengaluru (a major hub the JD does
    NOT list) gets strong-but-not-full credit, lifted to full only with explicit
    relocation intent toward the JD's Pune/Noida offices.
    """
    profile = candidate.get("profile", {})
    location = (profile.get("location") or "").lower()
    signals = candidate.get("redrob_signals") or {}
    if _TARGET_CITIES.search(location):
        return 1.0
    if _SECONDARY_CITIES.search(location):
        return 1.0 if signals.get("willing_to_relocate") else config.SECONDARY_CITY_SCORE
    if (profile.get("country") or "").lower() == "india":
        return 0.6
    return 0.5 if signals.get("willing_to_relocate") else 0.2


def education_score(candidate: dict) -> float:
    """Small institution-tier bonus. Deliberately low-weight: the JD values
    shipped systems over pedigree, and explicitly does not prize pure academia."""
    tiers = {entry.get("tier") for entry in candidate.get("education") or []}
    if "tier_1" in tiers:
        return 1.0
    if "tier_2" in tiers:
        return 0.7
    return 0.5


def weighted_fit(candidate: dict, text: str, semantic: float | None) -> tuple[float, dict]:
    """Blend all features with config.WEIGHTS; return (fit, trace).

    ``semantic`` is the pool-normalized JD similarity in [0, 1], or None for
    lexical-only scoring (see ``technical_fit``).
    """
    evidence_text, support_text = build_evidence_texts(candidate)
    technical, technical_parts = technical_fit(
        candidate, evidence_text.lower(), support_text.lower(), text, semantic
    )
    seniority = seniority_score(candidate, text)
    product = product_score(candidate)
    location = location_score(candidate)
    education = education_score(candidate)
    stability = stability_score(candidate)
    weights = config.WEIGHTS
    fit = (
        weights["tech"] * technical
        + weights["seniority"] * seniority
        + weights["product"] * product
        + weights["location"] * location
        + weights["education"] * education
        + weights["stability"] * stability
    )
    trace = {
        **technical_parts,
        "technical": technical,
        "seniority": seniority,
        "product": product,
        "location": location,
        "education": education,
        "stability": stability,
        "fit": fit,
    }
    return fit, trace
