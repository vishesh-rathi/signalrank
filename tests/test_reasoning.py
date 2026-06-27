"""Tests for ranker.reasoning — extractive, trace-grounded reasoning strings."""

from ranker.reasoning import generate_reasoning


def make_candidate(**signal_overrides) -> dict:
    signals = {"recruiter_response_rate": 0.8, "github_activity_score": 50}
    signals.update(signal_overrides)
    return {
        "candidate_id": "CAND_0000000",
        "profile": {"years_of_experience": 7.2, "current_title": "ML Engineer"},
        "skills": [{"name": "NLP"}],
        "redrob_signals": signals,
    }


def make_strong_trace() -> dict:
    return {
        "evidence": 1.0,
        "evidence_concepts": ["ranking", "retrieval"],
        "domain_gate": 1.0,
        "semantic": 0.95,
        "seniority": 1.0,
        "product": 1.0,
        "location": 1.0,
        "recency": 0.9,
        "mult": 0.9,
        "score": 0.85,
    }


def make_weak_trace() -> dict:
    return {
        "evidence": 0.5,
        "evidence_concepts": [],
        "domain_gate": 0.6,
        "semantic": None,
        "seniority": 0.8,
        "product": 0.4,
        "location": 0.2,
        "recency": 0.2,
        "mult": 0.45,
        "score": 0.21,
    }


def test_cites_real_profile_facts():
    text = generate_reasoning(make_candidate(), make_strong_trace(), rank=1)
    assert "ML Engineer" in text
    assert "7.2" in text


def test_never_mentions_fields_absent_from_profile():
    # The candidate has no Pinecone skill and no notice-period signal —
    # neither may appear in the output.
    text = generate_reasoning(make_candidate(), make_strong_trace(), rank=1)
    assert "Pinecone" not in text
    assert "notice" not in text.lower()


def test_weak_candidate_gets_honest_concerns():
    candidate = make_candidate(recruiter_response_rate=0.3, notice_period_days=120)
    text = generate_reasoning(candidate, make_weak_trace(), rank=92)
    # Concerns should be expressed — may not use the exact word "concern" but
    # must name at least one real gap from the trace
    lower = text.lower()
    has_concern_language = any(
        phrase in lower
        for phrase in [
            "concern",
            "however",
            "though",
            "but",
            "caveat",
            "although",
            "120 day",
            "120-day",
            "requires 120",
            "notice",
            "constrained",
        ]
    )
    assert has_concern_language, f"Weak candidate should have concern language, got: {text}"


def test_strong_and_weak_outputs_differ_substantively():
    strong = generate_reasoning(make_candidate(), make_strong_trace(), rank=1)
    weak = generate_reasoning(
        make_candidate(recruiter_response_rate=0.3, notice_period_days=120),
        make_weak_trace(),
        rank=92,
    )
    assert strong != weak
    # rank-1 tone is confident: no negative language about the candidate
    lower_strong = strong.lower()
    assert not any(
        word in lower_strong for word in ["concern", "however", "though", "but", "caveat"]
    ), f"Rank-1 should be confident, got: {strong}"


def test_deep_rank_names_the_weakest_real_axis_not_filler():
    # Rank consistency (Stage-4): a deep row with a real gap names that gap —
    # never the circular "outscored on JD-specific evidence" filler.
    trace = make_strong_trace()
    trace["location"] = 0.6  # India, outside the JD's cities
    text = generate_reasoning(make_candidate(), trace, rank=80)
    lower = text.lower()
    # Should mention the location issue
    has_location_concern = any(
        phrase in lower
        for phrase in ["outside", "location", "city", "cities", "named", "preferred", "bangalore"]
    )
    assert has_location_concern, f"Should mention location issue, got: {text}"
    assert "outscored" not in lower


def test_deep_rank_with_flawless_axes_omits_concern_rather_than_fabricating():
    # When every fit axis is genuinely strong, an honest row has no concern
    # clause — inventing one (or stating the ranking outcome as if it were a
    # candidate fact) is the Stage-4 filler smell.
    text = generate_reasoning(make_candidate(), make_strong_trace(), rank=80)
    lower = text.lower()
    # Should not fabricate concerns when all axes are strong
    assert "concern" not in lower
    top_text = generate_reasoning(make_candidate(), make_strong_trace(), rank=1)
    assert "concern" not in top_text.lower()


def test_deep_rank_corroboration_only_evidence_is_an_honest_concern():
    # evidence == 0.6 means ranking keywords exist only in recycled role blurbs;
    # the concern says so instead of implying the candidate built the system.
    trace = make_strong_trace()
    trace["evidence"] = 0.6
    trace["evidence_concepts"] = []
    text = generate_reasoning(make_candidate(), trace, rank=70)
    lower = text.lower()
    # Should mention the corroboration-only nature of evidence — may say
    # "role blurbs", "summary", "no direct", "no ranking", "general ML" etc.
    has_evidence_gap = any(
        phrase in lower
        for phrase in [
            "role blurb",
            "no direct",
            "no ranking",
            "general ml",
            "lacks specific",
            "without clear",
            "ranking/search mentions appear in role blurbs",
        ]
    )
    assert has_evidence_gap, f"Should mention weak evidence, got: {text}"


def test_strong_reasoning_cites_a_concrete_signal_value():
    # Stage-4 specificity: a strong row must carry a real number, not only
    # qualitative praise, so top rows do not all read identically.
    text = generate_reasoning(make_candidate(), make_strong_trace(), rank=1)
    # Should contain a concrete signal value — response rate percentage or GitHub score
    has_concrete = any(
        val in text for val in ["80%", "0.80", "50/100", "50 ", "response rate", "GitHub"]
    )
    assert has_concrete, f"Should cite a concrete signal, got: {text}"


def test_job_hopping_surfaces_as_an_honest_concern():
    # A low stability component must produce a tenure concern connected to the JD.
    # Strong on every other axis so tenure is the sole concern (concerns cap at 2).
    trace = make_strong_trace()
    trace["stability"] = 0.3
    text = generate_reasoning(make_candidate(), trace, rank=70)
    lower = text.lower()
    has_stability_concern = any(
        phrase in lower
        for phrase in ["tenure", "job-hopping", "role change", "frequent", "3+ year"]
    )
    assert has_stability_concern, f"Should mention stability/tenure issue, got: {text}"


def test_reasoning_names_matched_concepts_not_a_fixed_phrase():
    # Stage-4 variation: the evidence clause must quote the concepts that fired,
    # so two builders with different specialties read differently.
    ranking = generate_reasoning(make_candidate(), make_strong_trace(), rank=1)
    # Should contain the concept names "ranking" and "retrieval"
    assert "ranking" in ranking.lower() and "retrieval" in ranking.lower(), (
        f"Should name matched concepts, got: {ranking}"
    )

    search_trace = make_strong_trace()
    search_trace["evidence_concepts"] = ["search"]
    search = generate_reasoning(make_candidate(), search_trace, rank=1)
    assert "search" in search.lower(), f"Should name 'search' concept, got: {search}"
    assert ranking != search


def test_reasoning_cites_current_employer_only_when_marked_current():
    current = make_candidate()
    current["career_history"] = [
        {"company": "Swiggy", "is_current": True},
        {"company": "OldCorp", "is_current": False},
    ]
    text = generate_reasoning(current, make_strong_trace(), rank=1)
    # Should mention Swiggy (the current employer) somewhere
    assert "Swiggy" in text, f"Should cite current employer, got: {text}"
    assert "OldCorp" not in text  # a non-current role must not be cited as present
    # No is_current role -> no employer clause (career order is never assumed).
    no_current = make_candidate()
    no_current["career_history"] = [{"company": "OldCorp", "is_current": False}]
    text_no = generate_reasoning(no_current, make_strong_trace(), rank=1)
    assert "OldCorp" not in text_no or "currently" not in text_no.lower()


def test_output_is_compact_and_csv_friendly():
    text = generate_reasoning(make_candidate(), make_strong_trace(), rank=1)
    assert 30 < len(text) < 300
    assert '"' not in text
    assert text.endswith(".")


def test_reasoning_handles_sparse_trace_and_profile():
    text = generate_reasoning({}, {}, rank=1)
    assert isinstance(text, str)
    assert len(text) > 0
    assert text.endswith(".")


def test_structural_variation_across_different_candidates():
    """Different candidate_ids produce structurally different reasoning strings."""
    reasonings = []
    for i in range(10):
        cand = make_candidate()
        cand["candidate_id"] = f"CAND_{i:07d}"
        text = generate_reasoning(cand, make_strong_trace(), rank=5)
        reasonings.append(text)

    # Should have high uniqueness
    unique = len(set(reasonings))
    assert unique >= 5, f"Expected >=5 unique reasonings from 10 candidates, got {unique}"


def test_rank_tier_affects_tone():
    """Top-tier, mid-tier, and bottom-tier reasonings have different structural patterns."""
    cand = make_candidate()
    top = generate_reasoning(cand, make_strong_trace(), rank=1)
    mid = generate_reasoning(cand, make_strong_trace(), rank=30)
    bottom = generate_reasoning(cand, make_strong_trace(), rank=80)

    # All three should be different (different tier builders are used)
    assert len({top, mid, bottom}) == 3, (
        f"Different ranks should produce different reasoning: top={top}, mid={mid}, bottom={bottom}"
    )


# ─── Grammar invariants (battle-hardening regression suite) ───────────
import re  # noqa: E402  (kept local to the grammar-invariant tests below)

# Patterns that must NEVER appear in any generated reasoning string. Each encodes
# a real templating defect this module was hardened against.
_FORBIDDEN_PATTERNS = {
    "doubled verb (has has / has brings / having has …)":
        re.compile(r"\b(?:has|having)\s+(?:has|have|having|brings?|demonstrated|shows?)\b", re.I),
    "shows-has double verb": re.compile(r"\bshows?\s+has\b", re.I),
    "stilted 'in having'": re.compile(r"\bin having\b", re.I),
    "double space": re.compile(r"  +"),
    "space before punctuation": re.compile(r"\s[,;.]"),
    "orphaned semicolon-dash": re.compile(r";\s*—"),
    "nested parentheses": re.compile(r"\([^()]*\([^()]*\)[^()]*\)"),
    "wrong article before a vowel-sound initialism":
        re.compile(r"\ba\s+(?:AI|ML|NLP|IR)\b"),
    "wrong article before a vowel-letter word":
        re.compile(r"\ba\s+(?:Applied|Engineer|Analyst|Architect)\b"),
}

# A matrix that exercises every fragment path — crucially the generic-evidence
# floor (evidence >= 0.5 with no concepts), which is the exact source of the
# historical "has has" / "has brings" defects.
_TITLES = [
    "ML Engineer",
    "AI Engineer",
    "NLP Engineer",
    "Applied ML Engineer",
    "Senior Machine Learning Engineer",
    "Recommendation Systems Engineer",
    "Search Engineer",
    "Data Scientist",
    "AI Research Engineer",
]


def _trace(**over):
    base = {
        "evidence": 1.0,
        "evidence_concepts": ["ranking", "retrieval"],
        "domain_gate": 1.0,
        "semantic": 0.9,
        "seniority": 1.0,
        "product": 1.0,
        "location": 1.0,
        "recency": 0.9,
        "stability": 1.0,
        "mult": 0.9,
        "score": 0.8,
    }
    base.update(over)
    return base


_TRACES = [
    _trace(),  # deep concept builder
    _trace(evidence_concepts=["recommendation", "hybrid search", "evaluation"]),
    _trace(evidence=0.5, evidence_concepts=[]),  # generic floor — the bug source
    _trace(evidence=0.6, evidence_concepts=[]),  # corroboration-only
    _trace(evidence=0.5, evidence_concepts=[], location=0.2, recency=0.2, product=0.4),
    _trace(stability=0.3),
]


def _matrix_rows():
    for t, title in enumerate(_TITLES):
        for tr, trace in enumerate(_TRACES):
            for rank in (1, 5, 12, 30, 55, 80, 100):
                cand = make_candidate(notice_period_days=120 if tr % 2 else 0)
                cand["candidate_id"] = f"CAND_{(t * 100 + tr * 10 + rank):07d}"
                cand["profile"] = {"years_of_experience": 8.6 if t % 2 else 7.2,
                                   "current_title": title}
                cand["career_history"] = [{"company": "Acme", "is_current": True}]
                yield generate_reasoning(cand, trace, rank)


def test_no_grammar_artifacts_across_matrix():
    """No generated reasoning may contain any known templating defect."""
    offenders = {}
    for text in _matrix_rows():
        for label, pattern in _FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                offenders.setdefault(label, text)
    assert not offenders, "Grammar artifacts found:\n" + "\n".join(
        f"  {label}: {sample}" for label, sample in offenders.items()
    )


def test_every_reasoning_is_well_formed():
    """Every row capitalizes, ends in a period, carries no stray quote, stays compact."""
    for text in _matrix_rows():
        assert text[0].isupper(), f"not capitalized: {text}"
        assert text.endswith("."), f"no terminal period: {text}"
        assert '"' not in text, f"stray quote: {text}"
        assert 20 < len(text) < 300, f"length out of range ({len(text)}): {text}"


def test_article_helper_handles_numbers_and_initialisms():
    """`_article` chooses a/an by spoken sound, including suffixed numbers."""
    from ranker.reasoning import _article

    assert _article("ML Engineer") == "an"
    assert _article("AI Engineer") == "an"
    assert _article("NLP Engineer") == "an"
    assert _article("Applied ML Engineer") == "an"
    assert _article("Data Scientist") == "a"
    assert _article("Senior Machine Learning Engineer") == "a"
    # Numbers are judged by how they are spoken, even with %/-day/-year suffixes.
    assert _article("8.6-year tenure") == "an"
    assert _article("7.2-year tenure") == "a"
    assert _article("11-year veteran") == "an"
    assert _article("18-day notice") == "an"
    assert _article("180-day notice") == "a"  # "one hundred eighty"
    assert _article("120-day notice") == "a"
    assert _article("88% response rate") == "an"
    assert _article("80% response rate") == "an"
    assert _article("72% response rate") == "a"
    assert _article("90% response rate") == "a"


def test_article_matches_spoken_sound():
    """Indefinite articles follow spoken sound: 'an ML Engineer', 'a Data Scientist'."""
    for title in ("ML Engineer", "AI Engineer", "NLP Engineer", "Applied ML Engineer"):
        cand = make_candidate()
        cand["profile"] = {"years_of_experience": 7.2, "current_title": title}
        text = generate_reasoning(cand, make_strong_trace(), rank=1).lower()
        first = title.split()[0].lower()
        assert f"an {first}" in text, f"expected 'an {first}' for {title!r}, got: {text}"
        assert f"a {first} " not in text, f"wrong article for {title!r}, got: {text}"
    cand = make_candidate()
    cand["profile"] = {"years_of_experience": 7.2, "current_title": "Data Scientist"}
    text = generate_reasoning(cand, make_strong_trace(), rank=1).lower()
    assert "a data scientist" in text and "an data" not in text, text


def test_immediate_availability_reads_naturally():
    """A zero-day notice surfaces as 'immediate availability', never 'within 0 days'."""
    cand = make_candidate(notice_period_days=0, recruiter_response_rate=0.2,
                          github_activity_score=0)
    text = generate_reasoning(cand, make_strong_trace(), rank=10)
    assert "within 0 days" not in text, text
    assert "immediate availability" in text, text
