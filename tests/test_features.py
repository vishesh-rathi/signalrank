"""Tests for ranker.features — evidence, domain gating, and contextual fit."""

from ranker.features import (
    domain_gate,
    education_score,
    evidence_concepts,
    evidence_score,
    location_score,
    product_score,
    seniority_score,
    stability_score,
    technical_fit,
    tenure_chaser_penalty,
    weighted_fit,
)


def test_evidence_grades_by_concept_depth():
    # One distinct concept is real but shallow (0.7); depth raises the grade so
    # the top cohort does not saturate at a flat 1.0.
    assert evidence_score("we built a ranking system at scale") == 0.7
    assert evidence_score("built a ranking system and a semantic search engine") == 0.85
    assert (
        evidence_score("ranking systems, a recommendation engine, and vector search infra") == 1.0
    )
    # Generic ML prose with no STRONG concept is mid; nothing is zero.
    assert evidence_score("trained a deep learning model for fraud") == 0.5
    assert evidence_score("managed a sales team and quarterly targets") == 0.0


def test_evidence_concepts_dedupes_synonyms_within_a_bucket():
    # "ranking system" and "re-ranking" are the same concept -> counted once.
    one_concept = evidence_concepts("ranking system with re-ranking and reranked candidates")
    assert one_concept == ["ranking"]
    concepts = evidence_concepts("ranking system and a recommendation engine")
    assert set(concepts) == {"ranking", "recommendation"}


def test_evidence_ignores_research_engineer_substring_trap():
    # "search engine" must not fire inside "research engineer".
    assert evidence_score("ai research engineer focused on theory") == 0.0


def test_evidence_ignores_negated_mention():
    # "lighter weight than ranking systems at FAANG" disclaims the system —
    # the candidate is saying what they did NOT build.
    assert evidence_concepts("our features are lighter weight than ranking systems at faang") == []
    assert evidence_concepts("never built a ranking system, focused on dashboards") == []


def test_evidence_ignores_aspirational_sentence():
    # A sentence about the role the candidate WANTS is not work they did.
    aspiration = (
        "i'm looking to grow into a deeper ai/ml system-building role — "
        "closer to retrieval, llms, and modern ranking systems."
    )
    assert evidence_concepts(aspiration) == []
    # Suppression is sentence-scoped: a real build claim in an earlier sentence
    # still counts even when a later sentence is aspirational.
    built_then_aspires = (
        "built a ranking system for product search. open to senior roles in applied ml."
    )
    assert evidence_concepts(built_then_aspires) == ["ranking"]


def test_evidence_keeps_legitimate_comparative_claims():
    # The negation window is tight: a cue more than two words before the phrase
    # (or after it) must not suppress a genuine scale claim.
    assert evidence_concepts("scaled ranking systems to more than 50m queries a day") == ["ranking"]
    assert evidence_concepts("served more than 100m monthly users with ranking systems") == [
        "ranking"
    ]


def test_description_only_concepts_corroborate_but_cannot_establish():
    # Role descriptions in this pool are recycled boilerplate shared across
    # candidates, so a concept appearing ONLY there grades 0.6 — above generic
    # ML prose, below a claim the candidate makes in their own summary — and is
    # NOT exposed as a built-systems claim in the trace.
    _tech_cand = {"profile": {"current_title": "ML Engineer"}, "career_history": []}
    technical, parts = technical_fit(
        _tech_cand, "ml engineer", "owned the ranking systems layer end-to-end", "ml engineer", None
    )
    assert parts["evidence"] == 0.6
    assert parts["evidence_concepts"] == []
    summary_backed, _ = technical_fit(
        _tech_cand, "built ranking systems", "", "built ranking systems", None
    )
    assert summary_backed > technical


def test_domain_gate_positive_negative_neutral():
    assert domain_gate("nlp and information retrieval work") == 1.0
    assert domain_gate("computer vision and object detection pipelines") == 0.2
    assert domain_gate("wrote my phd thesis and academic papers") == 0.2  # research, no production
    assert domain_gate("backend systems for payments") == 0.6


def test_technical_fit_nlp_builder_beats_cv_builder():
    _tech_cand = {"profile": {"current_title": "ML Engineer"}, "career_history": []}
    nlp_text = "built a ranking system for product search"
    cv_text = "built image classification and object detection"
    nlp_fit, _ = technical_fit(_tech_cand, nlp_text, "", nlp_text, None)
    cv_fit, _ = technical_fit(_tech_cand, cv_text, "", cv_text, None)
    assert nlp_fit > cv_fit


def test_cv_primary_without_ir_build_evidence_is_clamped():
    candidate = {
        "profile": {
            "current_title": "Computer Vision Engineer",
            "summary": (
                "Worked across predictive modeling, NLP, analytics, and lightweight "
                "deployment workflows. I want to get serious about LLMs and retrieval "
                "beyond the surface level."
            ),
        },
        "career_history": [{"title": "Computer Vision Engineer"}],
    }
    text = candidate["profile"]["summary"].lower()
    technical, parts = technical_fit(candidate, text, "", text, 1.0)
    assert parts["evidence_concepts"] == []
    assert parts["domain_gate"] == 1.0  # generic NLP wording is present
    assert parts["domain_title_gate"] == 0.2
    assert technical < 0.2


def test_cv_title_with_real_ir_build_evidence_is_not_clamped():
    candidate = {
        "profile": {
            "current_title": "Computer Vision Engineer",
            "summary": (
                "Built a recommendation system, retrieval system, and semantic search engine."
            ),
        },
        "career_history": [{"title": "Computer Vision Engineer"}],
    }
    text = candidate["profile"]["summary"].lower()
    technical, parts = technical_fit(candidate, text, "", text, None)
    assert set(parts["evidence_concepts"]) == {"recommendation", "retrieval", "search"}
    assert parts["domain_title_gate"] == 1.0
    assert technical > 0.8


def test_technical_fit_blends_pool_normalized_semantic():
    _tech_cand = {"profile": {"current_title": "ML Engineer"}, "career_history": []}
    text = "built a ranking system for search"
    lexical_only, parts_none = technical_fit(_tech_cand, text, "", text, None)
    assert parts_none["semantic"] is None
    blended, parts = technical_fit(_tech_cand, text, "", text, 1.0)  # top-of-pool semantic
    assert parts["semantic"] == 1.0
    assert blended >= lexical_only  # a strong semantic percentile can only help here


def test_semantic_is_minority_share_of_technical():
    _tech_cand = {"profile": {"current_title": "ML Engineer"}, "career_history": []}
    # The pool-percentile semantic saturates (~0.99) across the whole top cohort,
    # so it must not be able to halve the graded-evidence gap: evidence carries
    # 1 - TECH_SEMANTIC_SHARE of the blend. Two candidates one evidence band
    # apart (0.85 vs 0.7) must stay separated even when the weaker one holds the
    # maximum semantic percentile.
    deep_text = "built a ranking system and semantic search engine"  # 0.85
    shallow_text = "built a ranking system"  # 0.7
    deep, _ = technical_fit(_tech_cand, deep_text, "", deep_text, 0.9)
    shallow_max_semantic, _ = technical_fit(_tech_cand, shallow_text, "", shallow_text, 1.0)
    assert deep > shallow_max_semantic


def test_hybrid_search_vocabulary_is_a_depth_concept():
    # The JD's own infrastructure words ("hybrid retrieval", "BM25", FAISS...)
    # mark the exact must-have capability, so they deepen the evidence grade:
    # an engineer narrating a BM25 + dense hybrid build reads as a deep builder.
    summary = (
        "designed the company's first hybrid retrieval system combining "
        "bm25 with dense vector recall, plus learning-to-rank"
    )
    assert set(evidence_concepts(summary)) == {"ranking", "retrieval", "hybrid search"}
    assert evidence_score(summary) == 1.0


def test_technical_fit_exposes_matched_concepts_in_trace():
    _tech_cand = {"profile": {"current_title": "ML Engineer"}, "career_history": []}
    text = "built a ranking system and a recommendation engine"
    _, parts = technical_fit(_tech_cand, text, "", text, None)
    assert set(parts["evidence_concepts"]) == {"ranking", "recommendation"}


def test_seniority_band_peak_and_junior_penalty():
    senior = {"profile": {"years_of_experience": 7.0, "current_title": "ML Engineer"}}
    junior = {"profile": {"years_of_experience": 7.0, "current_title": "Junior ML Engineer"}}
    text = "led production systems end-to-end"
    assert seniority_score(senior, text) > seniority_score(junior, text)
    fresh = {"profile": {"years_of_experience": 1.0, "current_title": "ML Engineer"}}
    assert seniority_score(senior, text) > seniority_score(fresh, text)


def test_stability_rewards_tenure_and_penalizes_job_hopping():
    stable = {
        "career_history": [
            {"company": "A", "duration_months": 40},
            {"company": "B", "duration_months": 36},
        ]
    }
    hopper = {
        "career_history": [
            {"company": "A", "duration_months": 14},
            {"company": "B", "duration_months": 16},
            {"company": "C", "duration_months": 15},
            {"company": "D", "duration_months": 17},
        ]
    }
    assert stability_score(stable) == 1.0
    assert stability_score(hopper) <= 0.3
    assert stability_score(stable) > stability_score(hopper)


def test_stability_is_neutral_on_sparse_history():
    # Too little history to judge — neither rewarded nor punished as a hopper.
    assert stability_score({"career_history": [{"duration_months": 10}]}) == 0.7
    assert stability_score({}) == 0.7
    # Two genuine startup stints should not be capped like a serial title-chaser.
    two_short = {"career_history": [{"duration_months": 14}, {"duration_months": 15}]}
    assert stability_score(two_short) == 0.4


def test_product_beats_services_only_career():
    product = {
        "profile": {"current_industry": "Fintech"},
        "career_history": [{"company": "Swiggy"}],
    }
    services_only = {
        "profile": {"current_industry": "IT Services"},
        "career_history": [{"company": "Infosys"}, {"company": "TCS"}],
    }
    assert product_score(product) > product_score(services_only)


def test_weighted_fit_returns_score_and_full_trace():
    # Built-evidence is read from the candidate's own summary (+ role titles),
    # not from the blended narrative — see util.build_evidence_texts.
    candidate = {
        "profile": {
            "years_of_experience": 7.0,
            "current_title": "ML Engineer",
            "current_industry": "Fintech",
            "location": "Pune, MH",
            "country": "India",
            "summary": "Led a ranking system for semantic search in production.",
        },
        "career_history": [{"company": "Swiggy"}],
        "education": [{"tier": "tier_1"}],
        "redrob_signals": {"willing_to_relocate": False},
    }
    text = "led a ranking system for semantic search in production"
    fit, trace = weighted_fit(candidate, text, None)
    assert 0.0 < fit <= 1.0
    assert trace["evidence"] == 0.85  # two distinct concepts: ranking + search
    assert set(trace["evidence_concepts"]) == {"ranking", "search"}
    assert trace["domain_gate"] == 1.0
    assert trace["location"] == 1.0
    assert trace["education"] == 1.0
    assert set(trace) == {
        "evidence",
        "evidence_concepts",
        "domain_gate",
        "title_gate",
        "domain_title_gate",
        "semantic",
        "has_disclaimer",
        "technical",
        "seniority",
        "product",
        "location",
        "education",
        "stability",
        "chaser_penalty",
        "fit",
    }


def test_tenure_chaser_penalty_fires_only_on_the_unambiguous_pattern():
    # The JD's title-chaser: four+ roles averaging under 18 months ("switching
    # companies every 1.5 years"). The penalty fires here and scales fit down.
    chaser = {"career_history": [{"duration_months": m} for m in (14, 16, 27, 13)]}
    assert tenure_chaser_penalty(chaser) == 0.75
    # Accelerating job-hopping (five short stints) also fires.
    accel = {"career_history": [{"duration_months": m} for m in (33, 19, 14, 13, 8)]}
    assert tenure_chaser_penalty(accel) == 0.75
    # Same short average but only three roles: too little history to call it
    # chasing (matches stability_score's >=4 requirement) — no penalty.
    short_history = {"career_history": [{"duration_months": m} for m in (12, 14, 13)]}
    assert tenure_chaser_penalty(short_history) == 1.0
    # Four roles but a healthy 30-month average is a stable career — no penalty.
    stable = {"career_history": [{"duration_months": m} for m in (30, 36, 24, 30)]}
    assert tenure_chaser_penalty(stable) == 1.0
    # Empty / missing history is neutral, never penalized.
    assert tenure_chaser_penalty({}) == 1.0


def test_tenure_chaser_penalty_multiplies_into_weighted_fit():
    profile = {
        "profile": {
            "years_of_experience": 7.0,
            "current_title": "ML Engineer",
            "current_industry": "Fintech",
            "summary": "Led a ranking system for semantic search in production.",
        },
        "redrob_signals": {},
    }
    text = "led a ranking system for semantic search in production"
    chaser = {**profile, "career_history": [{"duration_months": m} for m in (14, 16, 13, 15)]}
    stable = {**profile, "career_history": [{"duration_months": m} for m in (40, 44, 38, 42)]}
    chaser_fit, chaser_trace = weighted_fit(chaser, text, None)
    stable_fit, stable_trace = weighted_fit(stable, text, None)
    assert chaser_trace["chaser_penalty"] == 0.75
    assert stable_trace["chaser_penalty"] == 1.0
    # The chaser is down-weighted below the otherwise-identical stable career
    # (both share evidence/seniority/product/location; only tenure differs).
    assert chaser_fit < stable_fit


def test_location_target_city_beats_india_beats_abroad():
    pune = {"profile": {"location": "Pune, MH", "country": "India"}}
    chennai = {"profile": {"location": "Chennai, TN", "country": "India"}}
    abroad = {"profile": {"location": "Toronto", "country": "Canada"}, "redrob_signals": {}}
    relocating = {
        "profile": {"location": "Toronto", "country": "Canada"},
        "redrob_signals": {"willing_to_relocate": True},
    }
    assert location_score(pune) == 1.0
    assert location_score(pune) > location_score(chennai) > location_score(abroad)
    assert location_score(relocating) > location_score(abroad)


def test_bangalore_is_secondary_not_full_target_city():
    # Bangalore is not on the JD's list: strong-but-not-full credit, below the
    # JD-named cities and above generic India, lifted to full only on relocation.
    pune = {"profile": {"location": "Pune, MH", "country": "India"}}
    blr = {"profile": {"location": "Bangalore, KA", "country": "India"}, "redrob_signals": {}}
    blr_reloc = {
        "profile": {"location": "Bengaluru, KA", "country": "India"},
        "redrob_signals": {"willing_to_relocate": True},
    }
    chennai = {"profile": {"location": "Chennai, TN", "country": "India"}}
    assert location_score(pune) == 1.0
    assert location_score(blr) == 0.8
    assert location_score(pune) > location_score(blr) > location_score(chennai)
    assert location_score(blr_reloc) == 1.0


def test_education_tier_ladder():
    tier1 = {"education": [{"tier": "tier_1"}, {"tier": "tier_3"}]}
    tier2 = {"education": [{"tier": "tier_2"}]}
    tier3 = {"education": [{"tier": "tier_3"}]}
    no_education = {"education": []}
    assert education_score(tier1) > education_score(tier2) > education_score(tier3)
    assert education_score(no_education) == education_score(tier3)  # neutral, not punitive


def test_weighted_fit_on_sparse_candidate_stays_in_range():
    fit, trace = weighted_fit({}, "", None)
    assert 0.0 <= fit <= 1.0
    assert trace["semantic"] is None


def test_product_score_unknown_industry_is_neutral():
    candidate = {"profile": {"current_industry": "Widgets"}, "career_history": []}
    assert product_score(candidate) == 0.6


def test_prior_product_experience_counts_from_services_present():
    # JD: "currently at a consulting firm but have prior product-company
    # experience, that's fine" — the best industry across the career wins.
    candidate = {
        "profile": {"current_industry": "IT Services"},
        "career_history": [
            {"company": "Genpact", "industry": "IT Services", "is_current": True},
            {"company": "LinkedIn", "industry": "Internet", "is_current": False},
        ],
    }
    assert product_score(candidate) == 1.0
    never_product = {
        "profile": {"current_industry": "IT Services"},
        "career_history": [
            {"company": "Genpact", "industry": "IT Services"},
            {"company": "Quess Corp", "industry": "Consulting"},
        ],
    }
    assert product_score(never_product) == 0.4


def test_services_only_firms_cap_wins_over_industry_labels():
    # All-services-by-name is the strongest down-weight even when a row carries
    # a product-ish industry label.
    candidate = {
        "profile": {"current_industry": "Software"},
        "career_history": [
            {"company": "Infosys", "industry": "Software"},
            {"company": "TCS", "industry": "IT Services"},
        ],
    }
    assert product_score(candidate) == 0.2


def test_domain_gate_research_with_production_is_not_penalized():
    # Research language is only penalized when no production signal accompanies it.
    assert domain_gate("research prototypes deployed to production for users") == 0.6
    assert domain_gate("phd thesis and academic papers only") == 0.2
    assert domain_gate("nlp research published") == 1.0  # a positive domain term wins


def test_domain_gate_positive_beats_co_occurring_negative():
    # Precedence lock: a builder who mentions BOTH NLP/IR and computer vision
    # keeps the gate open — positive evidence wins over a co-occurring negative.
    assert domain_gate("semantic search and ranking plus some computer vision") == 1.0


def test_evidence_suppresses_course_taking_context():
    # The non-tech candidate template mentions AI keywords in a course-taking
    # context. "Taking online courses on" must suppress evidence in that sentence.
    course_text = (
        "i've been taking online courses on rag and vector databases, "
        "experimenting with langchain and the openai api for side projects"
    )
    assert evidence_concepts(course_text) == []

    # "Exploring" and "learning about" should also suppress.
    assert evidence_concepts("exploring how to build a recommendation system") == []
    assert evidence_concepts("learning about semantic search and retrieval") == []

    # But a real build claim in a different sentence still counts.
    real_then_course = (
        "built a recommendation engine for product search. now taking courses on vector databases."
    )
    assert evidence_concepts(real_then_course) == ["recommendation"]


def test_disclaimer_caps_evidence_grade():
    # A self-disclaiming summary should grade no higher than 0.55 regardless
    # of how many concepts match.
    from ranker.features import _grade_evidence

    # One primary concept without disclaimer = 0.7
    assert _grade_evidence(["recommendation"], [], "recommendation system built") == 0.7
    # One primary concept WITH disclaimer = 0.55
    assert (
        _grade_evidence(["recommendation"], [], "recommendation system built", has_disclaimer=True)
        == 0.55
    )
    # Even deep evidence (3+ concepts) caps at 0.55 with a disclaimer
    assert (
        _grade_evidence(
            ["ranking", "recommendation", "retrieval"],
            [],
            "deep builder text",
            has_disclaimer=True,
        )
        == 0.55
    )


def test_title_coherence_gate_separates_tech_from_non_tech():
    # A career with at least one tech title passes (1.0).
    tech_career = {
        "profile": {"current_title": "ML Engineer"},
        "career_history": [{"title": "ML Engineer"}, {"title": "Data Analyst"}],
    }
    non_tech_career = {
        "profile": {"current_title": "Content Writer"},
        "career_history": [
            {"title": "Content Writer"},
            {"title": "Graphic Designer"},
            {"title": "Mechanical Engineer"},
        ],
    }
    from ranker.features import _title_coherence_gate

    assert _title_coherence_gate(tech_career) == 1.0
    assert _title_coherence_gate(non_tech_career) == 0.2

    # "Mechanical Engineer" does NOT match — it's a non-tech engineering title.
    mech_only = {
        "profile": {"current_title": "Mechanical Engineer"},
        "career_history": [{"title": "Mechanical Engineer"}],
    }
    assert _title_coherence_gate(mech_only) == 0.2

    # Purely non-tech titles without any "engineer" variant
    hr_career = {
        "profile": {"current_title": "HR Manager"},
        "career_history": [{"title": "HR Manager"}, {"title": "Recruiter"}],
    }
    assert _title_coherence_gate(hr_career) == 0.2


def test_content_writer_trap_candidate_gets_low_score():
    # End-to-end: a Content Writer with AI course mentions should score far
    # below a genuine ML engineer.
    content_writer = {
        "profile": {
            "years_of_experience": 7.7,
            "current_title": "Content Writer",
            "current_industry": "Software",
            "location": "Bangalore, KA",
            "country": "India",
            "summary": (
                "Content Writer with 7.7+ years. Taking online courses on "
                "RAG and vector databases, experimenting with LangChain."
            ),
        },
        "career_history": [
            {"title": "Content Writer", "company": "Hooli", "industry": "Software"},
            {"title": "Graphic Designer", "company": "Pied Piper", "industry": "Software"},
        ],
        "education": [{"tier": "tier_3"}],
        "redrob_signals": {"willing_to_relocate": True},
    }
    ml_engineer = {
        "profile": {
            "years_of_experience": 7.0,
            "current_title": "ML Engineer",
            "current_industry": "Fintech",
            "location": "Bangalore, KA",
            "country": "India",
            "summary": "Built a recommendation system and semantic search engine.",
        },
        "career_history": [
            {"title": "ML Engineer", "company": "Swiggy", "industry": "Fintech"},
        ],
        "education": [{"tier": "tier_1"}],
        "redrob_signals": {"willing_to_relocate": False},
    }
    from ranker.util import build_narrative

    cw_text = build_narrative(content_writer).lower()
    ml_text = build_narrative(ml_engineer).lower()
    cw_score, _ = weighted_fit(content_writer, cw_text, None)
    ml_score, _ = weighted_fit(ml_engineer, ml_text, None)
    assert ml_score > cw_score, (
        f"ML engineer ({ml_score:.4f}) should score higher than Content Writer ({cw_score:.4f})"
    )
