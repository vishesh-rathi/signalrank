"""Tests for ranker.util — parsing, narrative assembly, lexicon matching."""

from datetime import date

from ranker.util import build_narrative, iter_candidates, months_between, pdate, text_has


def test_pdate_valid_and_invalid():
    assert pdate("2024-03-08") == date(2024, 3, 8)
    assert pdate(None) is None
    assert pdate("not-a-date") is None


def test_months_between():
    assert months_between(date(2022, 1, 1), date(2024, 1, 1)) == 24
    assert months_between(date(2023, 9, 10), date(2024, 1, 8)) == 4


def test_build_narrative_uses_summary_and_descriptions_not_skills():
    candidate = {
        "profile": {"summary": "Built a ranking system.", "headline": "ML person"},
        "career_history": [{"title": "ML Engineer", "description": "Deployed semantic search."}],
        "skills": [{"name": "SecretSkillNotInText"}],
    }
    text = build_narrative(candidate).lower()
    assert "ranking system" in text
    assert "semantic search" in text
    assert "ml engineer" in text
    assert "secretskillnotintext" not in text


def test_text_has_requires_word_boundaries():
    # The trap this guards against: "search engine" must NOT match
    # inside "research engineer", and "search" must NOT match "research".
    assert text_has("built a search engine for products", ["search engine"])
    assert not text_has("ai research engineer at a lab", ["search engine"])
    assert not text_has("published research papers", ["search"])
    # Phrases with punctuation still match as whole tokens.
    assert text_has("worked on re-rank pipelines", ["re-rank"])
    assert text_has("did a/b test analysis", ["a/b test"])


def test_text_has_is_case_insensitive_on_pre_lowered_text():
    # Contract: callers pass already-lowercased text; terms are lowercase.
    assert text_has("Information Retrieval focus".lower(), ["information retrieval"])


def test_iter_candidates_skips_malformed(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"candidate_id": "CAND_0000001"}\nNOT JSON\n\n{"candidate_id": "CAND_0000002"}\n')
    ids = [c["candidate_id"] for c in iter_candidates(str(p))]
    assert ids == ["CAND_0000001", "CAND_0000002"]


def test_iter_candidates_tolerates_non_utf8_bytes(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_bytes(
        b'{"candidate_id": "CAND_0000001"}\n'
        b'{"candidate_id": "CAND_\xff_bad"}\n'  # invalid byte -> replaced, still valid JSON
        b'{"candidate_id": "CAND_0000002"}\n'
    )
    ids = [c["candidate_id"] for c in iter_candidates(str(path))]
    assert len(ids) == 3
    assert ids[0] == "CAND_0000001"
    assert ids[-1] == "CAND_0000002"


def test_build_narrative_handles_missing_sections():
    assert build_narrative({}) == ""
    assert build_narrative({"profile": {"summary": "x"}, "career_history": None}) == "x"


def test_text_has_empty_inputs_are_false():
    assert text_has("anything at all", []) is False
    assert text_has("", ["ranking"]) is False


def test_pdate_truncates_datetime_strings():
    assert pdate("2024-03-08T10:30:00") == date(2024, 3, 8)
