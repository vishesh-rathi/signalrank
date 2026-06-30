"""End-to-end test: rank.py on a synthetic pool produces a valid CSV.

The pool is generated in-process (``build_synthetic_pool``) rather than sliced
from the real ``candidates.jsonl`` — that file is 465 MB and deliberately not
committed, so a fresh clone has no copy of it. Self-generating keeps these tests
green on a clean checkout and exercises the full pipeline against profiles whose
facts we control (varied built-evidence, seniority, location, and engagement
signals; zero honeypots).

Every date is derived from ``config.DATA_AS_OF`` — the dataset's fixed snapshot
date — so the synthetic profiles are reproducible and honor the same
"never call date.today()" invariant the ranker itself does.
"""

import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from ranker import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Concept phrases that fire distinct STRONG_CONCEPTS buckets (config.py). Each
# tuple grades to a different built-evidence depth (1 concept -> 0.7, 2 -> 0.85,
# 3 -> 1.0), so candidates spread across the fit range instead of saturating.
_CONCEPT_SETS = [
    ["ranking systems"],
    ["recommendation engines", "semantic search"],
    ["ranking systems", "vector search", "hybrid search"],
    ["semantic search", "evaluation framework"],
    ["learning to rank"],
    ["recommendation engines", "information retrieval", "ndcg"],
]
# Titles that all match the tech-title coherence gate (features.py).
_TITLES = [
    "ML Engineer",
    "Machine Learning Engineer",
    "Search Engineer",
    "NLP Engineer",
    "Data Scientist",
    "Applied Scientist",
]
# JD-named cities, Bangalore (secondary), and metros — exercises location_score.
_CITIES = ["Pune", "Bangalore", "Hyderabad", "Noida", "Mumbai", "Bengaluru"]
# Product industries only (no services firms), so product_score stays high.
_INDUSTRIES = ["Software", "Fintech", "E-commerce", "SaaS", "AI/ML", "HealthTech"]
# Company names; none appear in config.SERVICES_FIRMS.
_EMPLOYERS = ["Acme Labs", "Nova AI", "Vector Dynamics", "Quill", "Beacon", "Synth"]


def _months_ago(months: int) -> date:
    """Snapshot-relative date ``months`` whole months before ``DATA_AS_OF``.

    Day is pinned to the 1st so ``months_between`` round-trips exactly — the
    honeypot tenure check compares claimed ``duration_months`` against this span,
    and an off-by-a-day would risk a spurious flag.
    """
    base = config.DATA_AS_OF
    index = base.year * 12 + (base.month - 1) - months
    return date(index // 12, index % 12 + 1, 1)


def _make_candidate(i: int) -> dict:
    """One internally-consistent, non-honeypot candidate, varied by index ``i``."""
    concepts = _CONCEPT_SETS[i % len(_CONCEPT_SETS)]
    title = _TITLES[i % len(_TITLES)]
    employer = _EMPLOYERS[i % len(_EMPLOYERS)]
    prior_employer = _EMPLOYERS[(i + 3) % len(_EMPLOYERS)]

    cur_months = (20, 26, 32, 38, 44)[i % 5]
    prior_months = (24, 30, 36, 42)[i % 4]
    total_months = cur_months + prior_months
    # years_of_experience tracks the listed span (rounded), so career_exceeds_life
    # and experience_exceeds_career_span both stay well inside their slack.
    yoe = round(total_months / 12)

    cur_start = _months_ago(cur_months)
    prior_start = _months_ago(total_months)

    summary = (
        f"Senior engineer who built {', '.join(concepts)} in production for real "
        f"users at scale, owning the systems end-to-end."
    )
    return {
        "candidate_id": f"CAND_{i:07d}",
        "profile": {
            "summary": summary,
            "headline": f"{title} | NLP & IR systems",
            "current_title": title,
            "years_of_experience": yoe,
            "location": _CITIES[i % len(_CITIES)],
            "country": "India",
            "current_industry": _INDUSTRIES[i % len(_INDUSTRIES)],
        },
        "career_history": [
            {
                "title": title,
                "description": "Built and operated ML systems serving production traffic.",
                "company": prior_employer,
                "industry": _INDUSTRIES[(i + 2) % len(_INDUSTRIES)],
                "start_date": prior_start.isoformat(),
                "end_date": cur_start.isoformat(),
                "duration_months": prior_months,
                "is_current": False,
            },
            {
                "title": title,
                "description": "Own the ranking and retrieval stack end-to-end.",
                "company": employer,
                "industry": _INDUSTRIES[i % len(_INDUSTRIES)],
                "start_date": cur_start.isoformat(),
                "end_date": None,
                "duration_months": cur_months,
                "is_current": True,
            },
        ],
        "education": [
            {
                "tier": ("tier_1", "tier_2", None)[i % 3],
                "start_year": 2012,
                "end_year": 2016,
            }
        ],
        "skills": [],  # empty: no phantom-expertise or skill-duration inflation
        "redrob_signals": {
            "open_to_work_flag": True,
            "notice_period_days": (15, 30, 45, 60, 90)[i % 5],
            "willing_to_relocate": i % 2 == 0,
            "recruiter_response_rate": (0.9, 0.75, 0.6, 0.85, 0.5)[i % 5],
            "avg_response_time_hours": (12, 24, 48, 72)[i % 4],
            "last_active_date": _months_ago(i % 5).isoformat(),
            "github_activity_score": (60, 40, 0, 80, 20)[i % 5],
            "verified_email": True,
            "verified_phone": i % 2 == 0,
            "linkedin_connected": True,
            "skill_assessment_scores": {},
        },
    }


def build_synthetic_pool(count: int = 30) -> list[dict]:
    """A deterministic pool of valid, varied, honeypot-free candidates."""
    return [_make_candidate(i) for i in range(1, count + 1)]


def write_synthetic_pool(tmp_path: Path, count: int = 30) -> Path:
    """Write the synthetic pool to a JSONL file and return its path."""
    jsonl_path = tmp_path / "synthetic_pool.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as outfile:
        for candidate in build_synthetic_pool(count):
            outfile.write(json.dumps(candidate) + "\n")
    return jsonl_path


def run_ranker(jsonl_path: Path, out_path: Path, top: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "rank.py"),
            "--candidates",
            str(jsonl_path),
            "--out",
            str(out_path),
            "--artifacts",
            str(out_path.parent / "no_such_dir"),  # forces lexical-only
            "--top",
            str(top),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


def test_produces_exact_header_and_row_count(tmp_path):
    out_path = tmp_path / "team_test.csv"
    result = run_ranker(write_synthetic_pool(tmp_path), out_path, top=20)
    assert result.returncode == 0, result.stderr
    assert "lexical-only" in result.stdout  # missing artifacts must be announced
    with out_path.open() as handle:
        assert handle.readline().rstrip("\n") == "candidate_id,rank,score,reasoning"
    rows = list(csv.DictReader(out_path.open()))
    assert len(rows) == 20


def test_ranks_sequential_and_scores_non_increasing(tmp_path):
    out_path = tmp_path / "team_test.csv"
    run_ranker(write_synthetic_pool(tmp_path), out_path, top=20)
    rows = list(csv.DictReader(out_path.open()))
    assert [int(row["rank"]) for row in rows] == list(range(1, 21))
    scores = [float(row["score"]) for row in rows]
    assert scores == sorted(scores, reverse=True)


def test_every_row_has_nonempty_specific_reasoning(tmp_path):
    out_path = tmp_path / "team_test.csv"
    run_ranker(write_synthetic_pool(tmp_path), out_path, top=20)
    rows = list(csv.DictReader(out_path.open()))
    reasonings = [row["reasoning"] for row in rows]
    assert all(len(reasoning) > 20 for reasoning in reasonings)
    # Variation check: identical strings everywhere is a Stage-4 penalty.
    assert len(set(reasonings)) > 10


def test_inconsistent_artifacts_fall_back_to_lexical(tmp_path):
    import numpy as np

    jsonl = write_synthetic_pool(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    # 3 embedding rows but only 1 id -> inconsistent; rank.py must reject and warn.
    np.save(artifacts / "cand_embeddings.npy", np.zeros((3, 4), dtype=np.float16))
    np.save(artifacts / "jd_probe_embeddings.npy", np.zeros((2, 4), dtype=np.float16))
    (artifacts / "cand_ids.json").write_text(json.dumps(["CAND_0000001"]))
    out_path = tmp_path / "team_test.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "rank.py"),
            "--candidates",
            str(jsonl),
            "--out",
            str(out_path),
            "--artifacts",
            str(artifacts),
            "--top",
            "20",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "lexical-only" in result.stdout
    assert "inconsistent" in result.stderr
    assert len(list(csv.DictReader(out_path.open()))) == 20


def test_output_satisfies_validator_tie_break_invariants(tmp_path):
    out_path = tmp_path / "team_test.csv"
    run_ranker(write_synthetic_pool(tmp_path), out_path, top=20)
    rows = list(csv.DictReader(out_path.open()))
    for row in rows:
        assert len(row["score"].split(".")[1]) == 6  # exactly six decimals
    for earlier, later in zip(rows, rows[1:], strict=False):
        assert float(earlier["score"]) >= float(later["score"])
        if earlier["score"] == later["score"]:
            assert earlier["candidate_id"] < later["candidate_id"]  # validator tie-break rule
