"""End-to-end test: rank.py on the 50-candidate sample produces a valid CSV."""

import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_JSONL = PROJECT_ROOT / "candidates.jsonl"


def write_sample_jsonl(tmp_path: Path) -> Path:
    jsonl_path = tmp_path / "sample.jsonl"
    with CANDIDATES_JSONL.open() as infile, jsonl_path.open("w") as outfile:
        for i, line in enumerate(infile):
            if i >= 50:
                break
            outfile.write(line)
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
    result = run_ranker(write_sample_jsonl(tmp_path), out_path, top=20)
    assert result.returncode == 0, result.stderr
    assert "lexical-only" in result.stdout  # missing artifacts must be announced
    with out_path.open() as handle:
        assert handle.readline().rstrip("\n") == "candidate_id,rank,score,reasoning"
    rows = list(csv.DictReader(out_path.open()))
    assert len(rows) == 20


def test_ranks_sequential_and_scores_non_increasing(tmp_path):
    out_path = tmp_path / "team_test.csv"
    run_ranker(write_sample_jsonl(tmp_path), out_path, top=20)
    rows = list(csv.DictReader(out_path.open()))
    assert [int(row["rank"]) for row in rows] == list(range(1, 21))
    scores = [float(row["score"]) for row in rows]
    assert scores == sorted(scores, reverse=True)


def test_every_row_has_nonempty_specific_reasoning(tmp_path):
    out_path = tmp_path / "team_test.csv"
    run_ranker(write_sample_jsonl(tmp_path), out_path, top=20)
    rows = list(csv.DictReader(out_path.open()))
    reasonings = [row["reasoning"] for row in rows]
    assert all(len(reasoning) > 20 for reasoning in reasonings)
    # Variation check: identical strings everywhere is a Stage-4 penalty.
    assert len(set(reasonings)) > 10


def test_inconsistent_artifacts_fall_back_to_lexical(tmp_path):
    import numpy as np

    jsonl = write_sample_jsonl(tmp_path)
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
    run_ranker(write_sample_jsonl(tmp_path), out_path, top=20)
    rows = list(csv.DictReader(out_path.open()))
    for row in rows:
        assert len(row["score"].split(".")[1]) == 6  # exactly six decimals
    for earlier, later in zip(rows, rows[1:], strict=False):
        assert float(earlier["score"]) >= float(later["score"])
        if earlier["score"] == later["score"]:
            assert earlier["candidate_id"] < later["candidate_id"]  # validator tie-break rule
