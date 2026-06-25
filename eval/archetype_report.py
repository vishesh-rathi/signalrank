"""Archetype-vs-rank diagnostic: do the dataset's builder tiers surface at the top?

The pool encodes a candidate's true builder tier in the SUMMARY field via three
deliberate templates (career_history descriptions are shuffled boilerplate shared
across candidates, so the summary is the only candidate-coherent tier signal):

  ELITE   (n=21)   "...focus on search, retrieval, and ranking..."
  STRONG  (n=150)  "...strong background in nlp..."
  GENERIC (n=1000) "...predictive modeling, nlp, analytics..." (self-disclaims depth)

A ranking aligned with the JD ("we'd rather see 10 great matches than 1000
maybes") should place available ELITEs first, STRONGs next, and keep the
self-disclaiming GENERICs out of the top-10. This report measures exactly that,
plus the honeypot count, so every scoring change can be checked against the same
yardstick. Run:

    uv run python eval/archetype_report.py --artifacts artifacts
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rank import load_artifacts
from ranker import config
from ranker.honeypot import is_honeypot
from ranker.score import rank_candidates
from ranker.util import iter_candidates

CANDIDATES_PATH = "candidates.jsonl"

# Distinctive substrings of the three summary templates (verified counts:
# 21 / 150 / 1000 in the 100K pool). Matching is on the summary only.
ARCHETYPE_SIGNATURES = {
    "ELITE": "focus on search, retrieval, and ranking",
    "STRONG": "strong background in nlp",
    "SENIOR_ENG": "building systems that connect users with relevant information",
    "GENERIC": "predictive modeling, nlp, analytics",
}


def archetype(candidate: dict) -> str:
    summary = ((candidate.get("profile") or {}).get("summary") or "").lower()
    for name, signature in ARCHETYPE_SIGNATURES.items():
        if signature in summary:
            return name
    return "OTHER"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--candidates", default=CANDIDATES_PATH)
    args = parser.parse_args()

    started = time.time()
    candidates = list(iter_candidates(args.candidates))
    embeddings, id_index, probe_matrix = load_artifacts(args.artifacts)
    ranked = rank_candidates(
        candidates,
        embeddings,
        id_index,
        probe_matrix,
        config.DATA_AS_OF,
        top_n=len(candidates),
    )

    by_id_rank = {row["candidate_id"]: row for row in ranked}
    tags = {c["candidate_id"]: archetype(c) for c in candidates}
    pool = Counter(tags.values())
    print(f"pool: {dict(pool)}   ({time.time() - started:.0f}s)")

    for cut in (10, 50, 100):
        dist = Counter(tags[row["candidate_id"]] for row in ranked[:cut])
        print(
            f"top-{cut}: "
            + "  ".join(
                f"{k}={dist.get(k, 0)}"
                for k in ("ELITE", "STRONG", "SENIOR_ENG", "GENERIC", "OTHER")
            )
        )

    honeypots = sum(1 for row in ranked[:100] if is_honeypot(row["candidate"], config.DATA_AS_OF))
    print(f"honeypots in top-100: {honeypots}")

    # Every ELITE: rank plus the behavioral facts that legitimately bury a trap
    # (unreachable) elite. An AVAILABLE elite deep in the list is a scoring miss;
    # an unavailable one buried is the system working.
    print("\nELITE placement (rank | mult | open_to_work resp_rate | concern):")
    elites = sorted((by_id_rank[cid]["rank"], cid) for cid, tag in tags.items() if tag == "ELITE")
    for rank, cid in elites:
        row = by_id_rank[cid]
        signals = row["candidate"].get("redrob_signals") or {}
        trace = row["trace"]
        mult = trace.get("mult", 0.0)
        resp = signals.get("recruiter_response_rate")
        resp_str = f"{resp:.2f}" if isinstance(resp, (int, float)) else "?"
        weak = "available" if mult >= 0.75 else "low-engagement"
        print(
            f"  {rank:>6}  {cid}  mult={mult:.2f}  "
            f"otw={bool(signals.get('open_to_work_flag'))} resp={resp_str}  [{weak}]"
        )

    # GENERICs that broke into the top-10 — the S1/S2 failure surface.
    intruders = [row for row in ranked[:10] if tags[row["candidate_id"]] == "GENERIC"]
    if intruders:
        print("\nGENERIC in top-10 (rank, id, evidence, concepts):")
        for row in intruders:
            trace = row["trace"]
            print(
                f"  {row['rank']:>3}  {row['candidate_id']}  "
                f"evidence={trace.get('evidence')}  {trace.get('evidence_concepts')}"
            )


if __name__ == "__main__":
    main()
