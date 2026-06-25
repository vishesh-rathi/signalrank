"""Rank the candidate pool against the JD and write the submission CSV.

Contest reproduction entrypoint — runs CPU-only, offline, in well under the
5-minute budget:

    uv run python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Requires only numpy + the local ``ranker`` package. Embedding artifacts from
``precompute.py`` are optional: without them the ranker degrades to
lexical-only scoring (and says so) instead of failing.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from ranker import config
from ranker.reasoning import generate_reasoning
from ranker.score import rank_candidates
from ranker.util import iter_candidates


def load_artifacts(
    artifact_dir: str,
) -> tuple[np.ndarray | None, dict[str, int], np.ndarray | None]:
    """Load (embeddings, id->row index, probe matrix); (None, {}, None) if unusable.

    Missing OR inconsistent artifacts (row count != id count, or candidate and
    probe dimensions differ) degrade to lexical-only scoring with a warning,
    rather than silently mis-indexing embeddings onto the wrong candidates.
    """
    directory = Path(artifact_dir)
    cand_path = directory / "cand_embeddings.npy"
    probe_path = directory / "jd_probe_embeddings.npy"
    ids_path = directory / "cand_ids.json"
    if not (cand_path.exists() and probe_path.exists() and ids_path.exists()):
        return None, {}, None

    embeddings = np.load(cand_path)
    probe_matrix = np.load(probe_path)
    candidate_ids = json.loads(ids_path.read_text())
    if embeddings.shape[0] != len(candidate_ids) or embeddings.shape[1] != probe_matrix.shape[1]:
        print(
            f"warning: inconsistent embedding artifacts "
            f"(candidates {embeddings.shape}, probes {probe_matrix.shape}, "
            f"ids {len(candidate_ids)}); falling back to lexical-only scoring",
            file=sys.stderr,
        )
        return None, {}, None
    return embeddings, {cid: row for row, cid in enumerate(candidate_ids)}, probe_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    started = time.time()
    embeddings, id_index, probe_matrix = load_artifacts(args.artifacts)
    mode = "hybrid" if embeddings is not None else "lexical-only (artifacts missing)"

    # Materializing all 100K parsed candidate dicts is the run's dominant memory
    # cost (~1.3 GB peak RSS) — far more than the embedding matrix or the scored
    # list. At an 11x margin under the 16 GB ceiling on a fixed-size pool this is
    # the simplest correct choice; streaming the candidates end-to-end is the
    # lever to reach for only if the pool grows by an order of magnitude.
    candidates = list(iter_candidates(args.candidates))
    if embeddings is not None:
        uncovered = sum(1 for c in candidates if c.get("candidate_id") not in id_index)
        if uncovered:
            print(
                f"warning: {uncovered} of {len(candidates)} candidates have no embedding; "
                f"scored lexical-only",
                file=sys.stderr,
            )
    ranked = rank_candidates(
        candidates, embeddings, id_index, probe_matrix, config.DATA_AS_OF, top_n=args.top
    )

    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in ranked:
            writer.writerow(
                [
                    row["candidate_id"],
                    row["rank"],
                    f"{row['score']:.6f}",
                    generate_reasoning(row["candidate"], row["trace"], row["rank"]),
                ]
            )

    elapsed = time.time() - started
    print(
        f"[{mode}] ranked {len(candidates)} candidates -> {args.out} "
        f"({len(ranked)} rows) in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
