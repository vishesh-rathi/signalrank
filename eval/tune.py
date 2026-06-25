"""Coordinate-descent weight tuning against the hand-labeled dev set.

Reads ``eval/dev_labels.jsonl`` (candidate_id -> tier 0-5), scores those
candidates with the current pipeline, and searches config.WEIGHTS for the
combination that maximizes the judges' composite. Run AFTER precompute so
tuning sees the same hybrid (lexical + semantic) scores the final ranking
will use:

    uv run python eval/tune.py --artifacts artifacts

Prints the baseline and tuned composites plus the winning weights; updating
config.WEIGHTS with them is a deliberate manual step (the diff should be
reviewed, not auto-applied).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rank import load_artifacts
from ranker import config
from ranker.embeddings import semantic_fit_scores
from ranker.metrics import composite
from ranker.score import score_candidate
from ranker.util import iter_candidates

CANDIDATES_PATH = "candidates.jsonl"
LABELS_PATH = Path("eval/dev_labels.jsonl")
WEIGHT_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.55, 0.7]
COORDINATE_PASSES = 3


def load_labels() -> dict[str, int]:
    labels: dict[str, int] = {}
    if not LABELS_PATH.exists():
        return labels
    for line in LABELS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("tier") is not None:
            labels[row["candidate_id"]] = int(row["tier"])
    return labels


def evaluate(
    dev_candidates: list[dict],
    labels: dict[str, int],
    semantic_by_id: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Composite achieved on the dev set under a candidate weight vector.

    ``semantic_by_id`` holds each dev candidate's POOL-normalized semantic score
    (percentile over the full 100K pool, not the dev subset) so tuning sees the
    same recall axis the production ranker will.
    """
    config.WEIGHTS = weights
    scored = []
    for candidate in dev_candidates:
        candidate_id = candidate["candidate_id"]
        score, _ = score_candidate(candidate, semantic_by_id.get(candidate_id), config.DATA_AS_OF)
        scored.append((score, labels[candidate_id]))
    scored.sort(key=lambda pair: -pair[0])
    predicted = [tier for _, tier in scored]
    return composite(predicted, list(labels.values()))


def normalized(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    labels = load_labels()
    if not labels:
        raise SystemExit("eval/dev_labels.jsonl is missing or has no filled-in tiers")
    dev_candidates = [
        candidate
        for candidate in iter_candidates(CANDIDATES_PATH)
        if candidate.get("candidate_id") in labels
    ]
    embeddings, id_index, probe_matrix = load_artifacts(args.artifacts)
    mode = "hybrid" if embeddings is not None else "lexical-only"

    # Pool-normalize semantic over the FULL matrix once, then index the dev rows
    # (matches production; normalizing over only the dev subset would shift every
    # percentile).
    semantic_by_id: dict[str, float] = {}
    if embeddings is not None and probe_matrix is not None:
        pool_semantic = semantic_fit_scores(embeddings, probe_matrix)
        semantic_by_id = {
            candidate["candidate_id"]: float(pool_semantic[id_index[candidate["candidate_id"]]])
            for candidate in dev_candidates
            if candidate["candidate_id"] in id_index
        }

    best_weights = dict(config.WEIGHTS)
    best_score = evaluate(dev_candidates, labels, semantic_by_id, dict(best_weights))
    print(f"[{mode}] dev set n={len(dev_candidates)}  baseline composite={best_score:.4f}")

    for pass_number in range(COORDINATE_PASSES):
        for key in best_weights:
            for value in WEIGHT_GRID:
                trial = normalized({**best_weights, key: value})
                trial_score = evaluate(dev_candidates, labels, semantic_by_id, trial)
                if trial_score > best_score:
                    best_score, best_weights = trial_score, trial
        print(f"pass {pass_number + 1}: composite={best_score:.4f}")

    print(f"tuned composite={best_score:.4f}")
    print("tuned WEIGHTS =", {key: round(value, 3) for key, value in best_weights.items()})


if __name__ == "__main__":
    main()
