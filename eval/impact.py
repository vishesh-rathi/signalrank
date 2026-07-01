"""Impact harness for GATED scoring experiments (C1/C2).

Measures, but does not apply, a candidate scoring change: it scores the full pool
under baseline and under each variant and reports top-100 churn, top-10 archetype
mix, and the dev-set composite/NDCG. Nothing here mutates committed config — every
variant is applied transiently and restored. Decisions are made from the output.

    uv run python eval/impact.py --artifacts artifacts
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rank import load_artifacts
from ranker import behavioral, config, features
from ranker.embeddings import semantic_fit_scores
from ranker.metrics import composite, ndcg_at_k
from ranker.score import score_candidate
from ranker.util import compile_lexicon, iter_candidates

CANDIDATES_PATH = "candidates.jsonl"
LABELS_PATH = Path("eval/dev_labels.jsonl")
ARCHETYPES = [("ELITE", "search, retrieval, and ranking"),
              ("STRONG", "strong background in nlp"), ("GENERIC", "predictive modeling")]


def archetype(c):
    summ = ((c.get("profile") or {}).get("summary") or "").lower()
    for name, sig in ARCHETYPES:
        if sig in summ:
            return name
    return "other"


def load_labels():
    out = {}
    for raw in LABELS_PATH.read_text().splitlines():
        line = raw.strip()
        if line:
            r = json.loads(line)
            if r.get("tier") is not None:
                out[r["candidate_id"]] = int(r["tier"])
    return out


def score_pool(candidates, semantic_by_id, today):
    scored = []
    for c in candidates:
        cid = c.get("candidate_id") or ""
        s, _ = score_candidate(c, semantic_by_id.get(cid), today)
        scored.append((round(s, 6), cid, c))
    scored.sort(key=lambda r: (-r[0], r[1]))
    return scored


def dev_metrics(candidates, labels, semantic_by_id, today):
    dev = [(score_candidate(c, semantic_by_id.get(c["candidate_id"]), today)[0], c["candidate_id"])
           for c in candidates if c.get("candidate_id") in labels]
    dev.sort(key=lambda r: -r[0])
    predicted = [labels[cid] for _, cid in dev]
    pool = list(labels.values())
    return (
        composite(predicted, pool),
        ndcg_at_k(predicted, pool, 10),
        ndcg_at_k(predicted, pool, 50),
    )


# ── variant mutations (return a restore callable) ──────────────────────────────
def mutate_cf():
    orig = config.STRONG_CONCEPTS["recommendation"]
    config.STRONG_CONCEPTS["recommendation"] = orig + [
        "collaborative filtering",
        "collaborative-filtering",
    ]
    features.rebuild_concept_patterns()

    def restore():
        config.STRONG_CONCEPTS["recommendation"] = orig
        features.rebuild_concept_patterns()

    return restore


def mutate_skill_assessment():
    orig = behavioral.credibility_score
    relevant = compile_lexicon(tuple(config.DOMAIN_POSITIVE))

    def variant(candidate: dict) -> float:
        base = orig(candidate)
        signals = candidate.get("redrob_signals") or {}
        assessments = signals.get("skill_assessment_scores") or {}
        strong = sum(
            1 for name, sc in assessments.items()
            if isinstance(sc, int | float) and sc >= 70 and relevant.search((name or "").lower())
        )
        return min(1.0, base + 0.1 * min(strong, 2))

    # Intentional eval-only monkeypatch of the credibility scorer; restored below.
    behavioral.credibility_score = variant  # ty: ignore[invalid-assignment]

    def restore():
        behavioral.credibility_score = orig

    return restore


def mix_of(top):
    m = {}
    for _, _, c in top[:10]:
        m[archetype(c)] = m.get(archetype(c), 0) + 1
    return m


def mixstr(m):
    return " ".join(f"{m.get(k,0)}{k[0]}" for k in ("ELITE", "STRONG", "GENERIC", "other"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()
    today = config.DATA_AS_OF

    emb, id_index, probe = load_artifacts(args.artifacts)
    if emb is None or probe is None:
        raise SystemExit("artifacts required")
    candidates = list(iter_candidates(CANDIDATES_PATH))
    pool_sem = semantic_fit_scores(emb, probe)
    sem_by_id = {cid: float(pool_sem[r]) for cid, r in id_index.items() if r < len(pool_sem)}
    labels = load_labels()

    base = score_pool(candidates, sem_by_id, today)
    base_ids = [cid for _, cid, _ in base[:100]]
    base_set = set(base_ids)
    bc, b10, b50 = dev_metrics(candidates, labels, sem_by_id, today)
    print(f"BASELINE  dev composite={bc:.4f} ndcg@10={b10:.4f} ndcg@50={b50:.4f} "
          f"top10={mixstr(mix_of(base[:10]))}")

    for name, mut in (("C1 collaborative-filtering->concept", mutate_cf),
                      ("C2 skill_assessment positive", mutate_skill_assessment)):
        restore = mut()
        try:
            v = score_pool(candidates, sem_by_id, today)
            vc, v10, v50 = dev_metrics(candidates, labels, sem_by_id, today)
        finally:
            restore()
        v_ids = [cid for _, cid, _ in v[:100]]
        entered = [cid for cid in v_ids if cid not in base_set]
        left = [cid for cid in base_ids if cid not in set(v_ids)]
        print(f"\n{name}")
        print(f"  dev composite={vc:.4f} ({vc-bc:+.4f})  ndcg@10={v10:.4f} ({v10-b10:+.4f})  "
              f"ndcg@50={v50:.4f} ({v50-b50:+.4f})")
        print(f"  top-10 mix={mixstr(mix_of(v[:10]))}  top-100 churn: "
              f"{len(entered)} entered / {len(left)} left")
        ent_arch = {}
        for _, cid, c in v[:100]:
            if cid in set(entered):
                ent_arch[archetype(c)] = ent_arch.get(archetype(c), 0) + 1
        if entered:
            print(f"  entrants by archetype: {ent_arch}")


if __name__ == "__main__":
    main()
