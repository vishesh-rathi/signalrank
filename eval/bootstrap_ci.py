"""Bootstrap confidence intervals on the dev composite and the C1/C2 deltas.

The dev composite (`eval/tune.py`, `eval/impact.py`) is always reported as a single
point estimate over 160 labeled candidates. That hides two things: how much
sampling noise a set that small carries, and whether the C1/C2 "REJECT" calls
are backed by a real effect or by noise smaller than n=160 can resolve. Paired
bootstrap answers both without touching scoring:

    uv run python eval/bootstrap_ci.py --artifacts artifacts

Generates eval/bootstrap_ci_report.md. Read-only w.r.t. config — every mutation is
applied transiently and restored, same pattern as eval/impact.py.

Resampling is by CANDIDATE ID, not rank position: C1/C2 reorder candidates
relative to baseline, so "index 5" means a different candidate under each
system if you resample rank positions. Sampling ids and re-deriving each
system's own rank order from its own scores keeps the pairing valid.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rank import load_artifacts
from ranker import behavioral, config, features
from ranker.embeddings import semantic_fit_scores
from ranker.metrics import composite
from ranker.score import score_candidate
from ranker.util import compile_lexicon, iter_candidates

CANDIDATES_PATH = "candidates.jsonl"
LABELS_PATH = Path("eval/dev_labels.jsonl")
B = 5000
SEED = 1234  # fixed: a reproducible report, not a production ranking decision


def load_labels() -> dict[str, int]:
    out = {}
    for raw in LABELS_PATH.read_text().splitlines():
        line = raw.strip()
        if line:
            row = json.loads(line)
            if row.get("tier") is not None:
                out[row["candidate_id"]] = int(row["tier"])
    return out


# ── variant mutations, identical to eval/impact.py's C1/C2 ─────────────────────
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
            1
            for name, sc in assessments.items()
            if isinstance(sc, int | float) and sc >= 70 and relevant.search((name or "").lower())
        )
        return min(1.0, base + 0.1 * min(strong, 2))

    behavioral.credibility_score = variant  # ty: ignore[invalid-assignment]

    def restore():
        behavioral.credibility_score = orig

    return restore


def scores_by_id(dev_candidates, semantic_by_id, today):
    """candidate_id -> score under whichever scoring function is active right now."""
    return {
        c["candidate_id"]: score_candidate(c, semantic_by_id.get(c["candidate_id"]), today)[0]
        for c in dev_candidates
    }


def composite_for(ids, scores, labels):
    """Composite over exactly this (possibly-resampled, possibly-duplicated) id
    multiset: rank it by this system's own scores, read off tiers in that order.
    The same multiset of tiers is also the ideal-DCG pool (pool order is inert)."""
    pairs = sorted(((scores[cid], labels[cid]) for cid in ids), key=lambda p: -p[0])
    predicted = [tier for _, tier in pairs]
    pool = [labels[cid] for cid in ids]
    return composite(predicted, pool)


def percentile(sorted_vals: list[float], p: float) -> float:
    idx = int(p * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    today = config.DATA_AS_OF

    embeddings, id_index, probe_matrix = load_artifacts(args.artifacts)
    if embeddings is None or probe_matrix is None:
        raise SystemExit("artifacts required for a representative bootstrap")
    labels = load_labels()
    candidates = list(iter_candidates(CANDIDATES_PATH))
    dev_candidates = [c for c in candidates if c.get("candidate_id") in labels]
    pool_semantic = semantic_fit_scores(embeddings, probe_matrix)
    semantic_by_id = {
        cid: float(pool_semantic[row]) for cid, row in id_index.items() if row < len(pool_semantic)
    }

    base_scores = scores_by_id(dev_candidates, semantic_by_id, today)

    variant_scores = {}
    for name, mut in (("C1 collaborative-filtering->concept", mutate_cf),
                      ("C2 skill_assessment positive", mutate_skill_assessment)):
        restore = mut()
        try:
            variant_scores[name] = scores_by_id(dev_candidates, semantic_by_id, today)
        finally:
            restore()

    ids = list(labels.keys())
    n = len(ids)
    base_composite = composite_for(ids, base_scores, labels)

    random.seed(SEED)
    base_boot = []
    delta_boot = {name: [] for name in variant_scores}
    for _ in range(B):
        sample_ids = [ids[random.randrange(n)] for _ in range(n)]
        b_val = composite_for(sample_ids, base_scores, labels)
        base_boot.append(b_val)
        for name, scores in variant_scores.items():
            v_val = composite_for(sample_ids, scores, labels)
            delta_boot[name].append(v_val - b_val)

    base_boot.sort()
    lo, hi = percentile(base_boot, 0.025), percentile(base_boot, 0.975)

    out = [
        "# Bootstrap CI Report (generated by eval/bootstrap_ci.py)\n",
        f"Paired bootstrap over the {n} `eval/dev_labels.jsonl` candidates, B={B}, seed={SEED}.\n",
        f"**Baseline dev composite = {base_composite:.4f}**, 95% CI [{lo:.4f}, {hi:.4f}] "
        f"(width {hi - lo:.4f}).\n",
        "| Variant | point delta | 95% CI | P(delta<=0) | zero in CI? |",
        "|---|---|---|---|---|",
    ]
    for name, deltas in delta_boot.items():
        deltas.sort()
        d_lo, d_hi = percentile(deltas, 0.025), percentile(deltas, 0.975)
        frac_le0 = sum(1 for d in deltas if d <= 0) / len(deltas)
        point = composite_for(ids, variant_scores[name], labels) - base_composite
        in_ci = d_lo <= 0 <= d_hi
        out.append(f"| {name} | {point:+.4f} | [{d_lo:+.4f}, {d_hi:+.4f}] | "
                   f"{frac_le0:.3f} | {'yes' if in_ci else 'no'} |")
    out.append(
        f"\n**Reading this:** a {n}-candidate dev set carries real sampling noise — the "
        f"baseline composite's true value could plausibly be anywhere in the CI above. Where "
        f"zero falls inside a variant's CI, that delta is statistically indistinguishable from "
        f'noise; "REJECT" for such a variant means "not proven to help, and kept out on the '
        f'qualitative JD-alignment argument recorded in eval/impact.py," not "proven harmful."'
    )
    report = "\n".join(out) + "\n"
    Path("eval/bootstrap_ci_report.md").write_text(report)
    print(f"wrote eval/bootstrap_ci_report.md  baseline={base_composite:.4f} "
          f"CI=[{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
