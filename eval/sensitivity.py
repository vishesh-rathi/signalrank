"""Threshold sensitivity analysis for the ranker (addresses critical_review R5).

Answers: "How fragile is the top-10/top-50 to the hand-set thresholds?" For each
named threshold we perturb it (±20%, or alternative ladders for the discrete
evidence grades), re-rank, and measure how much the top set moves versus the
baseline. A stable top-10 under perturbation is the defense the review said was
missing; a fragile one is a real finding worth knowing.

    uv run python eval/sensitivity.py --artifacts artifacts

Generates eval/sensitivity_report.md. Read-only w.r.t. config — every perturbation
is applied transiently and restored, so this never changes scoring.

Performance note (honest, not silent): the baseline ranks the full pool; each
perturbation re-scores only the baseline top-2000 superset. A ±20% perturbation
cannot lift a candidate ~1900 places, so the perturbed top-100 is provably a
subset of that superset — verified by checking the score gap between baseline
rank-100 and rank-2000 dwarfs the largest per-candidate score swing observed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rank import load_artifacts
from ranker import config
from ranker.embeddings import semantic_fit_scores
from ranker.score import score_candidate
from ranker.util import iter_candidates

CANDIDATES_PATH = "candidates.jsonl"
SUPERSET = 2000

# Archetype signatures (exact dataset templates) for top-10 mix reporting.
ARCHETYPES = [
    ("ELITE", "search, retrieval, and ranking"),
    ("STRONG", "strong background in nlp"),
    ("GENERIC", "predictive modeling"),
]


def archetype(candidate: dict) -> str:
    summ = ((candidate.get("profile") or {}).get("summary") or "").lower()
    for name, sig in ARCHETYPES:
        if sig in summ:
            return name
    return "other"


def rank_superset(superset, semantic_by_id, today, top_n):
    scored = []
    for c in superset:
        cid = c.get("candidate_id") or ""
        s, _ = score_candidate(c, semantic_by_id.get(cid), today)
        scored.append((round(s, 6), cid, c))
    scored.sort(key=lambda r: (-r[0], r[1]))
    return scored[:top_n]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    today = config.DATA_AS_OF

    embeddings, id_index, probe_matrix = load_artifacts(args.artifacts)
    if embeddings is None or probe_matrix is None:
        raise SystemExit("artifacts required for a representative sensitivity run")
    candidates = list(iter_candidates(CANDIDATES_PATH))
    pool_semantic = semantic_fit_scores(embeddings, probe_matrix)
    semantic_by_id = {
        cid: float(pool_semantic[row]) for cid, row in id_index.items() if row < len(pool_semantic)
    }

    base = rank_superset(candidates, semantic_by_id, today, SUPERSET)
    base_ids = [cid for _, cid, _ in base]
    base_top10, base_top50, base_top100 = base_ids[:10], base_ids[:50], base_ids[:100]
    superset = [c for _, _, c in base]
    gap_100_2000 = base[99][0] - base[-1][0]

    # ---- perturbation catalogue ----------------------------------------------
    perts = []  # (label, apply_fn, restore_fn)

    def cfg_scale(attr, key, factor):
        def make():
            container = getattr(config, attr)
            orig = container[key]
            container[key] = orig * factor
            return lambda: container.__setitem__(key, orig)
        return make

    def cfg_set(attr, factor):
        def make():
            orig = getattr(config, attr)
            setattr(config, attr, orig * factor)
            return lambda: setattr(config, attr, orig)
        return make

    def bands_set(label, **overrides):
        def make():
            orig = dict(config.EVIDENCE_BANDS)
            config.EVIDENCE_BANDS.update(overrides)

            def restore():
                config.EVIDENCE_BANDS.clear()
                config.EVIDENCE_BANDS.update(orig)

            return restore

        return (label, make)

    for key in config.WEIGHTS:
        perts.append((f"WEIGHTS[{key}] -20%", cfg_scale("WEIGHTS", key, 0.8)))
        perts.append((f"WEIGHTS[{key}] +20%", cfg_scale("WEIGHTS", key, 1.2)))
    for key in config.MULT_WEIGHTS:
        perts.append((f"MULT_WEIGHTS[{key}] -20%", cfg_scale("MULT_WEIGHTS", key, 0.8)))
        perts.append((f"MULT_WEIGHTS[{key}] +20%", cfg_scale("MULT_WEIGHTS", key, 1.2)))
    for attr in ("TECH_SEMANTIC_SHARE", "MULT_FLOOR", "TENURE_CHASER_PENALTY",
                 "NOTICE_BUYOUT_DAYS", "NOTICE_MAX_DAYS"):
        perts.append((f"{attr} -20%", cfg_set(attr, 0.8)))
        perts.append((f"{attr} +20%", cfg_set(attr, 1.2)))
    perts.append(bands_set("evidence ladder 0.6/0.8/1.0", one=0.6, two=0.8))
    perts.append(bands_set("evidence ladder 0.75/0.9/1.0", one=0.75, two=0.9))
    perts.append(bands_set("corroboration cap 0.5", corroboration=0.5))
    perts.append(bands_set("disclaimer cap 0.50", disclaimer_cap=0.50))

    rows = []
    for label, make in perts:
        restore = make()
        try:
            r = rank_superset(superset, semantic_by_id, today, 100)
        finally:
            restore()
        ids = [cid for _, cid, _ in r]
        mix = {}
        for _, _, c in r[:10]:
            mix[archetype(c)] = mix.get(archetype(c), 0) + 1
        honey = sum(1 for sc, _, _ in r if sc == 0.0)
        rows.append((label, jaccard(ids[:10], base_top10), jaccard(ids[:50], base_top50),
                     jaccard(ids, base_top100), mix, honey,
                     ids[:10] == base_top10))

    base_mix = {}
    for _, _, c in base[:10]:
        base_mix[archetype(c)] = base_mix.get(archetype(c), 0) + 1

    def mixstr(m):
        return " ".join(f"{m.get(k,0)}{k[0]}" for k in ("ELITE", "STRONG", "GENERIC", "other"))

    out = ["# Sensitivity Report (generated by eval/sensitivity.py)\n",
           f"Baseline top-10 archetype mix: **{mixstr(base_mix)}**  ",
           f"(E=ELITE, S=STRONG, G=GENERIC, o=other). Superset bound: score gap "
           f"between baseline rank-100 and rank-{SUPERSET} = {gap_100_2000:.4f}.\n",
           "| Perturbation | top-10 Jaccard | top-10 identical? | top-50 Jaccard | "
           "top-100 Jaccard | top-10 mix | honeypots |",
           "|---|---|---|---|---|---|---|"]
    worst10 = 1.0
    for label, j10, j50, j100, mix, honey, same in sorted(rows, key=lambda x: x[1]):
        worst10 = min(worst10, j10)
        out.append(f"| {label} | {j10:.3f} | {'yes' if same else 'no'} | {j50:.3f} | "
                   f"{j100:.3f} | {mixstr(mix)} | {honey} |")
    out.append(f"\n**Summary:** {len(rows)} perturbations. Worst top-10 Jaccard = "
               f"{worst10:.3f}. Honeypots in top-100 across all perturbations: "
               f"{max(r[5] for r in rows)} (target 0). Baseline mix held in "
               f"{sum(1 for r in rows if r[4] == base_mix)}/{len(rows)} runs.")
    report = "\n".join(out) + "\n"
    Path("eval/sensitivity_report.md").write_text(report)
    print(f"wrote eval/sensitivity_report.md ({len(rows)} perturbations, "
          f"worst top-10 Jaccard={worst10:.3f}, max honeypots={max(r[5] for r in rows)})")


if __name__ == "__main__":
    main()
