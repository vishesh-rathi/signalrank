# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Package manager is `uv`. All commands use `uv run`.

```bash
# Tests
uv run pytest tests/                          # full suite (114 tests)
uv run pytest tests/test_features.py -k name  # single test

# Lint
uv run ruff check .                           # check
uv run ruff check --fix .                     # auto-fix

# Ranking pipeline
uv run python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Precompute embeddings (offline only — needs network, torch)
uv run python precompute.py

# Evaluation
uv run python eval/verify_submission.py       # regenerates eval/verify_report.md
uv run python eval/archetype_report.py        # tier distribution check after scoring changes

# Streamlit UI
uv run streamlit run app.py

# Validate submission format
uv run python validate_submission.py
```

## Architecture

Two-script design forced by contest constraints (5 min, CPU-only, no network):

**`precompute.py`** — runs once offline (network + GPU OK). Embeds 100K candidate narratives with `bge-small-en-v1.5` via sentence-transformers, writes fp16 `.npy` artifacts to `artifacts/` (committed to repo).

**`rank.py`** — contest entrypoint. Loads numpy artifacts only (no torch). Degrades gracefully to lexical-only scoring if artifacts are missing.

### `ranker/` package modules

| Module | Role |
|---|---|
| `config.py` | JD-as-config layer — all JD-specific lexicons (`STRONG_CONCEPTS`, `MED_PHRASES`), feature weights, thresholds. Swapping this retargets to a different JD. |
| `honeypot.py` | 5 deterministic consistency checks; impossible profiles → score 0 before scoring runs |
| `features.py` | Technical fit: concept-depth grading, domain gate, title coherence gate, title-chaser penalty |
| `behavioral.py` | Multiplier [0.30, 1.0] — availability (notice-period convex decay), responsiveness, recency, credibility |
| `score.py` | Combines fit × behavioral multiplier; sorts by (−score, candidate_id) |
| `reasoning.py` | Extractive reasoning generation for the submission CSV's `reasoning` column |
| `embeddings.py` | Semantic similarity: pool-normalized cosine against JD probes (20% of fit score) |
| `metrics.py` | NDCG and precision metrics against dev labels |
| `util.py` | `iter_candidates()`, `text_has()` (word-boundary matching — never naive substring) |

### `eval/` scripts

- `verify_submission.py` — generates `eval/verify_report.md` (edit the script, not the .md)
- `archetype_report.py` — checks tier distribution; run after any scoring change
- `tune.py` — weight optimization; **never auto-apply its output** (zeroes JD-named axes, overfits)
- `dev_labels.jsonl` — ground-truth labels for NDCG evaluation

## Critical Invariants

**Reproducibility**: Never use `date.today()`. All date math is relative to `config.DATA_AS_OF = date(2026, 6, 9)` — the dataset's fixed snapshot date.

**Skills array is ignored**: `skills[]` is uniform noise by design. All technical fit comes from career narrative (summary + role titles/descriptions).

**Behavioral signals multiply, never add**: `score = fit × multiplier`. Good behavior cannot lift a weak technical fit. The multiplier range is [0.30, 1.0] — it only suppresses.

**Evidence depth grading**: 1 distinct concept → 0.7, 2 → 0.85, 3+ → 1.0. Never flatten to presence/absence — that saturates the top cohort.

**Notice-period curve**: Full credit ≤30 days (JD buy-out window), then squared convex decay to 0 at 180 days. Do not linearize.

**Title-chaser penalty**: ≥4 roles averaging <18 months → ×0.75 multiplicative on fit. The `stability_score` weight (0.08) is too light to carry this signal alone.

**After any scoring change**: run `eval/archetype_report.py` to verify tier distribution (target: ~7 ELITE / ~3 STRONG in top-10, zero honeypots).

**`verify_report.md` is generated**: edit `eval/verify_submission.py`, then run it to regenerate.
