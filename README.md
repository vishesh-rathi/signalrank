---
title: SignalRank — Candidate Discovery Sandbox
emoji: 🏆
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Redrob Ranker

**An interpretable, CPU-only top-100 candidate ranker for the  Redrob India Runs Hackathon 2026 : Track 1 - The Data & AI Challenge - Intelligent Candidate Discovery.**

Given 100,000 candidate profiles and a job description for "Senior AI
Engineer — Founding Team", this system identifies the 100 best-fit candidates,
ranks them, and explains every ranking decision — all within 53 seconds on a
single CPU core, using only numpy and the standard library.

---

## The Problem

The Redrob dataset is not a straightforward ranking task. It is an adversarial
puzzle designed to test whether a system *understands* what a job description
means, or merely pattern-matches on keywords.

The pool contains four deliberate traps:

| Trap | How it fools naive rankers |
|---|---|
| **Skill-tag noise** | Every candidate carries 8–15 AI/ML skill tags. The tags are drawn from the same distribution regardless of actual ability — keyword matching promotes random candidates. |
| **~80 honeypot profiles** | Internally impossible facts (8 years at a 3-year-old company, "expert" skills used for 0 months). Ranking >10% of these in the top 100 **disqualifies** the submission. |
| **Recycled boilerplate** | Role descriptions are copy-pasted across unrelated candidates. A candidate who *never* built a ranking system can carry the phrase "owned the ranking layer" in their job history. |
| **Self-disclaiming generics** | 1,000 candidates with real but shallow ML experience whose summaries explicitly say *"lighter on the deep-learning side"*. They have excellent behavioral signals, tempting a ranker to promote them over deeper but less responsive builders. |

The JD itself warns: *"The right answer is not 'find candidates whose skills
section contains the most AI keywords.' That's a trap we've explicitly built
into the dataset."*

---

## The Core Insight

The "right answer" involves reading between the lines:

> **A Tier-5 candidate may not use the words "RAG" or "Pinecone" in their
> profile, but if their career history shows they built a recommendation
> system at a product company, they're a fit.** — Job Description

So the ranker ignores skill tags entirely. Instead, it reads the candidate's
*own words* — their summary and role titles — looking for evidence of
*built systems* in ranking, retrieval, search, and recommendation. And it
treats a *"perfect-on-paper candidate who hasn't logged in for 6 months and
has a 5% recruiter response rate"* as what they actually are: unavailable.

---

## Architecture

The system has two scripts, split by the contest's compute constraints
(5 min, 16 GB RAM, CPU-only, no network):

```
╔═══════════════════════════════════════════════════════════════════════════╗
║ OFFLINE PRECOMPUTE (one-time, network OK)                                 ║
║                                                                           ║
║ 100K candidate narratives ──┐                      ┌─ artifacts/ ───────┐ ║
║                             ├──> bge-small-en-v1.5 ┼─ cand_embeddings.npy ║
║ 8 JD requirement probes ────┘    (sentence-tr.)    ├─ jd_probes.npy       ║
║                                                    └─ cand_ids.json       ║
╚═══════════════════════════════════════════════════════════════════════════╝
                                      │
                            numpy load│(no torch, no network)
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════╗
║ rank.py — RANKING STEP (CPU-only, no network, ~53 s)                      ║
║                                                                           ║
║               ┌───────────────────────────────────────────┐               ║
║               │  [INPUT] candidates.jsonl (100K profiles) │               ║
║               └─────────────────────┬─────────────────────┘               ║
║                                     ▼                                     ║
║ ┌───────────────────────────────────────────────────────────────────────┐ ║
║ │ 1. HONEYPOT FILTER                                                    │ ║
║ │    5 deterministic consistency checks                                 │ ║
║ │    -> impossible profiles forced to score 0                           │ ║
║ └───────────────────────────────────┬───────────────────────────────────┘ ║
║                                     ▼                                     ║
║ ┌───────────────────────────────────────────────────────────────────────┐ ║
║ │ 2. DOMAIN-GATED TECHNICAL FIT                                         │ ║
║ │    Built-evidence graded by concept DEPTH from summary + titles       │ ║
║ │    (negated & aspirational mentions suppressed)                       │ ║
║ │    x domain gate (CV/research-only -> x0.2)                           │ ║
║ │    x title coherence gate (non-tech careers -> x0.2)                  │ ║
║ │    + pool-normalized semantic similarity (20% share)                  │ ║
║ └───────────────────────────────────┬───────────────────────────────────┘ ║
║                                     ▼                                     ║
║ ┌───────────────────────────────────────────────────────────────────────┐ ║
║ │ 3. CONTEXTUAL FEATURES                                                │ ║
║ │    Seniority (5-9 yr sweet spot) - Product-company experience         │ ║
║ │    Location (Pune/Noida preferred) - Tenure stability                 │ ║
║ │    Education tier                                                     │ ║
║ └───────────────────────────────────┬───────────────────────────────────┘ ║
║                                     ▼                                     ║
║ ┌───────────────────────────────────────────────────────────────────────┐ ║
║ │ 4. BEHAVIORAL MULTIPLIER  [0.30 , 1.0] — down-weights only            │ ║
║ │    Availability - Responsiveness - Recency - Credibility              │ ║
║ └───────────────────────────────────┬───────────────────────────────────┘ ║
║                                     ▼                                     ║
║ ┌───────────────────────────────────────────────────────────────────────┐ ║
║ │ 5. SCORE = fit x multiplier                                           │ ║
║ │    sorted by (-score, candidate_id)                                   │ ║
║ └───────────────────────────────────┬───────────────────────────────────┘ ║
║                                     ▼                                     ║
║               ┌───────────────────────────────────────────┐               ║
║               │     [OUTPUT] submission.csv (top 100)     │               ║
║               └───────────────────────────────────────────┘               ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### Why each piece exists

| Component | Defends against |
|---|---|
| **Honeypot filter** (`ranker/honeypot.py`) | ~80 impossible profiles that are tier-0 in the ground truth; >10% in top-100 = disqualification |
| **Built-evidence grading** (`ranker/features.py`) | Skill-tag noise, recycled boilerplate, aspirational/negated keyword mentions |
| **Domain gate** | CV/speech/robotics engineers with strong ML prose but explicitly JD-rejected |
| **Title coherence gate** | 4,164 non-tech professionals (Content Writers, HR Managers) whose summaries mention AI from courses |
| **Self-disclaimer cap** | 1,000 GENERIC candidates whose evidence should grade below non-disclaiming builders |
| **Behavioral multiplier** (`ranker/behavioral.py`) | Great-on-paper but unreachable candidates (low response rate, months-inactive); long notice periods (full credit ≤30d, convex decay beyond — JD: "buy out up to 30 days") |
| **Stability score** (`ranker/features.py`) | Short-average-tenure careers — a smooth retention grade across the whole pool |
| **Title-chaser penalty** (`ranker/features.py`) | The JD's explicit rejection of "switching companies every 1.5 years" — ≥4 roles averaging <18 months scale fit down multiplicatively (×0.75), the enforcement stability's 0.08 weight is too light to deliver alone |
| **Extractive reasoning** (`ranker/reasoning.py`) | Stage-4 manual review penalizes hallucination, templates, and rank-inconsistent tone |

---

## Key Design Decisions & Tradeoffs

### 1. Skills are completely ignored

The `skills[]` array is uniform noise by design. Every candidate carries similar
AI/ML tags regardless of actual depth. The ranker reads the career *narrative*
(summary + role titles) instead — the only candidate-coherent signal in the
dataset.

### 2. Evidence is graded by concept *depth*, not presence

A single mention of "ranking systems" scores 0.7. Two distinct concepts
(e.g., "ranking" + "retrieval") score 0.85. Three or more score 1.0. This
separates the 21 elite hybrid-retrieval builders from the 150 strong
NLP/recommendation builders and the 1,000 self-disclaiming generics — a
flat 1.0-for-any-mention would saturate the top cohort and let behavioral
signals decide NDCG@10 instead of technical depth.

### 3. Semantic similarity is a 20% minority share

The embedding cosine similarities span only ~[0.60, 0.69] across the whole
pool — every plausible builder is indistinguishable at ~0.99 percentile.
Pool-normalized semantic joins the technical blend at a 1:4 ratio: enough
recall to float plain-language builders the lexicon misses, not enough to
dilute the graded evidence that actually separates tiers.

### 4. Behavioral signals multiply, never add

`score = fit × behavioral_multiplier` where the multiplier lives in
[0.30, 1.0]. This ensures that poor engagement sinks a profile but good
engagement can *never* lift a weak technical fit above a strong one. This is
exactly the JD's stated priority: *"skills-fit vs. actually available."*

### 5. Role descriptions corroborate, never establish

Career descriptions in this dataset are recycled boilerplate shared verbatim
across candidates. A candidate might carry "owned the ranking layer" in their
role history without ever having built one. So description-only evidence caps
at 0.6 (corroboration), while summary + title evidence can reach 1.0.

### 6. Two-script architecture for the compute constraint

Embedding 100K narratives with a transformer would blow the 5-minute CPU
budget. Instead, `precompute.py` runs *once* offline (network + GPU OK) and
writes fp16 `.npy` artifacts (~75 MB total, committed to the repo). The
ranking step (`rank.py`) loads these with numpy alone — no torch, no network,
no GPU. If the artifacts are missing, it degrades gracefully to lexical-only
scoring.

### 7. Notice period: full credit inside the buy-out window, convex decay beyond

The JD says it will "buy out up to 30 days" and that "30+ day notice candidates
are still in scope but the bar gets higher." So the availability sub-score gives
**full credit to any notice ≤30 days** and then decays the notice factor
*convexly* (squared remaining-fraction) to zero at the dataset's 180-day ceiling.
A 90-day notice scores 0.36 (was 0.50 under a linear decay) and a 120-day notice
0.16 (was 0.33) — long-notice candidates stay in scope but must clear a higher
bar on every other signal, rather than being excluded outright.

### 8. Title-chasers are down-weighted multiplicatively, not just graded

The JD explicitly rejects candidates "switching companies every 1.5 years." The
graded `stability_score` carries this signal smoothly but at weight 0.08 cannot,
on its own, keep a strong-fit chaser out of the top ranks. So the *unambiguous*
pattern — four or more roles averaging under 18 months — applies a fixed
multiplicative penalty (×0.75) to the blended fit, mirroring how the JD's other
named rejections (CV/speech via the domain-title gate, consulting-only via
product) are enforced. It is a down-weight, never a honeypot-style zero, and is
inert for every candidate who does not match the pattern.

---

## Results

Independently verified by `eval/verify_submission.py` (imports nothing from
`ranker/`, rebuilds every metric from scratch):

| Metric | Value |
|---|---|
| **Top-10 composition** | 7 ELITE, 3 STRONG, **0 GENERIC** |
| **Top-50 composition** | 12 ELITE, 35 STRONG, 2 SENIOR_ENG, 1 GENERIC |
| **Top-100 composition** | 13 ELITE, 60 STRONG, 5 SENIOR_ENG, 22 GENERIC |
| Honeypots in top-100 | **0** (DQ threshold: >10%) |
| Trap candidates in top-100 | **0** services-only, **0** keyword-stuffers, **0** non-tech, **0** title-chasers |
| Unjustified swaps | **0** (no excluded builder dominates an included weaker candidate on all JD axes) |
| Unique reasoning strings | **100/100** |
| Spearman (rank vs. independent strength) | **0.781** |

### Missing ELITE candidates — every exclusion justified

Of 21 ELITE candidates in the pool, 13 are in the submission. The 8 excluded:
- 6 are **unreachable** (response rates 0.07–0.16, inactive 113–214 days,
  not open to work) — exactly the candidates the JD says to down-weight
- 2 are **out of the JD's 5–9-year band** (16.2 and 2.9 years of experience),
  reachable but outside the seniority window the role targets

### The 22 GENERIC in top-100 — legitimate behavioral advantage

The 22 GENERIC-archetype candidates persist on overwhelming engagement, not
evidence: avg recruiter-response 0.82, 100% open-to-work, 91% in target cities,
avg notice 52 days — versus the excluded coherent-FIT builders at 0.65 response,
48% open-to-work, 48% in target cities, 69-day notice. The head-to-head analysis
confirms **0 unjustified swaps**: no excluded builder dominates an included
GENERIC on *all* JD axes simultaneously.

---

## Reproduce the Submission

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.13 (uv installs it).

```bash
# Install dependencies (numpy only for ranking)
uv sync --locked

# Rank — CPU-only, no network, ~53 seconds
uv run python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

This is the **single command** that reproduces the submission CSV. It reads
`candidates.jsonl` from wherever you place it (the file is not committed due
to its 465 MB size — see "Candidate Pool" below), loads precomputed
embedding artifacts from `artifacts/`, and writes the spec-compliant CSV.

The output is **byte-identical** across reruns: all scoring uses a fixed
snapshot date (`config.DATA_AS_OF = 2026-06-09`), never `date.today()`.

### Candidate Pool

`candidates.jsonl` is not committed (465 MB, exceeds GitHub's 100 MB limit).
Stage-3 reproducers: place the hackathon-provided `candidates.jsonl` in the
repository root before running `rank.py`.

---

## Interactive Sandbox Dashboard

A Streamlit application provides an interactive UI to run the ranking pipeline,
inspect individual candidate scores, and export submission CSVs.

**Supported input formats:** JSON, JSONL, CSV, and XLSX.

### Run Locally

```bash
uv run streamlit run app.py
```

This opens the dashboard at `http://localhost:8501`. Select the pre-loaded
sample (`sample_candidates.jsonl`, 100 candidates) or upload a custom file
in any supported format. Click **Run Discovery Pipeline** to produce rankings.

### Run via Docker

```bash
docker build -t redrob-ranker .
docker run -p 7860:7860 redrob-ranker
```

Then open `http://localhost:7860`.

### Deploy to Hugging Face Spaces

The repository is pre-configured for HF Spaces with Docker SDK (see the YAML
frontmatter in this README and the `Dockerfile`). To deploy:

1. **Install and authenticate the HF CLI:**
   ```bash
   uv add huggingface_hub
   hf auth login
   ```
   Enter your HF write-access token when prompted.

2. **Deploy the Space:**
   Use the `hf upload` command. This will automatically create the Space if it doesn't exist, read the `sdk: docker` setting from the YAML frontmatter, and respect the `.gitignore` rules:
   ```bash
   hf upload YOUR_USERNAME/YOUR_SPACE_NAME . --repo-type space
   ```

4. **Wait for build:** HF detects the `Dockerfile`, builds the image, and
   starts Streamlit on port `7860`. The Space URL will be:
   `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

> **Note:** The `app_port: 7860` in the YAML frontmatter tells HF which port
> to proxy. The embedding artifacts are committed (~75 MB), so the ranking
> step reproduces with numpy alone — no torch, no network needed at runtime.

### Rebuild Artifacts from Scratch

To regenerate the embedding artifacts (offline, needs network + torch):

```bash
uv sync --group precompute
uv run --group precompute python precompute.py \
    --candidates ./candidates.jsonl --out artifacts
# On Apple Silicon: add --device mps --batch-size 32
```

---

## Infrastructure Constraints

| Constraint | Budget | Measured | Margin |
|---|---|---|---|
| Wall-clock | 300 s | **~53 s** | 5.6× |
| Peak RAM | 16 GB | **~1.4 GB** | 11× |
| CPU-only | required | ✅ | `rank.py` imports only numpy + stdlib |
| Network OFF | required | ✅ | Precomputed `.npy` artifacts, local files only |
| Disk | 5 GB | **~75 MB** | 66× |
| Determinism | required | ✅ | Byte-identical reruns; fixed `config.DATA_AS_OF`, never `date.today()` |

---

## Evaluation & Tuning

There is no live leaderboard, so the approach is tuned against a
rubric-labeled dev set (`eval/dev_labels.jsonl`) and a full-pool archetype
audit, not by submitting variations.

```bash
# Coordinate-descent weight tuning against the dev set
uv run python eval/tune.py --artifacts artifacts

# Full-pool archetype audit (ELITE/STRONG/GENERIC placement, honeypot count)
uv run python eval/archetype_report.py --artifacts artifacts

# Independent submission verification (no ranker/ import)
uv run python eval/verify_submission.py
```

Tuned weights are reviewed and applied to `ranker/config.py` by hand — never
auto-committed. The dev set is builder-heavy and the tuner reliably overfits
(e.g., zeroing education and seniority axes the JD names explicitly).

---

## Tests

```bash
uv run pytest -q          # 109 unit + end-to-end tests
uv run ruff check .       # lint gate (zero suppressions)
```

The end-to-end test (`tests/test_rank_e2e.py`) generates a synthetic pool,
runs the full pipeline, and validates the output against the contest's format
rules — the same `validate_submission.py` the organizers provide.

---

## Project Layout

```
rank.py               Ranking entrypoint (CPU-only, numpy + stdlib)
precompute.py          Offline embedding-artifact builder (torch)
app.py                 Streamlit sandbox dashboard
Dockerfile             HF Spaces / local Docker deployment
ranker/
  config.py            JD-as-config: lexicons, weights, thresholds
  util.py              Parsing, narrative assembly, lexicon matching
  honeypot.py          Deterministic impossible-profile detection
  features.py          Domain-gated technical fit + contextual features
  behavioral.py        Multiplicative engagement modifier
  embeddings.py        Cosine similarity + pool normalization
  score.py             Score composition + top-N ranking
  reasoning.py         Extractive, per-candidate reasoning strings
  metrics.py           NDCG, MAP, P@k (matching the judges' formula)
eval/
  tune.py              Coordinate-descent weight tuner
  archetype_report.py  Full-pool archetype placement diagnostic
  verify_submission.py Independent output verification (no ranker/ import)
  dev_labels.jsonl     Rubric-labeled dev set
tests/                 One suite per module + end-to-end CSV test
artifacts/             Committed embedding artifacts (~75 MB)
```
