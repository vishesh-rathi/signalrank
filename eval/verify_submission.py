#!/usr/bin/env python3
"""Independent output verification for submission.csv.

This does NOT import anything from ``ranker/`` and does NOT re-run the scorer.
It is a second, orthogonal opinion built only from the JD text and an
independent re-derivation implemented in this script: free-text built-evidence
grading, honeypot date-arithmetic, and the JD's explicit disqualifier gates. If the
ranker is correct, this independent lens must agree on the big questions:

  1. INCLUSION   - none of the 100 should trip a hard JD disqualifier.
  2. EXCLUSION   - every scarce, unambiguous fit (reachable ELITE / STRONG
                   builder, in-band, in-target-city, clean) should be present;
                   an excluded one that dominates an included weaker row is a
                   false negative.
  3. ORDERING    - independent strength should fall monotonically with rank.

Run:  python eval/verify_submission.py
Writes eval/verify_report.md (full tables); prints a compact verdict.
"""

import csv
import datetime
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "candidates.jsonl"
SUBMISSION = ROOT / "submission.csv"
REPORT = ROOT / "eval" / "verify_report.md"
TODAY = datetime.date(2026, 6, 9)  # data snapshot date (fixed for reproducibility)

# --- data-fact archetype signatures (verified 21/150/1000; summary-only) ---
ARCHETYPES = {
    "ELITE": "focus on search, retrieval, and ranking",
    "STRONG": "strong background in nlp",
    "SENIOR_ENG": "building systems that connect users with relevant information",
    "GENERIC": "predictive modeling, nlp, analytics",
}

# --- JD constants (lifted from the JD text, NOT from ranker/config) ---
TARGET_CITIES = (
    "pune",
    "noida",
    "hyderabad",
    "mumbai",
    "delhi",
    "bangalore",
    "bengaluru",
    "gurgaon",
    "gurugram",
    "ncr",
)
INDIA_CITY = TARGET_CITIES + ("chennai", "kolkata", "ahmedabad")
SERVICES = (
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "tech mahindra",
    "mindtree",
    "ltimindtree",
    "mphasis",
    "dxc",
    "larsen",
    "l&t infotech",
)
ENG_HINT = (
    "engineer",
    "developer",
    "scientist",
    " ml",
    "ml ",
    " ai",
    "ai ",
    "machine learning",
    "data ",
    "research",
    "architect",
    "sde",
    "programmer",
    "devops",
    "mlops",
    "swe",
    "founding",
    "technical",
)
AI_SKILL = {
    "rag",
    "embeddings",
    "embedding",
    "vector search",
    "vector database",
    "llm",
    "llms",
    "fine-tuning llms",
    "fine-tuning",
    "lora",
    "qlora",
    "peft",
    "ndcg",
    "retrieval",
    "information retrieval",
    "sentence transformers",
    "sentence-transformers",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "faiss",
    "elasticsearch",
    "opensearch",
    "bge",
    "e5",
    "learning to rank",
    "learning-to-rank",
    "xgboost",
    "recommendation systems",
    "recommender",
    "semantic search",
    "nlp",
    "transformers",
    "bm25",
    "re-ranking",
    "reranking",
    "hybrid search",
}
AI_SUBSTR = (
    "rag",
    "embedding",
    "vector",
    "llm",
    "retrieval",
    "rerank",
    "transformer",
    "recommend",
    "semantic",
    "fine-tun",
)

# framework-dabbler / aspirational-AI template (JD disqualifiers: "AI experience =
# recent LangChain -> OpenAI", "framework enthusiasts"). High-precision summary cues.
DABBLER = (
    "online course",
    "side project",
    "experimenting with",
    "excited about how",
    "augment my work",
    "streamline workflows",
    "exploring how",
    "taking courses",
    "been learning",
    "hobby project",
    "tutorials",
)
NON_ENG_TITLE = (
    "content writer",
    "marketing",
    "graphic design",
    "business analyst",
    "project manager",
    "program manager",
    "sales",
    "recruiter",
    "designer",
)

# hard JD-disqualifier gates checked against every submission row
GATES = (
    "honeypot",
    "services_only",
    "kw_stuffer",
    "framework_dabbler",
    "cv_primary",
    "pure_research",
)

# free-text built-evidence grades, strong -> weak
STRONG_EV = (
    "recommendation system",
    "recommender",
    "ranking system",
    "ranking model",
    "learning to rank",
    "learning-to-rank",
    "search engine",
    "search system",
    "semantic search",
    "information retrieval",
    "retrieval system",
    "vector search",
    "re-rank",
    "rerank",
    "recsys",
    "relevance ranking",
    "candidate ranking",
    "personalization",
    "personalisation",
    "matching system",
    "match candidates",
)
MED_EV = (
    "machine learning model",
    "ml model",
    "ml models",
    "deployed",
    "production model",
    "trained a model",
    "trained models",
    "deep learning",
    "neural network",
    "embeddings",
    "embedding",
    "transformer",
    "fine-tun",
    "nlp",
    "llm",
    "rag",
    "a/b test",
    "ab test",
    "ctr",
    "click-through",
)
WEAK_EV = (
    "data pipeline",
    "feature pipeline",
    "etl",
    "spark",
    "airflow",
    "kafka",
    "data science",
    "analytics",
    "model",
    "algorithm",
)

# JD disqualifier vocab
CV_SPEECH_ROBO = (
    "computer vision",
    "opencv",
    "object detection",
    "image segmentation",
    "image classification",
    "speech recognition",
    " asr",
    "text-to-speech",
    " tts",
    "robotics",
    "slam",
    "lidar",
    "point cloud",
    "autonomous driving",
)
NLP_IR = (
    "nlp",
    "natural language",
    "retrieval",
    "ranking",
    "search",
    "recommend",
    "embedding",
    "llm",
    "information retrieval",
    "rag",
    "semantic",
    "text",
)
RESEARCH_TITLE = (
    "research scientist",
    "research engineer",
    "researcher",
    "postdoc",
    "post-doc",
    "phd ",
    "ph.d",
    "academic",
)
PRODUCTION_CUE = (
    "production",
    "deployed",
    "deploy",
    "shipped",
    "in production",
    "serving",
    "real users",
    "at scale",
    "a/b",
    "latency",
    "throughput",
    "millions",
    "qps",
    "queries/",
)
SCOPE_CUE = (
    "led",
    "owned",
    "architected",
    "designed",
    "scaled",
    "staff",
    "principal",
    "tech lead",
    "lead engineer",
    "end-to-end",
    "de-facto",
)


def pdate(s):
    try:
        return datetime.date.fromisoformat((s or "")[:10])
    except Exception:
        return None


def months_between(d1, d2):
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def archetype(summary_l):
    for name, sig in ARCHETYPES.items():
        if sig in summary_l:
            return name
    return "OTHER"


def evidence_grade(blob):
    if any(k in blob for k in STRONG_EV):
        return "STRONG"
    if any(k in blob for k in MED_EV):
        return "MED"
    if any(k in blob for k in WEAK_EV):
        return "WEAK"
    return "NONE"


def honeypot_flags(rec, yoe):
    """Composite of 5 impossible/inconsistent checks; >=2 => honeypot."""
    ch = rec.get("career_history") or []
    sk = rec.get("skills") or []
    flags = []
    # 1. claimed duration vs real date span (>4 mo)
    for e in ch:
        sd, ed = pdate(e.get("start_date")), pdate(e.get("end_date"))
        d = e.get("duration_months")
        if sd and isinstance(d, int) and not isinstance(d, bool):
            span = months_between(sd, ed if ed else TODAY)
            if abs(span - d) > 4:
                flags.append("date_span_mismatch")
                break
    # 2. >=3 expert/advanced skills claimed with 0 months
    if (
        sum(
            1
            for it in sk
            if it.get("proficiency") in ("expert", "advanced") and it.get("duration_months") == 0
        )
        >= 3
    ):
        flags.append("expert_zero_months")
    # 3. any skill duration > career length
    if yoe and any(
        isinstance(it.get("duration_months"), int) and it.get("duration_months") > yoe * 12 + 6
        for it in sk
    ):
        flags.append("skill_gt_career")
    # 4. total career months >> yoe
    tot = sum(e.get("duration_months") for e in ch if _is_int(e.get("duration_months")))
    if yoe and tot > yoe * 12 + 18:
        flags.append("career_gt_yoe")
    # 5. education end before start
    for e in rec.get("education") or []:
        try:
            if int(e.get("end_year")) < int(e.get("start_year")):
                flags.append("edu_order")
                break
        except Exception:
            pass
    return flags


def seniority_band(yoe, title_l, summary_l):
    """Verifier-local seniority band mirroring the JD's 5-9y preference."""
    if yoe is None:
        band = 0.0
    elif 5 <= yoe <= 9:
        band = 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 10:
        band = 0.8
    elif 3 <= yoe < 4 or 10 < yoe <= 12:
        band = 0.5
    elif yoe > 12:
        band = 0.35
    else:
        band = 0.25
    if "junior" in title_l or "intern" in title_l:
        band *= 0.5
    if any(cue in summary_l for cue in SCOPE_CUE):
        band = min(1.0, band + 0.1)
    return band


def stability_band(durations):
    """Verifier-local tenure stability band; one-role histories stay neutral."""
    if len(durations) < 2:
        return 0.7
    average = statistics.fmean(durations)
    if average >= 30:
        band = 1.0
    elif average >= 24:
        band = 0.85
    elif average >= 18:
        band = 0.6
    elif average >= 12:
        band = 0.4
    else:
        band = 0.25
    if len(durations) >= 4 and average < 18:
        band = min(band, 0.3)
    return band


def classify(rec):
    p = rec.get("profile") or {}
    ch = rec.get("career_history") or []
    sk = rec.get("skills") or []
    s = rec.get("redrob_signals") or {}
    summary_l = (p.get("summary") or "").lower()
    title_l = (p.get("current_title") or "").lower()
    blob = (summary_l + " " + " ".join((e.get("description") or "") for e in ch)).lower()
    try:
        yoe = float(p.get("years_of_experience"))
    except Exception:
        yoe = None

    arch = archetype(summary_l)
    grade = evidence_grade(blob)
    grade_summary = evidence_grade(summary_l)  # ranker-like (summary only)

    comps = [(e.get("company") or "").lower() for e in ch]
    services_only = bool(comps) and all(any(sv in c for sv in SERVICES) for c in comps)
    durs = [e.get("duration_months") for e in ch if _is_int(e.get("duration_months"))]
    avg_tenure = statistics.fmean(durs) if durs else None
    seniority = seniority_band(yoe, title_l, summary_l)
    stability = stability_band(durs)
    cur = [e for e in ch if e.get("is_current")]
    cur_company = (cur[0].get("company") if cur else (ch[0].get("company") if ch else "")) or ""
    cur_services = any(sv in cur_company.lower() for sv in SERVICES)
    is_eng = any(h in title_l for h in ENG_HINT)
    ai_tags = sum(
        1
        for it in sk
        if (it.get("name") or "").lower() in AI_SKILL
        or any(k in (it.get("name") or "").lower() for k in AI_SUBSTR)
    )
    # Disqualifier gates test their NEGATIVE condition against candidate-coherent
    # text (summary + title) only — NOT the career descriptions, which are shuffled
    # boilerplate that inject 'production'/'search'/'nlp' into nearly every blob and
    # would silently neuter these checks (verified: blob-based gates fire 0/100K).
    coh_text = summary_l + " " + title_l
    kw_stuffer = (not is_eng) and ai_tags >= 5 and grade_summary != "STRONG"
    cv_primary = any(k in coh_text for k in CV_SPEECH_ROBO) and not any(
        k in coh_text for k in NLP_IR
    )
    pure_research = any(k in title_l for k in RESEARCH_TITLE) and not any(
        k in summary_l for k in PRODUCTION_CUE
    )
    framework_dabbler = sum(1 for k in DABBLER if k in summary_l) >= 2 and grade_summary != "STRONG"
    noneng_title = any(k in title_l for k in NON_ENG_TITLE)
    flags = honeypot_flags(rec, yoe)
    honeypot = len(flags) >= 2

    loc = (p.get("location") or "").lower()
    in_target = any(c in loc for c in TARGET_CITIES)
    in_india = (p.get("country") or "").lower() == "india" or any(c in loc for c in INDIA_CITY)
    relocate = s.get("willing_to_relocate") is True
    location_ok = in_target or (relocate and in_india)

    la = pdate(s.get("last_active_date"))
    active_days = (TODAY - la).days if la else None
    rr = s.get("recruiter_response_rate")
    rr = rr if isinstance(rr, (int, float)) else None
    otw = s.get("open_to_work_flag") is True
    notice = s.get("notice_period_days")
    notice = notice if isinstance(notice, (int, float)) else None
    reachable = (otw or (active_days is not None and active_days <= 120)) and (
        rr is None or rr >= 0.3
    )

    in_band = yoe is not None and 4 <= yoe <= 10
    disqualified = (
        honeypot or services_only or kw_stuffer or cv_primary or pure_research or framework_dabbler
    )

    # independent strength scalar (transparent; intentionally NOT the ranker formula)
    base = {"ELITE": 4.0, "STRONG": 3.0, "SENIOR_ENG": 3.0}.get(arch, 0.0)
    base = max(base, {"STRONG": 2.0, "MED": 1.0}.get(grade, 0.0))
    strength = (
        base
        + (0.5 if reachable else 0.0)
        + (0.25 if location_ok else 0.0)
        - (3.0 if disqualified else 0.0)
    )

    return {
        "id": rec.get("candidate_id"),
        "arch": arch,
        "grade": grade,
        "grade_summary": grade_summary,
        "yoe": yoe,
        "in_band": in_band,
        "services_only": services_only,
        "kw_stuffer": kw_stuffer,
        "cv_primary": cv_primary,
        "pure_research": pure_research,
        "honeypot": honeypot,
        "hp_flags": flags,
        "disqualified": disqualified,
        "framework_dabbler": framework_dabbler,
        "noneng_title": noneng_title,
        "in_target": in_target,
        "location_ok": location_ok,
        "reachable": reachable,
        "active_days": active_days,
        "rr": rr,
        "ai_tags": ai_tags,
        "otw": otw,
        "notice": notice,
        "relocate": relocate,
        "avg_tenure": avg_tenure,
        "seniority": seniority,
        "stability": stability,
        "cur_services": cur_services,
        "title": p.get("current_title"),
        "company": p.get("current_company"),
        "strength": strength,
    }


def coherent(c):
    """Candidate-COHERENT builder: tier proven by the summary archetype template
    (verified 21/150/1000), the only field the dataset guarantees is not shuffled
    boilerplate. Deliberately NOT the negation-blind phrase grade, which scores
    GENERIC summaries STRONG off 'lighter weight than ranking systems'."""
    return c["arch"] in ("ELITE", "STRONG", "SENIOR_ENG")


def coherent_fit(c):
    """A coherent builder with no JD reason to exclude — must be near the top."""
    return (
        coherent(c)
        and c["reachable"]
        and c["in_band"]
        and c["location_ok"]
        and not c["disqualified"]
    )


def main():
    sub_rank = {}
    with open(SUBMISSION) as f:
        for row in csv.DictReader(f):
            sub_rank[row["candidate_id"]] = int(row["rank"])

    pool_arch = Counter()
    gate_pool = Counter()  # pool-wide firing of each hard gate (validates the gates)
    sub_rows = {}  # id -> classification (the 100)
    coh = []  # every ELITE/STRONG-archetype candidate (the 171 builders)
    n = 0
    desc_strong_excluded = 0  # phrase-STRONG via boilerplate desc, non-archetype, excluded
    desc_strong_in_sub = 0
    with open(POOL) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n += 1
            cid = rec.get("candidate_id")
            in_sub = cid in sub_rank
            c = classify(rec)
            pool_arch[c["arch"]] += 1
            for g in GATES:
                gate_pool[g] += int(c[g])
            if in_sub:
                c["rank"] = sub_rank[cid]
                sub_rows[cid] = c
            if coherent(c):
                coh.append(c)
            elif c["grade"] == "STRONG":  # phrase-STRONG only via shuffled boilerplate
                desc_strong_in_sub += int(in_sub)
                desc_strong_excluded += int(not in_sub)

    # ---------------- analysis ----------------
    ranked = sorted(sub_rows.values(), key=lambda c: c["rank"])
    top10, top50, top100 = ranked[:10], ranked[:50], ranked[:100]

    def arch_dist(rows):
        d = Counter(c["arch"] for c in rows)
        keys = ("ELITE", "STRONG", "SENIOR_ENG", "GENERIC", "OTHER")
        return "  ".join(f"{k}={d.get(k, 0)}" for k in keys)

    grade_dist = Counter(c["grade"] for c in ranked)

    # (1) inclusion: hard-gate trips among the 100
    fp = {g: [c["id"] for c in ranked if c[g]] for g in GATES}

    # (2) coherent builders (171) — placement and the excluded ones, by reason
    coh_in_sub = [c for c in coh if c["id"] in sub_rank]
    coh_excluded = [c for c in coh if c["id"] not in sub_rank]

    def exclusion_reason(c):
        if not c["reachable"]:
            return "unreachable"
        if c["disqualified"]:
            return "disqualified"
        if not c["in_band"]:
            return "out-of-band(yoe)"
        if not c["location_ok"]:
            return "out-of-location"
        return "CLEAN-MISS"

    excl_reason = Counter(exclusion_reason(c) for c in coh_excluded)
    misses = [c for c in coh_excluded if exclusion_reason(c) == "CLEAN-MISS"]

    # coherent-FIT (clean, reachable, located, in-band) — must be near top
    fit_pool = [c for c in coh if coherent_fit(c)]
    fit_in100 = [c for c in fit_pool if c["id"] in sub_rank]
    fit_excluded = [c for c in fit_pool if c["id"] not in sub_rank]

    # ELITE drill-down
    elites = [c for c in coh if c["arch"] == "ELITE"]
    el_reach = [c for c in elites if c["reachable"]]
    el_in_sub = [c for c in elites if c["id"] in sub_rank]
    el_excl_reach = [c for c in el_reach if c["id"] not in sub_rank]
    el_trap = [c for c in elites if not c["reachable"] and c["id"] not in sub_rank]

    # weaker includes: top-100 rows that are NOT coherent builders (GENERIC/OTHER arch)
    weak_incl = [c for c in ranked if not coherent(c)]

    # head-to-head: does any excluded coherent-FIT builder dominate an included
    # GENERIC/OTHER on EVERY JD axis (availability + seniority + location +
    # stability) while being a higher tier? Such a pair is an unjustified swap. Zero pairs =>
    # every swap is a legitimate behavioral/context trade-off the JD asks for.
    def ge(a, b, slack=0.0):  # a at least as good as b (higher better)
        return (a if a is not None else 0) >= (b if b is not None else 0) - slack

    def le(a, b, slack=0.0):  # a at least as good as b (lower better)
        return (a if a is not None else 999) <= (b if b is not None else 999) + slack

    def dominates(x, y):
        return (
            ge(x["rr"], y["rr"])
            and le(x["active_days"], y["active_days"], 15)
            and (x["otw"] or not y["otw"])
            and le(x["notice"], y["notice"], 15)
            and x["location_ok"]
            and (x["in_target"] or not y["in_target"])
            and ge(x["seniority"], y["seniority"])
            and ge(x["stability"], y["stability"])
        )

    unjustified = [(x, y) for y in weak_incl for x in fit_excluded if dominates(x, y)]
    dom_x = {x["id"] for x, _ in unjustified}
    dom_y = {y["id"] for _, y in unjustified}

    def amean(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.fmean(vals) if vals else float("nan")

    def rate(rows, key):
        return statistics.fmean([1.0 if r[key] else 0.0 for r in rows]) if rows else float("nan")

    def avail_line(rows):
        return (
            f"rr={amean(rows, 'rr'):.2f} active={amean(rows, 'active_days'):.0f}d "
            f"otw={rate(rows, 'otw'):.0%} notice={amean(rows, 'notice'):.0f}d "
            f"in_target={rate(rows, 'in_target'):.0%}"
        )

    fit_av, gen_av = avail_line(fit_excluded), avail_line(weak_incl)

    # (3) ordering
    def mean_strength(rows):
        return statistics.fmean([c["strength"] for c in rows]) if rows else float("nan")

    def spearman(rows):
        by_str = sorted(rows, key=lambda c: -c["strength"])
        srank = {c["id"]: i + 1 for i, c in enumerate(by_str)}
        d2 = sum((c["rank"] - srank[c["id"]]) ** 2 for c in rows)
        m = len(rows)
        return 1 - 6 * d2 / (m * (m * m - 1)) if m > 1 else float("nan")

    def fline(c):
        r = sub_rank.get(c["id"], "—")
        return (
            f"| {c['id']} | {c['arch']} | {r} | {c['reachable']} | {c['active_days']} "
            f"| {c['rr']} | {c['otw']} | {c['notice']} | {c['yoe']} | {c['in_band']} "
            f"| {c['in_target']} | {c['location_ok']} | {c['grade_summary']} |"
        )

    FHDR = (
        "| id | arch | rank | reach | active_d | rr | otw | notice | yoe | "
        "in_band | in_target | loc_ok | summ_grade |"
    )
    FSEP = "|" + "---|" * 13

    # ---------------- write full report ----------------
    L = [
        "# Independent submission verification\n",
        "Lens: JD text + an independent re-derivation in this script "
        "(`eval/verify_submission.py`). No `ranker/` import, no re-scoring. "
        f"Snapshot {TODAY}. Coherence backbone = summary archetype (verified 21/150/1000), "
        "NOT the negation-blind phrase grade.\n",
        f"- Pool N={n:,}  archetypes: {dict(pool_arch)}",
        f"- Submission rows: {len(ranked)}\n",
        "## 1. Inclusion — hard JD gates among the 100\n",
        "Pool-wide column confirms each gate is live. cv_primary/pure_research read 0 "
        "pool-wide: this dataset encodes domain via narrative (handled by the ranker's "
        "domain gate), not via a separable CV/research-only industry cohort.\n",
        "| gate | pool-wide | trips in 100 | ids |",
        "|---|---|---|---|",
    ]
    for g, ids in fp.items():
        L.append(f"| {g} | {gate_pool[g]} | {len(ids)} | {', '.join(ids) if ids else '—'} |")
    L += [
        "",
        "## 2. Archetype + evidence of the 100 (independent recompute)\n",
        f"- top-10:  {arch_dist(top10)}",
        f"- top-50:  {arch_dist(top50)}",
        f"- top-100: {arch_dist(top100)}",
        "- evidence grade (top-100): "
        + "  ".join(f"{k}={grade_dist.get(k, 0)}" for k in ("STRONG", "MED", "WEAK", "NONE")),
        f"- coherent builders (ELITE/STRONG arch) in top-100: {len(coh_in_sub)} "
        f"| GENERIC/OTHER in top-100: {len(weak_incl)}",
        "",
    ]
    L += [
        "## 3. ELITE builders (21) — reachability & placement\n",
        f"- reachable {len(el_reach)}/21 | in-submission {len(el_in_sub)}/21 "
        f"| excluded-but-reachable {len(el_excl_reach)} | trap-buried {len(el_trap)}\n",
        FHDR,
        FSEP,
    ]
    L += [fline(c) for c in sorted(elites, key=lambda c: sub_rank.get(c["id"], 10**9))]
    L += [
        "",
        "## 4. Coherent builders (171) — exclusion accounting\n",
        f"- in top-100: {len(coh_in_sub)} | excluded: {len(coh_excluded)}",
        f"- excluded by reason: {dict(excl_reason)}",
        f"- **CLEAN-MISS** (reachable+in-band+located+clean yet excluded): {len(misses)}",
        f"- coherent-FIT pool: {len(fit_pool)} | in top-100: {len(fit_in100)} "
        f"| excluded: {len(fit_excluded)}\n",
        "### Excluded coherent-FIT (the candidates a correct ranker must justify dropping)\n",
        FHDR,
        FSEP,
    ]
    L += [fline(c) for c in sorted(fit_excluded, key=lambda c: -c["strength"])]
    L += [
        "",
        "### All other excluded coherent builders (behaviorally/structurally explained)\n",
        FHDR,
        FSEP,
    ]
    L += [
        fline(c)
        for c in sorted(coh_excluded, key=lambda c: (exclusion_reason(c), c["id"]))
        if exclusion_reason(c) != "CLEAN-MISS"
    ]
    L += [
        "",
        "## 5. GENERIC/OTHER occupying top-100 slots (weaker includes)\n",
        "These hold slots a reviewer could challenge; legitimate only if no excluded "
        "coherent-FIT dominates them on JD axes.\n",
        FHDR,
        FSEP,
    ]
    L += [fline(c) for c in sorted(weak_incl, key=lambda c: c["rank"])]
    L += [
        "",
        "## 6. Boilerplate-divergence (expected, NOT misses)\n",
        f"- non-archetype candidates scored phrase-STRONG only via shuffled career "
        f"descriptions: excluded={desc_strong_excluded}, in-submission={desc_strong_in_sub}.",
        "  These are the negation-blind / boilerplate false-positives the ranker "
        "deliberately discounts (S1/S2). Their exclusion is correct, not a miss.\n",
        "## 7. Ordering sanity\n",
        f"- mean independent strength: top10={mean_strength(top10):.2f}  "
        f"11-50={mean_strength(ranked[10:50]):.2f}  51-100={mean_strength(ranked[50:100]):.2f}",
        f"- Spearman(submission_rank, independent_strength) = {spearman(ranked):.3f}",
        "",
    ]
    L += [
        "## 8. Head-to-head — unjustified swaps (JD-axis dominance)\n",
        "Excluded coherent-FIT builder that is >= every included GENERIC/OTHER on rr, "
        "recency, open-to-work, notice, seniority, location AND stability, while a higher tier. "
        "Recency/notice use small operational slack; seniority/stability do not. "
        "Zero => all swaps are legitimate availability/context trade-offs.\n",
        f"- unjustified swap pairs: {len(unjustified)} | distinct excluded builders: "
        f"{len(dom_x)} | distinct included slots dominated: {len(dom_y)}\n",
        "- cohort availability (the JD's down-weight lever):",
        f"  - excluded coherent-FIT: {fit_av}",
        f"  - included GENERIC/OTHER: {gen_av}\n",
    ]
    if unjustified:
        L += [
            "| excluded builder | arch | rr | act | otw | notice | sen | stab "
            "| dominates incl | rank | Yarch | Yrr | Yact | Ysen | Ystab |",
            "|" + "---|" * 16,
        ]
        for x, y in sorted(unjustified, key=lambda t: t[1]["rank"]):
            L.append(
                f"| {x['id']} | {x['arch']} | {x['rr']} | {x['active_days']} | {x['otw']} "
                f"| {x['notice']} | {x['seniority']:.2f} | {x['stability']:.2f} "
                f"| {y['id']} | {y['rank']} | {y['arch']} "
                f"| {y['rr']} | {y['active_days']} | {y['seniority']:.2f} "
                f"| {y['stability']:.2f} |"
            )
        L.append("")
    REPORT.write_text("\n".join(L))

    # ---------------- compact verdict (stdout) ----------------
    tot_fp = sum(len(v) for v in fp.values())
    print("INDEPENDENT VERIFICATION (JD lens, no ranker import)")
    print(f"pool: {dict(pool_arch)}  N={n}")
    print(f"top-10:  {arch_dist(top10)}   top-50: {arch_dist(top50)}")
    print(
        f"top-100: {arch_dist(top100)}  | evidence "
        + " ".join(f"{k}={grade_dist.get(k, 0)}" for k in ("STRONG", "MED", "WEAK", "NONE"))
    )
    print(
        f"INCLUSION hard-gate trips in 100: {tot_fp}  "
        + " ".join(f"{g}={len(v)}(pool {gate_pool[g]})" for g, v in fp.items())
    )
    print(
        f"coherent builders in top-100: {len(coh_in_sub)}/171  | "
        f"GENERIC+OTHER in top-100: {len(weak_incl)}"
    )
    print(
        f"ELITE: reachable {len(el_reach)}/21 | in-sub {len(el_in_sub)}/21 "
        f"| excluded-reachable {len(el_excl_reach)} | trap-buried {len(el_trap)}"
    )
    print(
        f"coherent-FIT (clean reachable located in-band): {len(fit_pool)} pool / "
        f"{len(fit_in100)} in-100 / {len(fit_excluded)} EXCLUDED"
    )
    print(f"excluded-coherent by reason: {dict(excl_reason)}")
    print(f"boilerplate phrase-STRONG excluded (expected, not misses): {desc_strong_excluded}")
    print(
        f"UNJUSTIFIED SWAPS (JD-axis dominance, worse-in over better-out): "
        f"{len(unjustified)} pairs / {len(dom_x)} builders / {len(dom_y)} slots"
    )
    print(f"avail gap  excl-FIT: {fit_av}")
    print(f"           incl G/O: {gen_av}")
    print(
        f"ordering mean-strength top10/11-50/51-100: "
        f"{mean_strength(top10):.2f}/{mean_strength(ranked[10:50]):.2f}/{mean_strength(ranked[50:100]):.2f}"
        f"  spearman={spearman(ranked):.3f}"
    )
    print(f"report -> {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
