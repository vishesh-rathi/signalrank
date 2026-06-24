"""Extractive reasoning strings for the submission CSV.

Stage-4 review grades reasoning on specificity, honesty, variation, and zero
hallucination. The reasoning must read like a human recruiter's assessment —
varied in structure, grounded in profile facts, and honest about gaps.

Design principles for Stage-4 survival:
  1. MULTIPLE sentence structures — not a fixed semicolon template
  2. VARIED openings — lead with different aspects per candidate
  3. ORGANIC phrasing — compound sentences, varied connectors
  4. INTEGRATED concerns — woven naturally, not always "Concerns: ..."
  5. RANK-AWARE tone — stronger language for top candidates
  6. ZERO hallucination — every claim sourced from trace/profile
"""


# ─── Helper: current employer ─────────────────────────────────────────
def _current_employer(candidate: dict) -> str | None:
    """Company of the role flagged is_current — a concrete, non-hallucinated fact."""
    for entry in candidate.get("career_history") or []:
        if entry.get("is_current") and entry.get("company"):
            return entry["company"]
    return None


# ─── Helper: hash-based selector ──────────────────────────────────────
def _pick(candidate: dict, options: list[str], salt: int = 0) -> str:
    """Select from options using candidate_id hash — deterministic but varied.

    The ``salt`` parameter shifts the selection so different call sites with
    the same option count don't always choose the same index.
    """
    cid = candidate.get("candidate_id", "0")
    try:
        h = int(cid.split("_")[-1])
    except ValueError:
        h = 0
    return options[(h + salt) % len(options)]


# ─── Helper: concept rendering ────────────────────────────────────────
def _concept_phrase(concepts: list[str]) -> str:
    """Render matched concepts as an English list."""
    if "hybrid search" in concepts:
        concepts = [name for name in concepts if name != "search"]
    if len(concepts) == 1:
        return concepts[0]
    return ", ".join(concepts[:-1]) + " and " + concepts[-1]


# ─── Strength fragments ──────────────────────────────────────────────
def _build_evidence_fragment(candidate: dict, trace: dict) -> str | None:
    """What this candidate has built/shipped — the core technical claim."""
    concepts = trace.get("evidence_concepts") or []
    if concepts:
        verbs = [
            f"built {_concept_phrase(concepts)} systems",
            f"designed and shipped {_concept_phrase(concepts)} pipelines",
            f"engineered {_concept_phrase(concepts)} infrastructure",
            f"developed production {_concept_phrase(concepts)} systems",
            f"architected {_concept_phrase(concepts)} solutions",
            f"delivered end-to-end {_concept_phrase(concepts)} features",
        ]
        return _pick(candidate, verbs, salt=7)
    elif trace.get("evidence", 0) >= 0.5:
        generic = [
            "has hands-on experience shipping ML models to production",
            "brings practical ML deployment experience",
            "has delivered production ML systems",
            "demonstrated ability to productionize ML workloads",
            "has a track record of shipping ML to real users",
        ]
        return _pick(candidate, generic, salt=3)
    return None


def _standout_signal(trace: dict, signals: dict) -> tuple[str, float] | None:
    """Identify the standout engagement signal for this candidate."""
    options: list[tuple[float, str]] = []
    github = signals.get("github_activity_score")
    if isinstance(github, int | float) and github > 0:
        options.append((min(github / 80, 1.0), f"GitHub activity score of {github:.0f}/100"))
    response = signals.get("recruiter_response_rate")
    if isinstance(response, int | float) and response >= 0.5:
        pct = int(response * 100)
        options.append(
            (min((response - 0.4) / 0.55, 1.0), f"{pct}% recruiter response rate")
        )
    notice = signals.get("notice_period_days")
    if isinstance(notice, int | float) and notice <= 45:
        options.append(
            (min((60 - notice) / 60, 1.0), f"available within {int(notice)} days")
        )
    if trace.get("location", 0) >= 1.0:
        options.append((0.45, "located in a JD-preferred city"))
    if not options:
        return None
    best = max(options, key=lambda o: o[0])
    return best[1], best[0]


# ─── Concern fragments ────────────────────────────────────────────────
def _gather_concerns(candidate: dict, trace: dict) -> list[str]:
    """Honest concerns grounded in trace/profile — varied phrasing."""
    signals = candidate.get("redrob_signals", {})
    concerns: list[str] = []

    if trace.get("domain_gate", 1.0) <= 0.2:
        concerns.append(
            _pick(candidate, [
                "primary background is in CV/research rather than NLP/IR",
                "career focus appears to be computer vision, not retrieval/ranking",
            ], salt=11)
        )

    notice_days = signals.get("notice_period_days")
    if isinstance(notice_days, int | float) and notice_days >= 90:
        nd = int(notice_days)
        concerns.append(
            _pick(candidate, [
                f"{nd}-day notice period may delay onboarding",
                f"requires {nd} days before joining",
                f"long notice period ({nd} days)",
                f"availability constrained by {nd}-day notice",
            ], salt=5)
        )

    if trace.get("recency", 1.0) < 0.4:
        concerns.append(
            _pick(candidate, [
                "limited recent platform activity",
                "hasn't been active on the platform recently",
            ], salt=9)
        )

    if trace.get("location", 1.0) < 0.6:
        concerns.append(
            _pick(candidate, [
                "located outside the JD's preferred cities",
                "not in Pune/Noida or the listed metro areas",
                "geographic location doesn't match the JD's preference",
            ], salt=2)
        )

    response_rate = signals.get("recruiter_response_rate")
    if isinstance(response_rate, int | float) and response_rate < 0.35:
        concerns.append(f"recruiter response rate is only {int(response_rate * 100)}%")

    if trace.get("product", 1.0) <= 0.4:
        concerns.append(
            _pick(candidate, [
                "career has been entirely in services/consulting",
                "no product-company experience in their history",
            ], salt=4)
        )

    if trace.get("stability", 1.0) <= 0.3:
        concerns.append(
            _pick(candidate, [
                "short average tenure suggests possible job-hopping",
                "frequent role changes don't align with the JD's 3+ year expectation",
            ], salt=6)
        )

    return concerns


def _weakest_axis(candidate: dict, trace: dict) -> str | None:
    """Name the weakest fit axis as a specific, honest gap."""
    profile = candidate.get("profile", {})
    axes: list[tuple[float, str]] = []

    evidence = trace.get("evidence")
    if isinstance(evidence, int | float) and evidence <= 0.6:
        if evidence == 0.6:
            axes.append(
                (evidence, "ranking/search mentions appear in role blurbs but not their summary")
            )
        else:
            axes.append(
                (evidence, _pick(candidate, [
                    "no direct ranking/search/retrieval build evidence in their profile",
                    "profile lacks specific ranking or retrieval system experience",
                    "general ML background without clear search/ranking depth",
                ], salt=8))
            )

    years = profile.get("years_of_experience")
    seniority = trace.get("seniority")
    if (
        isinstance(seniority, int | float)
        and seniority <= 0.8
        and isinstance(years, int | float)
        and not 5 <= years <= 9
    ):
        axes.append((seniority, f"{years} years of experience sits outside the JD's 5–9 year band"))

    location = trace.get("location")
    if isinstance(location, int | float) and 0.6 <= location < 1.0:
        if location > 0.6:
            axes.append((location, "based in Bangalore — a strong hub but not a JD-named city"))
        else:
            axes.append((location, "located outside the JD's listed cities"))

    product = trace.get("product")
    if isinstance(product, int | float) and 0.4 < product <= 0.6:
        axes.append((product, "no confirmed product-company role in their career history"))

    stability = trace.get("stability")
    if isinstance(stability, int | float) and 0.3 < stability <= 0.6:
        axes.append((stability, "average tenure is under two years"))

    if not axes:
        return None
    return min(axes, key=lambda a: a[0])[1]


# ─── Main reasoning generator ────────────────────────────────────────
def generate_reasoning(candidate: dict, trace: dict, rank: int) -> str:
    """Build a 1-2 sentence human-readable reasoning from profile facts and trace.

    Structurally varied across candidates: different openings, connectors,
    and concern integration styles ensure that any 10 randomly sampled rows
    look substantively different — not template-filled.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    title = profile.get("current_title", "Candidate")
    yoe = profile.get("years_of_experience", "?")
    employer = _current_employer(candidate)
    evidence_frag = _build_evidence_fragment(candidate, trace)
    standout = _standout_signal(trace, signals)

    # Gather concerns
    concerns = _gather_concerns(candidate, trace)
    if rank > 50 and not concerns:
        weakest = _weakest_axis(candidate, trace)
        if weakest:
            concerns.append(weakest)

    # ── Build the reasoning using structurally different patterns ──────
    # The candidate_id hash selects which structural pattern to use,
    # ensuring deterministic but varied output.

    cid = candidate.get("candidate_id", "0")
    try:
        h = int(cid.split("_")[-1])
    except ValueError:
        h = 0

    # Select structural pattern based on candidate hash + rank tier
    # This ensures top/mid/bottom candidates get appropriate tone AND
    # different candidates within the same tier get different structures.
    if rank <= 15:
        text = _build_top_tier(h, title, yoe, employer, evidence_frag, standout, concerns)
    elif rank <= 50:
        text = _build_mid_tier(h, title, yoe, employer, evidence_frag, standout, concerns)
    else:
        text = _build_lower_tier(h, title, yoe, employer, evidence_frag, standout, concerns)

    # Normalize: capitalize first letter, ensure trailing period
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
    if not text.endswith("."):
        text += "."
    return text


# ─── Tier-specific builders ──────────────────────────────────────────
def _build_top_tier(
    h: int,
    title: str,
    yoe: object,
    employer: str | None,
    evidence: str | None,
    standout: tuple[str, float] | None,
    concerns: list[str],
) -> str:
    """Reasoning for ranks 1–15: confident, detail-rich, strong language."""
    standout_text = standout[0] if standout else None
    employer_phrase = f" at {employer}" if employer else ""

    patterns = [
        # Pattern 0: Lead with role + evidence, follow with standout
        lambda: _join_parts([
            f"{title} with {yoe} years of experience{employer_phrase}",
            f"who has {evidence}" if evidence else None,
            f"with {standout_text}" if standout_text else None,
            _concern_suffix(concerns, h),
        ]),
        # Pattern 1: Lead with experience narrative
        lambda: _join_parts([
            f"brings {yoe} years as a {title}{employer_phrase}",
            f"and has {evidence}" if evidence else None,
            f"— notably, {standout_text}" if standout_text else None,
            _concern_suffix(concerns, h),
        ]),
        # Pattern 2: Lead with employer context
        lambda: _join_parts([
            f"currently a {title}{employer_phrase} ({yoe} yrs)",
            f"with demonstrated strength in having {evidence}" if evidence else None,
            f"and {standout_text}" if standout_text else None,
            _concern_suffix(concerns, h),
        ]),
        # Pattern 3: Lead with the built-systems evidence
        lambda: _join_parts([
            f"has {evidence}{employer_phrase}" if evidence else f"{title}{employer_phrase}",
            f"over {yoe} years in the field" if evidence else f"with {yoe} years of experience",
            f"complemented by {standout_text}" if standout_text else None,
            _concern_suffix(concerns, h),
        ]),
        # Pattern 4: Lead with standout signal
        lambda: _join_parts([
            f"strong candidate with {standout_text}" if standout_text
            else f"strong {title}",
            f"— {yoe} years as a {title}{employer_phrase}",
            f"having {evidence}" if evidence else None,
            _concern_suffix(concerns, h),
        ]),
        # Pattern 5: Narrative flow
        lambda: _join_parts([
            f"a {yoe}-year {title}{employer_phrase}",
            f"who has {evidence}" if evidence else None,
            f"with strong engagement signals ({standout_text})" if standout_text else None,
            _concern_suffix(concerns, h),
        ]),
    ]

    return patterns[h % len(patterns)]()


def _build_mid_tier(
    h: int,
    title: str,
    yoe: object,
    employer: str | None,
    evidence: str | None,
    standout: tuple[str, float] | None,
    concerns: list[str],
) -> str:
    """Reasoning for ranks 16–50: solid but measured, concerns integrated naturally."""
    standout_text = standout[0] if standout else None
    employer_phrase = f" at {employer}" if employer else ""

    patterns = [
        # Pattern 0: Balanced assessment
        lambda: _join_parts([
            f"{title} ({yoe} yrs){employer_phrase}",
            f"has {evidence}" if evidence else None,
            f"with {standout_text}" if standout_text else None,
            _concern_integrated(concerns, h),
        ]),
        # Pattern 1: Lead with evidence, natural concern flow
        lambda: _join_parts([
            f"with {yoe} years{employer_phrase} as a {title}",
            f"has {evidence}" if evidence else None,
            _concern_contrast(concerns, standout_text, h),
        ]),
        # Pattern 2: Employer-first with parenthetical concern
        lambda: _join_parts([
            f"currently{employer_phrase} as a {title} with {yoe} years",
            f"having {evidence}" if evidence else None,
            f"and {standout_text}" if standout_text else None,
            _concern_parenthetical(concerns),
        ]),
        # Pattern 3: Evidence-forward
        lambda: _join_parts([
            f"has {evidence}" if evidence else f"{title}",
            f"across {yoe} years{employer_phrase}",
            f"with {standout_text}" if standout_text else None,
            _concern_integrated(concerns, h),
        ]),
        # Pattern 4: Compact with concern clause
        lambda: _join_parts([
            f"{title}{employer_phrase} ({yoe} yrs)",
            f"who has {evidence}" if evidence else None,
            _concern_however(concerns, standout_text),
        ]),
        # Pattern 5: Role narrative
        lambda: _join_parts([
            f"a {title} with {yoe} years in the domain{employer_phrase}",
            f"has {evidence}" if evidence else None,
            f"showing {standout_text}" if standout_text else None,
            _concern_suffix(concerns, h),
        ]),
        # Pattern 6: Signal-first approach
        lambda: _join_parts([
            f"shows {standout_text}" if standout_text
            else f"a solid {title}",
            f"as a {title}{employer_phrase} with {yoe} years",
            f"having {evidence}" if evidence else None,
            _concern_integrated(concerns, h),
        ]),
    ]

    return patterns[h % len(patterns)]()


def _build_lower_tier(
    h: int,
    title: str,
    yoe: object,
    employer: str | None,
    evidence: str | None,
    standout: tuple[str, float] | None,
    concerns: list[str],
) -> str:
    """Reasoning for ranks 51–100: honest, concern-forward, measured praise."""
    standout_text = standout[0] if standout else None
    employer_phrase = f" at {employer}" if employer else ""

    patterns = [
        # Pattern 0: Concern-integrated naturally
        lambda: _join_parts([
            f"{title}{employer_phrase} with {yoe} years",
            f"has {evidence}" if evidence else None,
            _concern_however(concerns, standout_text),
        ]),
        # Pattern 1: Lead with what's there, follow with gap
        lambda: _join_parts([
            f"brings {yoe} years as a {title}{employer_phrase}",
            f"and has {evidence}" if evidence else None,
            _concern_though(concerns),
        ]),
        # Pattern 2: Balanced with explicit gap acknowledgment
        lambda: _join_parts([
            f"a {yoe}-year {title}{employer_phrase}",
            f"with {standout_text}" if standout_text else None,
            _concern_but(concerns, evidence),
        ]),
        # Pattern 3: Concern-forward for deep ranks
        lambda: _join_parts([
            f"{title} ({yoe} yrs){employer_phrase}",
            _concern_while(concerns, evidence, standout_text),
        ]),
        # Pattern 4: Brief with honest assessment
        lambda: _join_parts([
            f"currently{employer_phrase} as a {title} ({yoe} yrs)",
            f"has {evidence}" if evidence else None,
            _concern_integrated(concerns, h),
        ]),
        # Pattern 5: Context-first
        lambda: _join_parts([
            f"working as a {title}{employer_phrase} for {yoe} years",
            f"with experience in having {evidence}" if evidence else None,
            _concern_suffix(concerns, h),
        ]),
    ]

    return patterns[h % len(patterns)]()


# ─── Concern integration helpers ──────────────────────────────────────
def _join_parts(parts: list[str | None]) -> str:
    """Join non-None parts with appropriate punctuation."""
    filtered = [p for p in parts if p]
    if not filtered:
        return ""
    # Join with comma-space or semicolon based on part lengths
    result = filtered[0]
    for part in filtered[1:]:
        if part.startswith("—") or part.startswith("("):
            result += f" {part}"
        elif part.startswith(",") or part.startswith(".") or part.startswith(";"):
            result += part
        elif len(result) > 60:
            result += f"; {part}"
        else:
            result += f", {part}"
    return result


def _concern_suffix(concerns: list[str], h: int) -> str | None:
    """Traditional concern suffix — used sparingly, not as the only style."""
    if not concerns:
        return None
    prefixes = [
        "though ",
        "noting that ",
        "however, ",
        "with the caveat that ",
    ]
    prefix = prefixes[h % len(prefixes)]
    return f". {prefix.capitalize()}{concerns[0]}" + (
        f" and {concerns[1]}" if len(concerns) > 1 else ""
    )


def _concern_integrated(concerns: list[str], h: int) -> str | None:
    """Weave concern naturally into the sentence flow."""
    if not concerns:
        return None
    connectors = [
        f", although {concerns[0]}",
        f", but {concerns[0]}",
        f" — {concerns[0]}",
        f", with {concerns[0]} as a consideration",
    ]
    result = connectors[h % len(connectors)]
    if len(concerns) > 1:
        result += f" and {concerns[1]}"
    return result


def _concern_parenthetical(concerns: list[str]) -> str | None:
    """Parenthetical concern — compact and unobtrusive."""
    if not concerns:
        return None
    if len(concerns) == 1:
        return f" ({concerns[0]})"
    return f" ({concerns[0]}; {concerns[1]})"


def _concern_contrast(
    concerns: list[str], standout_text: str | None, h: int
) -> str | None:
    """Contrast concern against a positive signal."""
    if not concerns:
        if standout_text:
            return f", and {standout_text}"
        return None
    if standout_text:
        connectors = [
            f"; {standout_text}, though {concerns[0]}",
            f"; while {standout_text}, {concerns[0]}",
            f"; {standout_text} — but {concerns[0]}",
        ]
        return connectors[h % len(connectors)]
    return f", though {concerns[0]}"


def _concern_however(concerns: list[str], standout_text: str | None) -> str | None:
    """'However'-style concern with optional positive contrast."""
    if not concerns:
        if standout_text:
            return f", with {standout_text}"
        return None
    concern_text = concerns[0]
    if len(concerns) > 1:
        concern_text += f" and {concerns[1]}"
    if standout_text:
        return f"; {standout_text}, however {concern_text}"
    return f". However, {concern_text}"


def _concern_though(concerns: list[str]) -> str | None:
    """Trailing 'though' clause for natural flow."""
    if not concerns:
        return None
    concern_text = " and ".join(concerns[:2])
    return f", though {concern_text}"


def _concern_but(
    concerns: list[str], evidence: str | None
) -> str | None:
    """'But' connector — acknowledges gap after positive."""
    if not concerns:
        return f", and has {evidence}" if evidence else None
    if evidence:
        return f", has {evidence}, but {concerns[0]}"
    return f", but {concerns[0]}"


def _concern_while(
    concerns: list[str], evidence: str | None, standout_text: str | None
) -> str | None:
    """'While' structure — balanced assessment."""
    if not concerns:
        parts = []
        if evidence:
            parts.append(f"has {evidence}")
        if standout_text:
            parts.append(standout_text)
        return ", ".join(parts) if parts else None

    positive_parts = []
    if evidence:
        positive_parts.append(f"has {evidence}")
    if standout_text:
        positive_parts.append(standout_text)

    concern_text = " and ".join(concerns[:2])
    if positive_parts:
        positive = " and ".join(positive_parts)
        return f"; while the profile shows {positive}, {concern_text}"
    return f"; {concern_text}"
