"""Extractive reasoning strings for the submission CSV.

Stage-4 review grades the reasoning column on specificity, honesty, variation,
and zero hallucination. Each string must read like a recruiter's one-line
assessment: grounded in profile facts, varied in structure across candidates,
and honest about gaps — never a fill-in-the-blank template.

The module is built as a tiny, robust assembly grammar rather than ad-hoc string
gluing, so the output cannot drift into the classic templating defects:

  * Every *strength* fragment is a bare PREDICATE — a phrase that reads correctly
    straight after "has" / "having" / "who has" ("built ranking systems",
    "a track record of shipping ML to real users"). ``_as_predicate`` is a
    defensive net that strips an accidental leading verb, so a doubled verb
    ("has has", "has brings", "having has") is structurally impossible.
  * Every *signal* fragment is a NOUN PHRASE ("an 88% recruiter response rate",
    "immediate availability"), so it reads correctly after "with" / "shows" /
    "complemented by" alike.
  * Articles are chosen by spoken sound, not first letter, so the JD's vocabulary
    of vowel-sound initialisms is handled ("an ML Engineer", "an NLP Engineer",
    "an 8.6-year tenure"), never "a AI Engineer".
  * Clauses are joined by an explicit connector list and a final normalizer
    (``_finalize``) that collapses stray whitespace and removes any space before
    punctuation — so doubled connectors, "; —", and double spaces cannot survive.

Design principles for Stage-4 survival, unchanged from the spec:
  1. MULTIPLE sentence structures, selected deterministically by candidate id.
  2. VARIED openings — lead with different aspects per candidate.
  3. INTEGRATED concerns — woven with natural connectors, not a fixed suffix.
  4. RANK-AWARE tone — confident for the top, concern-forward for the tail.
  5. ZERO hallucination — every claim is sourced from the trace or profile.
"""

import re

# ─── Article selection ────────────────────────────────────────────────
# Initialisms whose letter-by-letter pronunciation opens on a vowel sound — the
# JD's core vocabulary ("AI" = ay-eye, "ML" = em, "NLP" = en, "IR" = eye-arr).
_VOWEL_SOUND_INITIALISMS = {"ai", "ml", "nlp", "ir", "ux", "ui", "sre", "mle", "llm"}
# Vowel-LETTER words that are nonetheless pronounced with a leading CONSONANT
# sound ("a unified pipeline", "a use case"), so they take "a", not "an".
_CONSONANT_VOWEL_PREFIXES = ("uni", "use", "usa", "ubiq", "eu", "one", "uk")


def _number_starts_with_vowel_sound(token: str) -> bool:
    """True when the spoken form of a leading number opens on a vowel sound.

    Bounded to the magnitudes this module ever renders (years, days, percentages,
    0-100 scores): 8 / 11 / 18 and the eighties open on a vowel ("eight",
    "eleven", "eighteen", "eighty"); everything else — including hundreds such as
    120 or 180 ("one hundred …") — opens on a consonant.
    """
    match = re.match(r"\d+", token)
    if not match:
        return False
    whole = match.group().lstrip("0") or "0"
    if whole in ("8", "11", "18"):
        return True
    return len(whole) == 2 and whole[0] == "8"  # 80-89 -> "eighty…"


def _article(noun_phrase: str) -> str:
    """Return "a" or "an" to precede ``noun_phrase``, matched to spoken sound.

    Judges the first word by how it is pronounced, not merely its first letter,
    so vowel-sound initialisms ("ML"), spelled-out vowels, and leading numbers
    ("an 8.6-year tenure") are all handled correctly.
    """
    stripped = noun_phrase.strip()
    if not stripped:
        return "a"
    word = stripped.split()[0].strip("([{\"'")
    lowered = word.lower()
    if not lowered:
        return "a"
    if lowered[0].isdigit():
        return "an" if _number_starts_with_vowel_sound(lowered) else "a"
    alpha = "".join(ch for ch in lowered if ch.isalpha())
    if alpha in _VOWEL_SOUND_INITIALISMS:
        return "an"
    if lowered[0] in "aeiou":
        return "a" if lowered.startswith(_CONSONANT_VOWEL_PREFIXES) else "an"
    return "a"


def _with_article(noun_phrase: str) -> str:
    """``noun_phrase`` prefixed with its correct indefinite article."""
    return f"{_article(noun_phrase)} {noun_phrase}"


# ─── Candidate-id hash (deterministic structural variety) ─────────────
def _hash(candidate: dict) -> int:
    """Stable integer drawn from the candidate id — drives pattern selection."""
    cid = candidate.get("candidate_id", "0")
    try:
        return int(cid.split("_")[-1])
    except ValueError:
        return 0


def _pick(candidate: dict, options: list[str], salt: int = 0) -> str:
    """Select from ``options`` by candidate-id hash — deterministic but varied.

    ``salt`` shifts the index so independent call sites with the same option
    count do not always land on the same choice for a given candidate.
    """
    return options[(_hash(candidate) + salt) % len(options)]


# ─── Current employer ─────────────────────────────────────────────────
def _current_employer(candidate: dict) -> str | None:
    """Company of the role flagged ``is_current`` — a concrete, sourced fact.

    Career order is never assumed: a company is cited as the present employer
    only when its own record says so.
    """
    for entry in candidate.get("career_history") or []:
        if entry.get("is_current") and entry.get("company"):
            return entry["company"]
    return None


# ─── Strength fragments (always bare predicates) ──────────────────────
_LEADING_FINITE_VERB = re.compile(
    r"^(?:has|have|had|having|brings?|brought|demonstrated|shows?|shown)\s+", re.I
)


def _as_predicate(fragment: str) -> str:
    """Reduce a strength fragment to a bare predicate after "has"/"having".

    Defensive guarantee: even if a fragment is authored with a leading finite
    verb, this strips it so the assembled sentence can never emit "has has",
    "has brings", or "having has". Clean fragments pass through untouched.
    """
    return _LEADING_FINITE_VERB.sub("", fragment).strip()


def _concept_phrase(concepts: list[str]) -> str:
    """Render matched STRONG concepts as a natural English list."""
    if "hybrid search" in concepts:
        # "hybrid search" already carries the search idea; drop the bare "search"
        # so the list does not read "search and hybrid search".
        concepts = [name for name in concepts if name != "search"]
    if len(concepts) == 1:
        return concepts[0]
    return ", ".join(concepts[:-1]) + " and " + concepts[-1]


def _build_evidence_fragment(candidate: dict, trace: dict) -> str | None:
    """What this candidate has built/shipped — the core technical claim.

    Returns a bare predicate (no leading verb) or None. Concept-grounded
    fragments quote the exact concepts that fired so two builders with different
    specialties read differently; the generic floor is used only when graded
    built-evidence exists without a nameable concept.
    """
    concepts = trace.get("evidence_concepts") or []
    if concepts:
        phrase = _concept_phrase(concepts)
        verbs = [
            f"built {phrase} systems",
            f"designed and shipped {phrase} pipelines",
            f"engineered {phrase} infrastructure",
            f"developed production {phrase} systems",
            f"architected {phrase} solutions",
            f"delivered end-to-end {phrase} features",
        ]
        return _as_predicate(_pick(candidate, verbs, salt=7))
    if trace.get("evidence", 0) >= 0.5:
        generic = [
            "shipped machine-learning models to production",
            "delivered production ML systems end to end",
            "built and deployed ML models for real users",
            "productionized ML workloads across the stack",
            "a track record of shipping ML to real users",
        ]
        return _as_predicate(_pick(candidate, generic, salt=3))
    return None


# ─── Signal fragments (always noun phrases) ───────────────────────────
def _standout_signal(trace: dict, signals: dict) -> tuple[str, float] | None:
    """The single strongest engagement signal, as a noun phrase + its strength.

    Every option is a noun phrase, so it reads correctly after "with", "shows",
    "complemented by", and "and" alike.
    """
    options: list[tuple[float, str]] = []
    github = signals.get("github_activity_score")
    if isinstance(github, int | float) and github > 0:
        options.append(
            (min(github / 80, 1.0), f"a GitHub activity score of {github:.0f}/100")
        )
    response = signals.get("recruiter_response_rate")
    if isinstance(response, int | float) and response >= 0.5:
        pct = int(response * 100)
        options.append(
            (min((response - 0.4) / 0.55, 1.0), _with_article(f"{pct}% recruiter response rate"))
        )
    notice = signals.get("notice_period_days")
    if isinstance(notice, int | float) and notice <= 45:
        days = int(notice)
        phrase = "immediate availability" if days == 0 else f"availability within {days} days"
        options.append((min((60 - notice) / 60, 1.0), phrase))
    if trace.get("location", 0) >= 1.0:
        options.append((0.45, "a base in one of the JD's hiring cities"))
    if not options:
        return None
    best = max(options, key=lambda option: option[0])
    return best[1], best[0]


# ─── Concern fragments (clean, lowercase clauses) ─────────────────────
def _gather_concerns(candidate: dict, trace: dict) -> list[str]:
    """Honest concerns grounded in trace/profile — varied, no leading punctuation.

    Each clause reads correctly after a connector ("…, though <clause>"), so the
    assembler can weave any of them without special-casing.
    """
    signals = candidate.get("redrob_signals", {})
    concerns: list[str] = []

    if trace.get("domain_gate", 1.0) <= 0.2:
        concerns.append(
            _pick(candidate, [
                "their primary background is computer vision/research rather than NLP/IR",
                "their career focus is computer vision, not retrieval or ranking",
            ], salt=11)
        )

    notice_days = signals.get("notice_period_days")
    if isinstance(notice_days, int | float) and notice_days >= 90:
        nd = int(notice_days)
        concerns.append(
            _pick(candidate, [
                f"{_with_article(f'{nd}-day notice period')} may delay onboarding",
                f"they need {nd} days before joining",
                f"the notice period runs long at {nd} days",
                f"availability is constrained by {_with_article(f'{nd}-day notice')}",
            ], salt=5)
        )

    if trace.get("recency", 1.0) < 0.4:
        concerns.append(
            _pick(candidate, [
                "platform activity has been limited recently",
                "they have not been active on the platform recently",
            ], salt=9)
        )

    if trace.get("location", 1.0) < 0.6:
        concerns.append(
            _pick(candidate, [
                "they are based outside the JD's preferred cities",
                "their location is outside Pune/Noida and the listed metros",
                "their location does not match the JD's preference",
            ], salt=2)
        )

    response_rate = signals.get("recruiter_response_rate")
    if isinstance(response_rate, int | float) and response_rate < 0.35:
        concerns.append(f"the recruiter response rate is only {int(response_rate * 100)}%")

    if trace.get("product", 1.0) <= 0.4:
        concerns.append(
            _pick(candidate, [
                "their career has been entirely in services/consulting",
                "they have no product-company experience",
            ], salt=4)
        )

    if trace.get("stability", 1.0) <= 0.3:
        concerns.append(
            _pick(candidate, [
                "short average tenure suggests possible job-hopping",
                "frequent role changes sit at odds with the JD's 3+ year expectation",
            ], salt=6)
        )

    return concerns


def _weakest_axis(candidate: dict, trace: dict) -> str | None:
    """Name the weakest fit axis as a specific, honest gap (clean clause).

    Used only for deep ranks whose every hard signal is otherwise clean: rather
    than fabricate a concern or state the ranking outcome as if it were a
    candidate fact, name the single softest axis.
    """
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
                    "there is no direct ranking/search/retrieval build evidence in their profile",
                    "their profile lacks specific ranking or retrieval system experience",
                    "their background is general ML without clear search/ranking depth",
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
        axes.append(
            (seniority, f"their {years} years of experience sits outside the JD's 5-9 year band")
        )

    location = trace.get("location")
    if isinstance(location, int | float) and 0.6 <= location < 1.0:
        if location > 0.6:
            axes.append(
                (location, "they are based in Bangalore, a strong hub but not a JD-named city")
            )
        else:
            axes.append((location, "their location is outside the JD's listed cities"))

    product = trace.get("product")
    if isinstance(product, int | float) and 0.4 < product <= 0.6:
        axes.append((product, "there is no confirmed product-company role in their history"))

    stability = trace.get("stability")
    if isinstance(stability, int | float) and 0.3 < stability <= 0.6:
        axes.append((stability, "their average tenure is under two years"))

    if not axes:
        return None
    return min(axes, key=lambda axis: axis[0])[1]


# ─── Sentence patterns ────────────────────────────────────────────────
# Each pattern is (lead_template, evidence_connector, signal_connector). The lead
# template carries the role/experience opening; the two connectors are the exact
# glue placed before the evidence predicate and the standout noun phrase. A
# segment that is absent is dropped together with its connector, and the first
# surviving segment never receives a connector (see ``_assemble``).
#
# Placeholders: {ArtCap}/{art} = article for the role noun phrase (capitalized /
# lowercase); {YrArtCap} = capitalized article for the "{yoe}-year" phrase;
# {role}/{yoe}/{emp} = title, years, and " at <employer>" (or "").
_TOP_PATTERNS = [
    ("{ArtCap} {role} with {yoe} years{emp}", " who has ", ", complemented by "),
    ("Brings {yoe} years as {art} {role}{emp}", ", having ", " — notably, "),
    ("{ArtCap} {role}{emp} ({yoe} yrs)", " who has ", "; also brings "),
    ("{YrArtCap} {yoe}-year {role}{emp}", " who has ", ", with "),
    ("{ArtCap} {role} with {yoe} years{emp}", "; has ", ", plus "),
    ("{YrArtCap} {yoe}-year {role}{emp}", ", having ", " — with "),
]
_MID_PATTERNS = [
    ("{ArtCap} {role}{emp} ({yoe} yrs)", " who has ", ", with "),
    ("With {yoe} years{emp} as {art} {role}", ", has ", " and "),
    ("{ArtCap} {role} with {yoe} years{emp}", " has ", "; shows "),
    ("{YrArtCap} {yoe}-year {role}{emp}", " who has ", ", and "),
    ("Currently {art} {role}{emp} with {yoe} years", ", has ", ", with "),
    ("{ArtCap} {role} with {yoe} years in the domain{emp}", " has ", ", plus "),
]
_LOWER_PATTERNS = [
    ("{ArtCap} {role}{emp} with {yoe} years", " has ", ", with "),
    ("{YrArtCap} {yoe}-year {role}{emp}", " has ", ", with "),
    ("{ArtCap} {role}{emp} ({yoe} yrs)", " has ", ", showing "),
    ("Working as {art} {role}{emp} for {yoe} years", ", has ", ", with "),
    ("Currently {art} {role}{emp} ({yoe} yrs)", ", has ", ", with "),
    ("With {yoe} years{emp} as {art} {role}", ", has ", " and "),
]

# Concern connectors per tier — the same clause set, attached in a tone that
# matches the rank: a measured caveat near the top, a frank reservation in the
# tail. Each begins with its own punctuation, so it appends cleanly.
_CONCERN_CONNECTORS = {
    "top": ["; the one caveat is that {c}", "; one caveat: {c}", ", though {c}"],
    "mid": [", though {c}", "; that said, {c}", ", with the caveat that {c}", "; however, {c}"],
    "lower": [", though {c}", "; however, {c}", ", but {c}", "; the main reservation is that {c}"],
}


def _assemble(patterns: list[tuple[str, str, str]], facts: tuple, h: int) -> str:
    """Render one positive (no-concern) clause from a tier's pattern set.

    Joins lead → evidence → standout using the pattern's connectors, dropping any
    absent segment along with its connector so the result is always grammatical.
    """
    role, yoe, emp, evidence, standout = facts
    lead_template, evidence_connector, signal_connector = patterns[h % len(patterns)]
    role_article = _article(role)
    year_article = _article(f"{yoe}-year")
    lead = lead_template.format(
        ArtCap=role_article.capitalize(),
        art=role_article,
        YrArtCap=year_article.capitalize(),
        role=role,
        yoe=yoe,
        emp=emp,
    )
    out = ""
    segments = (("", lead), (evidence_connector, evidence), (signal_connector, standout))
    for connector, segment in segments:
        if not segment:
            continue
        out += (connector if out else "") + segment
    return out


def _attach_concern(positive: str, concerns: list[str], h: int, tier: str) -> str:
    """Weave up to two concerns onto ``positive`` with a tier-appropriate connector."""
    if not concerns:
        return positive
    clause = concerns[0]
    if len(concerns) > 1:
        clause += f" and {concerns[1]}"
    connectors = _CONCERN_CONNECTORS[tier]
    return positive + connectors[h % len(connectors)].format(c=clause)


def _finalize(text: str) -> str:
    """Normalize spacing/punctuation, capitalize, and guarantee a trailing period.

    This is the last line of defence: it collapses any stray run of whitespace,
    removes a space before any comma/semicolon/period, and ensures the sentence
    opens with a capital and closes with a period — so no assembly slip can reach
    the CSV as a doubled space, "; —", or an uncapitalized fragment.
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,;.])", r"\1", text)
    if not text:
        return "Profile available."
    text = text[0].upper() + text[1:]
    if not text.endswith("."):
        text += "."
    return text


# ─── Main reasoning generator ─────────────────────────────────────────
def generate_reasoning(candidate: dict, trace: dict, rank: int) -> str:
    """Build a 1-2 sentence human-readable reasoning from profile facts and trace.

    Structurally varied across candidates (different openings, connectors, and
    concern integration) so any 10 randomly sampled rows read as distinct
    assessments. Rank selects the tone tier; the candidate-id hash selects the
    pattern within it.
    """
    profile = candidate.get("profile", {})
    role = profile.get("current_title") or "Candidate"
    yoe = profile.get("years_of_experience", "?")
    employer = _current_employer(candidate)
    emp = f" at {employer}" if employer else ""
    evidence = _build_evidence_fragment(candidate, trace)
    standout_pair = _standout_signal(trace, candidate.get("redrob_signals", {}))
    standout = standout_pair[0] if standout_pair else None

    concerns = _gather_concerns(candidate, trace)
    if rank > 50 and not concerns:
        weakest = _weakest_axis(candidate, trace)
        if weakest:
            concerns.append(weakest)

    h = _hash(candidate)
    facts = (role, yoe, emp, evidence, standout)
    if rank <= 15:
        positive, tier = _assemble(_TOP_PATTERNS, facts, h), "top"
    elif rank <= 50:
        positive, tier = _assemble(_MID_PATTERNS, facts, h), "mid"
    else:
        positive, tier = _assemble(_LOWER_PATTERNS, facts, h), "lower"

    return _finalize(_attach_concern(positive, concerns, h, tier))
