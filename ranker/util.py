"""Parsing, narrative assembly, and lexicon-matching helpers.

These are the dataset-facing primitives every other ranker module builds on.
They are deliberately defensive: candidate records come from a 100K-line JSONL
file, and one malformed record must never abort a ranking run.
"""

import json
import re
from collections.abc import Iterator
from datetime import date
from functools import lru_cache


def pdate(value: object) -> date | None:
    """Parse an ISO ``YYYY-MM-DD`` string; return None for anything unparseable."""
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def months_between(start: date, end: date) -> int:
    """Whole calendar months from ``start`` to ``end`` (day-of-month ignored)."""
    return (end.year - start.year) * 12 + (end.month - start.month)


def build_narrative(candidate: dict) -> str:
    """Concatenate the candidate's free-text career story.

    Uses summary + headline + every role title and description. The ``skills``
    array is excluded on purpose: in this dataset skill tags are uniform noise,
    and the job description warns that ranking on them is a trap.
    """
    profile = candidate.get("profile") or {}
    parts = [profile.get("summary") or "", profile.get("headline") or ""]
    for entry in candidate.get("career_history") or []:
        parts.append(entry.get("title") or "")
        parts.append(entry.get("description") or "")
    return " ".join(part for part in parts if part).strip()


def build_evidence_texts(candidate: dict) -> tuple[str, str]:
    """(primary, corroboration) sources for the built-evidence feature.

    primary = summary + role titles: the candidate-coherent self-description and
    the roles they actually held. corroboration = role descriptions: in this pool
    those are recycled boilerplate paragraphs shared verbatim across unrelated
    candidates (the same "owned the ranking layer" text appears under different
    candidates and companies), so they can support an evidence claim but never
    establish one alone. The headline is excluded outright — it is an
    aspirational tag-line, not a record of work.
    """
    profile = candidate.get("profile") or {}
    primary = [profile.get("summary") or ""]
    descriptions = []
    for entry in candidate.get("career_history") or []:
        primary.append(entry.get("title") or "")
        descriptions.append(entry.get("description") or "")
    return (
        " ".join(part for part in primary if part).strip(),
        " ".join(part for part in descriptions if part).strip(),
    )


@lru_cache(maxsize=64)
def compile_lexicon(terms: tuple[str, ...]) -> re.Pattern[str]:
    """Compile a whole lexicon into ONE whole-token alternation regex.

    A single search then replaces one search per term — the hot path when
    scoring 100K narratives against ~150 lexicon terms (the difference between
    the ranking step fitting the 5-minute budget and blowing past it).

    Word-boundary matching is load-bearing: substring matching would let
    "search engine" match inside "research engineer", silently promoting the
    research-lab candidates the JD rejects. ``\\b`` alone mishandles terms
    bounded by punctuation (e.g. "a/b test"), so the lookarounds wrap the whole
    alternation explicitly. ``terms`` must be a tuple so the result is cacheable.
    """
    alternation = "|".join(re.escape(term) for term in sorted(set(terms), key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)")


def text_has(text: str, terms: list[str]) -> bool:
    """True if any lexicon ``term`` occurs in ``text`` as a whole token.

    Callers pass lowercased text; terms are lowercase by convention (config.py).
    """
    terms_tuple = tuple(terms)
    if not terms_tuple:
        return False
    return compile_lexicon(terms_tuple).search(text) is not None


def iter_candidates(path: str) -> Iterator[dict]:
    """Stream candidate dicts from a JSONL file, skipping malformed lines.

    Opens with ``errors="replace"`` so a single non-UTF8 byte becomes the
    Unicode replacement character instead of aborting a 100K-line run.
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
