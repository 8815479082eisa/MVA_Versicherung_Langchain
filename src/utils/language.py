from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"[A-Za-z\u00c4\u00d6\u00dc\u00df\u00e4\u00f6\u00fc]+")
_CITATION_RE = re.compile(r"\[[^\]]+\]")

_ENGLISH_MARKERS = {
    "a",
    "an",
    "and",
    "applies",
    "are",
    "claim",
    "contract",
    "coverage",
    "covered",
    "deductible",
    "for",
    "from",
    "insurance",
    "is",
    "no",
    "of",
    "the",
    "this",
    "to",
    "under",
    "vehicle",
    "what",
    "which",
    "windshield",
    "with",
    "yes",
}

_GERMAN_MARKERS = {
    "das",
    "der",
    "die",
    "f\u00fcr",
    "gedeckt",
    "gilt",
    "im",
    "in",
    "ist",
    "ja",
    "mit",
    "nein",
    "oder",
    "schaden",
    "selbstbeteiligung",
    "sind",
    "unter",
    "und",
    "versichert",
    "versicherung",
    "vertrag",
    "von",
    "welche",
    "windschutzscheibe",
    "zu",
}


def detect_response_language(text: str, fallback: str = "Unknown") -> str:
    """Detect English or German from the final response text.

    Empty, marker-free, or ambiguous short responses use ``fallback`` instead of
    inferring a language from the query, prompt, or configuration.
    """

    cleaned = _CITATION_RE.sub(" ", text or "")
    tokens = [token.casefold() for token in _TOKEN_RE.findall(cleaned)]
    if not tokens:
        return fallback

    english_score = sum(token in _ENGLISH_MARKERS for token in tokens)
    german_score = sum(token in _GERMAN_MARKERS for token in tokens)
    if any(character in cleaned for character in "\u00c4\u00d6\u00dc\u00df\u00e4\u00f6\u00fc"):
        german_score += 2

    if english_score > german_score and english_score > 0:
        return "English"
    if german_score > english_score and german_score > 0:
        return "German"
    return fallback
