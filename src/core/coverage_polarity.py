"""Conservative, scope-aware coverage contradiction checks (no model calls)."""
from __future__ import annotations

import re
from enum import Enum


class CoveragePolarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


_ANCHOR = re.compile(r"\b(?:insurance|insured|cover\w*|liability|policy|versicher\w*|gedeckt)\b", re.I)
_PREDICATE = re.compile(
    r"\b(?:does not apply|doesn't apply|do not apply|don't apply|applies|apply|"
    r"(?:is|are) (?:not )?valid|cover(?:ed|s|ing)?|insured|excluded|excludes|"
    r"uncovered|versichert|gedeckt|abgedeckt|ausgeschlossen|übernommen)\b", re.I
)
_NEGATION = re.compile(r"\b(?:not|no|never|without|nicht|kein\w*|ohne)\b", re.I)
_SPLIT = re.compile(r"(?<=[.!?])\s+|[\r\n;]+")
_GENERIC = set("the a an in on at to for of by from with is are be being been was were your our their this that it there insurance policy cover coverage covered covers covering insured fully explicitly valid apply applies does do not no never without excluded excludes uncovered and or also generally nicht kein keine ist sind versicherung versichert gedeckt ausgeschlossen".split())


def coverage_polarity(text: str) -> CoveragePolarity:
    text = text.replace("’", "'")
    if re.search(r"\b(?:may|might|possibly|whether|if|unless|except|only)\b|\?", text, re.I):
        return CoveragePolarity.UNKNOWN
    values = set()
    for match in _PREDICATE.finditer(text):
        predicate = match.group().lower()
        if predicate == "cover" and re.match(r"\s+(?:is|are)\b", text[match.end():], re.I):
            continue  # noun subject in 'cover is excluded', not an assertion
        if "appl" in predicate and not _ANCHOR.search(text):
            continue
        before = text[max(0, match.start() - 35):match.start()]
        # Negation belongs to this predicate, not to a distant clause.
        before = re.split(r"[,;:]|\b(?:and|but)\b", before)[-1]
        negative = bool(_NEGATION.search(before + " " + predicate)) or "n't" in predicate
        if predicate in {"excluded", "excludes", "uncovered", "ausgeschlossen"}:
            if negative:  # 'not excluded' does not establish affirmative coverage.
                return CoveragePolarity.UNKNOWN
            negative = True
        values.add(CoveragePolarity.NEGATIVE if negative else CoveragePolarity.POSITIVE)
    if not values and re.search(r"\bno\s+(?:\w+\s+){0,3}cover(?:age)?\b", text, re.I):
        values.add(CoveragePolarity.NEGATIVE)
    return next(iter(values)) if len(values) == 1 else CoveragePolarity.UNKNOWN


def _scope(text: str) -> set[str]:
    text = text.lower().replace("doesn't", "does not")
    return {word for word in re.findall(r"[a-zäöüß]+", text) if word not in _GENERIC}


def _locations(text: str) -> set[str]:
    # Include the direction: 'outside Kosovo' is not 'in Kosovo'. Case
    # insensitive so lowercasing a question cannot remove the scope guard.
    return {
        ("outside:" if m.group(1).lower() == "outside" else "in:") + m.group(2).lower()
        for m in re.finditer(
            r"\b(in|within|outside|throughout)\s+(?:the\s+)?([a-zäöüß]+)", text, re.I
        )
    }


def _subjects(text: str) -> set[str]:
    return set(re.findall(
        r"\b(?:liability|comprehensive|collision|motor|household|health|legal|"
        r"personal|business|partial|full|third.party)\b", text.lower()
    ))


def _named_entities(text: str) -> set[str]:
    # Conservative proper-name guard, including customers and insurers, not
    # only locations. Unrecognised spelling/case differences stay neutral.
    return {
        word.lower() for word in re.findall(r"\b[A-Z][a-z]+\b", text)
        if word.lower() not in _GENERIC and word.lower() not in _subjects(text)
    }


def coverage_relation(claim: str, evidence: str) -> str:
    """Return support/contradiction/unknown; generic overlap cannot contradict."""
    polarity = coverage_polarity(claim)
    scope = _scope(claim)
    if polarity is CoveragePolarity.UNKNOWN or not scope:
        return "unknown"
    locations = _locations(claim)
    candidates = []
    for statement in _SPLIT.split(evidence):
        other = coverage_polarity(statement)
        if other is CoveragePolarity.UNKNOWN:
            continue
        other_scope = _scope(statement)
        # Mutual scope agreement prevents a general rule from overriding a
        # local exception, or one insurance type from contradicting another.
        if not other_scope or locations != _locations(statement):
            continue
        if _subjects(claim) != _subjects(statement):
            continue
        if _named_entities(claim) != _named_entities(statement):
            continue
        if re.findall(r"\d+", claim) != re.findall(r"\d+", statement):
            continue
        shared = len(scope & other_scope)
        if shared / len(scope) < 0.8 or shared / len(other_scope) < 0.8:
            continue
        candidates.append(other)
    if not candidates or len(set(candidates)) != 1:
        return "unknown"
    return "support" if candidates[0] == polarity else "contradiction"
