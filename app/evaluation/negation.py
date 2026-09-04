"""Negation scope detection.

A NegEx-style algorithm: find negation cues, project a forward scope, and
terminate that scope at conjunctions and assertion verbs. This is what lets
the evaluator tell "no history of diabetes" from "history of diabetes" - the
distinction that lexical overlap is blind to and that this platform exists to
catch.

Deliberately simple and inspectable rather than learned. The full algorithm:

1. Normalize the text and split it into clauses at sentence and clause
   boundaries (`.`, `;`, `,` before a terminator, newline).
2. Within the clause containing the term, look backwards for a negation cue.
3. If a cue is found, check no *terminator* sits between the cue and the term -
   "no chest pain but reports dyspnea" must not mark dyspnea as negated.
4. Reject pseudo-cues ("no change", "no interval change") which look like
   negations but assert continuity.

Known limits, documented rather than hidden: no handling of double negation,
subject scope ("mother has no diabetes"), or hypothetical framing ("if the
patient develops diabetes"). These do not occur in this benchmark's generated
text; they would matter for real notes. See docs/EVALUATION.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.evaluation.normalization import (
    NEGATION_CUES,
    NEGATION_TERMINATORS,
    POST_NEGATION_CUES,
    POST_NEGATION_WINDOW,
    PSEUDO_NEGATION_CUES,
    normalize_text,
)

_CLAUSE_SPLIT = re.compile(r"[.;:\n]|(?<=\s)-\s")


@dataclass(frozen=True)
class NegationResult:
    """Whether a term is negated, and the evidence for that decision."""

    negated: bool
    cue: str | None = None
    clause: str | None = None

    @property
    def evidence(self) -> str:
        if self.clause is None:
            return ""
        return self.clause.strip()


def split_clauses(text: str) -> list[str]:
    """Split normalized text into clauses for scope-bounded cue search."""
    return [clause.strip() for clause in _CLAUSE_SPLIT.split(text) if clause.strip()]


def _terminator_between(segment: str) -> bool:
    """True when an assertion terminator appears in `segment`.

    `segment` is the text between a negation cue and the term. A terminator
    there means the cue's scope closed before reaching the term.
    """
    words = segment.split()
    for terminator in NEGATION_TERMINATORS:
        if " " in terminator:
            if terminator in segment:
                return True
        elif terminator in words:
            return True
    return False


def is_negated(term: str, text: str) -> NegationResult:
    """Determine whether `term` appears negated in `text`.

    Returns the first negated occurrence found. A term appearing both negated
    and affirmed ("denies chest pain ... reports chest pain") resolves to
    negated=False, because an affirmed mention anywhere means the output does
    assert it - which for contradiction detection is the answer that matters.
    """
    normalized_term = normalize_text(term)
    normalized_text = normalize_text(text)
    if not normalized_term or normalized_term not in normalized_text:
        return NegationResult(negated=False)

    found_affirmed = False
    negated_hit: NegationResult | None = None

    for clause in split_clauses(normalized_text):
        start = clause.find(normalized_term)
        while start != -1:
            term_end = start + len(normalized_term)
            post_cue = _find_post_cue(clause[term_end:])
            if post_cue is not None:
                if negated_hit is None:
                    negated_hit = NegationResult(negated=True, cue=post_cue, clause=clause)
                start = clause.find(normalized_term, start + 1)
                continue

            cue, cue_position = _find_cue(clause[:start])
            if cue is None:
                found_affirmed = True
            elif _is_pseudo_negation(clause, cue_position, term_end):
                found_affirmed = True
            elif _terminator_between(clause[cue_position + len(cue) : start]):
                found_affirmed = True
            elif negated_hit is None:
                negated_hit = NegationResult(negated=True, cue=cue.strip(), clause=clause)
            start = clause.find(normalized_term, start + 1)

    if found_affirmed:
        return NegationResult(negated=False)
    return negated_hit or NegationResult(negated=False)


def _find_cue(preceding: str) -> tuple[str | None, int]:
    """The nearest negation cue at the end of `preceding`, and its position.

    Scans for the cue closest to the term so that the innermost scope wins.
    """
    best_cue: str | None = None
    best_position = -1
    for cue in NEGATION_CUES:
        position = preceding.rfind(cue)
        if position > best_position:
            best_cue, best_position = cue, position
    return best_cue, best_position


def _find_post_cue(following: str) -> str | None:
    """A negation cue shortly after the term, which negates it backwards.

    "Chest X-ray was not performed" negates the study even though no cue
    precedes it. Bounded to a short window so a later clause's cue does not
    reach back across the sentence.
    """
    window = following[:POST_NEGATION_WINDOW]
    for cue in POST_NEGATION_CUES:
        if cue in window:
            return cue
    return None


def _is_pseudo_negation(clause: str, cue_position: int, term_end: int) -> bool:
    """True when the cue is part of a phrase that only looks like a negation.

    The span checked runs from the cue through the end of the term, because a
    pseudo-cue typically *contains* the term - "no interval change" is a
    statement of continuity, not a denial of change, and checking only the
    text before the term would never see it.
    """
    span = clause[cue_position:term_end]
    return any(pseudo in span for pseudo in PSEUDO_NEGATION_CUES)


def mentions(term: str, text: str) -> bool:
    """True when `term` appears in `text` at all, negated or not."""
    normalized_term = normalize_text(term)
    return bool(normalized_term) and normalized_term in normalize_text(text)


def containing_sentence(term: str, text: str) -> str | None:
    """The original-cased sentence containing `term`, for evidence quoting.

    Returns text as the model wrote it, not the normalized form, because a
    finding's evidence should show what the output actually said.
    """
    normalized_term = normalize_text(term)
    if not normalized_term:
        return None
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if normalized_term in normalize_text(sentence):
            return sentence.strip()
    return None
