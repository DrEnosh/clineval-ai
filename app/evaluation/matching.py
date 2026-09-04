"""Layered fact matching.

Naive exact string matching would fail on almost every real summary: a model
writing "CKD stage 3" has retained "Chronic kidney disease stage 3", and
scoring that as an omission would make the evaluator useless. So matching is a
ladder, tried in order, with the level recorded on every result:

| Level | Mechanism | Example |
|---|---|---|
| `EXACT` | Identical strings | "Creatinine" -> "Creatinine" |
| `NORMALIZED` | Equal after case/punctuation/spelling folding | "gastro-oesophageal reflux disease" -> "gastroesophageal reflux disease" |
| `ALIAS` | Curated synonym registry | "CKD" -> "chronic kidney disease stage 3" |
| `PARTIAL` | Enough content tokens present | "heart failure, reduced EF" -> "heart failure with reduced ejection fraction" |
| `NONE` | Not found at any level | |

Recording the level matters as much as the match itself. A finding can then
say "not found at any matching level", and a reviewer can see whether a claimed
retention rested on an exact string or on a partial token overlap. Level also
drives `confidence` on findings: a PARTIAL match is real evidence but weaker
evidence, and the finding says so.

Level 4 - LLM-assisted adjudication - is deliberately absent. `MatchResult`
and `FactMatcher.match` are shaped so an adjudicating matcher can be composed
in front of this one in Phase 4 without changing any evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.enums import MatchLevel
from app.evaluation.negation import NegationResult, containing_sentence, is_negated
from app.evaluation.normalization import normalize_text, surface_forms

# Tokens too generic to carry clinical meaning on their own. A PARTIAL match
# resting only on these would be noise.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "with", "and", "or", "in", "on", "at", "to", "for",
        "is", "was", "were", "are", "has", "have", "had", "no", "not", "patient",
        "history", "stage", "type", "disease", "disorder", "chronic", "acute",
        "mg", "ml", "dl", "l", "g", "mcg", "units", "daily", "oral",
    }
)

# Fraction of a term's content tokens that must appear for a PARTIAL match.
DEFAULT_PARTIAL_THRESHOLD = 0.65


@dataclass
class MatchResult:
    """Whether a term was found in output text, how, and with what evidence."""

    matched: bool
    level: MatchLevel
    term: str
    matched_form: str | None = None
    evidence: str | None = None
    negated: bool = False
    negation_cue: str | None = None
    details: dict[str, object] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """Confidence in this match, by the level it was made at."""
        return {
            MatchLevel.EXACT: 1.0,
            MatchLevel.NORMALIZED: 1.0,
            MatchLevel.ALIAS: 0.9,
            MatchLevel.PARTIAL: 0.7,
            MatchLevel.NONE: 1.0,
        }[self.level]

    @property
    def affirmed(self) -> bool:
        """Matched and asserted (not under a negation cue)."""
        return self.matched and not self.negated


def _content_tokens(text: str) -> list[str]:
    return [token for token in normalize_text(text).split() if token not in _STOPWORDS]


class FactMatcher:
    """Matches source terms against model output text.

    Stateless apart from configuration, so one instance is safely shared by
    every evaluator in a run.
    """

    def __init__(self, partial_threshold: float = DEFAULT_PARTIAL_THRESHOLD):
        if not 0.0 < partial_threshold <= 1.0:
            raise ValueError(f"partial_threshold must be in (0, 1], got {partial_threshold}")
        self.partial_threshold = partial_threshold

    def match(self, term: str, text: str) -> MatchResult:
        """Find `term` in `text`, climbing down the match ladder."""
        if not term or not text or not text.strip():
            return MatchResult(matched=False, level=MatchLevel.NONE, term=term)

        if term in text:
            return self._hit(term, text, term, MatchLevel.EXACT)

        normalized_text = normalize_text(text)
        normalized_term = normalize_text(term)

        if normalized_term and normalized_term in normalized_text:
            return self._hit(term, text, normalized_term, MatchLevel.NORMALIZED)

        for form in sorted(surface_forms(term), key=len, reverse=True):
            if form and form != normalized_term and self._contains_form(normalized_text, form):
                return self._hit(term, text, form, MatchLevel.ALIAS)

        partial = self._partial_match(normalized_term, normalized_text)
        if partial is not None:
            return self._hit(term, text, partial, MatchLevel.PARTIAL)

        return MatchResult(matched=False, level=MatchLevel.NONE, term=term)

    def _contains_form(self, normalized_text: str, form: str) -> bool:
        """Substring containment with word boundaries.

        Boundary checking prevents short aliases from matching inside longer
        words - without it the potassium alias "k" would match every "k" in
        the text, and "na" would match "nausea".
        """
        start = normalized_text.find(form)
        while start != -1:
            before_ok = start == 0 or not normalized_text[start - 1].isalnum()
            end = start + len(form)
            after_ok = end == len(normalized_text) or not normalized_text[end].isalnum()
            if before_ok and after_ok:
                return True
            start = normalized_text.find(form, start + 1)
        return False

    def _partial_match(self, normalized_term: str, normalized_text: str) -> str | None:
        """Match when enough of a multi-word term's content tokens are present."""
        tokens = [token for token in normalized_term.split() if token not in _STOPWORDS]
        if len(tokens) < 2:
            return None
        present = [token for token in tokens if self._contains_form(normalized_text, token)]
        if len(present) / len(tokens) >= self.partial_threshold:
            return " ".join(present)
        return None

    def _hit(self, term: str, text: str, form: str, level: MatchLevel) -> MatchResult:
        negation = self._negation_for(form, text)
        return MatchResult(
            matched=True,
            level=level,
            term=term,
            matched_form=form,
            evidence=containing_sentence(form, text) or containing_sentence(term, text),
            negated=negation.negated,
            negation_cue=negation.cue,
        )

    def _negation_for(self, form: str, text: str) -> NegationResult:
        return is_negated(form, text)

    def match_any(self, terms: list[str], text: str) -> MatchResult:
        """Best match across several equivalent terms.

        Used where a fact can be expressed by any of a set of surface forms
        and finding any one of them counts as retention.
        """
        best = MatchResult(matched=False, level=MatchLevel.NONE, term=terms[0] if terms else "")
        order = [
            MatchLevel.EXACT, MatchLevel.NORMALIZED, MatchLevel.ALIAS,
            MatchLevel.PARTIAL, MatchLevel.NONE,
        ]
        for term in terms:
            result = self.match(term, text)
            if result.matched and order.index(result.level) < order.index(best.level):
                best = result
        return best
