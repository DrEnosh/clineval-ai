"""Tests for the evaluation primitives: normalization, negation, numeric, matching.

These are the layers every evaluator rests on, so their edge cases are tested
here directly rather than only through the evaluators that use them.
"""

from __future__ import annotations

import pytest

from app.core.enums import MatchLevel, ValueComparison
from app.evaluation.matching import FactMatcher
from app.evaluation.negation import (
    containing_sentence,
    is_negated,
    mentions,
    split_clauses,
)
from app.evaluation.normalization import (
    are_equivalent,
    canonical_term,
    normalize_text,
    sentences_containing,
    split_sentences,
    surface_forms,
    tokenize,
)
from app.evaluation.numeric import (
    compare_blood_pressure,
    compare_term_value,
    compare_values,
    parse_blood_pressure,
    parse_numbers,
    values_near_term,
)


class TestNormalization:
    def test_lowercases_and_collapses_whitespace(self):
        assert normalize_text("  Chronic   Kidney   Disease  ") == "chronic kidney disease"

    def test_preserves_clinical_punctuation(self):
        assert normalize_text("BP 168/94") == "bp 168/94"
        assert normalize_text("Creatinine 1.9 mg/dL") == "creatinine 1.9 mg/dl"

    def test_folds_hyphens_to_spaces(self):
        assert normalize_text("gastro-oesophageal") == normalize_text("gastro oesophageal")

    def test_folds_british_spellings(self):
        assert normalize_text("Haemoglobin") == "hemoglobin"
        assert normalize_text("anaemia") == "anemia"
        assert normalize_text("oedema") == "edema"

    def test_spelling_folds_are_word_anchored(self):
        """Regression: unanchored 'oesophag' -> 'esophag' rewrote the American
        'gastroesophageal' (gastr-oesophag-eal) into 'gastresophageal',
        breaking every match for that condition."""
        assert normalize_text("Gastroesophageal reflux disease") == "gastroesophageal reflux disease"
        assert normalize_text("gastroesophageal") == "gastroesophageal"

    def test_handles_empty_input(self):
        assert normalize_text("") == ""
        assert tokenize("") == []

    def test_tokenize_splits_content(self):
        assert tokenize("Chronic kidney disease") == ["chronic", "kidney", "disease"]

    def test_split_sentences(self):
        assert split_sentences("One thing. Two things! Three?") == [
            "One thing.", "Two things!", "Three?",
        ]

    def test_sentences_containing_is_bounded(self):
        text = "Creatinine rose to 1.9. Potassium fell to 3.2."
        found = sentences_containing("potassium", text)
        assert len(found) == 1
        assert "creatinine" not in found[0]


class TestAliases:
    @pytest.mark.parametrize(
        "left,right",
        [
            ("Chronic kidney disease stage 3", "CKD"),
            ("Gastroesophageal reflux disease", "GERD"),
            ("Type 2 diabetes mellitus", "diabetes"),
            ("Hemoglobin", "Hgb"),
            ("Creatinine", "Cr"),
            ("Blood pressure", "BP"),
            ("Hydrochlorothiazide", "HCTZ"),
            ("Metoprolol succinate", "metoprolol"),
            ("Heart failure with reduced ejection fraction", "HFrEF"),
        ],
    )
    def test_known_aliases_are_equivalent(self, left, right):
        assert are_equivalent(left, right)
        assert are_equivalent(right, left)

    def test_unrelated_terms_are_not_equivalent(self):
        assert not are_equivalent("Creatinine", "Potassium")
        assert not are_equivalent("Asthma", "Cellulitis")

    def test_surface_forms_include_the_term_itself(self):
        assert "creatinine" in surface_forms("Creatinine")

    def test_unregistered_term_falls_back_to_itself(self):
        assert surface_forms("Zorblaxine") == frozenset({"zorblaxine"})

    def test_canonical_term_resolves_an_alias(self):
        assert canonical_term("CKD") == "chronic kidney disease stage 3"

    def test_canonical_term_passes_through_unknown_terms(self):
        assert canonical_term("Zorblaxine") == "zorblaxine"


class TestNegation:
    @pytest.mark.parametrize(
        "text",
        [
            "No history of diabetes.",
            "Patient denies chest pain.",
            "Negative for fever.",
            "Without any evidence of pneumonia.",
            "There is no chest pain.",
        ],
    )
    def test_detects_negated_terms(self, text):
        term = {"diabetes": "diabetes", "chest": "chest pain", "fever": "fever",
                "pneumonia": "pneumonia"}
        for key, value in term.items():
            if key in text.lower():
                assert is_negated(value, text).negated, text
                break

    def test_affirmed_terms_are_not_negated(self):
        assert not is_negated("diabetes", "Patient has a history of diabetes.").negated
        assert not is_negated("chest pain", "Patient reports chest pain.").negated

    def test_terminator_bounds_the_negation_scope(self):
        """'no chest pain but reports dyspnea' must not mark dyspnea negated."""
        text = "Patient has no chest pain but reports dyspnea."
        assert is_negated("chest pain", text).negated
        assert not is_negated("dyspnea", text).negated

    def test_pseudo_negation_is_ignored(self):
        """'Not only X' does not deny X."""
        assert not is_negated("diabetes", "Not only diabetes but also hypertension.").negated

    def test_no_change_is_a_genuine_negation_not_a_pseudo_cue(self):
        """'no change in urine output' denies a change. Treating "no change"
        as a pseudo-cue made preserved negations read as reversals."""
        assert is_negated("change in urine output", "No change in urine output.").negated

    def test_clause_boundary_bounds_scope(self):
        text = "Denies fever. Reports cough."
        assert is_negated("fever", text).negated
        assert not is_negated("cough", text).negated

    def test_affirmed_mention_anywhere_wins(self):
        """A term both denied and asserted counts as asserted, because the
        output does make the claim."""
        text = "Initially denies chest pain. On reassessment reports chest pain."
        assert not is_negated("chest pain", text).negated

    def test_absent_term_is_not_negated(self):
        assert not is_negated("diabetes", "Patient is well.").negated

    def test_empty_inputs_are_safe(self):
        assert not is_negated("", "text").negated
        assert not is_negated("term", "").negated

    def test_negation_result_carries_evidence(self):
        result = is_negated("diabetes", "No history of diabetes.")
        assert result.cue is not None
        assert "diabetes" in result.evidence

    def test_mentions_ignores_polarity(self):
        assert mentions("diabetes", "No history of diabetes.")
        assert not mentions("asthma", "No history of diabetes.")

    def test_containing_sentence_returns_original_casing(self):
        text = "Patient is well. Creatinine is 1.9 mg/dL today."
        assert containing_sentence("creatinine", text) == "Creatinine is 1.9 mg/dL today."

    def test_split_clauses(self):
        assert split_clauses("a. b; c") == ["a", "b", "c"]


class TestNumericComparison:
    def test_parses_numbers(self):
        assert [item.value for item in parse_numbers("Creatinine 1.9 and potassium 5.1")] == [1.9, 5.1]

    def test_parses_blood_pressure(self):
        assert parse_blood_pressure("BP 168/94 mmHg") == (168.0, 94.0)

    def test_blood_pressure_absent_returns_none(self):
        assert parse_blood_pressure("no reading") is None

    def test_exact_value_is_correct(self):
        result = compare_values(1.9, [1.9], unit="mg/dL")
        assert result.comparison is ValueComparison.CORRECT

    def test_trailing_zero_is_still_correct(self):
        assert compare_values(1.9, [1.90]).comparison is ValueComparison.CORRECT

    def test_small_difference_is_approximate(self):
        result = compare_values(1.90, [1.92], relative_tolerance=0.02)
        assert result.comparison is ValueComparison.APPROXIMATE

    def test_large_difference_is_incorrect(self):
        result = compare_values(1.9, [1.4])
        assert result.comparison is ValueComparison.INCORRECT
        assert result.is_error

    def test_no_candidates_is_omitted(self):
        assert compare_values(1.9, []).comparison is ValueComparison.OMITTED

    def test_closest_candidate_wins(self):
        """Reporting both endpoints of a trend must not be scored an error."""
        result = compare_values(1.9, [1.4, 1.9])
        assert result.comparison is ValueComparison.CORRECT

    def test_zero_source_value_does_not_divide_by_zero(self):
        assert compare_values(0.0, [0.0]).comparison is ValueComparison.CORRECT
        assert compare_values(0.0, [5.0]).comparison is ValueComparison.INCORRECT

    def test_describe_is_human_readable(self):
        result = compare_values(1.9, [1.4], unit="mg/dL")
        assert "1.9" in result.describe() and "1.4" in result.describe()
        assert "no value found" in compare_values(1.9, [], unit="mg/dL").describe()

    def test_values_near_term_is_sentence_bounded(self):
        """Regression: a fixed character window read the next sentence's
        values, letting one analyte's comparison see another's numbers."""
        text = "Creatinine is 1.9 mg/dL. Potassium is 5.7 mEq/L."
        assert [item.value for item in values_near_term("Creatinine", text)] == [1.9]
        assert [item.value for item in values_near_term("Potassium", text)] == [5.7]

    def test_values_near_term_resolves_aliases(self):
        assert 1.9 in [item.value for item in values_near_term("Creatinine", "Cr 1.9 mg/dL.")]

    def test_compare_term_value_end_to_end(self):
        result = compare_term_value("Creatinine", 1.9, "Creatinine rose to 1.9 mg/dL.", unit="mg/dL")
        assert result.comparison is ValueComparison.CORRECT

    def test_compare_term_value_missing_term(self):
        result = compare_term_value("Creatinine", 1.9, "Patient is well.")
        assert result.comparison is ValueComparison.OMITTED

    def test_blood_pressure_matching_pair_is_correct(self):
        assert compare_blood_pressure("168/94", "BP was 168/94 mmHg.").comparison is ValueComparison.CORRECT

    def test_blood_pressure_partial_match_is_incorrect(self):
        """Keeping the systolic and altering the diastolic changes the reading."""
        result = compare_blood_pressure("168/94", "BP was 168/72 mmHg.")
        assert result.comparison is ValueComparison.INCORRECT

    def test_blood_pressure_absent_is_omitted(self):
        assert compare_blood_pressure("168/94", "No vitals stated.").comparison is ValueComparison.OMITTED

    def test_blood_pressure_unparseable_source(self):
        assert compare_blood_pressure("normal", "BP 120/80").comparison is ValueComparison.NOT_COMPARABLE


class TestFactMatcher:
    @pytest.fixture
    def matcher(self) -> FactMatcher:
        return FactMatcher()

    def test_exact_match(self, matcher):
        result = matcher.match("Creatinine", "Creatinine is elevated.")
        assert result.matched and result.level is MatchLevel.EXACT

    def test_normalized_match(self, matcher):
        result = matcher.match("Chronic Kidney Disease", "history of chronic kidney disease")
        assert result.matched and result.level is MatchLevel.NORMALIZED

    def test_alias_match(self, matcher):
        result = matcher.match("Chronic kidney disease stage 3", "Known CKD.")
        assert result.matched and result.level is MatchLevel.ALIAS

    def test_partial_match(self, matcher):
        result = matcher.match(
            "Heart failure with reduced ejection fraction", "heart failure with reduced EF"
        )
        assert result.matched
        assert result.level in {MatchLevel.ALIAS, MatchLevel.PARTIAL}

    def test_no_match(self, matcher):
        result = matcher.match("Creatinine", "Patient is well.")
        assert not result.matched and result.level is MatchLevel.NONE

    def test_empty_inputs_do_not_match(self, matcher):
        assert not matcher.match("", "text").matched
        assert not matcher.match("term", "").matched
        assert not matcher.match("term", "   ").matched

    def test_short_alias_respects_word_boundaries(self, matcher):
        """The potassium alias 'k' must not match inside other words."""
        assert not matcher.match("Potassium", "Patient walks and talks.").matched

    def test_sodium_alias_does_not_match_inside_nausea(self, matcher):
        assert not matcher.match("Sodium", "Reports nausea.").matched

    def test_match_records_negation(self, matcher):
        result = matcher.match("diabetes", "No history of diabetes.")
        assert result.matched and result.negated
        assert not result.affirmed

    def test_affirmed_property(self, matcher):
        assert matcher.match("diabetes", "Has diabetes.").affirmed

    def test_confidence_reflects_match_level(self, matcher):
        assert matcher.match("Creatinine", "Creatinine 1.9").confidence == 1.0
        assert matcher.match("Chronic kidney disease stage 3", "CKD noted").confidence < 1.0

    def test_match_carries_evidence(self, matcher):
        result = matcher.match("Creatinine", "Patient is well. Creatinine is 1.9 mg/dL.")
        assert result.evidence == "Creatinine is 1.9 mg/dL."

    def test_match_any_prefers_the_strongest_level(self, matcher):
        result = matcher.match_any(["Chronic kidney disease stage 3", "Creatinine"], "Creatinine 1.9")
        assert result.level is MatchLevel.EXACT

    def test_match_any_with_no_hits(self, matcher):
        assert not matcher.match_any(["Creatinine", "Potassium"], "Patient is well.").matched

    def test_invalid_threshold_is_rejected(self):
        with pytest.raises(ValueError, match="partial_threshold"):
            FactMatcher(partial_threshold=0.0)
        with pytest.raises(ValueError, match="partial_threshold"):
            FactMatcher(partial_threshold=1.5)
