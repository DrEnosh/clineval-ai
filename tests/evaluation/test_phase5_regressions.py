"""Regression tests for defects found by the Phase 5 methodology challenges.

Every evaluator change made because of a Phase 5 observation becomes a test
here, so the specific weakness cannot silently return. See
docs/REAL_MODEL_EVALUATION.md for the observations these came from and
CHANGELOG.md for the version increments.

Two of the three findings were evaluator bugs. The third was a defect in the
*challenge*, and is recorded here too, because a test suite that only remembers
the times the system was wrong will drift toward assuming it never is.
"""

from __future__ import annotations

import pytest

from app.core.enums import (
    AbnormalFlag,
    DifficultyTier,
    FindingType,
    MedicationStatus,
    SupportStatus,
    TemporalTrend,
)
from app.core.enums import CaseType, DataProvenance
from app.evaluation.evaluators import FactSupportEvaluator, MedicationConsistencyEvaluator
from app.evaluation.matching import FactMatcher
from app.evaluation.support import analyze_trend, detect_medication_status
from app.models.schemas import AIOutput, BenchmarkCase
from app.models.structured import (
    EncounterInfo,
    LabResultEntry,
    LabTrendEntry,
    MedicationEntry,
    PatientProfile,
    StructuredCase,
)


def _structured(**overrides) -> StructuredCase:
    payload = dict(
        case_id="SYN_P5", seed="p5:0", generator_version="0.2.0",
        patient=PatientProfile(patient_id="SYN-PT-1", age=62, age_group="55-64", sex="female"),
        encounter=EncounterInfo(
            encounter_id="E1", encounter_type="Follow-up", setting="outpatient_primary_care"
        ),
        difficulty=DifficultyTier.MODERATE,
    )
    payload.update(overrides)
    return StructuredCase(**payload)


def _case(structured, **overrides) -> BenchmarkCase:
    payload = dict(
        case_id=structured.case_id, case_type=CaseType.NORMAL, benchmark_version="v",
        provenance=DataProvenance.SYNTHETIC_TEMPLATE, age_group="55-64", sex="female",
        encounter_type="Follow-up", clinical_note="Note.", reference_summary="Summary.",
        structured_case=structured,
    )
    payload.update(overrides)
    return BenchmarkCase(**payload)


def _output(text: str) -> AIOutput:
    return AIOutput(case_id="SYN_P5", model_id="test", summary_text=text)


class TestNumericEquivalence:
    """P5-01: values were compared as strings, so notation changed the verdict."""

    def test_padded_decimal_is_not_an_unsupported_value(self):
        """Bug: "12.50" was reported as a fabricated value in a case stating
        12.5. A model padding a decimal has invented nothing."""
        structured = _structured(
            medications=[
                MedicationEntry(
                    entity_id="MED-1", drug_name="Spironolactone",
                    status=MedicationStatus.CHANGED_DOSE, dose="25 mg", prior_dose="12.5 mg",
                )
            ]
        )
        result = FactSupportEvaluator().evaluate(
            _case(structured),
            _output("Spironolactone was increased from 12.50 mg to 25 mg."),
        )
        unsupported = [
            f for f in result.findings if f.finding_type is FindingType.UNSUPPORTED_ADDITION
        ]
        assert unsupported == [], [f.explanation for f in unsupported]

    def test_a_genuinely_invented_value_is_still_caught(self):
        """The fix must not blunt real hallucination detection."""
        structured = _structured(
            labs=[LabResultEntry(entity_id="LAB-1", name="Creatinine", value=1.9, unit="mg/dL")]
        )
        result = FactSupportEvaluator().evaluate(
            _case(structured), _output("Creatinine was 7.77 mg/dL.")
        )
        assert any(f.details.get("value") == "7.77" for f in result.findings)

    @pytest.mark.parametrize("written", ["1.9", "1.90", "1.900"])
    def test_trend_endpoints_match_regardless_of_notation(self, written):
        """Bug: trend endpoints were matched as strings, so a padded decimal
        broke endpoint detection and reported information loss on a summary
        that stated the trend correctly."""
        entity = LabTrendEntry(
            entity_id="TRND-1", lab_name="Creatinine", unit="mg/dL",
            earlier_ref="LAB-1", later_ref="LAB-2",
            earlier_value=1.4, later_value=1.9,
            earlier_timing="3 months ago", later_timing="today",
            trend=TemporalTrend.INCREASING, clinically_significant=True,
        )
        result = analyze_trend(
            entity, f"Creatinine rose from 1.40 to {written} mg/dL.", FactMatcher()
        )
        assert result.status is SupportStatus.SUPPORTED

    def test_a_different_value_is_still_not_a_match(self):
        entity = LabTrendEntry(
            entity_id="TRND-1", lab_name="Creatinine", unit="mg/dL",
            earlier_ref="LAB-1", later_ref="LAB-2",
            earlier_value=1.4, later_value=1.9,
            earlier_timing="3 months ago", later_timing="today",
            trend=TemporalTrend.INCREASING, clinically_significant=True,
        )
        result = analyze_trend(entity, "Creatinine rose from 1.4 to 2.7 mg/dL.", FactMatcher())
        assert result.status is not SupportStatus.SUPPORTED


class TestMedicationRegister:
    """P5-02: the status cue lists were too narrow for real clinical phrasing."""

    @pytest.mark.parametrize(
        "phrasing",
        [
            "Amlodipine has been held.",
            "Amlodipine was withheld.",
            "Amlodipine is no longer being taken.",
            "Amlodipine came off the regimen.",
            "Amlodipine was ceased.",
            "Amlodipine has been suspended.",
            "Amlodipine was omitted.",
        ],
    )
    def test_natural_discontinuation_phrasing_is_recognised(self, phrasing):
        """Bug: only "discontinued"/"stopped" were detected, so summaries that
        stated the status perfectly clearly were flagged as status errors."""
        assert detect_medication_status("Amlodipine", phrasing) == "discontinued"

    @pytest.mark.parametrize(
        "phrasing",
        [
            "Furosemide dose was stepped up from 20 mg to 40 mg.",
            "Furosemide was up titrated from 20 mg to 40 mg.",
            "Furosemide was escalated from 20 mg to 40 mg.",
            "Furosemide was doubled from 20 mg to 40 mg.",
        ],
    )
    def test_natural_dose_change_phrasing_is_recognised(self, phrasing):
        assert detect_medication_status("Furosemide", phrasing) == "changed_dose"

    def test_a_stopped_drug_written_naturally_is_not_a_status_error(self):
        structured = _structured(
            medications=[
                MedicationEntry(
                    entity_id="MED-1", drug_name="Amlodipine",
                    status=MedicationStatus.DISCONTINUED, dose="5 mg",
                    change_timing="1 month ago",
                )
            ]
        )
        result = MedicationConsistencyEvaluator().evaluate(
            _case(structured), _output("Amlodipine has been held since last month.")
        )
        assert result.findings == [], [f.explanation for f in result.findings]

    def test_a_stopped_drug_reported_as_current_is_still_an_error(self):
        """Widening the cue lists must not blunt real status detection."""
        structured = _structured(
            medications=[
                MedicationEntry(
                    entity_id="MED-1", drug_name="Amlodipine",
                    status=MedicationStatus.DISCONTINUED, dose="5 mg",
                )
            ]
        )
        result = MedicationConsistencyEvaluator().evaluate(
            _case(structured), _output("Ongoing therapy includes Amlodipine 5 mg daily.")
        )
        assert any(
            f.finding_type is FindingType.MEDICATION_STATUS_ERROR for f in result.findings
        )


class TestChallengeConstruction:
    """P5-03: a defect in the challenge, not the evaluator."""

    def test_dropping_a_current_medication_is_correctly_flagged(self):
        """The compression challenge originally dropped current medications and
        called the resulting findings false positives. They were not: removing
        the antibiotic from a cellulitis summary is real information loss. The
        challenge was corrected, and this records why."""
        structured = _structured(
            medications=[
                MedicationEntry(
                    entity_id="MED-1", drug_name="Cephalexin",
                    status=MedicationStatus.CURRENT, dose="500 mg",
                )
            ]
        )
        result = MedicationConsistencyEvaluator().evaluate(
            _case(structured), _output("30-year-old female seen for cellulitis.")
        )
        assert any(
            f.finding_type is FindingType.MEDICATION_OMISSION for f in result.findings
        )

    def test_abstraction_regex_does_not_truncate_at_a_decimal_point(self):
        """A `[^.]*\\.` pattern stopped at the decimal in "4.1", leaving
        mangled text like "potassium is now deranged.1 to 5.7". Both the
        challenge suite and the Phase 4 mock provider had this bug, so the mock
        `abstractive` behaviour had been producing malformed summaries."""
        from app.experiments.challenges import abstraction
        from app.providers.mock import _abstractive

        structured = _structured(
            labs=[
                LabResultEntry(entity_id="LAB-1", name="Potassium", value=4.1, unit="mEq/L"),
                LabResultEntry(
                    entity_id="LAB-2", name="Potassium", value=5.7, unit="mEq/L",
                    abnormal_flag=AbnormalFlag.HIGH,
                ),
            ],
            lab_trends=[
                LabTrendEntry(
                    entity_id="TRND-1", lab_name="Potassium", unit="mEq/L",
                    earlier_ref="LAB-1", later_ref="LAB-2",
                    earlier_value=4.1, later_value=5.7,
                    earlier_timing="2 weeks ago", later_timing="today",
                    trend=TemporalTrend.INCREASING, clinically_significant=True,
                )
            ],
        )
        case = _case(
            structured,
            reference_summary=(
                "Potassium rose from 4.1 to 5.7 mEq/L between 2 weeks ago and today. "
                "Plan: repeat testing."
            ),
        )
        for produced in (abstraction(case), _abstractive(case)):
            assert "4.1" not in produced, produced
            assert "deranged.1" not in produced
            assert ".1 to 5.7" not in produced
            assert "Plan: repeat testing." in produced, "later sentences must survive"


class TestChallengeSuiteHolds:
    """The suite that found these defects must keep passing."""

    def test_every_decidable_challenge_passes(self):
        from app.benchmark import generate_benchmark
        from app.experiments.challenges import run_challenges, summarize_outcomes
        from app.experiments.selection import select_experiment_sample

        cases = generate_benchmark(case_count=100, seed=42)
        outcomes = run_challenges(select_experiment_sample(cases).cases)
        summary = summarize_outcomes(outcomes)

        failures = [o.report() for o in outcomes if not o.passed]
        assert summary["failed"] == 0, failures
        assert summary["decidable"] > 30, "the suite must actually exercise the evaluator"
