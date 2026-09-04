"""Regression tests: every bug found during Phase 3 development.

Each of these was a real defect, most of them surfaced by evaluating the
reference summary against itself (the oracle test) rather than by inspection.
They are collected here so the specific failures cannot silently return.

The oracle run went from 89 false positives to 0 through these fixes. That
number is the argument for building the false-positive check before trusting
any sensitivity result.
"""

from __future__ import annotations

from app.core.enums import (
    AbnormalFlag,
    DifficultyTier,
    FindingType,
    MedicationStatus,
    TemporalTrend,
)
from app.evaluation.evaluators import (
    FactSupportEvaluator,
    MedicationConsistencyEvaluator,
    NegationConsistencyEvaluator,
    TemporalConsistencyEvaluator,
)
from app.evaluation.negation import is_negated
from app.evaluation.normalization import normalize_text, sentences_containing
from app.evaluation.numeric import values_near_term
from app.evaluation.support import (
    analyze_finding,
    analyze_lab,
    analyze_trend,
    detect_medication_status,
)
from app.evaluation.matching import FactMatcher
from app.core.enums import CaseType, DataProvenance, FactImportance, FactPolarity, FactType
from app.models.schemas import AIOutput, BenchmarkCase, ClinicalFact
from app.models.structured import (
    ClinicalFindingEntry,
    ConditionEntry,
    EncounterInfo,
    LabResultEntry,
    LabTrendEntry,
    MedicationEntry,
    PatientProfile,
    ProcedureEntry,
    StructuredCase,
)


def _structured(**overrides) -> StructuredCase:
    payload = dict(
        case_id="SYN_R", seed="r:0", generator_version="0.2.0",
        patient=PatientProfile(patient_id="SYN-PT-1", age=60, age_group="55-64", sex="male"),
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
        provenance=DataProvenance.SYNTHETIC_TEMPLATE, age_group="55-64", sex="male",
        encounter_type="Follow-up", clinical_note="Note.", reference_summary="Summary.",
        structured_case=structured,
    )
    payload.update(overrides)
    return BenchmarkCase(**payload)


class TestNormalizationRegressions:
    def test_spelling_fold_does_not_corrupt_gastroesophageal(self):
        """Bug: the unanchored fold 'oesophag' -> 'esophag' rewrote the
        American 'gastroesophageal' (gastr-oesophag-eal) to
        'gastresophageal', so the term matched nothing and every case with
        GERD reported a false unsupported-condition finding."""
        assert normalize_text("Gastroesophageal reflux disease") == "gastroesophageal reflux disease"

    def test_british_spelling_still_folds(self):
        assert normalize_text("Haemoglobin") == "hemoglobin"

    def test_normalized_term_matches_a_normalized_corpus(self):
        """Bug: a normalized term was compared against a raw corpus, so the
        hyphen in 'Community-acquired pneumonia' made a present condition look
        unsupported."""
        evaluator = FactSupportEvaluator()
        structured = _structured(
            conditions=[
                ConditionEntry(
                    entity_id="COND-1", name="Community-acquired pneumonia", is_primary=True
                )
            ]
        )
        result = evaluator.evaluate(
            _case(structured),
            AIOutput(case_id="SYN_R", model_id="m", summary_text="Community-acquired pneumonia."),
        )
        assert not any(f.finding_type is FindingType.UNSUPPORTED_ADDITION for f in result.findings)


class TestContextWindowRegressions:
    def test_medication_status_cues_do_not_leak_across_sentences(self):
        """Bug: a fixed +/-120 character window read a neighbouring sentence's
        status cue, so a continued drug next to a newly started one was
        reported as a status error."""
        text = "Nitrofurantoin 100 mg was newly started. Continued on Lisinopril 20 mg."
        assert detect_medication_status("Lisinopril", text) == "current"
        assert detect_medication_status("Nitrofurantoin", text) == "new"

    def test_lab_values_do_not_leak_across_sentences(self):
        """Bug: a fixed forward window read the next sentence's numbers, so
        one analyte was compared against another's value."""
        text = "Creatinine is 1.9 mg/dL. Potassium is 5.7 mEq/L."
        assert [v.value for v in values_near_term("Creatinine", text)] == [1.9]

    def test_sentences_containing_is_the_bounding_primitive(self):
        text = "Creatinine rose. Potassium fell."
        found = sentences_containing("potassium", text)
        assert len(found) == 1
        assert "creatinine" not in found[0]

    def test_trend_direction_is_read_from_the_trend_own_sentence(self):
        """Bug: a 160-character window crossed into the next sentence and read
        another analyte's direction word, reporting a correctly-described
        falling eGFR as a reversed trend."""
        entity = LabTrendEntry(
            entity_id="TRND-1", lab_name="eGFR", unit="mL/min/1.73m2",
            earlier_ref="LAB-1", later_ref="LAB-2",
            earlier_value=100, later_value=45,
            earlier_timing="6 months ago", later_timing="today",
            trend=TemporalTrend.DECREASING, clinically_significant=True,
        )
        text = (
            "eGFR fell from 100 to 45 mL/min/1.73m2 between 6 months ago and today. "
            "Creatinine rose from 1.1 to 2.1 mg/dL."
        )
        result = analyze_trend(entity, text, FactMatcher())
        assert result.status.value == "supported"


class TestSeriesAttributionRegressions:
    def test_repeated_measurements_do_not_cross_attribute(self):
        """Bug: a creatinine series (admission, day 2, discharge) had each
        trend matched on the analyte name alone, so the first trend's sentence
        was attributed to all of them."""
        structured = _structured(
            labs=[
                LabResultEntry(entity_id="LAB-1", name="Creatinine", value=1.1, unit="mg/dL"),
                LabResultEntry(entity_id="LAB-2", name="Creatinine", value=2.1, unit="mg/dL"),
                LabResultEntry(entity_id="LAB-3", name="Creatinine", value=1.6, unit="mg/dL"),
            ],
            lab_trends=[
                LabTrendEntry(
                    entity_id="TRND-2", lab_name="Creatinine", unit="mg/dL",
                    earlier_ref="LAB-2", later_ref="LAB-3",
                    earlier_value=2.1, later_value=1.6,
                    earlier_timing="on hospital day 2", later_timing="at discharge",
                    trend=TemporalTrend.DECREASING, clinically_significant=True,
                )
            ],
        )
        text = (
            "Creatinine rose from 1.1 to 2.1 mg/dL between on admission and on hospital day 2. "
            "Creatinine fell from 2.1 to 1.6 mg/dL between on hospital day 2 and at discharge."
        )
        result = analyze_trend(structured.lab_trends[0], text, FactMatcher(), structured)
        assert result.status.value == "supported"

    def test_day_number_is_not_read_as_a_trend_endpoint(self):
        """Bug: substring matching found the '2' of 'hospital day 2' and
        treated it as this trend's endpoint value."""
        entity = LabTrendEntry(
            entity_id="TRND-1", lab_name="Creatinine", unit="mg/dL",
            earlier_ref="LAB-1", later_ref="LAB-2",
            earlier_value=3.1, later_value=2,
            earlier_timing="on hospital day 2", later_timing="at discharge",
            trend=TemporalTrend.DECREASING, clinically_significant=True,
        )
        text = "Creatinine rose from 0.8 to 3.1 mg/dL between on admission and on hospital day 2."
        result = analyze_trend(entity, text, FactMatcher())
        assert result.status.value != "supported"

    def test_unstated_measurement_in_a_series_is_omitted_not_contradicted(self):
        structured = _structured(
            labs=[
                LabResultEntry(entity_id="LAB-1", name="Creatinine", value=1.1, unit="mg/dL"),
                LabResultEntry(entity_id="LAB-2", name="Creatinine", value=2.1, unit="mg/dL"),
            ]
        )
        result = analyze_lab(structured.labs[0], "Creatinine is 2.1 mg/dL.", FactMatcher(), structured)
        assert result.status.value == "omitted"

    def test_value_matching_no_measurement_is_still_a_contradiction(self):
        """A series licenses ambiguity about which timepoint is meant, not
        about values the case never records."""
        structured = _structured(
            labs=[
                LabResultEntry(entity_id="LAB-1", name="Creatinine", value=1.1, unit="mg/dL"),
                LabResultEntry(entity_id="LAB-2", name="Creatinine", value=2.1, unit="mg/dL"),
            ]
        )
        result = analyze_lab(structured.labs[0], "Creatinine is 9.9 mg/dL.", FactMatcher(), structured)
        assert result.status.value == "contradicted"


class TestNegationRegressions:
    def test_reversal_is_detected_not_reported_as_omission(self):
        """Bug: matching the full wording 'no history of diabetes' meant a
        reversal ('patient has a history of diabetes') failed to match at all
        and was reported as an omission - inverting the severity of the worst
        failure mode the benchmark models. Detection rate was 3.6%."""
        structured = _structured(
            findings=[
                ClinicalFindingEntry(
                    entity_id="FIND-1", description="no history of diabetes", present=False
                )
            ]
        )
        case = _case(
            structured,
            negations=[
                ClinicalFact(
                    fact_id="N1", type=FactType.OBSERVATION, content="No history of diabetes",
                    importance=FactImportance.CRITICAL, polarity=FactPolarity.NEGATED,
                    source_ref="FIND-1",
                )
            ],
        )
        result = NegationConsistencyEvaluator().evaluate(
            case,
            AIOutput(
                case_id="SYN_R", model_id="m",
                summary_text="The patient has a history of diabetes.",
            ),
        )
        assert any(f.finding_type is FindingType.NEGATION_REVERSAL for f in result.findings)

    def test_post_posed_negation_is_detected(self):
        """Bug: cues were only searched before the term, so 'Chest X-ray was
        not performed' read as an assertion that it was performed."""
        assert is_negated("chest x-ray", "Chest X-ray was not performed.").negated

    def test_procedure_correctly_reported_as_not_performed(self):
        entity = ProcedureEntry(entity_id="PROC-1", name="Chest X-ray", performed=False)
        result = analyze_procedure_helper(entity, "Chest X-ray was not performed.")
        assert result.status.value == "supported"

    def test_pseudo_negation_spanning_the_term_is_ignored(self):
        """Bug: the pseudo-cue check looked only at text before the term, so a
        phrase containing the term could never be recognised as a pseudo-cue."""
        assert not is_negated("diabetes", "Not only diabetes but also hypertension.").negated

    def test_no_change_negative_is_preserved_not_reversed(self):
        """Bug: 'no change' was listed as a pseudo-negation, borrowed from
        radiology's 'no interval change'. In this vocabulary 'no change in
        urine output' is a genuine negative, so the pseudo-cue made the
        evaluator report a correctly preserved negation as a reversal."""
        entity = ClinicalFindingEntry(
            entity_id="FIND-1", description="no change in urine output", present=False
        )
        result = analyze_finding(
            entity,
            "Documented negatives: no syncope; no flank pain; no change in urine output.",
            FactMatcher(),
        )
        assert result.status.value == "supported"

    def test_negated_finding_preserved_is_supported(self):
        entity = ClinicalFindingEntry(
            entity_id="FIND-1", description="no history of diabetes", present=False
        )
        result = analyze_finding(entity, "Documented negatives: no history of diabetes.", FactMatcher())
        assert result.status.value == "supported"


def analyze_procedure_helper(entity, text):
    from app.evaluation.support import analyze_procedure

    return analyze_procedure(entity, text, FactMatcher())


class TestDoseRegressions:
    def test_time_expression_is_not_read_as_a_dose(self):
        """Bug: 'Insulin glargine was discontinued 2 weeks ago' had its '2'
        compared against the 30-unit dose, producing a false dose error."""
        structured = _structured(
            medications=[
                MedicationEntry(
                    entity_id="MED-1", drug_name="Insulin glargine",
                    status=MedicationStatus.DISCONTINUED, dose="30 units",
                    change_timing="2 weeks ago",
                )
            ]
        )
        result = MedicationConsistencyEvaluator().evaluate(
            _case(structured),
            AIOutput(
                case_id="SYN_R", model_id="m",
                summary_text="Insulin glargine was discontinued 2 weeks ago.",
            ),
        )
        assert result.findings == []

    def test_a_genuinely_wrong_dose_is_still_caught(self):
        structured = _structured(
            medications=[
                MedicationEntry(
                    entity_id="MED-1", drug_name="Lisinopril",
                    status=MedicationStatus.CURRENT, dose="20 mg",
                )
            ]
        )
        result = MedicationConsistencyEvaluator().evaluate(
            _case(structured),
            AIOutput(
                case_id="SYN_R", model_id="m", summary_text="Continued on Lisinopril 80 mg."
            ),
        )
        assert any(f.finding_type is FindingType.MEDICATION_DOSE_ERROR for f in result.findings)


class TestBenchmarkRegressions:
    def test_no_case_both_denies_and_asserts_the_same_thing(self):
        """Bug in the Phase 2 generator, found by evaluating reference
        summaries against themselves: a case could document 'no recent falls'
        as a relevant negative while a conflict resolution asserted that falls
        had occurred. Ground truth contradicting itself made the evaluator
        report a correct summary as a negation reversal - and the evaluator
        was right."""
        from app.benchmark import generate_benchmark
        from app.benchmark.integrity import check_no_self_contradiction

        for case in generate_benchmark(case_count=100, seed=42):
            assert check_no_self_contradiction(case) == [], case.case_id

    def test_the_self_contradiction_check_fires(self):
        """A validator that never fails is worth nothing."""
        from app.benchmark.integrity import check_no_self_contradiction
        from app.models.structured import ConflictEntry

        structured = _structured(
            findings=[
                ClinicalFindingEntry(
                    entity_id="FIND-1", description="no recent falls", present=False
                )
            ],
            conflicts=[
                ConflictEntry(
                    entity_id="CONF-1", topic="fall history",
                    earlier_statement="patient denies recent falls",
                    earlier_timing="on initial assessment",
                    later_statement="family reports two falls in the past month",
                    later_timing="on reassessment",
                    resolution="Recent falls have occurred.",
                )
            ],
        )
        assert check_no_self_contradiction(_case(structured))


    def test_reference_summary_states_a_lab_no_trend_covers(self):
        """Bug in the Phase 2 summary renderer, found by evaluating the
        reference summary against itself: standalone abnormal labs were
        suppressed by analyte *name* whenever any trend used that analyte, so
        a case with both a standalone creatinine and a creatinine series
        silently dropped the standalone value from its reference summary."""
        from app.benchmark.summary_renderer import render_reference_summary

        structured = _structured(
            labs=[
                LabResultEntry(
                    entity_id="LAB-1", name="Creatinine", value=1.4, unit="mg/dL",
                    day_offset=-90, timing_description="3 months ago",
                ),
                LabResultEntry(
                    entity_id="LAB-2", name="Creatinine", value=1.9, unit="mg/dL",
                    abnormal_flag=AbnormalFlag.HIGH,
                ),
                LabResultEntry(
                    entity_id="LAB-3", name="Creatinine", value=2.3, unit="mg/dL",
                    abnormal_flag=AbnormalFlag.HIGH, timing_description="today",
                ),
            ],
            lab_trends=[
                LabTrendEntry(
                    entity_id="TRND-1", lab_name="Creatinine", unit="mg/dL",
                    earlier_ref="LAB-1", later_ref="LAB-2",
                    earlier_value=1.4, later_value=1.9,
                    earlier_timing="3 months ago", later_timing="today",
                    trend=TemporalTrend.INCREASING, clinically_significant=True,
                )
            ],
        )
        summary = render_reference_summary(structured)
        assert "2.3" in summary, "a measurement no trend covers must survive into the summary"
