"""Synthetic clinical note-summarization benchmark generator.

Phase 1 ships one hand-authored case per `CaseType` (10 cases total), each
built to exercise exactly the failure mode that case type is named for (see
docs/ARCHITECTURE.md, Benchmark design). Cases are entirely synthetic: no
real patient data is used or required.

Why hand-authored rather than randomly templated for Phase 1: the gold-standard
facts (critical_facts, negations, temporal_facts, medication_facts) must be
clinically coherent with the note for the evaluator to be testable at all.
Later phases (see docs/ARCHITECTURE.md, Phase 5) scale each case type out to
many randomized variants and may draw structured context from Synthea; this
module is the seam that work extends - `generate_default_benchmark` is the
stable entry point, and adding a case type only requires adding a new
`_build_*` function and registering it.

Every case declares `provenance=DataProvenance.MANUAL_SYNTHETIC` -- authored
by hand for this benchmark, not sourced from Synthea or any real record.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.enums import (
    CaseType,
    DataProvenance,
    FactImportance,
    FactPolarity,
    FactType,
    MedicationStatus,
    Severity,
    TemporalTrend,
)
from app.models.schemas import BenchmarkCase, ClinicalFact, MedicationFact, RiskFlag, TemporalFact

BENCHMARK_VERSION = "v1"


def _fact(
    case_id: str,
    fact_num: int,
    type: FactType,
    content: str,
    importance: FactImportance,
    polarity: FactPolarity = FactPolarity.AFFIRMED,
    **attributes: str,
) -> ClinicalFact:
    return ClinicalFact(
        fact_id=f"{case_id}-F{fact_num:03d}",
        type=type,
        content=content,
        importance=importance,
        polarity=polarity,
        attributes=attributes,
    )


def _med(
    case_id: str,
    fact_num: int,
    drug_name: str,
    status: MedicationStatus,
    content: str,
    importance: FactImportance,
    dose: str | None = None,
    route: str | None = None,
) -> MedicationFact:
    return MedicationFact(
        fact_id=f"{case_id}-M{fact_num:03d}",
        content=content,
        importance=importance,
        drug_name=drug_name,
        status=status,
        dose=dose,
        route=route,
    )


def _temporal(
    case_id: str,
    fact_num: int,
    content: str,
    importance: FactImportance,
    earlier_value: str,
    later_value: str,
    unit: str,
    trend: TemporalTrend,
) -> TemporalFact:
    return TemporalFact(
        fact_id=f"{case_id}-T{fact_num:03d}",
        content=content,
        importance=importance,
        earlier_value=earlier_value,
        later_value=later_value,
        unit=unit,
        trend=trend,
    )


def _risk(case_id: str, flag_num: int, description: str, severity: Severity) -> RiskFlag:
    return RiskFlag(flag_id=f"{case_id}-R{flag_num:03d}", description=description, severity=severity)


def _build_normal_case() -> BenchmarkCase:
    case_id = "CASE_001"
    return BenchmarkCase(
        case_id=case_id,
        case_type=CaseType.NORMAL,
        benchmark_version=BENCHMARK_VERSION,
        provenance=DataProvenance.MANUAL_SYNTHETIC,
        age_group="45-54",
        sex="female",
        encounter_type="Annual wellness visit",
        conditions=["Essential hypertension, well-controlled"],
        medications=["Lisinopril 10 mg oral daily"],
        allergies=["No known drug allergies"],
        procedures=[],
        observations=["Well-appearing, no acute distress"],
        laboratory_values=["Basic metabolic panel within normal limits"],
        vital_signs=["BP 122/78 mmHg", "HR 72 bpm", "Temp 98.4 F"],
        clinical_timeline=["Annual wellness visit; no interval changes since visit 11 months ago"],
        clinical_note=(
            "45-year-old female presents for annual wellness visit. She reports feeling well "
            "with no new complaints. History of essential hypertension, well-controlled on "
            "lisinopril 10 mg daily. No chest pain, shortness of breath, or lower extremity "
            "edema. No known drug allergies. Continues regular exercise three times weekly. "
            "Vital signs: BP 122/78 mmHg, HR 72 bpm, Temp 98.4 F. Physical exam unremarkable. "
            "Basic metabolic panel within normal limits. Plan: continue lisinopril 10 mg daily, "
            "routine health maintenance labs ordered, return to clinic in one year or sooner "
            "if new symptoms develop."
        ),
        reference_summary=(
            "45-year-old female here for a routine annual wellness visit. Hypertension remains "
            "well-controlled on lisinopril 10 mg daily; no new symptoms and exam is unremarkable. "
            "Plan is to continue current medication and follow up in one year."
        ),
        critical_facts=[],
        non_critical_facts=[
            _fact(case_id, 1, FactType.OBSERVATION, "Patient reports feeling well, no new symptoms", FactImportance.NON_CRITICAL),
            _fact(case_id, 2, FactType.CONDITION, "Hypertension is well-controlled on current regimen", FactImportance.NON_CRITICAL),
        ],
        medication_facts=[
            _med(case_id, 1, "Lisinopril", MedicationStatus.CURRENT, "Lisinopril 10 mg daily continued", FactImportance.NON_CRITICAL, dose="10 mg oral daily"),
        ],
        temporal_facts=[],
        negations=[
            _fact(case_id, 3, FactType.OBSERVATION, "No chest pain, shortness of breath, or lower extremity edema", FactImportance.NON_CRITICAL, FactPolarity.NEGATED),
            _fact(case_id, 4, FactType.ALLERGY, "No known drug allergies", FactImportance.NON_CRITICAL, FactPolarity.NEGATED),
        ],
        expected_risk_flags=[],
        subgroup={"age_group": "45-54", "sex": "female", "encounter_setting": "outpatient_primary_care"},
    )


def _build_critical_omission_case() -> BenchmarkCase:
    case_id = "CASE_002"
    return BenchmarkCase(
        case_id=case_id,
        case_type=CaseType.CRITICAL_OMISSION,
        benchmark_version=BENCHMARK_VERSION,
        provenance=DataProvenance.MANUAL_SYNTHETIC,
        age_group="65-74",
        sex="male",
        encounter_type="Nephrology follow-up",
        conditions=["Chronic kidney disease, stage 3", "Type 2 diabetes mellitus", "Hypertension"],
        medications=["Lisinopril 20 mg daily", "Metformin 1000 mg twice daily"],
        allergies=["No known drug allergies"],
        laboratory_values=[
            "Creatinine 1.4 mg/dL (3 months ago)",
            "Creatinine 1.9 mg/dL (today)",
            "eGFR declined from 52 to 38 mL/min/1.73m2",
        ],
        vital_signs=["BP 138/86 mmHg"],
        clinical_timeline=["Follow-up 3 months after prior nephrology visit"],
        clinical_note=(
            "68-year-old male with chronic kidney disease stage 3, type 2 diabetes, and "
            "hypertension returns for nephrology follow-up. Serum creatinine has increased "
            "from 1.4 mg/dL three months ago to 1.9 mg/dL today, with a corresponding decline "
            "in estimated GFR from 52 to 38 mL/min/1.73m2. This represents a clinically "
            "significant decline in renal function since the last visit. Patient denies "
            "changes in urine output, hematuria, or flank pain. Blood pressure today 138/86 "
            "mmHg. Dietary sodium and protein counseling provided. Continue lisinopril 20 mg "
            "daily for blood pressure control. Plan: repeat basic metabolic panel in 2 weeks, "
            "consider metformin dose adjustment given renal decline, renal ultrasound if "
            "creatinine continues to rise."
        ),
        reference_summary=(
            "68-year-old male with CKD stage 3, diabetes, and hypertension. Creatinine "
            "increased from 1.4 to 1.9 mg/dL over 3 months (eGFR 52 to 38), a significant "
            "decline in renal function requiring close follow-up and possible metformin "
            "adjustment. Dietary counseling given; repeat labs in 2 weeks."
        ),
        critical_facts=[
            _fact(
                case_id, 1, FactType.LAB,
                "Creatinine increased from 1.4 to 1.9 mg/dL over 3 months, consistent with "
                "significant decline in renal function",
                FactImportance.CRITICAL,
            ),
            _fact(
                case_id, 2, FactType.LAB,
                "eGFR declined from 52 to 38 mL/min/1.73m2",
                FactImportance.CRITICAL,
            ),
        ],
        non_critical_facts=[
            _fact(case_id, 3, FactType.OBSERVATION, "Dietary sodium and protein counseling provided", FactImportance.NON_CRITICAL),
        ],
        medication_facts=[
            _med(case_id, 1, "Lisinopril", MedicationStatus.CURRENT, "Lisinopril 20 mg daily continued", FactImportance.NON_CRITICAL, dose="20 mg daily"),
            _med(case_id, 2, "Metformin", MedicationStatus.CURRENT, "Metformin 1000 mg twice daily; dose adjustment being considered given renal decline", FactImportance.CRITICAL, dose="1000 mg twice daily"),
        ],
        temporal_facts=[
            _temporal(
                case_id, 1,
                "Creatinine increased from 1.4 to 1.9 mg/dL over 3 months",
                FactImportance.CRITICAL,
                earlier_value="1.4 mg/dL", later_value="1.9 mg/dL", unit="mg/dL",
                trend=TemporalTrend.INCREASING,
            ),
            _temporal(
                case_id, 2,
                "eGFR declined from 52 to 38 mL/min/1.73m2",
                FactImportance.CRITICAL,
                earlier_value="52 mL/min/1.73m2", later_value="38 mL/min/1.73m2", unit="mL/min/1.73m2",
                trend=TemporalTrend.DECREASING,
            ),
        ],
        negations=[
            _fact(case_id, 4, FactType.OBSERVATION, "Denies changes in urine output, hematuria, or flank pain", FactImportance.NON_CRITICAL, FactPolarity.NEGATED),
        ],
        expected_risk_flags=[
            _risk(
                case_id, 1,
                "Omitting the renal function decline could delay medication dose adjustment "
                "(e.g. metformin) and appropriate nephrology follow-up",
                Severity.HIGH,
            ),
        ],
        subgroup={"age_group": "65-74", "sex": "male", "encounter_setting": "outpatient_specialty"},
    )


def _build_hallucination_case() -> BenchmarkCase:
    case_id = "CASE_003"
    return BenchmarkCase(
        case_id=case_id,
        case_type=CaseType.HALLUCINATION,
        benchmark_version=BENCHMARK_VERSION,
        provenance=DataProvenance.MANUAL_SYNTHETIC,
        age_group="35-44",
        sex="female",
        encounter_type="Urgent care visit",
        conditions=["Acute bronchitis"],
        medications=["Albuterol inhaler, as needed"],
        allergies=["Penicillin (rash)"],
        observations=["Chest clear to auscultation, no wheezes/rales/rhonchi", "No respiratory distress"],
        laboratory_values=[],
        vital_signs=["Temp 99.1 F", "SpO2 98% on room air"],
        clinical_timeline=["5-day history of cough prior to today's visit"],
        clinical_note=(
            "38-year-old female presents with a 5-day history of productive cough, mild "
            "fatigue, and low-grade subjective fevers. Denies shortness of breath at rest. "
            "Chest is clear to auscultation bilaterally with no wheezes, rales, or rhonchi; no "
            "signs of pneumonia on exam. SpO2 98% on room air, temp 99.1 F. Chest X-ray was not "
            "obtained as it is not indicated given a reassuring exam. Presentation is "
            "consistent with acute viral bronchitis. No antibiotics prescribed given likely "
            "viral etiology. Allergic to penicillin (rash). Albuterol inhaler prescribed for "
            "symptomatic relief as needed. Counseled on supportive care: fluids, rest, "
            "over-the-counter antitussives. Return if symptoms worsen or fever persists beyond "
            "3 days."
        ),
        reference_summary=(
            "38-year-old female with 5 days of cough consistent with acute viral bronchitis. "
            "Exam and vitals reassuring, no signs of pneumonia, no imaging or antibiotics "
            "indicated. Albuterol inhaler given for symptomatic relief; penicillin allergy "
            "noted. Supportive care advised."
        ),
        critical_facts=[
            _fact(case_id, 1, FactType.OBSERVATION, "No signs of pneumonia on exam; chest X-ray not indicated at this visit", FactImportance.CRITICAL, FactPolarity.NEGATED),
            _fact(case_id, 2, FactType.OBSERVATION, "No antibiotics prescribed; presentation consistent with viral etiology", FactImportance.CRITICAL, FactPolarity.NEGATED),
            _fact(case_id, 3, FactType.ALLERGY, "Allergic to penicillin (rash)", FactImportance.CRITICAL),
        ],
        non_critical_facts=[
            _fact(case_id, 4, FactType.OBSERVATION, "5-day history of productive cough and mild fatigue", FactImportance.NON_CRITICAL),
        ],
        medication_facts=[
            _med(case_id, 1, "Albuterol", MedicationStatus.NEW, "Albuterol inhaler prescribed as needed for symptomatic relief", FactImportance.NON_CRITICAL, dose="as needed"),
        ],
        temporal_facts=[],
        negations=[
            _fact(case_id, 5, FactType.OBSERVATION, "Denies shortness of breath at rest", FactImportance.NON_CRITICAL, FactPolarity.NEGATED),
        ],
        expected_risk_flags=[
            _risk(
                case_id, 1,
                "A hallucinated finding of pneumonia or an invented antibiotic prescription "
                "(especially given the documented penicillin allergy) would constitute an "
                "unnecessary or unsafe treatment recommendation",
                Severity.HIGH,
            ),
        ],
        subgroup={"age_group": "35-44", "sex": "female", "encounter_setting": "urgent_care"},
    )


def _build_negation_case() -> BenchmarkCase:
    case_id = "CASE_004"
    return BenchmarkCase(
        case_id=case_id,
        case_type=CaseType.NEGATION,
        benchmark_version=BENCHMARK_VERSION,
        provenance=DataProvenance.MANUAL_SYNTHETIC,
        age_group="55-64",
        sex="male",
        encounter_type="Preoperative evaluation",
        conditions=[],
        medications=[],
        allergies=[],
        observations=["No acute distress"],
        clinical_timeline=["Preoperative evaluation prior to elective knee arthroscopy"],
        clinical_note=(
            "58-year-old male presents for preoperative evaluation prior to elective knee "
            "arthroscopy. Patient denies any history of diabetes mellitus. No known drug "
            "allergies. Denies prior history of myocardial infarction or stroke. Denies "
            "current tobacco use; quit smoking 10 years ago. No history of bleeding disorders. "
            "Cardiac exam unremarkable, lungs clear. Cleared for surgery from a medical "
            "standpoint."
        ),
        reference_summary=(
            "58-year-old male cleared for elective knee arthroscopy. No history of diabetes, "
            "no known drug allergies, no prior MI or stroke, no bleeding disorder history, "
            "and a former smoker (quit 10 years ago). Exam unremarkable."
        ),
        critical_facts=[],
        non_critical_facts=[
            _fact(case_id, 1, FactType.OBSERVATION, "Cardiac exam unremarkable, lungs clear", FactImportance.NON_CRITICAL),
        ],
        medication_facts=[],
        temporal_facts=[],
        negations=[
            _fact(case_id, 2, FactType.CONDITION, "No history of diabetes mellitus", FactImportance.CRITICAL, FactPolarity.NEGATED),
            _fact(case_id, 3, FactType.ALLERGY, "No known drug allergies", FactImportance.CRITICAL, FactPolarity.NEGATED),
            _fact(case_id, 4, FactType.CONDITION, "No prior history of myocardial infarction or stroke", FactImportance.CRITICAL, FactPolarity.NEGATED),
            _fact(case_id, 5, FactType.CONDITION, "No history of bleeding disorders", FactImportance.CRITICAL, FactPolarity.NEGATED),
            _fact(case_id, 6, FactType.OBSERVATION, "Denies current tobacco use; quit smoking 10 years ago", FactImportance.NON_CRITICAL, FactPolarity.NEGATED),
        ],
        expected_risk_flags=[
            _risk(
                case_id, 1,
                "Reversing any of these negated diagnoses (e.g. reporting the patient HAS "
                "diabetes or a bleeding disorder) could trigger inappropriate perioperative "
                "management or unnecessary case cancellation",
                Severity.CRITICAL,
            ),
        ],
        subgroup={"age_group": "55-64", "sex": "male", "encounter_setting": "preoperative_clinic"},
    )


def _build_temporal_case() -> BenchmarkCase:
    case_id = "CASE_005"
    return BenchmarkCase(
        case_id=case_id,
        case_type=CaseType.TEMPORAL,
        benchmark_version=BENCHMARK_VERSION,
        provenance=DataProvenance.MANUAL_SYNTHETIC,
        age_group="70-79",
        sex="female",
        encounter_type="Cardiology follow-up",
        conditions=["Heart failure with reduced ejection fraction"],
        medications=["Furosemide", "Metoprolol succinate 50 mg daily"],
        laboratory_values=[],
        vital_signs=["BP 118/74 mmHg", "HR 68 bpm"],
        clinical_timeline=["Two weeks after furosemide dose increase"],
        clinical_note=(
            "74-year-old female with heart failure with reduced ejection fraction returns for "
            "follow-up. Two weeks ago, weight was 68 kg; today weight is 71 kg, consistent with "
            "fluid retention. Furosemide dose was increased from 20 mg to 40 mg daily two weeks "
            "ago in response to that weight gain. Patient reports improved but persistent lower "
            "extremity edema since the dose increase. Metoprolol succinate 50 mg daily "
            "unchanged. Vitals: BP 118/74 mmHg, HR 68 bpm. Plan: continue furosemide 40 mg "
            "daily, recheck weight and basic metabolic panel in one week."
        ),
        reference_summary=(
            "74-year-old female with HFrEF, weight increased from 68 kg to 71 kg over two "
            "weeks prompting a furosemide increase from 20 mg to 40 mg daily; edema improved "
            "but persistent since that change. Metoprolol unchanged. Recheck weight and labs "
            "in one week."
        ),
        critical_facts=[
            _fact(case_id, 1, FactType.OBSERVATION, "Weight increased from 68 kg to 71 kg over two weeks, consistent with fluid retention", FactImportance.CRITICAL),
        ],
        non_critical_facts=[],
        medication_facts=[
            _med(
                case_id, 1, "Furosemide", MedicationStatus.CHANGED_DOSE,
                "Furosemide dose increased from 20 mg to 40 mg daily two weeks ago",
                FactImportance.CRITICAL, dose="40 mg daily (increased from 20 mg)",
            ),
            _med(
                case_id, 2, "Metoprolol succinate", MedicationStatus.CURRENT,
                "Metoprolol succinate 50 mg daily, unchanged",
                FactImportance.NON_CRITICAL, dose="50 mg daily",
            ),
        ],
        temporal_facts=[
            _temporal(
                case_id, 1, "Weight increased from 68 kg to 71 kg over two weeks",
                FactImportance.CRITICAL,
                earlier_value="68 kg", later_value="71 kg", unit="kg", trend=TemporalTrend.INCREASING,
            ),
            _temporal(
                case_id, 2, "Furosemide dose increased from 20 mg to 40 mg daily",
                FactImportance.CRITICAL,
                earlier_value="20 mg daily", later_value="40 mg daily", unit="mg/day", trend=TemporalTrend.INCREASING,
            ),
        ],
        negations=[],
        expected_risk_flags=[
            _risk(
                case_id, 1,
                "Confusing the prior (20 mg) and current (40 mg) furosemide dose, or reversing "
                "the weight trend, could lead to under- or over-diuresis",
                Severity.HIGH,
            ),
        ],
        subgroup={"age_group": "70-79", "sex": "female", "encounter_setting": "outpatient_specialty"},
    )


def _build_medication_case() -> BenchmarkCase:
    case_id = "CASE_006"
    return BenchmarkCase(
        case_id=case_id,
        case_type=CaseType.MEDICATION,
        benchmark_version=BENCHMARK_VERSION,
        provenance=DataProvenance.MANUAL_SYNTHETIC,
        age_group="60-69",
        sex="male",
        encounter_type="Primary care follow-up",
        conditions=["Hypertension", "Type 2 diabetes mellitus", "Hyperlipidemia"],
        medications=["Lisinopril 20 mg daily", "Metformin 1000 mg twice daily", "Atorvastatin 20 mg daily"],
        clinical_timeline=["Follow-up one month after amlodipine was discontinued"],
        clinical_note=(
            "64-year-old male returns for follow-up. Amlodipine was discontinued last month "
            "due to peripheral edema. Lisinopril 20 mg daily continued for blood pressure "
            "control, now at goal. Metformin dose was increased from 500 mg to 1000 mg twice "
            "daily for improved glycemic control given an elevated A1c. Atorvastatin 20 mg "
            "daily newly prescribed today for hyperlipidemia. Patient has a documented history "
            "of taking hydrochlorothiazide in the past, discontinued approximately two years "
            "ago due to hyponatremia; not restarted. No peripheral edema on today's exam."
        ),
        reference_summary=(
            "64-year-old male with hypertension, diabetes, and hyperlipidemia. Amlodipine was "
            "stopped last month for edema, which has resolved. Lisinopril continued at goal "
            "BP. Metformin increased from 500 mg to 1000 mg twice daily for glycemic control. "
            "Atorvastatin 20 mg daily newly started today. Prior hydrochlorothiazide use "
            "(stopped ~2 years ago for hyponatremia) is not restarted."
        ),
        critical_facts=[],
        non_critical_facts=[],
        medication_facts=[
            _med(case_id, 1, "Amlodipine", MedicationStatus.DISCONTINUED, "Amlodipine discontinued last month due to peripheral edema", FactImportance.CRITICAL, dose="was 5 mg daily"),
            _med(case_id, 2, "Lisinopril", MedicationStatus.CURRENT, "Lisinopril 20 mg daily continued, blood pressure now at goal", FactImportance.CRITICAL, dose="20 mg daily"),
            _med(case_id, 3, "Metformin", MedicationStatus.CHANGED_DOSE, "Metformin dose increased from 500 mg to 1000 mg twice daily", FactImportance.CRITICAL, dose="1000 mg twice daily (increased from 500 mg)"),
            _med(case_id, 4, "Atorvastatin", MedicationStatus.NEW, "Atorvastatin 20 mg daily newly prescribed today", FactImportance.CRITICAL, dose="20 mg daily"),
            _med(case_id, 5, "Hydrochlorothiazide", MedicationStatus.HISTORICAL, "History of hydrochlorothiazide use, discontinued approximately two years ago due to hyponatremia; not restarted", FactImportance.NON_CRITICAL),
        ],
        temporal_facts=[
            _temporal(
                case_id, 1, "Metformin dose increased from 500 mg to 1000 mg twice daily",
                FactImportance.CRITICAL,
                earlier_value="500 mg twice daily", later_value="1000 mg twice daily", unit="mg",
                trend=TemporalTrend.INCREASING,
            ),
        ],
        negations=[
            _fact(case_id, 1, FactType.OBSERVATION, "No peripheral edema on today's exam", FactImportance.NON_CRITICAL, FactPolarity.NEGATED),
        ],
        expected_risk_flags=[
            _risk(
                case_id, 1,
                "Confusing discontinued amlodipine with a current medication, or the "
                "historical hydrochlorothiazide with a current one, could result in "
                "duplicate or inappropriate antihypertensive therapy",
                Severity.HIGH,
            ),
        ],
        subgroup={"age_group": "60-69", "sex": "male", "encounter_setting": "outpatient_primary_care"},
    )


def _build_multi_morbidity_case() -> BenchmarkCase:
    case_id = "CASE_007"
    return BenchmarkCase(
        case_id=case_id,
        case_type=CaseType.MULTI_MORBIDITY,
        benchmark_version=BENCHMARK_VERSION,
        provenance=DataProvenance.MANUAL_SYNTHETIC,
        age_group="65-74",
        sex="female",
        encounter_type="Comprehensive chronic disease management visit",
        conditions=[
            "Type 2 diabetes mellitus",
            "Hypertension",
            "Chronic kidney disease, stage 3",
            "Hyperlipidemia",
            "Osteoarthritis",
        ],
        medications=[
            "Metformin 1000 mg twice daily",
            "Lisinopril 10 mg daily",
            "Atorvastatin 40 mg daily",
            "Acetaminophen 500 mg as needed for joint pain",
        ],
        laboratory_values=["A1c 7.2%", "LDL 88 mg/dL", "Creatinine 1.3 mg/dL, stable"],
        vital_signs=["BP 128/80 mmHg"],
        clinical_timeline=["Routine chronic disease management, no acute changes"],
        clinical_note=(
            "69-year-old female with type 2 diabetes, hypertension, CKD stage 3, "
            "hyperlipidemia, and osteoarthritis presents for comprehensive chronic disease "
            "management. Diabetes control is adequate with A1c 7.2% on metformin 1000 mg "
            "twice daily. Blood pressure well-controlled at 128/80 mmHg on lisinopril 10 mg "
            "daily. LDL at goal (88 mg/dL) on atorvastatin 40 mg daily. Creatinine stable at "
            "1.3 mg/dL, consistent with known CKD stage 3, no interval change. Osteoarthritis "
            "of the knees managed with acetaminophen as needed; no NSAIDs given renal "
            "function. All conditions stable; continue current regimen and routine "
            "monitoring."
        ),
        reference_summary=(
            "69-year-old female with five stable chronic conditions: diabetes (A1c 7.2%), "
            "hypertension (128/80 on lisinopril), CKD stage 3 (creatinine 1.3, stable), "
            "hyperlipidemia (LDL at goal on atorvastatin), and osteoarthritis (managed with "
            "acetaminophen, NSAIDs avoided given renal function). All conditions stable; "
            "continue current regimen."
        ),
        critical_facts=[
            _fact(case_id, 1, FactType.CONDITION, "Type 2 diabetes mellitus, A1c 7.2%, adequately controlled", FactImportance.CRITICAL),
            _fact(case_id, 2, FactType.CONDITION, "Hypertension, well-controlled at 128/80 mmHg", FactImportance.CRITICAL),
            _fact(case_id, 3, FactType.CONDITION, "Chronic kidney disease stage 3, creatinine stable at 1.3 mg/dL", FactImportance.CRITICAL),
            _fact(case_id, 4, FactType.OBSERVATION, "NSAIDs avoided for osteoarthritis given renal function", FactImportance.CRITICAL, FactPolarity.NEGATED),
        ],
        non_critical_facts=[
            _fact(case_id, 5, FactType.CONDITION, "Hyperlipidemia, LDL at goal (88 mg/dL)", FactImportance.NON_CRITICAL),
            _fact(case_id, 6, FactType.CONDITION, "Osteoarthritis of the knees, managed with acetaminophen", FactImportance.NON_CRITICAL),
        ],
        medication_facts=[
            _med(case_id, 1, "Metformin", MedicationStatus.CURRENT, "Metformin 1000 mg twice daily continued", FactImportance.CRITICAL, dose="1000 mg twice daily"),
            _med(case_id, 2, "Lisinopril", MedicationStatus.CURRENT, "Lisinopril 10 mg daily continued", FactImportance.CRITICAL, dose="10 mg daily"),
            _med(case_id, 3, "Atorvastatin", MedicationStatus.CURRENT, "Atorvastatin 40 mg daily continued", FactImportance.NON_CRITICAL, dose="40 mg daily"),
            _med(case_id, 4, "Acetaminophen", MedicationStatus.CURRENT, "Acetaminophen 500 mg as needed for joint pain", FactImportance.NON_CRITICAL, dose="500 mg as needed"),
        ],
        temporal_facts=[],
        negations=[],
        expected_risk_flags=[
            _risk(
                case_id, 1,
                "With five conditions and four medications, a summary that drops any single "
                "condition-medication pairing (e.g. the renal-function rationale for avoiding "
                "NSAIDs) could lead to an inappropriate treatment choice",
                Severity.MODERATE,
            ),
        ],
        subgroup={"age_group": "65-74", "sex": "female", "encounter_setting": "outpatient_primary_care"},
    )


def _build_high_complexity_case() -> BenchmarkCase:
    case_id = "CASE_008"
    return BenchmarkCase(
        case_id=case_id,
        case_type=CaseType.HIGH_COMPLEXITY,
        benchmark_version=BENCHMARK_VERSION,
        provenance=DataProvenance.MANUAL_SYNTHETIC,
        age_group="75-84",
        sex="male",
        encounter_type="Hospital discharge summary",
        conditions=[
            "Acute decompensated heart failure",
            "Acute kidney injury, resolved",
            "Atrial fibrillation",
            "Type 2 diabetes mellitus",
        ],
        medications=[
            "Furosemide 40 mg twice daily (discharge)",
            "Metoprolol succinate 25 mg daily",
            "Apixaban 5 mg twice daily",
            "Insulin glargine 20 units nightly",
        ],
        laboratory_values=[
            "Creatinine peaked at 2.1 mg/dL on hospital day 2",
            "Creatinine 1.6 mg/dL at discharge",
            "Potassium 4.2 mEq/L at discharge",
            "BNP 850 pg/mL on admission",
        ],
        vital_signs=["Weight 82 kg at discharge, down from 88 kg on admission"],
        clinical_timeline=[
            "Day 1: admitted for acute decompensated heart failure with volume overload",
            "Day 2: creatinine rose to 2.1 mg/dL, attributed to aggressive diuresis (acute kidney injury)",
            "Day 2-3: IV furosemide held, fluids liberalized, creatinine improved",
            "Day 4: creatinine improved to 1.6 mg/dL, diuresis resumed at lower dose",
            "Day 5: transitioned to oral furosemide 40 mg twice daily, ready for discharge",
        ],
        clinical_note=(
            "81-year-old male admitted for acute decompensated heart failure with volume "
            "overload, weight 88 kg on admission, BNP 850 pg/mL. Also with known atrial "
            "fibrillation on apixaban and type 2 diabetes on insulin glargine. IV furosemide "
            "was started with good diuretic response. On hospital day 2, creatinine rose from "
            "a baseline of 1.2 mg/dL to a peak of 2.1 mg/dL, felt to represent acute kidney "
            "injury related to aggressive diuresis; IV furosemide was held and fluids "
            "liberalized. Creatinine improved to 1.6 mg/dL by day 4, at which point diuresis "
            "was cautiously resumed at a lower dose. Metoprolol succinate was continued "
            "throughout for rate control of atrial fibrillation, uptitrated from 12.5 mg to "
            "25 mg daily as the patient tolerated it hemodynamically. By discharge on day 5, "
            "weight had decreased to 82 kg, potassium was 4.2 mEq/L, and the patient was "
            "transitioned to oral furosemide 40 mg twice daily. Discharge medications: "
            "furosemide 40 mg twice daily, metoprolol succinate 25 mg daily, apixaban 5 mg "
            "twice daily (unchanged throughout admission), insulin glargine 20 units nightly. "
            "Follow-up with cardiology in one week including basic metabolic panel to "
            "reassess renal function and potassium."
        ),
        reference_summary=(
            "81-year-old male admitted for decompensated heart failure (88 to 82 kg with "
            "diuresis, BNP 850). Course complicated by AKI (creatinine 1.2 to 2.1 peak) "
            "attributed to diuresis, which resolved to 1.6 with a brief hold and fluid "
            "liberalization before diuresis cautiously resumed. Metoprolol uptitrated from "
            "12.5 to 25 mg daily for AFib rate control; apixaban unchanged throughout. "
            "Discharged on furosemide 40 mg twice daily, metoprolol succinate 25 mg daily, "
            "apixaban 5 mg twice daily, and insulin glargine 20 units nightly, with cardiology "
            "follow-up and repeat labs in one week."
        ),
        critical_facts=[
            _fact(case_id, 1, FactType.LAB, "Creatinine rose from a baseline of 1.2 mg/dL to a peak of 2.1 mg/dL (acute kidney injury attributed to diuresis)", FactImportance.CRITICAL),
            _fact(case_id, 2, FactType.LAB, "Creatinine improved to 1.6 mg/dL by discharge after IV furosemide was held and fluids liberalized", FactImportance.CRITICAL),
            _fact(case_id, 3, FactType.OBSERVATION, "Weight decreased from 88 kg on admission to 82 kg at discharge with diuresis", FactImportance.CRITICAL),
            _fact(case_id, 4, FactType.MEDICATION, "Apixaban dose unchanged throughout admission", FactImportance.CRITICAL),
        ],
        non_critical_facts=[
            _fact(case_id, 5, FactType.OBSERVATION, "Cardiology follow-up in one week with repeat basic metabolic panel", FactImportance.NON_CRITICAL),
        ],
        medication_facts=[
            _med(case_id, 1, "Furosemide", MedicationStatus.CHANGED_DOSE, "IV furosemide held during AKI, later transitioned to oral furosemide 40 mg twice daily at discharge", FactImportance.CRITICAL, dose="40 mg twice daily (oral, at discharge)"),
            _med(case_id, 2, "Metoprolol succinate", MedicationStatus.CHANGED_DOSE, "Uptitrated from 12.5 mg to 25 mg daily during admission", FactImportance.CRITICAL, dose="25 mg daily (increased from 12.5 mg)"),
            _med(case_id, 3, "Apixaban", MedicationStatus.CURRENT, "Apixaban 5 mg twice daily, unchanged throughout admission", FactImportance.CRITICAL, dose="5 mg twice daily"),
            _med(case_id, 4, "Insulin glargine", MedicationStatus.CURRENT, "Insulin glargine 20 units nightly, discharge medication", FactImportance.NON_CRITICAL, dose="20 units nightly"),
        ],
        temporal_facts=[
            _temporal(case_id, 1, "Creatinine rose from 1.2 to 2.1 mg/dL then improved to 1.6 mg/dL", FactImportance.CRITICAL, earlier_value="1.2 mg/dL", later_value="2.1 mg/dL peak, 1.6 mg/dL at discharge", unit="mg/dL", trend=TemporalTrend.INCREASING),
            _temporal(case_id, 2, "Weight decreased from 88 kg to 82 kg over the admission", FactImportance.CRITICAL, earlier_value="88 kg", later_value="82 kg", unit="kg", trend=TemporalTrend.DECREASING),
            _temporal(case_id, 3, "Metoprolol succinate uptitrated from 12.5 mg to 25 mg daily", FactImportance.CRITICAL, earlier_value="12.5 mg daily", later_value="25 mg daily", unit="mg/day", trend=TemporalTrend.INCREASING),
        ],
        negations=[],
        expected_risk_flags=[
            _risk(case_id, 1, "Confusing the AKI peak creatinine with the discharge creatinine could misrepresent current renal function to the outpatient team", Severity.HIGH),
            _risk(case_id, 2, "Missing the transient furosemide hold could obscure why diuresis was interrupted, or wrongly suggest apixaban was changed for the AKI", Severity.HIGH),
        ],
        subgroup={"age_group": "75-84", "sex": "male", "encounter_setting": "inpatient"},
    )


def _build_conflicting_information_case() -> BenchmarkCase:
    case_id = "CASE_009"
    return BenchmarkCase(
        case_id=case_id,
        case_type=CaseType.CONFLICTING_INFORMATION,
        benchmark_version=BENCHMARK_VERSION,
        provenance=DataProvenance.MANUAL_SYNTHETIC,
        age_group="50-59",
        sex="female",
        encounter_type="Emergency department visit",
        conditions=[],
        medications=[],
        clinical_timeline=[
            "10:00 AM triage note: patient denies chest pain",
            "11:30 AM physician reassessment: patient now reports intermittent chest pain since this morning",
        ],
        clinical_note=(
            "10:00 AM triage note: 54-year-old female presents with fatigue. Patient denies "
            "chest pain, denies shortness of breath. Vitals stable.\n\n"
            "11:30 AM physician reassessment: On further questioning, patient now reports "
            "intermittent chest pain since this morning, described as pressure-like, radiating "
            "to the left arm, associated with mild shortness of breath. This is a change from "
            "the initial triage assessment. EKG and troponin ordered emergently given this "
            "updated history. The 10:00 AM triage note's denial of chest pain is superseded by "
            "this reassessment and should not be treated as the current clinical picture."
        ),
        reference_summary=(
            "54-year-old female initially denied chest pain at triage, but on physician "
            "reassessment 90 minutes later reported intermittent pressure-like chest pain "
            "radiating to the left arm with mild shortness of breath, since this morning. The "
            "later history supersedes the initial denial; EKG and troponin ordered emergently."
        ),
        critical_facts=[
            _fact(
                case_id, 1, FactType.OBSERVATION,
                "On reassessment, patient reports intermittent pressure-like chest pain "
                "radiating to the left arm with mild shortness of breath (supersedes the "
                "earlier triage denial of chest pain)",
                FactImportance.CRITICAL,
                superseded_fact="No chest pain reported at 10:00 AM triage",
            ),
            _fact(case_id, 2, FactType.OBSERVATION, "EKG and troponin ordered emergently given the updated chest pain history", FactImportance.CRITICAL),
        ],
        non_critical_facts=[],
        medication_facts=[],
        temporal_facts=[],
        negations=[
            _fact(
                case_id, 3, FactType.OBSERVATION,
                "10:00 AM triage: patient denies chest pain and shortness of breath",
                FactImportance.NON_CRITICAL,
                FactPolarity.NEGATED,
                superseded_by="CASE_009-F001",
            ),
        ],
        expected_risk_flags=[
            _risk(
                case_id, 1,
                "A summary built only from the earlier triage note (or that fails to indicate "
                "the later report supersedes it) would omit an emergent, potentially "
                "cardiac symptom",
                Severity.CRITICAL,
            ),
        ],
        subgroup={"age_group": "50-59", "sex": "female", "encounter_setting": "emergency_department"},
    )


def _build_demographic_subgroup_case() -> BenchmarkCase:
    case_id = "CASE_010"
    return BenchmarkCase(
        case_id=case_id,
        case_type=CaseType.DEMOGRAPHIC_SUBGROUP,
        benchmark_version=BENCHMARK_VERSION,
        provenance=DataProvenance.MANUAL_SYNTHETIC,
        age_group="8-12",
        sex="male",
        encounter_type="Pediatric well-child visit",
        conditions=[],
        medications=[],
        allergies=["No known drug allergies"],
        observations=["Growth parameters tracking along established curve", "Development age-appropriate"],
        laboratory_values=[],
        vital_signs=["Height and weight at 60th percentile for age", "BP 98/60 mmHg"],
        clinical_timeline=["Routine annual well-child visit"],
        clinical_note=(
            "10-year-old male presents for routine well-child visit. Growth parameters "
            "(height and weight) tracking along the established curve at the 60th percentile. "
            "Development is age-appropriate; no concerns raised by parent or child. "
            "Immunizations are up to date per current schedule. No known drug allergies. "
            "Physical exam unremarkable. Vision and hearing screens passed. Anticipatory "
            "guidance provided on nutrition, physical activity, and screen time. Return for "
            "next well-child visit in one year."
        ),
        reference_summary=(
            "10-year-old male, routine well-child visit. Growth and development are "
            "age-appropriate (60th percentile), immunizations up to date, exam and screens "
            "unremarkable. Anticipatory guidance given; follow up in one year."
        ),
        critical_facts=[],
        non_critical_facts=[
            _fact(case_id, 1, FactType.OBSERVATION, "Growth parameters tracking along established curve at 60th percentile", FactImportance.NON_CRITICAL),
            _fact(case_id, 2, FactType.OBSERVATION, "Development is age-appropriate", FactImportance.NON_CRITICAL),
            _fact(case_id, 3, FactType.OBSERVATION, "Immunizations up to date per current schedule", FactImportance.NON_CRITICAL),
        ],
        medication_facts=[],
        temporal_facts=[],
        negations=[
            _fact(case_id, 4, FactType.ALLERGY, "No known drug allergies", FactImportance.NON_CRITICAL, FactPolarity.NEGATED),
        ],
        expected_risk_flags=[],
        subgroup={"age_group": "8-12", "sex": "male", "encounter_setting": "pediatric_primary_care"},
    )


_CASE_BUILDERS = [
    _build_normal_case,
    _build_critical_omission_case,
    _build_hallucination_case,
    _build_negation_case,
    _build_temporal_case,
    _build_medication_case,
    _build_multi_morbidity_case,
    _build_high_complexity_case,
    _build_conflicting_information_case,
    _build_demographic_subgroup_case,
]


def generate_default_benchmark() -> list[BenchmarkCase]:
    """Return the Phase 1 benchmark: one case per `CaseType`, fully annotated.

    Deterministic - calling this twice yields byte-identical output, which is
    what "reproducible benchmark" means for a hand-authored set like this one.
    """
    return [builder() for builder in _CASE_BUILDERS]


def save_benchmark(cases: list[BenchmarkCase], path: str | Path) -> Path:
    """Write `cases` to `path` as a JSON array. Creates parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [case.model_dump(mode="json") for case in cases]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_benchmark(path: str | Path) -> list[BenchmarkCase]:
    """Load and validate a benchmark JSON file into `BenchmarkCase` objects.

    Raises `FileNotFoundError` with a clear message if the file is missing,
    and lets pydantic's `ValidationError` surface (rather than swallowing it)
    if the file is malformed - failing loudly beats silently evaluating
    against corrupted ground truth.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}. Run `python scripts/generate_benchmark.py` "
            "to generate the default synthetic benchmark."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkCase.model_validate(item) for item in raw]
