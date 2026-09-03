"""Criticality rules: what makes a benchmark fact "critical".

Criticality is assigned by an explicit, enumerated rule - never inferred from
how a sentence is worded. Every fact the annotator marks critical carries the
id of the rule that marked it, so any importance label in the benchmark can be
traced back to a stated reason.

    IMPORTANT: these are BENCHMARK DESIGN LABELS, not a universal clinical
    standard. They encode "a summary that drops this is doing something the
    benchmark counts as a serious error". They are not triage criteria, not
    clinical practice guidance, and not validated against any external
    standard. A different institution would reasonably draw these lines
    differently, which is exactly why they are configurable rules with ids
    rather than hard-coded judgments.

Rules are also what gives the benchmark a defensible answer to "why is this
fact critical and that one not?" - the answer is always a rule id.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.enums import AllergySeverity, FindingSignificance, Severity
from app.models.structured import (
    AllergyEntry,
    ClinicalFindingEntry,
    ConflictEntry,
    DiagnosisEntry,
    LabResultEntry,
    LabTrendEntry,
    MedicationEntry,
    VitalSignEntry,
)


@dataclass(frozen=True)
class CriticalityRule:
    rule_id: str
    description: str
    rationale: str
    severity_if_missed: Severity


CRITICALITY_RULES: dict[str, CriticalityRule] = {
    "CR-01": CriticalityRule(
        "CR-01",
        "A laboratory value at or beyond a panic threshold is critical.",
        "A value in the panic range typically drives immediate clinical action; "
        "omitting it from a summary hides the reason for that action.",
        Severity.CRITICAL,
    ),
    "CR-02": CriticalityRule(
        "CR-02",
        "An abnormal value for a key analyte is critical.",
        "Key analytes (renal function, electrolytes, hemoglobin, cardiac and "
        "sepsis markers) commonly change management; abnormal values for "
        "non-key analytes are recorded but not treated as critical.",
        Severity.HIGH,
    ),
    "CR-03": CriticalityRule(
        "CR-03",
        "A clinically significant laboratory trend is critical.",
        "The direction and magnitude of change can matter more than either "
        "single value; reporting only one endpoint misrepresents the trajectory.",
        Severity.HIGH,
    ),
    "CR-04": CriticalityRule(
        "CR-04",
        "A major medication change (new, discontinued, or dose change) is critical.",
        "Medication changes are the most actionable content in a clinical "
        "summary; confusing a stopped drug for a current one risks duplicate "
        "or contraindicated therapy.",
        Severity.HIGH,
    ),
    "CR-05": CriticalityRule(
        "CR-05",
        "A severe allergy is critical.",
        "A missed severe allergy can lead directly to administration of an "
        "agent the patient reacts to.",
        Severity.CRITICAL,
    ),
    "CR-06": CriticalityRule(
        "CR-06",
        "A primary diagnosis is critical.",
        "The primary diagnosis is the organizing fact of the encounter.",
        Severity.HIGH,
    ),
    "CR-07": CriticalityRule(
        "CR-07",
        "A resolved contradiction between earlier and later documentation is critical.",
        "A summary that reports the superseded statement as current is wrong "
        "even though it quoted the note accurately.",
        Severity.CRITICAL,
    ),
    "CR-08": CriticalityRule(
        "CR-08",
        "A finding marked clinically critical is critical, including a negative finding.",
        "Relevant negatives ('no history of diabetes') carry as much weight as "
        "positives, and are the findings most often reversed by summarizers.",
        Severity.HIGH,
    ),
    "CR-09": CriticalityRule(
        "CR-09",
        "A vital sign at or beyond a panic threshold is critical.",
        "Panic-range vitals indicate acute physiological instability.",
        Severity.CRITICAL,
    ),
    "CR-10": CriticalityRule(
        "CR-10",
        "An acute deterioration event in the timeline is critical.",
        "Deterioration explains why management changed and must survive "
        "summarization for the course of the encounter to make sense.",
        Severity.HIGH,
    ),
}


def rule(rule_id: str) -> CriticalityRule:
    return CRITICALITY_RULES[rule_id]


# Each classifier returns the id of the rule that makes the item critical, or
# None if no rule applies. Keeping them tiny and pure makes each one directly
# testable against its rule.


def classify_lab(lab: LabResultEntry) -> str | None:
    from app.benchmark.content import LAB_TEMPLATES

    if lab.is_critical_value:
        return "CR-01"
    template = LAB_TEMPLATES.get(lab.name)
    if lab.is_abnormal and template is not None and template.is_key_lab:
        return "CR-02"
    return None


def classify_lab_trend(trend: LabTrendEntry) -> str | None:
    return "CR-03" if trend.clinically_significant else None


def classify_medication(medication: MedicationEntry) -> str | None:
    return "CR-04" if medication.is_change else None


def classify_allergy(allergy: AllergyEntry) -> str | None:
    return "CR-05" if allergy.severity is AllergySeverity.SEVERE else None


def classify_diagnosis(diagnosis: DiagnosisEntry) -> str | None:
    return "CR-06" if diagnosis.is_primary else None


def classify_conflict(conflict: ConflictEntry) -> str | None:
    return "CR-07"


def classify_finding(finding: ClinicalFindingEntry) -> str | None:
    return "CR-08" if finding.significance is FindingSignificance.CRITICAL else None


def classify_vital(vital: VitalSignEntry) -> str | None:
    from app.core.enums import AbnormalFlag

    if vital.abnormal_flag in {AbnormalFlag.CRITICAL_LOW, AbnormalFlag.CRITICAL_HIGH}:
        return "CR-09"
    return None
