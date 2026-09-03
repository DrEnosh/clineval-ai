"""The structured synthetic case: ClinEval's source of truth.

Everything a benchmark case contains is *derived from* a `StructuredCase`:

    StructuredCase -> clinical timeline -> source note
                   -> reference summary          (independently)
                   -> structured annotations     (independently)

The note, the reference summary, and the gold-standard annotations are three
independent renderings of the same structured object. None of them is derived
from another, and no model output ever feeds back into this layer. That is
what makes the ground truth trustworthy: an annotation cannot disagree with
the note, because neither was written by hand against the other.

Every clinical element is a `StructuredEntity` with a stable `entity_id`.
Annotations reference those ids, which is what lets the integrity checker
verify that every annotation traces to real structured content rather than
to something a renderer invented.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.enums import (
    AbnormalFlag,
    AllergySeverity,
    ConditionStatus,
    DifficultyTier,
    FailureModeCategory,
    FailureOpportunityType,
    FindingSignificance,
    MedicationStatus,
    Severity,
    TemporalTrend,
    TimelineEventType,
)


class StructuredEntity(BaseModel):
    """Base for every clinical element in a case.

    `entity_id` is unique within a case (e.g. "MED-002", "LAB-005") and is the
    anchor annotations point at.
    """

    entity_id: str


class PatientProfile(BaseModel):
    """Synthetic demographics only.

    `patient_id` is a generated surrogate (e.g. "SYN-PT-00042"). There is no
    name, no date of birth, no address, no contact detail, and no identifier
    that could correspond to a real person - asserted by the integrity checks.
    """

    patient_id: str
    age: int
    age_group: str
    sex: str


class EncounterInfo(BaseModel):
    """Encounter context. Timing is always relative (day offsets, "3 months
    ago"), never an absolute calendar date, so cases carry no date that could
    anchor to a real record."""

    encounter_id: str
    encounter_type: str
    setting: str
    length_of_stay_days: int | None = None


class ConditionEntry(StructuredEntity):
    name: str
    status: ConditionStatus = ConditionStatus.ACTIVE
    onset_description: str | None = None
    is_primary: bool = False


class MedicationEntry(StructuredEntity):
    drug_name: str
    status: MedicationStatus
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    prior_dose: str | None = None
    change_reason: str | None = None
    change_timing: str | None = None
    indication: str | None = None

    @property
    def is_change(self) -> bool:
        """True for the statuses criticality rule CR-04 treats as a major
        medication change."""
        return self.status in {
            MedicationStatus.NEW,
            MedicationStatus.DISCONTINUED,
            MedicationStatus.CHANGED_DOSE,
        }

    def describe(self) -> str:
        parts = [self.drug_name]
        if self.dose:
            parts.append(self.dose)
        if self.route:
            parts.append(self.route)
        if self.frequency:
            parts.append(self.frequency)
        return " ".join(parts)


class LabResultEntry(StructuredEntity):
    name: str
    value: float
    unit: str
    day_offset: int = 0
    timing_description: str = "today"
    abnormal_flag: AbnormalFlag = AbnormalFlag.NORMAL
    reference_low: float | None = None
    reference_high: float | None = None

    @property
    def is_abnormal(self) -> bool:
        return self.abnormal_flag is not AbnormalFlag.NORMAL

    @property
    def is_critical_value(self) -> bool:
        return self.abnormal_flag in {AbnormalFlag.CRITICAL_LOW, AbnormalFlag.CRITICAL_HIGH}

    def describe(self) -> str:
        return f"{self.name} {self.format_value()} {self.unit}"

    def format_value(self) -> str:
        """Render the value without a trailing ".0" on whole numbers, so a
        WBC reads "14" rather than "14.0"."""
        if self.value == int(self.value):
            return str(int(self.value))
        return f"{self.value:g}"


class LabTrendEntry(StructuredEntity):
    """An explicit link between two `LabResultEntry` rows for the same analyte.

    Temporal annotations derive from this, and the integrity checker verifies
    both referenced lab results actually exist - which is how "every temporal
    annotation corresponds to actual timeline data" is enforced rather than
    assumed.
    """

    lab_name: str
    unit: str
    earlier_ref: str
    later_ref: str
    earlier_value: float
    later_value: float
    earlier_timing: str
    later_timing: str
    trend: TemporalTrend
    clinically_significant: bool = False

    def format_value(self, value: float) -> str:
        if value == int(value):
            return str(int(value))
        return f"{value:g}"

    def describe(self) -> str:
        return (
            f"{self.lab_name} {self.trend.value} from "
            f"{self.format_value(self.earlier_value)} to "
            f"{self.format_value(self.later_value)} {self.unit} "
            f"({self.earlier_timing} to {self.later_timing})"
        )


class VitalSignEntry(StructuredEntity):
    name: str
    value: str
    unit: str | None = None
    day_offset: int = 0
    timing_description: str = "today"
    abnormal_flag: AbnormalFlag = AbnormalFlag.NORMAL

    @property
    def is_abnormal(self) -> bool:
        return self.abnormal_flag is not AbnormalFlag.NORMAL

    def describe(self) -> str:
        return f"{self.name} {self.value}{' ' + self.unit if self.unit else ''}"


class ProcedureEntry(StructuredEntity):
    name: str
    timing_description: str = "today"
    day_offset: int = 0
    outcome: str | None = None
    performed: bool = True


class AllergyEntry(StructuredEntity):
    substance: str
    reaction: str
    severity: AllergySeverity = AllergySeverity.MODERATE

    def describe(self) -> str:
        return f"{self.substance} - {self.reaction}"


class ClinicalFindingEntry(StructuredEntity):
    """An examination or history finding.

    `present=False` is a *relevant negative* - "no jugular venous distension",
    "no history of diabetes". These become negated gold-standard facts, and
    they are the single most important thing for a summarizer not to flip.
    """

    description: str
    present: bool = True
    significance: FindingSignificance = FindingSignificance.ROUTINE
    timing_description: str = "today"


class DiagnosisEntry(StructuredEntity):
    name: str
    is_primary: bool = False
    certainty: str = "confirmed"


class FollowUpPlanEntry(StructuredEntity):
    interval: str
    provider: str
    actions: list[str] = Field(default_factory=list)


class ConflictEntry(StructuredEntity):
    """Two statements from the same encounter that disagree, plus which one wins.

    Modelled explicitly (rather than as two loose findings) because the
    clinically important content is the *resolution*: a summary that reports
    the superseded statement as current is wrong even though it quoted the
    note accurately.
    """

    topic: str
    earlier_statement: str
    earlier_timing: str
    later_statement: str
    later_timing: str
    resolution: str


class TimelineEvent(BaseModel):
    day_offset: int
    event_type: TimelineEventType
    description: str
    entity_refs: list[str] = Field(default_factory=list)


class FailureOpportunity(BaseModel):
    """A documented way a summarizer could fail on this case.

    A benchmark condition, not a model output and not a defect in the case.
    It records what a model must get right and how bad it would be to get it
    wrong, so later phases can measure whether a model fell into a trap that
    was known in advance.
    """

    opportunity_id: str
    type: FailureOpportunityType
    description: str
    target_ref: str | None = None
    severity_if_missed: Severity = Severity.MODERATE


class StructuredCase(BaseModel):
    """The complete structured representation of one synthetic case."""

    case_id: str
    seed: str
    generator_version: str

    patient: PatientProfile
    encounter: EncounterInfo

    conditions: list[ConditionEntry] = Field(default_factory=list)
    medications: list[MedicationEntry] = Field(default_factory=list)
    allergies: list[AllergyEntry] = Field(default_factory=list)
    procedures: list[ProcedureEntry] = Field(default_factory=list)
    labs: list[LabResultEntry] = Field(default_factory=list)
    lab_trends: list[LabTrendEntry] = Field(default_factory=list)
    vitals: list[VitalSignEntry] = Field(default_factory=list)
    findings: list[ClinicalFindingEntry] = Field(default_factory=list)
    diagnoses: list[DiagnosisEntry] = Field(default_factory=list)
    follow_up: list[FollowUpPlanEntry] = Field(default_factory=list)
    conflicts: list[ConflictEntry] = Field(default_factory=list)

    timeline: list[TimelineEvent] = Field(default_factory=list)

    difficulty: DifficultyTier
    categories: list[FailureModeCategory] = Field(default_factory=list)
    failure_opportunities: list[FailureOpportunity] = Field(default_factory=list)

    def all_entities(self) -> list[StructuredEntity]:
        return [
            *self.conditions, *self.medications, *self.allergies, *self.procedures,
            *self.labs, *self.lab_trends, *self.vitals, *self.findings,
            *self.diagnoses, *self.follow_up, *self.conflicts,
        ]

    def entity_index(self) -> dict[str, StructuredEntity]:
        """Map entity_id -> entity, used by the integrity checker to resolve
        every annotation's `source_ref`."""
        return {entity.entity_id: entity for entity in self.all_entities()}

    @property
    def active_conditions(self) -> list[ConditionEntry]:
        return [c for c in self.conditions if c.status is ConditionStatus.ACTIVE]

    @property
    def historical_conditions(self) -> list[ConditionEntry]:
        return [c for c in self.conditions if c.status is not ConditionStatus.ACTIVE]

    @property
    def current_medications(self) -> list[MedicationEntry]:
        return [
            m for m in self.medications
            if m.status in {MedicationStatus.CURRENT, MedicationStatus.NEW, MedicationStatus.CHANGED_DOSE}
        ]

    @property
    def medication_changes(self) -> list[MedicationEntry]:
        return [m for m in self.medications if m.is_change]

    @property
    def abnormal_labs(self) -> list[LabResultEntry]:
        return [lab for lab in self.labs if lab.is_abnormal]

    @property
    def relevant_negatives(self) -> list[ClinicalFindingEntry]:
        return [f for f in self.findings if not f.present]

    def text_corpus(self) -> str:
        """All free text held in the structured case, lowercased.

        The integrity checker uses this to verify that nothing in a rendered
        note or reference summary was invented by a renderer.
        """
        parts: list[str] = [
            self.patient.age_group, self.patient.sex, str(self.patient.age),
            self.encounter.encounter_type, self.encounter.setting,
        ]
        if self.encounter.length_of_stay_days is not None:
            parts.append(str(self.encounter.length_of_stay_days))
        for condition in self.conditions:
            parts += [condition.name, condition.onset_description or ""]
        for med in self.medications:
            parts += [
                med.drug_name, med.dose or "", med.route or "", med.frequency or "",
                med.prior_dose or "", med.change_reason or "", med.change_timing or "",
                med.indication or "",
            ]
        for allergy in self.allergies:
            parts += [allergy.substance, allergy.reaction]
        for procedure in self.procedures:
            parts += [procedure.name, procedure.timing_description, procedure.outcome or ""]
        for lab in self.labs:
            parts += [
                lab.name, lab.format_value(), lab.unit, lab.timing_description,
                str(lab.reference_low or ""), str(lab.reference_high or ""),
            ]
        for trend in self.lab_trends:
            parts += [
                trend.lab_name, trend.unit, trend.earlier_timing, trend.later_timing,
                trend.format_value(trend.earlier_value), trend.format_value(trend.later_value),
            ]
        for vital in self.vitals:
            parts += [vital.name, vital.value, vital.unit or "", vital.timing_description]
        for finding in self.findings:
            parts += [finding.description, finding.timing_description]
        for diagnosis in self.diagnoses:
            parts.append(diagnosis.name)
        for plan in self.follow_up:
            parts += [plan.interval, plan.provider, *plan.actions]
        for conflict in self.conflicts:
            parts += [
                conflict.topic, conflict.earlier_statement, conflict.later_statement,
                conflict.earlier_timing, conflict.later_timing, conflict.resolution,
            ]
        for event in self.timeline:
            parts.append(event.description)
        return " ".join(parts).lower()
