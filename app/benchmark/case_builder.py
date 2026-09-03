"""Assembly of a `StructuredCase` from the content library.

`CaseDraft` is the mutable working object: it allocates stable entity ids,
holds the growing lists of clinical entities, and records timeline events and
failure opportunities. Feature builders (app/benchmark/features.py) mutate a
draft; this module provides the base patient/encounter scaffold they build on
and the sampling helpers they share.

All randomness flows through a single seeded `random.Random` passed in by the
synthesizer, so a case is fully determined by its seed.
"""

from __future__ import annotations

from random import Random

from app.benchmark import content
from app.benchmark.content import ConditionTemplate, LabTemplate, MedicationTemplate
from app.core.enums import (
    AbnormalFlag,
    ConditionStatus,
    DifficultyTier,
    FailureModeCategory,
    FailureOpportunityType,
    FindingSignificance,
    MedicationStatus,
    Severity,
    TimelineEventType,
)
from app.models.structured import (
    AllergyEntry,
    ClinicalFindingEntry,
    ConditionEntry,
    ConflictEntry,
    DiagnosisEntry,
    EncounterInfo,
    FailureOpportunity,
    FollowUpPlanEntry,
    LabResultEntry,
    LabTrendEntry,
    MedicationEntry,
    PatientProfile,
    ProcedureEntry,
    StructuredCase,
    TimelineEvent,
    VitalSignEntry,
)

GENERATOR_VERSION = "0.2.0"


class CaseDraft:
    """Mutable case under construction."""

    def __init__(self, case_id: str, seed: str, rng: Random, difficulty: DifficultyTier):
        self.case_id = case_id
        self.seed = seed
        self.rng = rng
        self.difficulty = difficulty

        self.patient: PatientProfile | None = None
        self.encounter: EncounterInfo | None = None

        self.conditions: list[ConditionEntry] = []
        self.medications: list[MedicationEntry] = []
        self.allergies: list[AllergyEntry] = []
        self.procedures: list[ProcedureEntry] = []
        self.labs: list[LabResultEntry] = []
        self.lab_trends: list[LabTrendEntry] = []
        self.vitals: list[VitalSignEntry] = []
        self.findings: list[ClinicalFindingEntry] = []
        self.diagnoses: list[DiagnosisEntry] = []
        self.follow_up: list[FollowUpPlanEntry] = []
        self.conflicts: list[ConflictEntry] = []
        self.timeline: list[TimelineEvent] = []
        self.failure_opportunities: list[FailureOpportunity] = []
        self.categories: list[FailureModeCategory] = []

        # Rendering hints set by features; the note renderer reads these
        # rather than re-deriving intent from the categories list.
        self.verbose_note = False
        self.terse_note = False
        self.duplicate_documentation = False

        self._counters: dict[str, int] = {}
        self._condition_templates: list[ConditionTemplate] = []

    def next_id(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}-{self._counters[prefix]:03d}"

    # -- registration helpers ------------------------------------------------

    def add_category(self, category: FailureModeCategory) -> None:
        if category not in self.categories:
            self.categories.append(category)

    def add_event(
        self,
        day_offset: int,
        event_type: TimelineEventType,
        description: str,
        refs: list[str] | None = None,
    ) -> None:
        self.timeline.append(
            TimelineEvent(
                day_offset=day_offset,
                event_type=event_type,
                description=description,
                entity_refs=refs or [],
            )
        )

    def add_opportunity(
        self,
        opportunity_type: FailureOpportunityType,
        description: str,
        target_ref: str | None = None,
        severity: Severity = Severity.MODERATE,
    ) -> FailureOpportunity:
        opportunity = FailureOpportunity(
            opportunity_id=self.next_id("OPP"),
            type=opportunity_type,
            description=description,
            target_ref=target_ref,
            severity_if_missed=severity,
        )
        self.failure_opportunities.append(opportunity)
        return opportunity

    # -- lookups used by features -------------------------------------------

    @property
    def condition_templates(self) -> list[ConditionTemplate]:
        return list(self._condition_templates)

    def register_condition_template(self, template: ConditionTemplate) -> None:
        self._condition_templates.append(template)

    def has_drug(self, drug_name: str) -> bool:
        return any(m.drug_name == drug_name for m in self.medications)

    def has_lab(self, lab_name: str) -> bool:
        return any(lab.name == lab_name for lab in self.labs)

    def condition_names(self) -> set[str]:
        return {c.name for c in self.conditions}

    def candidate_labs(self) -> list[str]:
        """Labs relevant to this case's conditions, in a stable order."""
        names: list[str] = []
        for template in self._condition_templates:
            for lab_name in template.labs:
                if lab_name not in names:
                    names.append(lab_name)
        return names

    # -- entity constructors -------------------------------------------------

    def add_condition(
        self,
        template: ConditionTemplate,
        status: ConditionStatus = ConditionStatus.ACTIVE,
        is_primary: bool = False,
        onset: str | None = None,
    ) -> ConditionEntry:
        entry = ConditionEntry(
            entity_id=self.next_id("COND"),
            name=template.name,
            status=status,
            onset_description=onset,
            is_primary=is_primary,
        )
        self.conditions.append(entry)
        self.register_condition_template(template)
        return entry

    def add_medication(
        self,
        template: MedicationTemplate,
        status: MedicationStatus,
        dose: str | None = None,
        prior_dose: str | None = None,
        change_reason: str | None = None,
        change_timing: str | None = None,
    ) -> MedicationEntry:
        entry = MedicationEntry(
            entity_id=self.next_id("MED"),
            drug_name=template.drug_name,
            status=status,
            dose=dose if dose is not None else self.rng.choice(template.doses),
            route=template.route,
            frequency=template.frequency,
            prior_dose=prior_dose,
            change_reason=change_reason,
            change_timing=change_timing,
            indication=template.indication,
        )
        self.medications.append(entry)
        return entry

    def add_lab(
        self,
        template: LabTemplate,
        value: float,
        day_offset: int = 0,
        timing: str = "today",
    ) -> LabResultEntry:
        entry = LabResultEntry(
            entity_id=self.next_id("LAB"),
            name=template.name,
            value=round(value, template.decimals),
            unit=template.unit,
            day_offset=day_offset,
            timing_description=timing,
            abnormal_flag=template.classify(round(value, template.decimals)),
            reference_low=template.reference_low,
            reference_high=template.reference_high,
        )
        self.labs.append(entry)
        self.add_event(day_offset, TimelineEventType.LAB_RESULT, f"{entry.describe()} ({timing})", [entry.entity_id])
        return entry

    def add_vital(
        self,
        name: str,
        value: str,
        unit: str | None,
        flag: AbnormalFlag = AbnormalFlag.NORMAL,
        timing: str = "today",
    ) -> VitalSignEntry:
        entry = VitalSignEntry(
            entity_id=self.next_id("VIT"),
            name=name, value=value, unit=unit,
            abnormal_flag=flag, timing_description=timing,
        )
        self.vitals.append(entry)
        return entry

    def add_finding(
        self,
        description: str,
        present: bool = True,
        significance: FindingSignificance = FindingSignificance.ROUTINE,
        timing: str = "today",
    ) -> ClinicalFindingEntry:
        entry = ClinicalFindingEntry(
            entity_id=self.next_id("FIND"),
            description=description, present=present,
            significance=significance, timing_description=timing,
        )
        self.findings.append(entry)
        return entry

    def add_allergy(self, template: content.AllergyTemplate) -> AllergyEntry:
        entry = AllergyEntry(
            entity_id=self.next_id("ALG"),
            substance=template.substance,
            reaction=template.reaction,
            severity=template.severity,
        )
        self.allergies.append(entry)
        return entry

    def add_procedure(
        self, name: str, timing: str = "today", outcome: str | None = None, performed: bool = True
    ) -> ProcedureEntry:
        entry = ProcedureEntry(
            entity_id=self.next_id("PROC"),
            name=name, timing_description=timing, outcome=outcome, performed=performed,
        )
        self.procedures.append(entry)
        if performed:
            self.add_event(0, TimelineEventType.PROCEDURE, f"{name} performed ({timing})", [entry.entity_id])
        return entry

    def add_diagnosis(self, name: str, is_primary: bool = False) -> DiagnosisEntry:
        entry = DiagnosisEntry(entity_id=self.next_id("DX"), name=name, is_primary=is_primary)
        self.diagnoses.append(entry)
        return entry

    def add_follow_up(self, interval: str, provider: str, actions: list[str]) -> FollowUpPlanEntry:
        entry = FollowUpPlanEntry(
            entity_id=self.next_id("FU"), interval=interval, provider=provider, actions=actions,
        )
        self.follow_up.append(entry)
        return entry

    def add_conflict(
        self,
        topic: str,
        earlier_statement: str,
        earlier_timing: str,
        later_statement: str,
        later_timing: str,
        resolution: str,
    ) -> ConflictEntry:
        entry = ConflictEntry(
            entity_id=self.next_id("CONF"),
            topic=topic,
            earlier_statement=earlier_statement, earlier_timing=earlier_timing,
            later_statement=later_statement, later_timing=later_timing,
            resolution=resolution,
        )
        self.conflicts.append(entry)
        self.add_event(0, TimelineEventType.REASSESSMENT, resolution, [entry.entity_id])
        return entry

    def add_lab_trend(
        self,
        template: LabTemplate,
        earlier: LabResultEntry,
        later: LabResultEntry,
        significant: bool,
    ) -> LabTrendEntry:
        from app.core.enums import TemporalTrend

        if later.value > earlier.value:
            trend = TemporalTrend.INCREASING
        elif later.value < earlier.value:
            trend = TemporalTrend.DECREASING
        else:
            trend = TemporalTrend.STABLE

        entry = LabTrendEntry(
            entity_id=self.next_id("TRND"),
            lab_name=template.name, unit=template.unit,
            earlier_ref=earlier.entity_id, later_ref=later.entity_id,
            earlier_value=earlier.value, later_value=later.value,
            earlier_timing=earlier.timing_description, later_timing=later.timing_description,
            trend=trend, clinically_significant=significant,
        )
        self.lab_trends.append(entry)
        return entry

    def build(self) -> StructuredCase:
        assert self.patient is not None and self.encounter is not None
        return StructuredCase(
            case_id=self.case_id,
            seed=self.seed,
            generator_version=GENERATOR_VERSION,
            patient=self.patient,
            encounter=self.encounter,
            conditions=self.conditions,
            medications=self.medications,
            allergies=self.allergies,
            procedures=self.procedures,
            labs=self.labs,
            lab_trends=self.lab_trends,
            vitals=self.vitals,
            findings=self.findings,
            diagnoses=self.diagnoses,
            follow_up=self.follow_up,
            conflicts=self.conflicts,
            timeline=sorted(self.timeline, key=lambda e: e.day_offset),
            difficulty=self.difficulty,
            categories=self.categories,
            failure_opportunities=self.failure_opportunities,
        )


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def sample_normal_value(template: LabTemplate, rng: Random) -> float:
    low, high = template.reference_low, template.reference_high
    span = high - low
    return round(rng.uniform(low + span * 0.15, high - span * 0.15), template.decimals)


def sample_abnormal_value(template: LabTemplate, rng: Random) -> float:
    """A value outside the reference range but inside the panic thresholds.

    Deliberately not critical: CR-02 cases must be distinguishable from CR-01
    cases, so this helper stays clear of the panic boundary.
    """
    if template.concern_direction == "low":
        ceiling = template.reference_low
        floor = template.critical_low if template.critical_low is not None else ceiling * 0.6
        floor = floor + (ceiling - floor) * 0.25
        return round(rng.uniform(floor, ceiling * 0.95), template.decimals)
    floor = template.reference_high
    cap = template.critical_high if template.critical_high is not None else floor * 1.8
    cap = floor + (cap - floor) * 0.75
    return round(rng.uniform(floor * 1.05, max(cap, floor * 1.1)), template.decimals)


def sample_critical_value(template: LabTemplate, rng: Random) -> float | None:
    """A value at or beyond a panic threshold, or None if this analyte has none."""
    if template.concern_direction == "low" and template.critical_low is not None:
        return round(rng.uniform(template.critical_low * 0.75, template.critical_low), template.decimals)
    if template.critical_high is not None:
        return round(rng.uniform(template.critical_high, template.critical_high * 1.3), template.decimals)
    if template.critical_low is not None:
        return round(rng.uniform(template.critical_low * 0.75, template.critical_low), template.decimals)
    return None


def next_dose_up(template: MedicationTemplate, current: str) -> str | None:
    """The next dose up the ladder, used for dose-increase cases."""
    doses = list(template.doses)
    if current not in doses:
        return None
    index = doses.index(current)
    return doses[index + 1] if index + 1 < len(doses) else None


def build_base_case(draft: CaseDraft, rng: Random) -> None:
    """Establish patient, encounter, a primary condition, its medication, and
    baseline vitals - the scaffold every case shares before features apply."""
    age_group, low, high = rng.choice(content.AGE_GROUPS)
    age = rng.randint(low, high)
    sex = rng.choice(content.SEXES)

    draft.patient = PatientProfile(
        patient_id=f"SYN-PT-{rng.randrange(10**5):05d}",
        age=age, age_group=age_group, sex=sex,
    )

    primary_template = rng.choice(content.CONDITION_TEMPLATES)
    encounter_type = (
        rng.choice(primary_template.encounter_types)
        if primary_template.encounter_types
        else "Primary care follow-up"
    )
    setting = content.ENCOUNTER_SETTINGS.get(encounter_type, "outpatient_primary_care")
    length_of_stay = rng.randint(2, 6) if setting == "inpatient" else None

    draft.encounter = EncounterInfo(
        encounter_id=f"SYN-ENC-{rng.randrange(10**5):05d}",
        encounter_type=encounter_type,
        setting=setting,
        length_of_stay_days=length_of_stay,
    )

    condition = draft.add_condition(primary_template, is_primary=True, onset=rng.choice(content.PAST_TIMINGS))
    draft.add_diagnosis(primary_template.name, is_primary=True)
    draft.add_event(
        0, TimelineEventType.ENCOUNTER_START,
        f"{encounter_type} for {primary_template.name}", [condition.entity_id],
    )

    if primary_template.medications:
        med_template = content.MEDICATION_TEMPLATES[rng.choice(primary_template.medications)]
        draft.add_medication(med_template, MedicationStatus.CURRENT)

    for finding in rng.sample(content.GENERIC_FINDINGS, k=min(2, len(content.GENERIC_FINDINGS))):
        draft.add_finding(finding, present=True)

    draft.add_vital(
        "Blood pressure", f"{rng.randint(110, 132)}/{rng.randint(66, 82)}", "mmHg", AbnormalFlag.NORMAL
    )
    heart_rate = content.VITAL_TEMPLATES[0]
    draft.add_vital(
        heart_rate.name, str(rng.randint(int(heart_rate.normal_range[0]), int(heart_rate.normal_range[1]))),
        heart_rate.unit, AbnormalFlag.NORMAL,
    )
