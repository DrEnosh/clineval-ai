"""Failure-mode feature builders.

Each function adds the clinical content for one `FailureModeCategory` to a
`CaseDraft`. Features compose: a HARD case might apply comorbidities, a lab
trend, a dose change, and a relevant negative, and the result is still one
coherent patient because every feature draws from the content library's
condition-medication-lab linkages rather than inventing free-floating facts.

Two invariants every feature must respect:

1. **Coherence.** A feature never asserts something that contradicts what is
   already in the draft. The negation trap, for instance, picks a condition
   the patient does *not* have - otherwise the case would claim both that the
   patient has diabetes and that they do not.

2. **Ground truth is untouched.** Features add `FailureOpportunity` records
   describing what a summarizer might get wrong. They never corrupt the
   structured case to manufacture an error. The perturbation framework
   describes traps; it does not plant lies.
"""

from __future__ import annotations

from random import Random

from app.benchmark import content
from app.benchmark.case_builder import (
    CaseDraft,
    next_dose_up,
    sample_abnormal_value,
    sample_critical_value,
    sample_normal_value,
)
from app.core.enums import (
    AbnormalFlag,
    AllergySeverity,
    ConditionStatus,
    FailureModeCategory,
    FailureOpportunityType,
    FindingSignificance,
    MedicationStatus,
    Severity,
    TimelineEventType,
)


def _pick_lab_template(draft: CaseDraft, rng: Random) -> content.LabTemplate:
    """A lab relevant to this case's conditions, falling back to a key analyte."""
    candidates = [name for name in draft.candidate_labs() if not draft.has_lab(name)]
    if not candidates:
        candidates = [name for name, t in content.LAB_TEMPLATES.items() if t.is_key_lab and not draft.has_lab(name)]
    if not candidates:
        candidates = list(content.LAB_TEMPLATES)
    return content.LAB_TEMPLATES[rng.choice(sorted(candidates))]


def _unused_condition(draft: CaseDraft, rng: Random) -> content.ConditionTemplate | None:
    """A condition the patient does NOT have - required for negation traps."""
    present = draft.condition_names()
    candidates = [t for t in content.CONDITION_TEMPLATES if t.name not in present and not t.is_acute]
    return rng.choice(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def apply_straightforward(draft: CaseDraft, rng: Random) -> None:
    """A clean encounter: a normal lab and a routine follow-up, no traps."""
    template = _pick_lab_template(draft, rng)
    draft.add_lab(template, sample_normal_value(template, rng))
    draft.add_follow_up(
        rng.choice(content.FOLLOW_UP_INTERVALS),
        rng.choice(content.FOLLOW_UP_PROVIDERS),
        ["routine health maintenance"],
    )


def apply_abnormal_labs(draft: CaseDraft, rng: Random) -> None:
    """One abnormal lab; occasionally a panic-range value (CR-01 vs CR-02)."""
    template = _pick_lab_template(draft, rng)
    make_critical = rng.random() < 0.35
    value = sample_critical_value(template, rng) if make_critical else None
    if value is None:
        value = sample_abnormal_value(template, rng)
    lab = draft.add_lab(template, value)

    if lab.is_critical_value:
        draft.add_opportunity(
            FailureOpportunityType.OMISSION,
            f"Panic-range {lab.name} ({lab.describe()}) must be preserved in any summary.",
            lab.entity_id, Severity.CRITICAL,
        )
    elif lab.is_abnormal and template.is_key_lab:
        draft.add_opportunity(
            FailureOpportunityType.OMISSION,
            f"Abnormal key analyte {lab.name} ({lab.describe()}) must be preserved.",
            lab.entity_id, Severity.HIGH,
        )


def apply_abnormal_vitals(draft: CaseDraft, rng: Random) -> None:
    """An out-of-range vital sign, sometimes in the panic range (CR-09)."""
    if rng.random() < 0.5:
        systolic, diastolic = rng.randint(150, 178), rng.randint(92, 104)
        draft.add_vital("Blood pressure", f"{systolic}/{diastolic}", "mmHg", AbnormalFlag.HIGH)
        draft.add_opportunity(
            FailureOpportunityType.OMISSION,
            f"Elevated blood pressure {systolic}/{diastolic} mmHg must be preserved.",
            draft.vitals[-1].entity_id, Severity.MODERATE,
        )
        return

    template = rng.choice(content.VITAL_TEMPLATES)
    critical = rng.random() < 0.3 and template.critical_high is not None
    if critical:
        value = round(rng.uniform(template.critical_high, template.critical_high * 1.08), template.decimals)
        flag = AbnormalFlag.CRITICAL_HIGH
    else:
        value = round(rng.uniform(template.normal_range[1] * 1.05, template.normal_range[1] * 1.25), template.decimals)
        flag = AbnormalFlag.HIGH
    rendered = str(int(value)) if template.decimals == 0 else f"{value:g}"
    vital = draft.add_vital(template.name, rendered, template.unit, flag)
    draft.add_opportunity(
        FailureOpportunityType.OMISSION,
        f"Abnormal {template.name.lower()} ({vital.describe()}) must be preserved.",
        vital.entity_id, Severity.CRITICAL if critical else Severity.MODERATE,
    )


def apply_temporal_progression(draft: CaseDraft, rng: Random) -> None:
    """Two results for one analyte plus an explicit trend linking them.

    The trap is that both values appear in the note; a summary that reports
    only the earlier one, or reverses the direction, is wrong.
    """
    template = _pick_lab_template(draft, rng)
    timing = rng.choice(content.PAST_TIMINGS)

    if template.concern_direction == "low":
        earlier_value = sample_normal_value(template, rng)
        later_value = sample_abnormal_value(template, rng)
    else:
        earlier_value = sample_normal_value(template, rng)
        later_value = sample_abnormal_value(template, rng)

    earlier = draft.add_lab(template, earlier_value, day_offset=-30, timing=timing)
    later = draft.add_lab(template, later_value, day_offset=0, timing="today")
    trend = draft.add_lab_trend(template, earlier, later, significant=True)

    draft.add_event(
        0, TimelineEventType.ASSESSMENT,
        f"{template.name} {trend.trend.value} from {trend.format_value(earlier_value)} to "
        f"{trend.format_value(later_value)} {template.unit} between {timing} and today",
        [trend.entity_id],
    )
    draft.add_opportunity(
        FailureOpportunityType.TEMPORAL_TRAP,
        f"{template.name} moved from {trend.format_value(earlier_value)} to "
        f"{trend.format_value(later_value)} {template.unit}; a summary must not report the "
        f"historical value as current or reverse the direction.",
        trend.entity_id, Severity.HIGH,
    )


def _add_medication_for_case(draft: CaseDraft, rng: Random) -> content.MedicationTemplate:
    """A drug plausibly linked to one of the case's conditions."""
    options: list[str] = []
    for template in draft.condition_templates:
        options.extend(template.medications)
    unused = [name for name in options if not draft.has_drug(name)]
    pool = unused or [name for name in content.MEDICATION_TEMPLATES if not draft.has_drug(name)]
    if not pool:
        pool = list(content.MEDICATION_TEMPLATES)
    return content.MEDICATION_TEMPLATES[rng.choice(sorted(set(pool)))]


def apply_medication_change(draft: CaseDraft, rng: Random) -> None:
    """A newly started medication (CR-04)."""
    template = _add_medication_for_case(draft, rng)
    med = draft.add_medication(
        template, MedicationStatus.NEW,
        change_reason=f"newly started for {template.indication}",
        change_timing="today",
    )
    draft.add_event(
        0, TimelineEventType.MEDICATION_CHANGE,
        f"{template.drug_name} {med.dose} started today", [med.entity_id],
    )
    draft.add_opportunity(
        FailureOpportunityType.MEDICATION_TRAP,
        f"{template.drug_name} was newly started; a summary must not report it as a "
        f"pre-existing home medication.",
        med.entity_id, Severity.HIGH,
    )


def apply_medication_discontinuation(draft: CaseDraft, rng: Random) -> None:
    """A stopped medication - the classic "still listed as current" trap."""
    template = _add_medication_for_case(draft, rng)
    reason = rng.choice(
        ["an adverse effect", "peripheral edema", "hyperkalemia", "hypotension", "intolerance"]
    )
    timing = rng.choice(content.PAST_TIMINGS)
    med = draft.add_medication(
        template, MedicationStatus.DISCONTINUED,
        change_reason=f"discontinued due to {reason}",
        change_timing=timing,
    )
    draft.add_event(
        -30, TimelineEventType.MEDICATION_CHANGE,
        f"{template.drug_name} discontinued {timing} due to {reason}", [med.entity_id],
    )
    draft.add_opportunity(
        FailureOpportunityType.MEDICATION_TRAP,
        f"{template.drug_name} was DISCONTINUED {timing}; listing it as a current "
        f"medication would risk duplicate or contraindicated therapy.",
        med.entity_id, Severity.HIGH,
    )


def apply_medication_dose_change(draft: CaseDraft, rng: Random) -> None:
    """A dose increase, with both the prior and current dose in the note."""
    template = _add_medication_for_case(draft, rng)
    if len(template.doses) < 2:
        apply_medication_change(draft, rng)
        return

    prior = rng.choice(template.doses[:-1])
    new_dose = next_dose_up(template, prior)
    timing = rng.choice(content.PAST_TIMINGS)
    reason = rng.choice(
        ["inadequate control", "worsening symptoms", "volume overload", "an elevated result"]
    )
    med = draft.add_medication(
        template, MedicationStatus.CHANGED_DOSE, dose=new_dose, prior_dose=prior,
        change_reason=f"increased due to {reason}", change_timing=timing,
    )
    draft.add_event(
        -14, TimelineEventType.MEDICATION_CHANGE,
        f"{template.drug_name} increased from {prior} to {new_dose} {timing}", [med.entity_id],
    )
    draft.add_opportunity(
        FailureOpportunityType.MEDICATION_TRAP,
        f"{template.drug_name} was increased from {prior} to {new_dose}; reporting the "
        f"prior dose as current would misstate the regimen.",
        med.entity_id, Severity.HIGH,
    )


def apply_negation(draft: CaseDraft, rng: Random) -> None:
    """A clinically important negative about a condition the patient lacks.

    Coherence matters here: the negated condition is chosen from those NOT in
    the case, so the note never both asserts and denies the same diagnosis.
    """
    template = _unused_condition(draft, rng)
    if template is None:
        return
    finding = draft.add_finding(
        f"no history of {template.short_name}", present=False,
        significance=FindingSignificance.CRITICAL,
    )
    draft.add_opportunity(
        FailureOpportunityType.NEGATION_TRAP,
        f"The note states there is NO history of {template.short_name}; asserting the "
        f"patient has it would reverse a clinically important negative.",
        finding.entity_id, Severity.CRITICAL,
    )

    if rng.random() < 0.5 and not draft.allergies:
        no_allergy = draft.add_finding(
            "no known drug allergies", present=False, significance=FindingSignificance.CRITICAL
        )
        draft.add_opportunity(
            FailureOpportunityType.NEGATION_TRAP,
            "The note documents NO known drug allergies; asserting an allergy would "
            "introduce a false contraindication.",
            no_allergy.entity_id, Severity.HIGH,
        )


def apply_relevant_negative_findings(draft: CaseDraft, rng: Random) -> None:
    """Pertinent negatives drawn from the case's own conditions."""
    pool: list[str] = []
    for template in draft.condition_templates:
        pool.extend(template.negatives)
    pool.extend(content.GENERIC_NEGATIVES)
    existing = {f.description for f in draft.findings}
    choices = [n for n in dict.fromkeys(pool) if n not in existing]
    if not choices:
        return
    for description in rng.sample(choices, k=min(rng.randint(2, 3), len(choices))):
        draft.add_finding(description, present=False, significance=FindingSignificance.IMPORTANT)


def apply_multiple_comorbidities(draft: CaseDraft, rng: Random) -> None:
    """Two or three additional chronic conditions, each with a medication."""
    for _ in range(rng.randint(2, 3)):
        template = _unused_condition(draft, rng)
        if template is None:
            break
        draft.add_condition(template, onset=rng.choice(content.PAST_TIMINGS))
        draft.add_diagnosis(template.name)
        if template.medications:
            med_template = content.MEDICATION_TEMPLATES[rng.choice(template.medications)]
            if not draft.has_drug(med_template.drug_name):
                draft.add_medication(med_template, MedicationStatus.CURRENT)
        if template.findings and rng.random() < 0.5:
            draft.add_finding(rng.choice(template.findings), present=True)


def apply_historical_vs_current(draft: CaseDraft, rng: Random) -> None:
    """A resolved past condition and a past medication, alongside active ones.

    The trap is conflating the historical item with the current picture.
    """
    template = _unused_condition(draft, rng)
    if template is None:
        return
    timing = rng.choice(["2 years ago", "3 years ago", "5 years ago"])
    condition = draft.add_condition(
        template, status=ConditionStatus.RESOLVED, onset=f"resolved {timing}"
    )
    draft.add_opportunity(
        FailureOpportunityType.HISTORICAL_CURRENT_TRAP,
        f"{template.name} is RESOLVED ({timing}), not active; a summary must not list "
        f"it among current problems.",
        condition.entity_id, Severity.HIGH,
    )

    if template.medications:
        med_template = content.MEDICATION_TEMPLATES[rng.choice(template.medications)]
        if not draft.has_drug(med_template.drug_name):
            med = draft.add_medication(
                med_template, MedicationStatus.HISTORICAL,
                change_reason="taken in the past, not currently prescribed",
                change_timing=timing,
            )
            draft.add_opportunity(
                FailureOpportunityType.HISTORICAL_CURRENT_TRAP,
                f"{med_template.drug_name} is a HISTORICAL medication ({timing}), not "
                f"current.",
                med.entity_id, Severity.MODERATE,
            )


def apply_conflicting_information(draft: CaseDraft, rng: Random) -> None:
    """An earlier statement superseded by a later reassessment (CR-07)."""
    topic, earlier, later, resolution = rng.choice(
        [
            (
                "chest pain",
                "patient denies chest pain",
                "patient reports intermittent pressure-like chest pain since this morning",
                "The later assessment supersedes the initial history; chest pain is present.",
            ),
            (
                "medication adherence",
                "patient reports taking all medications as prescribed",
                "patient reports having run out of medications two weeks ago",
                "The later history supersedes the initial report; the patient has not been "
                "taking medications for two weeks.",
            ),
            (
                "fever",
                "patient denies fever at triage",
                "patient reports subjective fevers for the past three days",
                "The later history supersedes the triage report; subjective fevers are present.",
            ),
            (
                "fall history",
                "patient denies recent falls",
                "family reports two falls in the past month",
                "The collateral history supersedes the patient's initial report; recent "
                "falls have occurred.",
            ),
        ]
    )
    conflict = draft.add_conflict(
        topic=topic,
        earlier_statement=earlier, earlier_timing="on initial assessment",
        later_statement=later, later_timing="on reassessment",
        resolution=resolution,
    )
    draft.add_opportunity(
        FailureOpportunityType.CONFLICT_TRAP,
        f"Documentation about {topic} changed during the encounter; a summary that "
        f"reports the superseded statement as current would be wrong despite quoting "
        f"the note accurately.",
        conflict.entity_id, Severity.CRITICAL,
    )


def apply_hallucination_opportunity(draft: CaseDraft, rng: Random) -> None:
    """A gap a model may be tempted to fill with plausible-sounding content.

    The case explicitly documents that a study was NOT done, or that a result
    is pending. Nothing false is added; the opportunity records what a model
    might invent.
    """
    choice = rng.random()
    if choice < 0.4:
        procedure = rng.choice(content.PROCEDURE_TEMPLATES)
        entry = draft.add_procedure(
            procedure, timing="not performed at this visit",
            outcome=None, performed=False,
        )
        draft.add_finding(
            f"{procedure.lower()} was not obtained at this visit", present=False,
            significance=FindingSignificance.IMPORTANT,
        )
        draft.add_opportunity(
            FailureOpportunityType.HALLUCINATION,
            f"{procedure} was NOT performed; reporting a result for it would be a "
            f"fabricated finding.",
            entry.entity_id, Severity.HIGH,
        )
        return

    if choice < 0.75:
        template = _pick_lab_template(draft, rng)
        finding = draft.add_finding(
            f"{template.name.lower()} result is pending at the time of this note",
            present=True, significance=FindingSignificance.IMPORTANT,
        )
        draft.add_opportunity(
            FailureOpportunityType.HALLUCINATION,
            f"The {template.name} result is PENDING; stating a numeric value would be "
            f"fabricated.",
            finding.entity_id, Severity.HIGH,
        )
        return

    finding = draft.add_finding(
        "no antibiotics were prescribed at this visit", present=False,
        significance=FindingSignificance.CRITICAL,
    )
    draft.add_opportunity(
        FailureOpportunityType.HALLUCINATION,
        "No antibiotic was prescribed; reporting one would invent a treatment the "
        "patient did not receive.",
        finding.entity_id, Severity.HIGH,
    )


def apply_procedures_and_followup(draft: CaseDraft, rng: Random) -> None:
    """A performed procedure with an outcome, plus a concrete follow-up plan."""
    pool: list[str] = []
    for template in draft.condition_templates:
        pool.extend(template.procedures)
    pool = pool or list(content.PROCEDURE_TEMPLATES)
    procedure_name = rng.choice(sorted(set(pool)))
    outcome = rng.choice(
        ["no acute abnormality identified", "findings consistent with the known diagnosis",
         "stable compared with prior study", "no interval change"]
    )
    draft.add_procedure(procedure_name, timing="today", outcome=outcome)

    actions = [
        rng.choice(["repeat laboratory testing", "reassess symptoms", "review medication tolerance"]),
    ]
    draft.add_follow_up(
        rng.choice(content.FOLLOW_UP_INTERVALS), rng.choice(content.FOLLOW_UP_PROVIDERS), actions
    )


def apply_critical_omission(draft: CaseDraft, rng: Random) -> None:
    """A high-stakes fact plus an explicit omission opportunity.

    Uses a panic-range analyte where one exists, otherwise a severe allergy -
    either way the case carries something a summary must not drop.
    """
    candidates = [t for t in content.LAB_TEMPLATES.values() if t.critical_high or t.critical_low]
    # Prefer an analyte not already measured, so the case never reports two
    # different values for the same analyte at the same timepoint.
    unused = [t for t in candidates if not draft.has_lab(t.name)]
    template = rng.choice(sorted(unused or candidates, key=lambda t: t.name))
    value = sample_critical_value(template, rng)
    if value is None:
        return
    lab = draft.add_lab(template, value)
    draft.add_event(
        0, TimelineEventType.DETERIORATION,
        f"Panic-range {template.name} identified: {lab.describe()}", [lab.entity_id],
    )
    draft.add_opportunity(
        FailureOpportunityType.OMISSION,
        f"Panic-range {template.name} ({lab.describe()}) is the most important fact in "
        f"this case; omitting it would hide the reason for urgent action.",
        lab.entity_id, Severity.CRITICAL,
    )

    severe = [a for a in content.ALLERGY_TEMPLATES if a.severity is AllergySeverity.SEVERE]
    if severe and not draft.allergies:
        allergy = draft.add_allergy(rng.choice(severe))
        draft.add_opportunity(
            FailureOpportunityType.OMISSION,
            f"Severe allergy to {allergy.substance} ({allergy.reaction}) must be preserved.",
            allergy.entity_id, Severity.CRITICAL,
        )


def apply_high_complexity(draft: CaseDraft, rng: Random) -> None:
    """A multi-day course with deterioration then improvement.

    Produces the interacting timeline that makes summarization genuinely hard:
    a value that rises and then partially recovers, so "the creatinine" has
    three different correct answers depending on which day is meant.
    """
    draft.verbose_note = True
    template = content.LAB_TEMPLATES["Creatinine"]
    baseline = draft.add_lab(template, sample_normal_value(template, rng), day_offset=-1, timing="on admission")
    peak_value = sample_abnormal_value(template, rng)
    peak = draft.add_lab(template, peak_value, day_offset=2, timing="on hospital day 2")
    recovery_value = round((baseline.value + peak_value) / 2, template.decimals)
    recovery = draft.add_lab(template, recovery_value, day_offset=4, timing="at discharge")

    draft.add_lab_trend(template, baseline, peak, significant=True)
    draft.add_lab_trend(template, peak, recovery, significant=True)

    draft.add_event(
        2, TimelineEventType.DETERIORATION,
        f"{template.name} rose to {peak.format_value()} {template.unit} on hospital day 2",
        [peak.entity_id],
    )
    draft.add_event(
        4, TimelineEventType.IMPROVEMENT,
        f"{template.name} improved to {recovery.format_value()} {template.unit} by discharge",
        [recovery.entity_id],
    )
    draft.add_opportunity(
        FailureOpportunityType.TEMPORAL_TRAP,
        f"{template.name} rose to {peak.format_value()} then improved to "
        f"{recovery.format_value()} {template.unit}; reporting the peak as the discharge "
        f"value would misrepresent current renal function to the receiving team.",
        recovery.entity_id, Severity.HIGH,
    )

    if draft.encounter is not None and draft.encounter.length_of_stay_days is None:
        draft.encounter = draft.encounter.model_copy(update={"length_of_stay_days": 5})


def apply_duplicate_documentation(draft: CaseDraft, rng: Random) -> None:
    """Redundant restatement of content already in the note.

    Real notes are full of copy-forward duplication. The trap is a summary
    that treats a repeated statement as two separate clinical events.
    """
    draft.duplicate_documentation = True
    if draft.findings:
        target = rng.choice(draft.findings)
        draft.add_opportunity(
            FailureOpportunityType.REDUNDANCY_TRAP,
            f"'{target.description}' is documented more than once in the note; a summary "
            f"must not report it as two separate findings.",
            target.entity_id, Severity.LOW,
        )


def apply_long_note(draft: CaseDraft, rng: Random) -> None:
    """Verbose documentation - extra review of systems and exam detail."""
    draft.verbose_note = True
    pool = [n for n in content.GENERIC_NEGATIVES if n not in {f.description for f in draft.findings}]
    for description in rng.sample(pool, k=min(3, len(pool))):
        draft.add_finding(description, present=False, significance=FindingSignificance.ROUTINE)


def apply_short_note(draft: CaseDraft, rng: Random) -> None:
    """Terse documentation - few facts, minimal sections."""
    draft.terse_note = True


def apply_combined_critical(draft: CaseDraft, rng: Random) -> None:
    """The adversarial combination: a negation, a temporal trap, and a
    medication trap in one case, which is where summarizers most often fail."""
    apply_negation(draft, rng)
    apply_temporal_progression(draft, rng)
    apply_medication_discontinuation(draft, rng)
    draft.add_opportunity(
        FailureOpportunityType.OMISSION,
        "This case combines a negated diagnosis, a laboratory trend, and a discontinued "
        "medication; each must survive summarization independently.",
        None, Severity.HIGH,
    )


FEATURE_BUILDERS = {
    FailureModeCategory.STRAIGHTFORWARD: apply_straightforward,
    FailureModeCategory.CRITICAL_OMISSION: apply_critical_omission,
    FailureModeCategory.HALLUCINATION_OPPORTUNITY: apply_hallucination_opportunity,
    FailureModeCategory.NEGATION: apply_negation,
    FailureModeCategory.TEMPORAL_PROGRESSION: apply_temporal_progression,
    FailureModeCategory.MEDICATION_CHANGE: apply_medication_change,
    FailureModeCategory.MEDICATION_DISCONTINUATION: apply_medication_discontinuation,
    FailureModeCategory.MEDICATION_DOSE_CHANGE: apply_medication_dose_change,
    FailureModeCategory.MULTIPLE_COMORBIDITIES: apply_multiple_comorbidities,
    FailureModeCategory.HIGH_COMPLEXITY: apply_high_complexity,
    FailureModeCategory.CONFLICTING_INFORMATION: apply_conflicting_information,
    FailureModeCategory.ABNORMAL_LABS: apply_abnormal_labs,
    FailureModeCategory.ABNORMAL_VITALS: apply_abnormal_vitals,
    FailureModeCategory.RELEVANT_NEGATIVE_FINDINGS: apply_relevant_negative_findings,
    FailureModeCategory.HISTORICAL_VS_CURRENT: apply_historical_vs_current,
    FailureModeCategory.PROCEDURES_AND_FOLLOWUP: apply_procedures_and_followup,
    FailureModeCategory.DUPLICATE_DOCUMENTATION: apply_duplicate_documentation,
    FailureModeCategory.LONG_NOTE: apply_long_note,
    FailureModeCategory.SHORT_NOTE: apply_short_note,
    FailureModeCategory.COMBINED_CRITICAL: apply_combined_critical,
}
