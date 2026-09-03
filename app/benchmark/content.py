"""Curated synthetic clinical content library.

This module holds the vocabulary the generator draws from. It is deliberately
*linked* rather than flat: a condition knows its typical medications, the labs
that are usually followed for it, and the findings and relevant negatives that
plausibly accompany it. That linkage is what keeps generated cases coherent -
a patient with CKD gets a creatinine trend and an ACE inhibitor, not a random
drug bolted to a random lab.

All content is fictional and written to be *clinically plausible* so that it
meaningfully exercises an evaluator. It is not medical reference material, it
is not exhaustive, and dose ladders here are illustrative rather than
prescribing guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.enums import AbnormalFlag, AllergySeverity


@dataclass(frozen=True)
class MedicationTemplate:
    drug_name: str
    doses: tuple[str, ...]
    route: str
    frequency: str
    indication: str


@dataclass(frozen=True)
class LabTemplate:
    """A laboratory analyte with its reference range and panic thresholds.

    `concern_direction` records which way is clinically worrying, so a
    generated trend moves in a direction that makes sense for the analyte
    (creatinine rising is bad; hemoglobin falling is bad).
    """

    name: str
    unit: str
    reference_low: float
    reference_high: float
    critical_low: float | None = None
    critical_high: float | None = None
    concern_direction: str = "high"
    decimals: int = 1
    is_key_lab: bool = False

    def classify(self, value: float) -> AbnormalFlag:
        """Classify a value against this analyte's thresholds.

        Panic thresholds are checked before the reference range, so a value
        that is both above the reference high and above the critical high is
        reported as CRITICAL_HIGH rather than merely HIGH.
        """
        if self.critical_low is not None and value <= self.critical_low:
            return AbnormalFlag.CRITICAL_LOW
        if self.critical_high is not None and value >= self.critical_high:
            return AbnormalFlag.CRITICAL_HIGH
        if value < self.reference_low:
            return AbnormalFlag.LOW
        if value > self.reference_high:
            return AbnormalFlag.HIGH
        return AbnormalFlag.NORMAL


@dataclass(frozen=True)
class ConditionTemplate:
    name: str
    short_name: str
    medications: tuple[str, ...]
    labs: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()
    negatives: tuple[str, ...] = ()
    encounter_types: tuple[str, ...] = ()
    procedures: tuple[str, ...] = ()
    is_acute: bool = False


@dataclass(frozen=True)
class AllergyTemplate:
    substance: str
    reaction: str
    severity: AllergySeverity


@dataclass(frozen=True)
class VitalTemplate:
    name: str
    unit: str | None
    normal_range: tuple[float, float]
    critical_low: float | None = None
    critical_high: float | None = None
    decimals: int = 0


# ---------------------------------------------------------------------------
# Laboratory analytes
# ---------------------------------------------------------------------------

LAB_TEMPLATES: dict[str, LabTemplate] = {
    "Creatinine": LabTemplate("Creatinine", "mg/dL", 0.6, 1.2, None, 4.0, "high", 1, True),
    "Potassium": LabTemplate("Potassium", "mEq/L", 3.5, 5.0, 2.5, 6.0, "high", 1, True),
    "Sodium": LabTemplate("Sodium", "mEq/L", 135, 145, 120, 160, "low", 0, True),
    "Hemoglobin": LabTemplate("Hemoglobin", "g/dL", 12.0, 16.0, 7.0, None, "low", 1, True),
    "White blood cell count": LabTemplate("White blood cell count", "K/uL", 4.0, 11.0, 1.0, 25.0, "high", 1, True),
    "Platelet count": LabTemplate("Platelet count", "K/uL", 150, 400, 50, None, "low", 0, False),
    "Glucose": LabTemplate("Glucose", "mg/dL", 70, 100, 40, 500, "high", 0, True),
    "Hemoglobin A1c": LabTemplate("Hemoglobin A1c", "%", 4.0, 5.6, None, None, "high", 1, False),
    "LDL cholesterol": LabTemplate("LDL cholesterol", "mg/dL", 0, 100, None, None, "high", 0, False),
    "TSH": LabTemplate("TSH", "uIU/mL", 0.4, 4.0, None, None, "high", 1, False),
    "BNP": LabTemplate("BNP", "pg/mL", 0, 100, None, 2000, "high", 0, True),
    "Troponin I": LabTemplate("Troponin I", "ng/mL", 0, 0.04, None, 1.0, "high", 2, True),
    "Lactate": LabTemplate("Lactate", "mmol/L", 0.5, 2.0, None, 4.0, "high", 1, True),
    "INR": LabTemplate("INR", "", 0.8, 1.2, None, 5.0, "high", 1, True),
    "eGFR": LabTemplate("eGFR", "mL/min/1.73m2", 60, 120, 15, None, "low", 0, True),
    "Calcium": LabTemplate("Calcium", "mg/dL", 8.5, 10.5, 6.5, 13.0, "high", 1, False),
}


# ---------------------------------------------------------------------------
# Medications
# ---------------------------------------------------------------------------

MEDICATION_TEMPLATES: dict[str, MedicationTemplate] = {
    "Lisinopril": MedicationTemplate("Lisinopril", ("5 mg", "10 mg", "20 mg", "40 mg"), "oral", "daily", "blood pressure control"),
    "Losartan": MedicationTemplate("Losartan", ("25 mg", "50 mg", "100 mg"), "oral", "daily", "blood pressure control"),
    "Amlodipine": MedicationTemplate("Amlodipine", ("2.5 mg", "5 mg", "10 mg"), "oral", "daily", "blood pressure control"),
    "Hydrochlorothiazide": MedicationTemplate("Hydrochlorothiazide", ("12.5 mg", "25 mg"), "oral", "daily", "blood pressure control"),
    "Metformin": MedicationTemplate("Metformin", ("500 mg", "850 mg", "1000 mg"), "oral", "twice daily", "glycemic control"),
    "Glipizide": MedicationTemplate("Glipizide", ("2.5 mg", "5 mg", "10 mg"), "oral", "daily", "glycemic control"),
    "Insulin glargine": MedicationTemplate("Insulin glargine", ("10 units", "20 units", "30 units"), "subcutaneous", "nightly", "glycemic control"),
    "Furosemide": MedicationTemplate("Furosemide", ("20 mg", "40 mg", "80 mg"), "oral", "daily", "volume management"),
    "Spironolactone": MedicationTemplate("Spironolactone", ("12.5 mg", "25 mg"), "oral", "daily", "heart failure management"),
    "Metoprolol succinate": MedicationTemplate("Metoprolol succinate", ("12.5 mg", "25 mg", "50 mg", "100 mg"), "oral", "daily", "rate control"),
    "Diltiazem": MedicationTemplate("Diltiazem", ("120 mg", "180 mg", "240 mg"), "oral", "daily", "rate control"),
    "Apixaban": MedicationTemplate("Apixaban", ("2.5 mg", "5 mg"), "oral", "twice daily", "anticoagulation"),
    "Warfarin": MedicationTemplate("Warfarin", ("2 mg", "5 mg"), "oral", "daily", "anticoagulation"),
    "Atorvastatin": MedicationTemplate("Atorvastatin", ("10 mg", "20 mg", "40 mg", "80 mg"), "oral", "daily", "lipid management"),
    "Rosuvastatin": MedicationTemplate("Rosuvastatin", ("5 mg", "10 mg", "20 mg"), "oral", "daily", "lipid management"),
    "Levothyroxine": MedicationTemplate("Levothyroxine", ("50 mcg", "75 mcg", "100 mcg"), "oral", "daily", "thyroid replacement"),
    "Albuterol": MedicationTemplate("Albuterol", ("2 puffs",), "inhaled", "as needed", "bronchodilation"),
    "Tiotropium": MedicationTemplate("Tiotropium", ("18 mcg",), "inhaled", "daily", "bronchodilation"),
    "Fluticasone": MedicationTemplate("Fluticasone", ("110 mcg", "220 mcg"), "inhaled", "twice daily", "airway inflammation"),
    "Prednisone": MedicationTemplate("Prednisone", ("10 mg", "20 mg", "40 mg"), "oral", "daily", "anti-inflammatory therapy"),
    "Omeprazole": MedicationTemplate("Omeprazole", ("20 mg", "40 mg"), "oral", "daily", "acid suppression"),
    "Pantoprazole": MedicationTemplate("Pantoprazole", ("20 mg", "40 mg"), "oral", "daily", "acid suppression"),
    "Acetaminophen": MedicationTemplate("Acetaminophen", ("500 mg", "650 mg"), "oral", "as needed", "pain control"),
    "Ibuprofen": MedicationTemplate("Ibuprofen", ("200 mg", "400 mg", "600 mg"), "oral", "as needed", "pain control"),
    "Ferrous sulfate": MedicationTemplate("Ferrous sulfate", ("325 mg",), "oral", "daily", "iron replacement"),
    "Cephalexin": MedicationTemplate("Cephalexin", ("500 mg",), "oral", "four times daily", "antibiotic therapy"),
    "Azithromycin": MedicationTemplate("Azithromycin", ("250 mg", "500 mg"), "oral", "daily", "antibiotic therapy"),
    "Ceftriaxone": MedicationTemplate("Ceftriaxone", ("1 g", "2 g"), "intravenous", "daily", "antibiotic therapy"),
    "Nitrofurantoin": MedicationTemplate("Nitrofurantoin", ("100 mg",), "oral", "twice daily", "antibiotic therapy"),
    "Levetiracetam": MedicationTemplate("Levetiracetam", ("250 mg", "500 mg", "750 mg"), "oral", "twice daily", "seizure prophylaxis"),
    "Sertraline": MedicationTemplate("Sertraline", ("25 mg", "50 mg", "100 mg"), "oral", "daily", "mood management"),
}


# ---------------------------------------------------------------------------
# Conditions, each linked to plausible medications, labs, findings, negatives
# ---------------------------------------------------------------------------

CONDITION_TEMPLATES: tuple[ConditionTemplate, ...] = (
    ConditionTemplate(
        "Essential hypertension", "hypertension",
        ("Lisinopril", "Amlodipine", "Hydrochlorothiazide", "Losartan"),
        ("Potassium", "Creatinine"),
        ("blood pressure above goal on today's reading",),
        ("no chest pain", "no headache", "no visual changes"),
        ("Primary care follow-up", "Annual wellness visit", "Hypertension follow-up"),
    ),
    ConditionTemplate(
        "Type 2 diabetes mellitus", "diabetes",
        ("Metformin", "Glipizide", "Insulin glargine"),
        ("Hemoglobin A1c", "Glucose", "Creatinine"),
        ("no diabetic foot ulceration on exam",),
        ("no polyuria", "no visual changes", "no hypoglycemic episodes"),
        ("Primary care follow-up", "Diabetes management visit"),
    ),
    ConditionTemplate(
        "Chronic kidney disease stage 3", "CKD",
        ("Lisinopril", "Furosemide"),
        ("Creatinine", "eGFR", "Potassium", "Hemoglobin"),
        ("trace lower extremity edema",),
        ("no hematuria", "no flank pain", "no change in urine output"),
        ("Nephrology follow-up", "Primary care follow-up"),
        ("Renal ultrasound",),
    ),
    ConditionTemplate(
        "Heart failure with reduced ejection fraction", "heart failure",
        ("Furosemide", "Metoprolol succinate", "Spironolactone"),
        ("BNP", "Potassium", "Creatinine", "Sodium"),
        ("trace lower extremity edema", "jugular venous distension"),
        ("no orthopnea", "no paroxysmal nocturnal dyspnea", "no chest pain"),
        ("Cardiology follow-up", "Heart failure clinic visit"),
        ("Echocardiogram",),
    ),
    ConditionTemplate(
        "Atrial fibrillation", "atrial fibrillation",
        ("Apixaban", "Metoprolol succinate", "Diltiazem"),
        ("Hemoglobin", "Creatinine"),
        ("irregularly irregular rhythm on exam",),
        ("no palpitations today", "no syncope", "no bleeding"),
        ("Cardiology follow-up", "Primary care follow-up"),
        ("Electrocardiogram",),
    ),
    ConditionTemplate(
        "Chronic obstructive pulmonary disease", "COPD",
        ("Albuterol", "Tiotropium", "Prednisone"),
        ("White blood cell count",),
        ("decreased breath sounds bilaterally",),
        ("no hemoptysis", "no fever", "no chest pain"),
        ("Pulmonary follow-up", "Urgent care visit"),
        ("Chest X-ray",),
    ),
    ConditionTemplate(
        "Community-acquired pneumonia", "pneumonia",
        ("Azithromycin", "Ceftriaxone"),
        ("White blood cell count", "Lactate"),
        ("crackles at the right lung base",),
        ("no hemoptysis", "no pleuritic chest pain"),
        ("Emergency department visit", "Hospital admission"),
        ("Chest X-ray",),
        is_acute=True,
    ),
    ConditionTemplate(
        "Acute kidney injury", "AKI",
        ("Furosemide",),
        ("Creatinine", "Potassium", "eGFR"),
        ("dry mucous membranes",),
        ("no hematuria", "no urinary retention"),
        ("Hospital admission", "Emergency department visit"),
        is_acute=True,
    ),
    ConditionTemplate(
        "Hyperlipidemia", "hyperlipidemia",
        ("Atorvastatin", "Rosuvastatin"),
        ("LDL cholesterol",),
        (),
        ("no myalgias on statin therapy",),
        ("Primary care follow-up", "Annual wellness visit"),
    ),
    ConditionTemplate(
        "Hypothyroidism", "hypothyroidism",
        ("Levothyroxine",),
        ("TSH",),
        (),
        ("no cold intolerance", "no unexplained weight gain"),
        ("Primary care follow-up", "Endocrinology follow-up"),
    ),
    ConditionTemplate(
        "Asthma", "asthma",
        ("Albuterol", "Fluticasone"),
        (),
        ("scattered expiratory wheezes",),
        ("no nocturnal symptoms", "no recent exacerbation"),
        ("Primary care follow-up", "Urgent care visit"),
    ),
    ConditionTemplate(
        "Gastroesophageal reflux disease", "GERD",
        ("Omeprazole", "Pantoprazole"),
        (),
        (),
        ("no dysphagia", "no melena", "no weight loss"),
        ("Primary care follow-up", "Gastroenterology follow-up"),
    ),
    ConditionTemplate(
        "Osteoarthritis", "osteoarthritis",
        ("Acetaminophen", "Ibuprofen"),
        (),
        ("reduced range of motion at the affected joint",),
        ("no joint erythema", "no joint effusion"),
        ("Primary care follow-up", "Orthopedic follow-up"),
        ("Knee arthroscopy",),
    ),
    ConditionTemplate(
        "Iron deficiency anemia", "anemia",
        ("Ferrous sulfate",),
        ("Hemoglobin",),
        ("pallor of the conjunctivae",),
        ("no melena", "no hematochezia", "no lightheadedness"),
        ("Primary care follow-up", "Hematology follow-up"),
        ("Colonoscopy",),
    ),
    ConditionTemplate(
        "Cellulitis", "cellulitis",
        ("Cephalexin",),
        ("White blood cell count",),
        ("erythema and warmth over the affected area",),
        ("no fluctuance", "no fever", "no lymphangitic streaking"),
        ("Urgent care visit", "Emergency department visit"),
        is_acute=True,
    ),
    ConditionTemplate(
        "Urinary tract infection", "UTI",
        ("Nitrofurantoin",),
        ("White blood cell count",),
        ("suprapubic tenderness",),
        ("no flank pain", "no fever"),
        ("Urgent care visit", "Primary care follow-up"),
        is_acute=True,
    ),
    ConditionTemplate(
        "Seizure disorder", "seizure disorder",
        ("Levetiracetam",),
        (),
        (),
        ("no breakthrough seizures", "no focal neurologic deficit"),
        ("Neurology follow-up",),
    ),
    ConditionTemplate(
        "Major depressive disorder", "depression",
        ("Sertraline",),
        (),
        (),
        ("no suicidal ideation", "no psychotic symptoms"),
        ("Primary care follow-up", "Behavioral health follow-up"),
    ),
)

CONDITIONS_BY_NAME: dict[str, ConditionTemplate] = {c.name: c for c in CONDITION_TEMPLATES}


# ---------------------------------------------------------------------------
# Allergies, vitals, procedures, generic findings
# ---------------------------------------------------------------------------

ALLERGY_TEMPLATES: tuple[AllergyTemplate, ...] = (
    AllergyTemplate("Penicillin", "rash", AllergySeverity.MODERATE),
    AllergyTemplate("Sulfa drugs", "hives", AllergySeverity.MODERATE),
    AllergyTemplate("Codeine", "nausea and vomiting", AllergySeverity.MILD),
    AllergyTemplate("Latex", "contact dermatitis", AllergySeverity.MILD),
    AllergyTemplate("Iodinated contrast", "anaphylaxis", AllergySeverity.SEVERE),
    AllergyTemplate("Aspirin", "bronchospasm", AllergySeverity.SEVERE),
    AllergyTemplate("Peanuts", "anaphylaxis", AllergySeverity.SEVERE),
)

VITAL_TEMPLATES: tuple[VitalTemplate, ...] = (
    VitalTemplate("Heart rate", "bpm", (60, 100), 40, 150),
    VitalTemplate("Respiratory rate", "breaths/min", (12, 20), 8, 30),
    VitalTemplate("Temperature", "F", (97.0, 99.5), 95.0, 103.0, 1),
    VitalTemplate("Oxygen saturation", "%", (94, 100), 88, None),
)

PROCEDURE_TEMPLATES: tuple[str, ...] = (
    "Chest X-ray", "Electrocardiogram", "Echocardiogram", "Renal ultrasound",
    "CT of the abdomen and pelvis", "Colonoscopy", "Knee arthroscopy",
    "Cardiac catheterization", "Pulmonary function testing",
)

GENERIC_FINDINGS: tuple[str, ...] = (
    "no acute distress",
    "lungs clear to auscultation bilaterally",
    "abdomen soft and non-tender",
    "regular rate and rhythm",
    "alert and oriented",
    "extremities warm and well perfused",
)

GENERIC_NEGATIVES: tuple[str, ...] = (
    "no fever",
    "no chest pain",
    "no shortness of breath at rest",
    "no syncope",
    "no recent falls",
    "no unintentional weight loss",
)

FOLLOW_UP_PROVIDERS: tuple[str, ...] = (
    "primary care", "cardiology", "nephrology", "pulmonology",
    "endocrinology", "the care team",
)

FOLLOW_UP_INTERVALS: tuple[str, ...] = (
    "1 week", "2 weeks", "4 weeks", "3 months", "6 months",
)

AGE_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("18-34", 18, 34),
    ("35-44", 35, 44),
    ("45-54", 45, 54),
    ("55-64", 55, 64),
    ("65-74", 65, 74),
    ("75-84", 75, 84),
)

SEXES: tuple[str, ...] = ("female", "male")

ENCOUNTER_SETTINGS: dict[str, str] = {
    "Primary care follow-up": "outpatient_primary_care",
    "Annual wellness visit": "outpatient_primary_care",
    "Hypertension follow-up": "outpatient_primary_care",
    "Diabetes management visit": "outpatient_primary_care",
    "Nephrology follow-up": "outpatient_specialty",
    "Cardiology follow-up": "outpatient_specialty",
    "Heart failure clinic visit": "outpatient_specialty",
    "Pulmonary follow-up": "outpatient_specialty",
    "Endocrinology follow-up": "outpatient_specialty",
    "Gastroenterology follow-up": "outpatient_specialty",
    "Hematology follow-up": "outpatient_specialty",
    "Neurology follow-up": "outpatient_specialty",
    "Behavioral health follow-up": "outpatient_specialty",
    "Orthopedic follow-up": "outpatient_specialty",
    "Urgent care visit": "urgent_care",
    "Emergency department visit": "emergency_department",
    "Hospital admission": "inpatient",
}

PAST_TIMINGS: tuple[str, ...] = (
    "2 weeks ago", "1 month ago", "6 weeks ago", "3 months ago", "6 months ago",
)


def all_drug_names() -> frozenset[str]:
    """Every drug the generator can produce.

    The integrity checker uses this to detect a rendered summary naming a drug
    that is not in the case - i.e. an unsupported fact introduced by a renderer.
    """
    return frozenset(MEDICATION_TEMPLATES)


def all_condition_names() -> frozenset[str]:
    return frozenset(CONDITIONS_BY_NAME)


def all_lab_names() -> frozenset[str]:
    return frozenset(LAB_TEMPLATES)
