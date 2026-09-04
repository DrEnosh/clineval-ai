"""Clinical text normalization and a curated alias registry.

This is deliberately *not* a medical ontology. It is a small, transparent,
hand-maintained table of the abbreviations and spelling variants this
benchmark's own content library actually produces, plus generic text
normalization. Everything in it is inspectable and testable.

Why not a real terminology service (SNOMED, RxNorm, UMLS)? Three reasons:
they are licensed or heavyweight; they would make evaluation results depend
on an external resource version; and an ontology lookup that silently maps two
terms together is exactly the kind of invisible judgment this project tries to
avoid. A curated table is auditable - you can read the whole thing.

Assumptions, stated because they are assumptions and not facts:

- **Aliases are bidirectional and context-free.** "CKD" is treated as
  equivalent to "chronic kidney disease" everywhere. This is safe for this
  benchmark's vocabulary but would not be safe in general text.
- **British/American spelling folding is anchored at word boundaries.** The
  folds convert haemoglobin/oesophageal correctly. They are anchored because
  an unanchored "oesophag" -> "esophag" also rewrites the *American*
  "gastroesophageal" (gastr-oesophag-eal) into "gastresophageal", which was a
  real defect caught by oracle testing. Anchoring means "gastro-oesophageal"
  folds to "gastro esophageal" rather than to the American spelling, so the
  alias registry carries that form explicitly.
- **Aliases capture equivalence, not hierarchy.** "diabetes" is listed as an
  alias of "type 2 diabetes mellitus" because a summary saying "diabetes" has
  retained the clinically relevant content for this benchmark. It is a
  deliberate loss of specificity, documented in docs/EVALUATION.md.
"""

from __future__ import annotations

import re
from functools import lru_cache

_PUNCTUATION = re.compile(r"[^a-z0-9/.\s-]+")
_WHITESPACE = re.compile(r"\s+")
_HYPHENS = re.compile(r"[-‐-―]")

# Applied after lowercasing, anchored at word starts. Anchoring is load-bearing:
# unanchored, "oesophag" rewrites the American "gastroesophageal" into
# "gastresophageal".
_SPELLING_FOLDS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rf"\b{source}"), target)
    for source, target in (
        ("oesophag", "esophag"),
        ("haemo", "hemo"),
        ("haema", "hema"),
        ("anaemi", "anemi"),
        ("oedema", "edema"),
        ("sulpha", "sulfa"),
        ("paediatr", "pediatr"),
        ("diarrhoea", "diarrhea"),
    )
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def normalize_text(text: str) -> str:
    """Lowercase, fold spelling variants, strip punctuation, collapse spaces.

    Slashes and periods survive because they carry meaning in clinical values
    ("168/94", "1.9"). Hyphens become spaces so "gastro-oesophageal" and
    "gastro oesophageal" normalize alike.
    """
    if not text:
        return ""
    lowered = _HYPHENS.sub(" ", text.lower())
    for pattern, target in _SPELLING_FOLDS:
        lowered = pattern.sub(target, lowered)
    lowered = _PUNCTUATION.sub(" ", lowered)
    return _WHITESPACE.sub(" ", lowered).strip()


def split_sentences(text: str) -> list[str]:
    """Split raw text into sentences, preserving original casing."""
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def sentences_containing(term: str, text: str) -> list[str]:
    """Normalized sentences that mention `term`.

    Context detection - medication status cues, trend direction words,
    historical framing - must be bounded to the sentence the term appears in.
    A fixed character window bleeds into neighbouring sentences and picks up
    cues belonging to a different drug or analyte, which produced a run of
    false positives against a known-correct summary.
    """
    normalized_term = normalize_text(term)
    if not normalized_term:
        return []
    return [
        normalized
        for sentence in split_sentences(text)
        if normalized_term in (normalized := normalize_text(sentence))
    ]


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


# ---------------------------------------------------------------------------
# Alias registry
# ---------------------------------------------------------------------------

# Each entry maps a canonical term to the surface forms a summary might use.
# The canonical form is the one the benchmark's content library emits.

CONDITION_ALIASES: dict[str, tuple[str, ...]] = {
    "essential hypertension": ("hypertension", "htn", "high blood pressure", "elevated blood pressure"),
    "type 2 diabetes mellitus": ("type 2 diabetes", "t2dm", "diabetes mellitus", "diabetes", "dm2", "niddm"),
    "chronic kidney disease stage 3": (
        "chronic kidney disease", "ckd", "ckd stage 3", "ckd 3", "stage 3 chronic kidney disease",
        "chronic renal disease", "chronic renal insufficiency",
    ),
    "heart failure with reduced ejection fraction": (
        "heart failure", "hfref", "chf", "congestive heart failure", "systolic heart failure",
    ),
    "atrial fibrillation": ("afib", "a fib", "af"),
    "chronic obstructive pulmonary disease": ("copd", "chronic obstructive lung disease"),
    "community-acquired pneumonia": ("pneumonia", "cap", "community acquired pneumonia"),
    "acute kidney injury": ("aki", "acute renal failure", "acute renal injury"),
    "hyperlipidemia": ("dyslipidemia", "high cholesterol", "hypercholesterolemia"),
    "hypothyroidism": ("underactive thyroid", "low thyroid"),
    "asthma": ("reactive airway disease",),
    "gastroesophageal reflux disease": (
        "gerd", "gord", "reflux", "acid reflux", "gastro esophageal reflux disease",
    ),
    "osteoarthritis": ("oa", "degenerative joint disease", "djd"),
    "iron deficiency anemia": ("anemia", "iron deficiency", "ida"),
    "cellulitis": ("skin infection",),
    "urinary tract infection": ("uti", "bladder infection"),
    "seizure disorder": ("epilepsy",),
    "major depressive disorder": ("depression", "mdd"),
}

LAB_ALIASES: dict[str, tuple[str, ...]] = {
    "creatinine": ("cr", "scr", "serum creatinine"),
    "potassium": ("k", "serum potassium"),
    "sodium": ("na", "serum sodium"),
    "hemoglobin": ("hgb", "hb", "haemoglobin"),
    "white blood cell count": ("wbc", "white count", "leukocyte count", "white blood cells"),
    "platelet count": ("platelets", "plt"),
    "glucose": ("blood glucose", "blood sugar", "bg"),
    "hemoglobin a1c": ("a1c", "hba1c", "hemoglobin a1c", "glycated hemoglobin"),
    "ldl cholesterol": ("ldl", "ldl c"),
    "tsh": ("thyroid stimulating hormone",),
    "bnp": ("b type natriuretic peptide", "brain natriuretic peptide"),
    "troponin i": ("troponin", "trop"),
    "lactate": ("lactic acid",),
    "inr": ("international normalized ratio",),
    "egfr": ("gfr", "estimated gfr", "estimated glomerular filtration rate"),
    "calcium": ("ca", "serum calcium"),
}

VITAL_ALIASES: dict[str, tuple[str, ...]] = {
    "blood pressure": ("bp", "systolic", "sbp"),
    "heart rate": ("hr", "pulse", "pulse rate"),
    "respiratory rate": ("rr", "resp rate", "breathing rate"),
    "temperature": ("temp", "t"),
    "oxygen saturation": ("spo2", "o2 sat", "oxygen sat", "sat", "sao2", "pulse oximetry"),
}

MEDICATION_ALIASES: dict[str, tuple[str, ...]] = {
    "hydrochlorothiazide": ("hctz",),
    "metoprolol succinate": ("metoprolol", "metoprolol xl", "toprol"),
    "insulin glargine": ("glargine", "lantus", "basal insulin"),
    "acetaminophen": ("paracetamol", "tylenol"),
    "ferrous sulfate": ("iron", "iron supplement", "ferrous sulphate"),
    "levothyroxine": ("synthroid", "l thyroxine"),
    "albuterol": ("salbutamol", "ventolin"),
}

# Cue phrases used by the negation and medication-status detectors. Kept here
# so every piece of clinical language the evaluator keys on lives in one file.

NEGATION_CUES: tuple[str, ...] = (
    "no ", "not ", "without ", "denies ", "denied ", "negative for ", "absent ",
    "no evidence of ", "no history of ", "free of ", "ruled out ", "rules out ",
    "never ", "none ", "unremarkable for ", "no known ", "declines ", "lacks ",
    "was not ", "were not ", "is not ", "are not ", "did not ", "does not ",
    "no longer ", "not performed", "not obtained", "not done",
)

# Cues that negate a term appearing BEFORE them: "chest x-ray was not
# performed", "troponin was negative". Without these, post-posed negation is
# invisible and a correctly-reported absent finding reads as an assertion.
POST_NEGATION_CUES: tuple[str, ...] = (
    "was not performed", "were not performed", "not performed",
    "was not obtained", "were not obtained", "not obtained",
    "was not done", "not done", "was not given", "not given",
    "was negative", "were negative", "is negative", "are negative",
    "was absent", "is absent", "were absent", "not detected",
    "was deferred", "deferred", "was declined", "declined",
    "is not present", "was not present", "not identified",
)

# How far after a term a post-negation cue may appear and still apply.
POST_NEGATION_WINDOW = 30

# Phrases that look like negation cues but do not deny the following term.
#
# Deliberately short. "no change" was listed here originally, borrowed from
# radiology where "no interval change" asserts stability rather than denying a
# finding. That was wrong for this vocabulary: "no change in urine output" is a
# genuine negative, and treating it as a pseudo-cue made the evaluator read a
# correctly preserved negation as a reversal. A pseudo-cue that swallows real
# negations is far more damaging than a missing one, so entries here must be
# phrases that cannot appear as genuine clinical negatives.
PSEUDO_NEGATION_CUES: tuple[str, ...] = (
    "not only",
    "gram negative",
)

# Tokens that end a negation's scope: after one of these, a cue no longer
# applies. Without termination, "no chest pain but reports dyspnea" would
# wrongly mark dyspnea as negated.
NEGATION_TERMINATORS: tuple[str, ...] = (
    "but", "however", "although", "though", "except", "aside from", "apart from",
    "reports", "complains", "endorses", "positive for", "presents with", "with",
)

MEDICATION_STATUS_CUES: dict[str, tuple[str, ...]] = {
    "discontinued": (
        "discontinued", "stopped", "held", "ceased", "no longer taking", "no longer on",
        "d/c", "dc'd", "taken off", "came off", "withdrawn",
    ),
    "new": (
        "newly started", "started", "initiated", "commenced", "new prescription",
        "newly prescribed", "begun", "added",
    ),
    "changed_dose": (
        "increased", "decreased", "titrated", "uptitrated", "adjusted", "reduced",
        "raised", "lowered", "changed from", "dose change",
    ),
    "historical": (
        "in the past", "previously", "historical", "former", "prior medication",
        "used to take", "not a current", "past medication", "no longer a current",
    ),
    "current": (
        "continued", "continues", "current", "ongoing", "maintained", "remains on",
        "stable on", "taking",
    ),
}


@lru_cache(maxsize=1)
def _build_alias_index() -> dict[str, frozenset[str]]:
    """Canonical (normalized) term -> every normalized surface form.

    Cached because it is derived from module constants and never changes at
    runtime.
    """
    index: dict[str, set[str]] = {}
    for registry in (CONDITION_ALIASES, LAB_ALIASES, VITAL_ALIASES, MEDICATION_ALIASES):
        for canonical, aliases in registry.items():
            key = normalize_text(canonical)
            forms = index.setdefault(key, {key})
            forms.update(normalize_text(alias) for alias in aliases)
    return {key: frozenset(value) for key, value in index.items()}


def surface_forms(term: str) -> frozenset[str]:
    """Every known surface form for `term`, including the term itself.

    Falls back to just the normalized term when it is not in the registry -
    an unknown term is matched literally rather than guessed at.
    """
    normalized = normalize_text(term)
    index = _build_alias_index()
    if normalized in index:
        return index[normalized]

    # A term may itself be an alias (output says "CKD", source says the long
    # form); resolve it back to the canonical family.
    for canonical, forms in index.items():
        if normalized in forms:
            return forms
    return frozenset({normalized})


def are_equivalent(left: str, right: str) -> bool:
    """True when two terms share a surface-form family or normalize alike."""
    left_normalized = normalize_text(left)
    right_normalized = normalize_text(right)
    if left_normalized == right_normalized:
        return True
    return right_normalized in surface_forms(left) or left_normalized in surface_forms(right)


def canonical_term(term: str) -> str:
    """The canonical form for `term`, or its normalized form if unregistered."""
    normalized = normalize_text(term)
    index = _build_alias_index()
    if normalized in index:
        return normalized
    for canonical, forms in index.items():
        if normalized in forms:
            return canonical
    return normalized
