# ClinEval AI — Benchmark design and methodology

> **Scope note.** Every case in this benchmark is synthetic and fictional. The
> criticality rules, difficulty tiers, and severity labels defined here are
> **benchmark design decisions**, not clinical standards, triage criteria, or
> validated instruments. Passing this benchmark would not establish that a
> system is safe for clinical use.

---

## 1. The ground-truth-first principle

The central design rule is that **the structured case is the source of truth**,
and everything else is derived from it:

```
                    StructuredCase
                  (source of truth)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    source note    reference summary   annotations
   (note_renderer) (summary_renderer)  (annotator)
```

The three renderings are **independent**. The summary is not written from the
note; the annotations are not extracted from the note. Each reads the same
structured object and writes its own output.

Two consequences follow, and both matter:

**The note and the annotations cannot disagree.** With hand annotation, the
annotator can misread the note, or the note can be edited without the
annotations following. Here a mismatch is structurally impossible, because
neither was written against the other.

**The summary tests information retention, not paraphrase.** If the reference
summary were derived from the note, a model could score well by copying
sentences. Generating it independently from structure means the question is
"did the model preserve the clinical content?", not "did it match our wording?"

The model under evaluation never enters this pipeline. It cannot define,
influence, or annotate its own ground truth.

## 2. Data flow in detail

```
seed ──> StructuredCase ──> clinical timeline
             │                     │
             │  base scaffold: patient, encounter, primary condition,
             │  a linked medication, baseline vitals
             │
             └──> feature builders apply, in fixed order, one per
                  failure-mode category the tier selected
                          │
      ┌───────────────────┼───────────────────┐
      ▼                   ▼                   ▼
 render_note()  render_reference_summary()  annotate()
      │                   │                   │
      └───────────────────┴───────────────────┘
                          ▼
                    BenchmarkCase
             (carries its StructuredCase for traceability)
```

Feature application order is fixed (`_FEATURE_ORDER` in `synthesizer.py`)
rather than draw order, so structural features (comorbidities, historical
conditions) land before features that read the case's existing content
(negation traps, medication changes). Without that, a negation trap could be
chosen before the comorbidity that contradicts it.

## 3. Reproducibility

Each case is seeded with the string `"<master_seed>:<index>"`. A case's content
is fully determined by that seed, its difficulty tier, and the configuration.

```bash
python -m scripts.generate_benchmark --cases 100 --seed 42
```

Running this twice produces byte-identical output, across processes and runs.

**The guarantee is scoped to `(seed, case_count, config)` — not the seed
alone.** Difficulty tiers are allocated by largest remainder across the whole
run, so changing `--cases` reassigns tiers and therefore changes case content.
Asking for 200 cases does *not* reproduce the first 100 of a 100-case run.

That is a deliberate trade. Largest-remainder allocation means a benchmark
advertising 20% adversarial cases contains *exactly* 20, not 20 in
expectation. For a benchmark whose distribution is quoted in results, an exact
mix is worth more than prefix stability.

## 4. Difficulty tiers

Tiers describe how much clinical content a case carries and how adversarially
it is constructed. They do **not** describe how hard the case would be for a
clinician.

| Tier | Definition | Features | Typical content |
|---|---|---|---|
| **EASY** | A simple encounter with few facts and no traps. Clean documentation, one problem, a normal or single abnormal result. | 1–2 | Straightforward notes, short notes, routine negatives, procedures and follow-up |
| **MODERATE** | Several clinical facts interacting: multiple medications, labs, or a temporal element. Realistic routine documentation. | 2–4 | Abnormal labs and vitals, medication changes, a lab trend, a negation, comorbidities |
| **HARD** | Multiple interacting conditions and medications, temporal events, negations, and abnormal values in one case. Always includes a multi-day course. | 4–6 (+ `high_complexity`) | Everything in MODERATE plus dose changes, discontinuations, historical-vs-current distinctions, long notes |
| **ADVERSARIAL** | Deliberately constructed to expose known LLM summarization failures. Always combines a negation, a temporal trap, and a medication trap. | 4–7 (+ `combined_critical`) | Critical omissions, hallucination opportunities, conflicting information, duplicate documentation |

Tier definitions live in `config/benchmark.yaml` and are editable. The mix
defaults to 20 / 35 / 30 / 15.

## 5. Failure-mode categories

Twenty categories, each with a feature builder in `app/benchmark/features.py`.
A case carries a **list** of them, because real documentation mixes failure
modes and the interesting cases are combinations.

| Category | What the case contains |
|---|---|
| `straightforward` | A clean encounter, normal result, routine follow-up |
| `critical_omission` | A panic-range value (and often a severe allergy) that must survive summarization |
| `hallucination_opportunity` | A documented gap — a study *not* performed, a pending result, or no antibiotic prescribed |
| `negation` | A clinically important negative about a condition the patient does **not** have |
| `temporal_progression` | Two results for one analyte plus an explicit trend linking them |
| `medication_change` | A newly started drug |
| `medication_discontinuation` | A stopped drug, with reason and timing |
| `medication_dose_change` | A dose increase, with both prior and current dose in the note |
| `multiple_comorbidities` | Two or three extra chronic conditions, each with a medication |
| `high_complexity` | A multi-day course: a value rises, then partially recovers |
| `conflicting_information` | An earlier statement superseded by a later reassessment |
| `abnormal_labs` | An out-of-range result, sometimes panic-range |
| `abnormal_vitals` | An out-of-range vital sign, sometimes panic-range |
| `relevant_negative_findings` | Pertinent negatives drawn from the case's own conditions |
| `historical_vs_current` | A resolved condition and a past medication alongside active ones |
| `procedures_and_followup` | A performed study with an outcome, plus a concrete plan |
| `duplicate_documentation` | A finding restated later, as copy-forward documentation does |
| `long_note` | Verbose documentation with extra review of systems |
| `short_note` | Terse documentation, minimal sections |
| `combined_critical` | Negation + temporal trap + medication trap in one case |

## 6. Clinical coherence

Cases are composed, not randomized. The content library
(`app/benchmark/content.py`) links each condition to its plausible
medications, the labs usually followed for it, and its typical findings and
relevant negatives. A CKD patient gets a creatinine trend and an ACE
inhibitor, not a random drug attached to a random analyte.

Two coherence invariants are enforced and tested:

**A case never both asserts and denies the same diagnosis.** Negation traps
draw from conditions the patient does *not* have. Without this, a case could
carry "type 2 diabetes mellitus" in its problem list and "no history of
diabetes" in its negatives.

**One analyte has one value per timepoint.** A case reporting both "sodium 125
today" and "sodium 110 today" is incoherent, and an evaluator could not decide
which value retention should be measured against. This is enforced by
`check_no_duplicate_measurements` — a check added after generation review
caught exactly this defect.

## 7. Criticality rules

Criticality is assigned by an explicit rule, never inferred from wording. Every
critical fact records the id of the rule that marked it, so "why is this
critical?" always has a stated answer.

| Rule | Condition | Rationale |
|---|---|---|
| **CR-01** | Lab at or beyond a panic threshold | Drives immediate action; omitting it hides the reason for that action |
| **CR-02** | Abnormal value for a **key** analyte | Key analytes commonly change management; abnormal non-key analytes are recorded but not critical |
| **CR-03** | Clinically significant lab trend | Direction and magnitude can matter more than either endpoint |
| **CR-04** | Major medication change (new / discontinued / dose change) | Most actionable content in a summary; confusing stopped for current risks duplicate therapy |
| **CR-05** | Severe allergy | A missed severe allergy can lead directly to administering the agent |
| **CR-06** | Primary diagnosis | The organizing fact of the encounter |
| **CR-07** | Resolved contradiction between earlier and later documentation | Reporting the superseded statement is wrong even when quoted accurately |
| **CR-08** | Finding marked clinically critical, **including negatives** | Relevant negatives are the findings most often reversed by summarizers |
| **CR-09** | Vital sign at or beyond a panic threshold | Indicates acute physiological instability |
| **CR-10** | Acute deterioration event | Explains why management changed |

**Key analytes** (CR-02): creatinine, potassium, sodium, hemoglobin, WBC,
glucose, BNP, troponin, lactate, INR, eGFR.

**These are benchmark design labels.** They encode "a summary that drops this
is doing something this benchmark counts as a serious error." A different
institution would reasonably draw these lines differently — which is why they
are enumerated rules with ids rather than hard-coded judgments.

A critical negated finding is filed in **both** `negations` and
`critical_facts`. A reversed critical negative is the worst failure the
benchmark models, so it must count toward critical retention as well as being
listed as a negation.

## 8. Reference summary inclusion rules

The summary renderer selects from structure using explicit rules:

1. **Always include** — primary diagnosis, every abnormal lab, every
   significant trend, every medication change, every severe allergy, every
   conflict resolution.
2. **Include when present** — active comorbidities, important relevant
   negatives, performed procedures with outcomes, studies explicitly *not*
   performed, the follow-up plan.
3. **Never include** — routine normal findings, boilerplate review of systems,
   normal labs carrying no trend, and anything absent from the structured case.
4. **Always distinguish current from historical** — a resolved condition or a
   discontinued drug is stated as such, never listed as current.

Rule 3's last clause is machine-checked: `check_summary_supported` verifies
every drug name, condition name, and numeric value in the summary appears in
the structured case.

## 9. The controlled perturbation framework

A `FailureOpportunity` records a known way a summarizer could fail on a case:
its type, a description, the entity it targets, and how severe missing it would
be.

**Opportunities never corrupt the ground truth.** The framework describes
traps; it does not plant lies. A hallucination opportunity does not add a false
finding — it documents that a chest X-ray was *not* performed, and records that
inventing a result would be a fabrication. The case remains entirely truthful.

| Type | The trap |
|---|---|
| `omission` | A high-stakes fact that must be preserved |
| `hallucination` | A gap a model may fill with plausible-sounding content |
| `negation_trap` | A negated statement that could be reversed |
| `temporal_trap` | Values that could be swapped or a trend reversed |
| `medication_trap` | A stopped/changed drug that could be reported as current |
| `historical_current_trap` | Past content that could be presented as active |
| `conflict_trap` | A superseded statement that could be reported as current |
| `redundancy_trap` | Repeated documentation that could be double-counted |

Opportunities with HIGH or CRITICAL severity are promoted to `RiskFlag`s. Minor
ones are not — a risk flag that applied to every redundancy would mean nothing.

## 10. Integrity checks

Run on every generation and by `python -m scripts.audit_benchmark`, which exits
non-zero on any failure so it can gate CI.

| # | Check | Verifies |
|---|---|---|
| 1 | `annotations_reference_source` | Every annotation's `source_ref` resolves to a real entity |
| 2 | `critical_facts_grounded` | Every critical fact is grounded and carries a rule id |
| 3 | `medication_annotations` | Drug name and status match the source entity, and the drug appears in the note |
| 4 | `temporal_annotations` | Each trend's endpoints resolve to real lab results, with distinct values |
| 5 | `negation_annotations` | Each negation points at genuinely absent content |
| 6 | `summary_supported` | No drug, condition, or number in the summary is absent from the structured case |
| 7 | `case_ids_unique` | No duplicate case ids |
| 8 | *(reproducibility)* | Same seed and count produce identical output — asserted in tests |
| 9 | `metadata_matches` | Recorded metadata actually describes the cases it ships with |
| 10 | `no_identifiable_information` | No email, phone, SSN, calendar date, named person, or MRN; patient ids are synthetic surrogates |
| + | `no_duplicate_measurements` | One analyte has one value per timepoint |
| + | `case_structure` | Note and summary non-empty; summary shorter than note |

Each check is proven to fire: `tests/benchmark/test_integrity.py` corrupts a
case in the specific way each check targets and asserts it is caught. A
validator that never fails is worth nothing.

## 11. Commands

Generate the default benchmark:

```bash
python -m scripts.generate_benchmark --cases 100 --seed 42
```

Generate a targeted subset:

```bash
python -m scripts.generate_benchmark --cases 40 --seed 7 --difficulty ADVERSARIAL
```

Audit a benchmark (exits non-zero on failure):

```bash
python -m scripts.audit_benchmark
```

Regenerate the curated Phase 1 seed set:

```bash
python -m scripts.generate_benchmark --curated
```

## 12. Versioning

Generated benchmarks are stamped `clineval-benchmark-v0.1` and written to
`data/synthetic/<version>/`. An existing version is **never overwritten
silently** — `save_generated_benchmark` raises unless `--force` is passed.
A published benchmark is a fixed artifact that results are reported against;
replacing one in place would invalidate any evaluation citing it.

To publish a changed benchmark, bump `version` in `config/benchmark.yaml` so
the new set lands alongside the old one.

The curated Phase 1 seed set keeps its own version (`v1`) because it is a
separate artifact with separate provenance: hand-authored rather than generated.

## 13. Limitations

- **Synthetic, and structurally so.** Cases are composed from a curated content
  library of 18 conditions and 31 medications. Real notes are messier, longer,
  more elliptical, and contain documentation patterns this generator does not
  model.
- **Template phrasing.** The note renderer varies its opening line and section
  content, but sentence structures repeat across cases. A model could learn
  surface regularities that would not transfer to real documentation.
- **Criticality is a benchmark judgment.** CR-01 through CR-10 are defensible
  but not validated against any external standard.
- **Subgroup dimensions are demographic labels on synthetic cases.** They
  support benchmark-level slicing only, and are **not** evidence about
  real-world population fairness.
- **No clinician review.** The clinical content is plausible, not
  authoritative, and has not been reviewed by a licensed clinician.
- **Coverage is not representativeness.** Twenty failure modes across 100 cases
  exercise twenty failure modes; they do not represent the distribution of real
  clinical documentation or of real summarization errors.
