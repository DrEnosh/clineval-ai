# ClinEval AI — Architecture

> **Scope note.** ClinEval AI is a research and portfolio project for evaluating
> clinical AI systems. It is **not** a medical device, is **not** clinically
> validated, and is **not** intended for diagnosis, treatment, medical advice, or
> regulatory submission. Nothing in this document should be read as a claim that
> an evaluated system is safe for patient care.

---

## 1. Problem

Evaluating a clinical AI summarizer is not the same as evaluating a general
summarizer. Two summaries can score identically on ROUGE while differing in ways
that matter enormously at the bedside:

- One says *"no history of diabetes"*; the other says *"patient has diabetes."*
- One reports the **current** creatinine of 1.9 mg/dL; the other reports the
  **three-month-old** value of 1.4 mg/dL.
- One notes amlodipine was **discontinued**; the other lists it as current.

A single aggregate score hides exactly the failures a deployment decision hinges
on. ClinEval AI is built around the opposite premise: **the unit of output is an
evidenced, explained failure, not a number.**

## 2. Layered design

Each layer depends only on the layer below it, and all layers share one set of
types (`app/models/schemas.py`). That shared contract is what allows the
benchmark, database, and API to be built and tested independently of any
concrete evaluator.

```
┌──────────────────────────────────────────────────────────────┐
│  dashboard/            Streamlit UI (Phase 8)                │
│  app/api/              FastAPI HTTP surface (Phase 7)        │
├──────────────────────────────────────────────────────────────┤
│  app/reporting/        Evaluation report generation (P10)    │
│  app/scoring/          Weighted scoring + risk class. (P6)   │
├──────────────────────────────────────────────────────────────┤
│  app/evaluation/       Deterministic evaluation engine (P3)  │
│    base.py             ← the contract                        │
│    support.py          ← shared entity analysis engine       │
├──────────────────────────────────────────────────────────────┤
│  app/benchmark/        Synthetic case generation (P2)        │
│  app/models/schemas.py Shared Pydantic contract  ◄─ everyone │
│  app/core/             Enums + env-driven configuration      │
├──────────────────────────────────────────────────────────────┤
│  app/database/         SQLite now, PostgreSQL-ready (P7)     │
└──────────────────────────────────────────────────────────────┘
```

### Data flow of one evaluation

```
BenchmarkCase          AIOutput
(gold standard,        (summary produced by
 derived from           the model under test)
 structure)
      │                        │
      └──────────┬─────────────┘
                 ▼
   normalization / negation / numeric / matching
                 ▼
        entity support analysis
                 ▼
         EvaluationRunner
                 │
    ┌────────────┼────────────┬─────────────┬──────────────┐
    ▼            ▼            ▼             ▼              ▼
CriticalRet.  Medication   Negation     Temporal      FactSupport
    │            │            │             │              │
    └────────────┴─────┬──────┴─────────────┴──────────────┘
                       ▼
         list[EvaluationFinding] + MetricValue
              (detection only, severity unset)
                       ▼
                Severity policy
           (configurable, applied once)
                       ▼
                EvaluationResult
                       ▼
       Scoring layer → RiskLevel  (Phase 6, not built)
                       ▼
              Report / Dashboard
```

Severity assignment sits deliberately *outside* the evaluators. Evaluators
detect; the runner classifies. Severity is the most contestable judgment in the
system, so it lives in one configurable place rather than being duplicated
across five evaluators that could disagree.

### The one-way rule for ground truth

The model under evaluation never contributes to ground truth. Gold-standard
facts come only from the `BenchmarkCase`; `AIOutput` is only ever the thing
being judged. This is enforced by the signature of
`BaseEvaluator.evaluate(case, ai_output)` and stated in that module's docstring
so a future evaluator implementation cannot quietly violate it.

## 3. Data model

`app/models/schemas.py` is the single source of truth for every shape crossing a
layer boundary. Redefining a subset of these fields elsewhere is a bug.

| Type | Role |
|---|---|
| `ClinicalFact` | One atomic annotated fact, with `importance` and `polarity` |
| `MedicationFact` | Adds `drug_name` / `status` / `dose` / `route` so drug errors are detectable independent of wording |
| `TemporalFact` | Adds `earlier_value` / `later_value` / `trend` so value confusion is detectable |
| `RiskFlag` | What could go wrong clinically if this case is summarized badly |
| `BenchmarkCase` | Structured context + note + reference summary + the gold-standard annotation layer |
| `Model` | A registered AI system under evaluation |
| `AIOutput` | One generated summary (`run_index` distinguishes repeated generations) |
| `EvaluationFinding` | An evidenced finding: type, explanation, source evidence, output evidence, fact id, evaluator name and version, match level, confidence. `severity` is unset until the policy assigns it |
| `MetricValue` | A metric with the numerator and denominator it came from; `value` is `None` when undefined |
| `CaseResult` | Per-case metrics + findings within a run |
| `EvaluationRun` | Everything needed to reproduce a run |
| `EvaluationResult` | A complete model-vs-benchmark evaluation |
| `Report` | Pointer to a generated report artifact |

Phase 3 replaced the Phase 1 `Failure` placeholder with `EvaluationFinding`.
The old type carried a single `evidence` string and a severity the evaluator
assigned itself; neither survives contact with the requirement that a finding
be judgeable without re-reading the case.

### Why polarity is a first-class field

`FactPolarity` (`AFFIRMED` / `NEGATED`) is stored on every fact rather than
inferred from text at evaluation time. Negation is the failure mode most likely
to be silently missed by overlap-based metrics — *"no history of diabetes"* and
*"history of diabetes"* have near-identical lexical overlap and opposite clinical
meaning. Making polarity structural means the negation evaluator compares a
boolean, not a phrase.

## 4. Benchmark design

Two artifacts, with different provenance. Full methodology in
[docs/BENCHMARK.md](BENCHMARK.md).

### The generated benchmark (Phase 2)

`app/benchmark/synthesizer.py` runs a ground-truth-first pipeline: a
`StructuredCase` is built from a seed, and the note, reference summary, and
annotations are each rendered *independently* from it. Nothing is written first
and back-annotated, so the note and its annotations cannot disagree, and the
summary tests information retention rather than paraphrase.

```
app/benchmark/
  content.py          curated clinical vocabulary, with condition-drug-lab links
  case_builder.py     CaseDraft: entity id allocation, sampling, base scaffold
  features.py         one builder per failure-mode category
  criticality.py      CR-01 .. CR-10, the enumerated importance rules
  note_renderer.py    structured case -> source note
  summary_renderer.py structured case -> reference summary   (independent)
  annotator.py        structured case -> gold-standard facts (independent)
  synthesizer.py      pipeline + persistence + metadata
  integrity.py        twelve checks, each proven to fire
  curated.py          the hand-authored Phase 1 seed set
```

Every annotation carries a `source_ref` (the structured entity it came from) and
a `criticality_rule` (the rule that set its importance), which is what makes the
integrity checks possible: an annotation that cannot be traced to real
structured content is a detectable error rather than a silent one.

### The curated seed set (Phase 1)

`app/benchmark/curated.py` produces one hand-authored case per `CaseType`.
Each case is built to exercise the specific failure mode it is named for:

| Case | Type | Exercises |
|---|---|---|
| CASE_001 | `normal` | Baseline; no expected risk flags |
| CASE_002 | `critical_omission` | Creatinine 1.4 → 1.9 mg/dL, eGFR 52 → 38 |
| CASE_003 | `hallucination` | Documents what must *not* be invented (pneumonia, antibiotics despite penicillin allergy) |
| CASE_004 | `negation` | Four critical negated diagnoses |
| CASE_005 | `temporal` | Prior vs. current furosemide dose; weight trend |
| CASE_006 | `medication` | All five `MedicationStatus` values in one case |
| CASE_007 | `multi_morbidity` | Five conditions, four medications |
| CASE_008 | `high_complexity` | 5-day admission, AKI that rises then resolves |
| CASE_009 | `conflicting_information` | Later assessment supersedes earlier triage note |
| CASE_010 | `demographic_subgroup` | Pediatric case for subgroup coverage |

**Why hand-authored for Phase 1.** Gold-standard annotations must be clinically
coherent with the note, or the evaluator is being tested against nonsense.
Templated randomization is added in Phase 2/5 on top of this validated seed set,
with `generate_default_benchmark()` as the stable entry point.

**Determinism.** Generation is deterministic — the same call produces
byte-identical output, asserted by a test. Later randomized generation will be
seeded from `settings.random_seed`.

**Provenance.** Every case declares `DataProvenance.MANUAL_SYNTHETIC`. No real
patient data is used anywhere. See `data/README.md`.

### Annotation integrity is enforced by tests

`tests/benchmark/test_generator.py` asserts the gold standard is internally
coherent — unique namespaced fact ids, correct importance labeling, negations
actually negated, temporal facts with distinct earlier/later values, and every
annotated drug name and temporal value grounded in the note text. These tests
caught two real annotation bugs during Phase 1 (a mislabeled fact and a
medication annotated but never mentioned in the note).

## 5. Evaluation dimensions

Each dimension is a separate evaluator class, independently testable and
swappable. Every result records `evaluator_version` so a past run can be
interpreted after the logic changes.

| Dimension | What it answers | Status |
|---|---|---|
| Critical retention | Is each critical fact retained / omitted / contradicted? | **Implemented (P3)** |
| Medication consistency | Invented / omitted / wrong drug, dose, or status? | **Implemented (P3)** |
| Negation consistency | Was a negated fact asserted as affirmed? | **Implemented (P3)** |
| Temporal consistency | Were values, trends, or prior-vs-current confused? | **Implemented (P3)** |
| Fact support | What does the output support, and what does it assert that the source does not? | **Implemented (P3)** |
| Consistency across runs | Do repeated generations agree? | Contract only; needs a stochastic model (P4) |
| LLM adjudication | Is a near-miss semantically equivalent? | Contract only (P4) |

All five implemented evaluators share one engine
(`app/evaluation/support.py`), so a fix to negation scope or numeric comparison
improves every dimension at once rather than being patched in five places.
Full methodology in [docs/EVALUATION.md](EVALUATION.md).

**Supplementary metrics.** ROUGE and embedding similarity may be reported, but
they are supplementary only. Lexical overlap is not clinical correctness, and
neither is embedding proximity; both are near-blind to exactly the negation and
temporal reversals this platform exists to catch.

**LLM-as-judge.** Permitted, but never as an unquestioned source of truth. Any
LLM judge must emit schema-validated JSON, record model name/version, prompt
version, timestamp and configuration, retry safely on malformed output, and be
labeled in the UI as a model-based judgment. Rule-based and LLM-based signals
are combined rather than the latter replacing the former.

## 6. Severity policy

| Severity | Definition |
|---|---|
| `INFO` | Cosmetic or stylistic; no clinical content affected |
| `LOW` | Non-critical detail imprecise or omitted |
| `MODERATE` | Potentially meaningful clinical distortion |
| `HIGH` | Clinically important misinformation, contradiction, hallucination, or omission of a critical fact |
| `CRITICAL` | Potentially dangerous misrepresentation of highly important information (e.g. reversing a negated diagnosis, inventing a contraindicated treatment) |

Severity attaches to the `EvaluationFinding`, not to the case, because the same
case can produce findings of different severities from different models.

Severity is assigned by a **configurable policy** (`app/evaluation/severity.py`,
`config/evaluation.yaml`) applied by the runner *after* detection — never by the
evaluator that made the finding. A finding about a fact the benchmark marked
critical escalates one step; so does one about a panic-range value. Low
confidence does **not** de-escalate: a probable serious error stays serious, and
confidence is reported separately so a reviewer can weigh it.

## 7. Scoring and risk classification (Phase 6)

Dimension scores are normalized to 0.0–1.0 (higher is better) and combined using
weights read from configuration, never hard-coded:

```yaml
scoring:
  critical_omission_weight: ...
  contradiction_weight: ...
  hallucination_weight: ...
  factuality_weight: ...
```

The resulting `RiskLevel` (`LOW` / `MODERATE` / `HIGH`) is a **research
framework, not a medical or regulatory standard**. Its derivation is documented
and its thresholds are configurable precisely so it is auditable rather than
authoritative.

## 8. Observability and reproducibility

Every `EvaluationRun` records model id, benchmark id and version, evaluation
version, prompt version, random seed, and full configuration. Reproducing a run
means replaying those recorded values — no field needed for reproduction is
inferred after the fact.

## 9. Persistence (Phase 7)

SQLite initially, with a schema deliberately kept PostgreSQL-compatible (no
SQLite-specific types, explicit column types, UTC timestamps). Tables mirror the
schema types: `models`, `benchmark_cases`, `evaluation_runs`, `case_results`,
`failures`, `reports`.

## 10. Configuration and security

All configuration flows through `app/core/config.py` (`pydantic-settings`),
which reads environment variables and an optional local `.env`. There are no
default secrets. With `LLM_API_KEY` unset, the application runs entirely
offline — `settings.is_llm_configured` is the single gate any LLM-calling code
must check. `.env` is gitignored; `.env.example` is committed.

## 11. Phase plan

| Phase | Scope | Status |
|---|---|---|
| 1 | Architecture, schemas, evaluator interfaces, 10-case benchmark, minimal API, tests | **Complete** |
| 2 | Ground-truth-first benchmark generation, criticality rules, integrity audit | **Complete** |
| 3 | Deterministic evaluation engine, findings, severity, self-validation | **Complete** |
| 4 | Evaluation engine implementations | Pending |
| 5 | Failure detection (hallucination, negation, temporal, medication) | Pending |
| 6 | Scoring and risk framework | Pending |
| 7 | Database + full API | Pending |
| 8 | Streamlit dashboard | Pending |
| 9 | Multi-model comparison | Pending |
| 10 | Report generation | Pending |
| 11 | Testing and quality hardening | Pending |
| 12 | Dockerization | Pending |
| 13 | Documentation and portfolio packaging | Pending |

## 12. Design decisions worth recording

**Interfaces before implementations.** Phase 1 defines the evaluator contract
without writing a single evaluator. This let the benchmark, API, and test suite
be built and validated against a stable contract, and it forced the ground-truth
directionality rule into the type signature rather than a convention.

**No stub endpoints returning fake data.** The Phase 1 API exposes only health
and benchmark reads — what actually exists. An endpoint returning invented
evaluation numbers would violate the project's core rule against fabricated
results, even in a placeholder.

**Structured fields over text matching.** `MedicationStatus`, `FactPolarity`,
and `TemporalTrend` are enums on the fact, so "did the model get the drug status
right" is a comparison, not a regex.
