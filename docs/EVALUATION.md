# ClinEval AI — Evaluation methodology

> **Scope note.** The severity labels, metric definitions, and matching rules
> described here are **research framework decisions**, not clinical or
> regulatory standards. Nothing this system reports establishes that a model
> is safe for clinical use.

---

## 1. Why lexical overlap is not enough

A summarizer can score well on ROUGE or BLEU while being clinically wrong, and
badly while being clinically right. Four cases make the point:

| Source | Output | Lexical overlap | Clinically |
|---|---|---|---|
| "No history of diabetes." | "Patient has diabetes." | **Very high** — one token differs | **Catastrophic** — fabricated diagnosis |
| "Creatinine increased from 1.4 to 1.9 mg/dL." | "Creatinine was 1.4 mg/dL." | **High** — most tokens shared | **Wrong** — stale value read as current |
| "Chronic kidney disease stage 3" | "CKD stage 3" | **Low** — almost no shared tokens | **Correct** |
| "Amlodipine was discontinued." | "Patient is on amlodipine." | **High** | **Dangerous** — risks duplicate therapy |

The first two are exactly inverted: the outputs closest to the source text are
the most harmful. This is not a defect in ROUGE — it measures what it claims
to. It is a reason not to use it as a safety signal.

ClinEval therefore compares against **structured clinical content**, not text.
A medication's status is compared against `MedicationStatus`; a trend against
its recorded endpoints and direction; a negative finding against its polarity.
Lexical metrics may be added later as supplementary signals, clearly labelled
as such.

## 2. Architecture

```
                    Benchmark Case
                          |
        +-----------------+-----------------+
        |                                   |
  Structured Ground Truth          Source Clinical Note
        |                                   |
        |                                   v
        |                        AI Generated Summary
        |                                   |
        +-----------------+-----------------+
                          v
        +-------------------------------------+
        |     normalization / negation /      |   primitives
        |     numeric comparison / matching   |
        +-------------------------------------+
                          v
        +-------------------------------------+
        |      entity support analysis        |   shared engine
        +-------------------------------------+
                          v
        +------------------+------------------+
        |                                     |
  Deterministic Evaluators            LLM Evaluators
   (five, implemented)                (contract only)
        |                                     |
        +------------------+------------------+
                           v
                    Finding Store
                           |
                    severity policy
                           |
                           v
                  Evaluation Result
```

Layers, bottom up:

| Module | Responsibility |
|---|---|
| `normalization.py` | Text normalization, curated alias registry, clinical cue lists |
| `negation.py` | NegEx-style negation scope detection |
| `numeric.py` | Clinical value parsing, tolerance-based comparison |
| `matching.py` | Layered fact matching with match levels |
| `support.py` | Per-entity support analysis — the shared engine |
| `evaluators/` | Five deterministic evaluators, one per dimension |
| `severity.py` | Configurable severity classification |
| `metrics.py` | Metric definitions and aggregation |
| `registry.py` | Name → evaluator factory |
| `runner.py` | Benchmark iteration, severity application, aggregation |

Deterministic and model-based evaluators are separated at the base class
(`DeterministicEvaluator` / `LLMEvaluator`) so the registry, runner, and result
schema already distinguish them before any LLM judge exists. **No LLM
evaluator is implemented** — Phase 3 is the deterministic foundation.

## 3. Core principle: the output never defines the truth

Every evaluator reads ground truth from `case.structured_case` and the claim
under test from `ai_output.summary_text`. The output cannot influence what
counts as correct.

Every finding is traceable to:

| Field | Traces to |
|---|---|
| `case_id`, `finding_id` | the benchmark case |
| `fact_id` | the gold-standard annotation |
| `source_ref` | the structured entity |
| `evaluator_name`, `evaluator_version` | the evaluator that produced it |
| `source_evidence` | what the source says |
| `output_evidence` | what the model said |
| `match_level`, `confidence` | how certain the match was |

Both evidence fields are required for clinical findings, so a reviewer can
judge a finding without re-reading the case.

## 4. Fact matching

Matching is a ladder, tried in order, with the level recorded:

| Level | Mechanism | Example | Confidence |
|---|---|---|---|
| `EXACT` | Identical strings | "Creatinine" → "Creatinine" | 1.0 |
| `NORMALIZED` | Case, punctuation, spelling folded | "gastro-oesophageal reflux disease" → "gastroesophageal reflux disease" | 1.0 |
| `ALIAS` | Curated synonym registry | "CKD" → "chronic kidney disease stage 3" | 0.9 |
| `PARTIAL` | ≥65% of content tokens present | "heart failure, reduced EF" → "heart failure with reduced ejection fraction" | 0.7 |
| `NONE` | Not found at any level | | — |

Recording the level matters as much as the match. A finding can say "not found
at any matching level", and a reviewer can see whether a claimed retention
rested on an exact string or a token overlap.

Short aliases are word-boundary bounded. Without that, the potassium alias
"k" matches inside every word containing k, and the sodium alias "na" matches
"nausea".

**Level 4 — LLM adjudication — is deliberately absent.** `MatchResult` and
`FactMatcher.match` are shaped so an adjudicating matcher can be composed in
front of this one without changing any evaluator.

## 5. Normalization and the alias registry

`normalization.py` holds a small, hand-maintained table of the abbreviations
and spelling variants this benchmark's content library actually produces.

**Why not SNOMED, RxNorm, or UMLS?** They are licensed or heavyweight; they
make results depend on an external resource version; and an ontology lookup
that silently maps two terms together is the kind of invisible judgment this
project avoids. A curated table is auditable — you can read all of it.

Stated assumptions:

- **Aliases are bidirectional and context-free.** "CKD" is treated as
  equivalent to "chronic kidney disease" everywhere. Safe for this vocabulary;
  not safe for general text.
- **Aliases capture equivalence, not hierarchy.** "diabetes" is an alias of
  "type 2 diabetes mellitus" because a summary saying "diabetes" has retained
  the clinically relevant content *for this benchmark*. A deliberate loss of
  specificity.
- **Spelling folds are word-anchored.** Unanchored, `oesophag → esophag`
  rewrites the American "gastr-**oesophag**-eal" into "gastresophageal". That
  was a real defect, caught by oracle testing.

## 6. Negation detection

A NegEx-style algorithm:

1. Normalize and split into clauses at `.`, `;`, `:`, newline.
2. Look **backwards** from the term for a negation cue.
3. Look **forwards** within 30 characters for a post-posed cue —
   "chest X-ray **was not performed**" negates a term that precedes it.
4. Terminate scope at conjunctions and assertion verbs — "no chest pain **but
   reports** dyspnea" must not mark dyspnea negated.
5. Reject pseudo-cues that span the term — "no interval change" asserts
   continuity, it does not deny change.

An affirmed mention anywhere wins over a negated one: if the output asserts a
term at all, it makes that claim.

**Known limits**, documented rather than hidden: no double negation, no
subject scope ("mother has no diabetes"), no hypothetical framing ("if the
patient develops diabetes"). None occur in this benchmark's generated text;
all would matter for real notes.

## 7. Clinical numeric comparison

| Result | Condition |
|---|---|
| `CORRECT` | Equal after normalization — 1.9 = 1.90 |
| `APPROXIMATE` | Within 2% relative tolerance — rounding only |
| `INCORRECT` | Beyond tolerance |
| `OMITTED` | No value found for this analyte |

**Why 2%?** On a creatinine of 1.9 it permits ±0.04 — rounding and nothing
else. The errors this platform exists to catch look small numerically:
reporting 1.4 instead of 1.9 is a 26% difference, and reporting the *previous*
value instead of the current one is caught by comparing against the right
source value, not by tolerance at all. A loose tolerance buys nothing and
hides real errors.

Additional rules:

- **Blood pressure compares as an ordered pair.** Both components must match;
  keeping the systolic and altering the diastolic changes the reading.
- **Values are read within the analyte's own sentence.** A fixed character
  window read the next sentence's numbers and compared one analyte against
  another's value.
- **Doses require a dose unit.** "Discontinued 2 weeks ago" must not have its
  "2" compared against a 30-unit prescription.
- **The closest candidate wins.** A summary correctly reporting both endpoints
  of a trend is not scored as an error.

## 8. The five deterministic evaluators

### Critical information retention
Classifies each critical fact as retained, omitted, or contradicted, by fact
id. A fact whose `source_ref` does not resolve is reported as
`missing_annotation` rather than silently skipped — a benchmark defect should
be visible in results, not absorbed by them.

### Medication consistency
Presence → status → dose, in that order, because the errors compound: a drug
that is absent cannot have a wrong dose, and reporting both would double-count
one failure. Distinguishes current, historical, discontinued, newly started,
and dose-changed.

Status compatibility is not strict equality. A dose-changed drug described as
"current" is acceptable — it *is* current — and the dose check catches a
summary reporting the old dose. A discontinued drug described as current is
never acceptable.

### Negation consistency
Four outcomes: preserved, omitted, reversed, unsupported positive assertion.
Omission and reversal are kept separate because a summary that said less
differs from one that said something false.

Negative findings are matched on their **core term** ("no history of diabetes"
→ "diabetes"), then polarity is read from the output. Matching the full wording
meant a reversal failed to match at all and was reported as a mere omission —
inverting the severity of the worst failure mode the benchmark models.

### Temporal consistency
Three distinct failures:

- **information loss** — the trend is flattened to one value; context gone,
  nothing false asserted.
- **value confusion** — the historical value is reported without the current
  one; actively misinforms.
- **trend reversal** — the direction is stated backwards.

A trend is attributed to a sentence only when both its endpoints appear there
as whole numeric tokens within 40 characters of each other. Without all three
constraints, a creatinine series (admission, day 2, discharge) had the first
trend's sentence attributed to every trend over that analyte.

Also covers resolved conditions presented as current, historical medications
presented as active, and superseded statements from resolved conflicts.

### Structured fact support
Both directions. Source → output classifies every analyzable entity. Output →
source flags clinical terms and values the summary asserts that the case does
not contain.

The hallucination direction is **vocabulary-bounded**: it detects invented
content drawn from the benchmark's own content library. Content invented in
other terms is not detected, which makes `unsupported_fact_rate` a **floor on
hallucination, not a ceiling**.

## 9. Severity: separated from detection

Evaluators emit `finding_type` and say nothing about how bad it is. The
severity policy classifies afterwards.

The separation is not tidiness. Severity is the most contestable judgment in
the system — an institution deploying this would set its own thresholds — and
burying it in each evaluator would mean changing detection code to change a
policy opinion, and would let two evaluators disagree about the same class of
error.

Base severities (`config/evaluation.yaml`, fully overridable):

| Finding type | Base | Reasoning |
|---|---|---|
| `empty_output` | CRITICAL | Nothing was produced |
| `critical_omission` | HIGH | A critical fact was dropped |
| `contradiction` | HIGH | False information created |
| `negation_reversal` | HIGH | Fabricated diagnosis, not a wording slip |
| `medication_status_error` | HIGH | Risks duplicate or contraindicated therapy |
| `medication_dose_error` | HIGH | Misstates the regimen |
| `temporal_value_confusion` | HIGH | Stale value read as current |
| `temporal_trend_reversal` | HIGH | Inverts the clinical trajectory |
| `conflict_resolution_error` | HIGH | Superseded statement reported as current |
| `medication_hallucination` | HIGH | Invented drug |
| `temporal_information_loss` | MODERATE | Context lost, nothing false asserted |
| `historical_current_confusion` | MODERATE | Past presented as active |
| `unsupported_addition` | MODERATE | Ranges from padding to fabrication |
| `negation_omission` | MODERATE | Negative dropped, not flipped |
| `medication_omission` | MODERATE | Drug dropped |
| `non_critical_omission` | LOW | Non-critical content dropped |
| `value_approximation` | INFO | Rounding is not a clinical error |

**Escalation.** A finding about a fact the benchmark marked CRITICAL escalates
one step; so does one about a panic-range value. Escalation saturates at
CRITICAL. `value_approximation` never escalates — a rounded value stays
informational however important the fact.

**Low confidence does not de-escalate.** A probable serious error stays
serious; confidence is reported separately so a reviewer can weigh it.

## 10. Metric definitions

Every metric declares its numerator, denominator, interpretation, edge cases,
and limitations **in code** (`app/evaluation/metrics.py`), so documentation and
computation cannot drift apart.

| Metric | Numerator / Denominator |
|---|---|
| `critical_information_retention` | critical facts retained / critical facts annotated |
| `critical_omission_rate` | critical facts absent / critical facts annotated |
| `critical_contradiction_rate` | critical facts contradicted / critical facts annotated |
| `medication_consistency` | medications correct in drug, status, dose / medications annotated |
| `medication_omission_rate` | medications absent / medications annotated |
| `medication_status_error_rate` | medications misframed / medications **mentioned** |
| `negation_consistency` | negations with polarity preserved / negations annotated |
| `negation_reversal_rate` | negations asserted positive / negations annotated |
| `temporal_consistency` | trends with endpoints and direction preserved / trends annotated |
| `temporal_information_loss_rate` | trends flattened to one endpoint / trends annotated |
| `supported_fact_rate` | entities supported / entities analyzable |
| `unsupported_fact_rate` | assertions absent from source / assertions detected |

Two rules run through all of it:

**An undefined metric is reported as undefined.** Zero critical facts means
retention has no value — not 1.0 ("perfect"), not 0.0 ("failed"). `value` is
`None` and aggregation skips it. Reporting 100% retention for a case with
nothing to retain would be flattering nonsense.

**Aggregation pools counts; it does not average averages.** Benchmark-level
retention of 0.9 means 90% of all critical facts across all cases were
retained. Macro-averaging would let a case with one critical fact weigh as
heavily as one with twenty. Both are computed — micro is the headline, macro
is reported alongside, because a large gap between them means performance
varies with how much is at stake in a case.

Note `medication_status_error_rate` uses **mentioned** drugs as its
denominator: a drug the summary omitted cannot have its status misstated, and
including it would dilute the rate.

## 11. Edge cases

| Case | Behavior |
|---|---|
| Empty / whitespace-only output | One `empty_output` finding per evaluator; retention 0.0; `unsupported_fact_rate` **undefined** (nothing asserted can be unsupported) |
| Malformed output (HTML, JSON, other languages) | Evaluated as text; never raises |
| Extremely long output | Evaluated normally |
| Missing annotation (`source_ref` unresolvable) | `missing_annotation` finding |
| No analyzer for an entity type | Excluded from the denominator, not counted as failure |
| Repeated mentions of one fact | Counted once |
| Repeated measurements of one analyte | Value not attributable to a timepoint → OMITTED, not CONTRADICTED — unless it matches **no** measurement in the case, which is a contradiction |
| Conflicting source facts | Resolution is ground truth; reporting the superseded statement is an error |
| Approximate values | `APPROXIMATE`, severity INFO |
| Ambiguous synonyms | Matched at `ALIAS` level with confidence 0.9 |
| Harmless extra prose | Not flagged — only vocabulary terms and clinical values are checked |
| Case with no output | Skipped, recorded in `configuration.cases_skipped_no_output`, never scored as failure |
| Evaluator raises | Caught, recorded on that evaluator's result, run continues |

## 12. Evaluating the evaluator

```bash
python -m scripts.validate_evaluators --cases 50
```

Two questions, both required.

**Specificity — does it stay quiet when nothing is wrong?** The reference
summary is correct by construction, so every finding against it is a false
positive.

**Sensitivity — does it fire when something is wrong?** Nine injections
corrupt a correct summary in one known way each.

An injection that cannot apply to a case is reported as **skipped, never as a
pass**. Counting an injection that never happened as a success would be
precisely the self-flattery this harness exists to prevent.

Current results (50 cases, seed 42):

```
False positives against correct summaries : 0  (0.0000 per case)

Injection                     Expected                       Det Miss Skip    Rate
add_unsupported_diagnosis     unsupported_addition            50    0    0  100.00%
change_dose                   medication_dose_error           50    0    0  100.00%
change_lab_value              contradiction                   34    1   15   97.14%
change_medication             medication_hallucination        50    0    0  100.00%
change_medication_status      medication_status_error         16    0   34  100.00%
present_resolved_as_current   historical_current_confusion    10    0   40  100.00%
remove_critical_fact          critical_omission               14    0   36  100.00%
reverse_negation              negation_reversal               28    0   22  100.00%
reverse_temporal_trend        temporal_trend_reversal         31    0   19  100.00%

OVERALL: PASS
```

The oracle check went from **89 false positives to 0** during development. That
number is the argument for building the specificity check before trusting any
sensitivity result — every one of those 89 was a real defect, and none would
have been visible from the sensitivity numbers alone.

Bugs it caught, all now regression-tested:

1. `oesophag → esophag` corrupting the American "gastroesophageal".
2. Medication status cues leaking across sentence boundaries.
3. Trend direction words read from a neighbouring analyte's sentence.
4. "Discontinued 2 weeks ago" read as a 2-unit dose.
5. A normalized term compared against a raw corpus (hyphens).
6. Repeated measurements of one analyte cross-attributed.
7. Negation reversal reported as omission (detection rate 3.6%).
8. Post-posed negation ("was not performed") undetected.
9. Pseudo-negation spanning the term ("no interval change").
10. A **benchmark** defect: reference summaries dropped a standalone lab value
    whenever any trend used that analyte.

## 13. Known limitations

**Matching.** Alias coverage is limited to this benchmark's vocabulary; a
model using an unregistered synonym would be scored as omitting a fact.
`PARTIAL` matching at 65% of content tokens is a threshold, not a fact.

**Negation.** No double negation, subject scope, or hypothetical framing.
Post-posed cues use a fixed 30-character window.

**Medication status.** Cue-based. A summary conveying discontinuation through
phrasing outside the cue list would be scored as a status error — a false
positive this evaluator can produce.

**Temporal.** Direction detection is keyword-based. A trend described across
several sentences may be scored as information loss where a reader would
follow it. The 40-character endpoint linkage distance is a heuristic.

**Repeated measurements.** The evaluator cannot attribute a stated value to a
timepoint without temporal parsing it does not do, so it declines to accuse
where it cannot attribute. This trades sensitivity for specificity on series.

**Hallucination.** Vocabulary-bounded — a floor, not a ceiling.

**Fact support weighting.** `supported_fact_rate` treats every entity as
equally important, which they are not. Use it alongside
`critical_information_retention`, which weights by the benchmark's own
importance labels.

**Validated on synthetic text only.** Every result here is against generated
summaries with regular structure. Real clinical documentation is messier, and
these detection rates should not be assumed to transfer.

**No LLM adjudication.** Semantic equivalence beyond the alias registry is not
assessed. That is Phase 4.

## 14. What Phase 3 does not do

Deliberately absent, and not to be inferred from any output:

- **No LLM-as-judge.** Contract only.
- **No overall risk score, safety score, or deployment recommendation.** Those
  come after the evaluator is validated, not before.
- **No real model has been evaluated.** All outputs come from test-only
  providers (`app/evaluation/providers.py`), labelled
  `is_real_model_output: False` in their metadata so a synthetic output can
  never be mistaken for a model generation.
