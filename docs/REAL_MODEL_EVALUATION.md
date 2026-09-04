# ClinEval AI — Phase 5: real-model evaluation and methodology validation

> **Status: apparatus complete, real-model results PENDING.**
>
> No provider credential is configured in this environment and no provider SDK
> is installed, so **no real model has been evaluated**. The experiment is
> fully built, pre-registered below, and runs the moment a credential is
> supplied. Every result section that requires a real model says PENDING rather
> than carrying a placeholder number.
>
> The methodology validation that does **not** require a model was completed,
> and it found and fixed four defects. That work is reported in full.

> **Scope note.** These are exploratory results on a small synthetic benchmark.
> They are not clinical validation, not evidence about real-world performance,
> and not grounds for any claim that a model is safe for clinical use.

---

## 1. Why there are no real-model numbers here

Phase 5 asks for three models evaluated on ten cases. That requires an API
credential, and this environment has none:

```
configured providers : ['mock']
anthropic / openai / gemini / local : not set
SDKs installed       : none
```

`ANTHROPIC_BASE_URL` is present in the environment, but it points at the
harness proxy that runs the assistant. That authentication exists to run this
session, not to be spent on evaluation calls, so it was not used.

Two options were available: fabricate plausible numbers, or build the
experiment and report the blocker. The first is excluded by Phase 5's own
definition of done ("no fabricated results exist") and by the premise of the
whole project. So the apparatus is built and pre-registered, and
`scripts/run_experiment.py` **refuses** to substitute a mock for a real
provider:

```
$ python -m scripts.run_experiment --models anthropic:<model-id>
Error building model 'anthropic:<model-id>': No credential for provider
'anthropic'. Set CLINEVAL_ANTHROPIC_API_KEY ...

No result is written. This script does not fall back to a mock provider,
because a fabricated result must never be mistaken for a real one.
$ echo $?
1
```

To run the experiment, set a credential and re-run — nothing else changes:

```bash
export CLINEVAL_ANTHROPIC_API_KEY=...        # and/or OPENAI / GOOGLE
pip install anthropic                        # SDKs are lazy, install as needed

python -m scripts.run_experiment \
  --models anthropic:<model-id> openai:<model-id> gemini:<model-id> \
  --adjudicate --judge anthropic:<model-id> \
  --repeat-model anthropic:<model-id> --repeats 3 --repeat-temperature 1.0
```

The experiment runs with **one, two, or three** models. It does not require all
three.

## 2. Experiment design (pre-registered)

### Sample

Ten cases, stratified rather than random, selected deterministically from the
100-case `clineval-benchmark-v0.1` benchmark. Every model sees the same ten.

| Stratum | Case | Why it is in the sample |
|---|---|---|
| simple | `SYN_0001` | Baseline: straightforward documentation |
| critical omission | `SYN_0086` | Carries a panic-range value |
| hallucination opportunity | `SYN_0087` | Documents a study *not* performed |
| negation | `SYN_0025` | Clinically important negative |
| temporal trend | `SYN_0023` | Lab trend with both endpoints |
| medication change | `SYN_0022` | Started / stopped / dose-changed drug |
| multimorbidity | `SYN_0024` | Several concurrent conditions |
| high complexity | `SYN_0056` | Multi-day course, rise then partial recovery |
| conflict resolution | `SYN_0089` | Superseded statement |
| adversarial | `SYN_0088` | Combined negation + temporal + medication traps |

Composition: 1 EASY, 4 MODERATE, 1 HARD, 4 ADVERSARIAL. 41 critical facts,
30 medication facts, 13 temporal facts, 17 negations. Mean note length 1,325
characters.

A stratum with no matching case is reported as **unfilled**, never back-filled
with a substitute — that would claim coverage the sample does not have.

### Generation

One prompt for every model (`prompts/summarization_v1.txt`), defining the task
only. The model receives the **source clinical note and nothing else**: no
critical-fact list, no benchmark labels, no annotations, no evaluator findings.
Leakage would make the experiment measure prompt engineering rather than
summarization.

Recorded per generation: provider, exact model identifier, model version where
exposed, temperature, max tokens, prompt version, prompt content hash, latency,
token usage, and timestamp.

### Evaluation

```
source note -> model -> summary -> deterministic evaluators -> routed findings
                                                                     |
                              informed adjudication  <---------------+
                              blind adjudication     <---------------+
```

### Blind adjudication (the anchoring experiment)

Two conditions over identical findings and identical summaries. Only the prompt
differs.

| | Prompt | Judge sees the deterministic finding |
|---|---|---|
| **Condition A — informed** | `adjudication_v1` | yes |
| **Condition B — blind** | `adjudication_blind_v1` | no |

Measured: judgment agreement rate, withdrawal delta, contradiction handling,
and `ADJUDICATION_REQUIRED` counts. **Neither condition is assumed better.** A
low agreement rate or a large withdrawal delta indicates the deterministic
result is anchoring the judge; that is a measurement, and it is reported
whichever way it comes out.

The comparison is verified able to *detect* a difference, not merely to report
agreement: `test_end_to_end_anchoring_is_detectable` drives a judge whose answer
depends on seeing the deterministic finding and confirms the comparison reports
0.0 agreement.

## 3. Results — PENDING

The following require a real model and are not yet available. No placeholder
values are given.

| Section | Status |
|---|---|
| Models evaluated | **PENDING** — no credential configured |
| Pre-adjudication metrics per model | **PENDING** |
| Post-adjudication finding counts | **PENDING** |
| Failure taxonomy per model | **PENDING** |
| Blind vs informed comparison | **PENDING** (apparatus verified) |
| Repeated-run stability | **PENDING** (apparatus verified) |
| Cost, tokens, latency | **PENDING** (apparatus verified) |
| Human inspection sample | **PENDING** (apparatus verified) |

The apparatus was exercised end-to-end with the deterministic mock provider to
confirm every artifact is produced correctly. Those runs are **not results**:
mock output is labelled `is_real_model_output: false`, and both
`scripts/analyze_runs.py` and the artifacts themselves state that nothing in
them is evidence about how any real model behaves.

## 4. Methodology validation — COMPLETE

This part needed no model, and it is where Phase 5 did its real work.

### The problem it addresses

Phase 3 validated the evaluator against the benchmark's **own** reference
summaries and controlled corruptions of them: 0 false positives, 9/9 injections
detected. That establishes self-consistency. It does **not** establish that the
evaluator survives clinical language written in a different register — which is
exactly what a real model produces.

So `app/experiments/challenges.py` writes clinically equivalent summaries in
realistic registers and asserts what the evaluator should conclude.

| Category | Probes | Expectation |
|---|---|---|
| `register` | Different words, same facts | must not fire |
| `compression` | Routine content dropped, critical facts kept | must not fire |
| `negation_forms` | "denies", "negative for", "is absent", "without" | must not fire |
| `numeric_forms` | `1.90`, "up to", decimal padding | must not fire |
| `medication_language` | "held", "stepped up", "no longer being taken" | must not fire |
| `historical_language` | Current-vs-past in other words | must not fire |
| `long_note` | Same content buried in padding | must not fire |
| `abstraction` | Numbers replaced by clinical characterization | **undecided** |

These are hand-designed transformations, **not model outputs**. They test
whether the evaluator understands clinical language; they say nothing about how
any particular model writes.

### First run: 19 failures out of 59 decidable challenges (67.8% pass)

| Category | Failures | Unexpected findings |
|---|---|---|
| `compression` | 10 | `medication_omission` ×10 |
| `numeric_forms` | 6 | `unsupported_addition` ×6, `temporal_information_loss` ×5 |
| `medication_language` | 3 | `medication_status_error` ×3 |
| `register`, `negation_forms`, `historical_language`, `long_note` | 0 | — |

Negation forms, register shifts, and long notes passed cleanly on the first
run. The alias registry and sentence-bounded matching from Phase 3 held up.

### What each failure turned out to be

Not all three were evaluator bugs, and separating them mattered.

**P5-01 — numeric values compared as strings (evaluator bug).** A summary
writing `12.50 mg` where the case says `12.5 mg` was reported as fabricating a
value, and `1.40 → 1.90` broke trend endpoint matching. Notation was changing
the verdict. Fixed by comparing numerically in both places.

**P5-02 — medication cues too narrow (evaluator bug).** "has been held", "is
no longer being taken", and "dose was stepped up" went undetected, so summaries
stating the status perfectly clearly were flagged as status errors. Cue lists
widened; shorter general cues now cover families of phrasings.

**P5-03 — the challenge was wrong, not the evaluator.** The compression
challenge dropped current medications and called the resulting findings false
positives. On inspection the evaluator was right: removing the antibiotic from
a cellulitis summary is real information loss, not harmless concision. The
challenge was corrected and a regression test records *why* — a suite that only
remembers the times the system was wrong drifts toward assuming it never is.

**P5-04 — a truncating regex (bug in Phase 4 code, found incidentally).** The
pattern `[^.]*\.` stopped at the decimal in `4.1`, producing mangled text like
`"potassium is now deranged.1 to 5.7 mEq/L"`. Present in the challenge suite
*and* in the Phase 4 mock provider, meaning the Phase 4 adjudication
demonstration had been running against slightly malformed summaries. It did not
affect any deterministic evaluator or any committed benchmark artifact. Fixed
at both sites.

### After the fixes: 56 of 56 decidable challenges pass

Full details, versions, and regression tests in [CHANGELOG.md](../CHANGELOG.md).

### The abstraction result — deliberately unresolved

Seven `abstraction` challenges are marked **undecided** and excluded from the
pass rate. Given a summary saying "Interval change: renal function has
worsened" where the source says "creatinine rose from 1.4 to 1.9 mg/dL", the
evaluator currently reports:

```
critical_omission        9
temporal_information_loss 12
contradiction             3
```

That is recorded, not asserted to be correct. Whether clinical abstraction
adequately conveys a numeric trend is a clinical judgment, not a technical one,
and encoding a contested opinion as a passing test would launder it into a
fact. **This is precisely the case LLM adjudication exists for**, and measuring
how a real judge rules on it is one of the more interesting things the pending
experiment will produce.

## 5. Evaluator self-challenge

| Evaluator | Synthetic validation (Phase 3) | Real-register observations (Phase 5) | Known weaknesses | Potential improvements |
|---|---|---|---|---|
| `critical_retention` | 0 FP on oracle; omission injection 100% | Register, compression, negation forms all clean | Treats abstraction as omission | LLM adjudication (built); a clinical-equivalence table |
| `medication_consistency` | 0 FP; status/dose injections 100% | **Failed** on "held", "stepped up", "no longer being taken" → cues widened | Still cue-based; unlisted phrasing will misfire | Dependency parsing, or adjudicating status questions |
| `negation_consistency` | 0 FP; reversal injection 100% | **Clean** across five negation constructions | No double negation, subject scope, or hypotheticals | Full NegEx/ConText port |
| `temporal_consistency` | 0 FP; reversal injection 100% | **Failed** on padded decimals → numeric matching | 40-char endpoint linkage is a heuristic; trends across sentences may be missed | Relation extraction over the whole summary |
| `fact_support` | 0 FP; hallucination injection 100% | **Failed** on padded decimals → numeric comparison | Vocabulary-bounded: a floor on hallucination, not a ceiling | Open-vocabulary entity extraction |

Every Phase 5 change became a regression test, took a version increment
(`0.3.0` → `0.5.0` across all five evaluators, since they share the engine that
changed), and is documented in the changelog — as required, so that optimizing
against the data that revealed a problem is at least visible when it happens.

## 6. Statistical discipline

Ten cases cannot support a percentage read as an estimate of real-world
performance, and the apparatus enforces that rather than trusting the reader:

- Every artifact carries `"exploratory": true`.
- Every rate is reported with its counts (`0.7561 (31/41)`).
- Every metric with a zero denominator is `null`, never `0.0` or `1.0`.
- The comparison artifact produces **no overall ranking**. Models with
  different failure profiles are not ordered on one axis — a test asserts the
  artifact contains no `rank`, `best_model`, `overall_score`, or `winner` key.
- `scripts/analyze_runs.py` says "**potential** false positive" and states that
  no item has been verified by a person.

Uncertainty intervals are deferred to the 100-case scale-up, where they would
mean something.

## 7. Privacy

Unchanged and verified: the experiment runs only on the synthetic benchmark. No
real patient note, identifier, or record is introduced. The repository remains
safe for public portfolio use. Data leaves the machine only when a remote
provider is deliberately configured, and the CLI names the provider and case
count before sending anything.

## 8. Limitations

**Of this phase:**

- **No real model has been evaluated.** Everything in §3 is pending.
- The challenge transformations are hand-designed, so they carry my assumptions
  about what realistic clinical language looks like. A real model will phrase
  things I did not anticipate, and that is the point of running the experiment.
- The `abstraction` question remains open by design.
- The blind/informed comparison has been verified to *work*; whether real
  anchoring occurs is unmeasured.

**Of the system, restated:**

- Results describe this synthetic benchmark only.
- Criticality rules (CR-01…CR-10) are benchmark design labels, not a clinical
  standard, and no clinician has reviewed them.
- Hallucination detection is vocabulary-bounded — a floor, not a ceiling.
- Prompt-injection defense is containment, not prevention.
- **No risk score, deployment classification, or safety score is produced.**
  Nothing here establishes that any model is safe for clinical use.

## 9. Reproducing this

```bash
# Methodology challenges (no credential needed)
python -m pytest tests/evaluation/test_phase5_regressions.py -v

# Apparatus check with the offline mock (produces artifacts, NOT results)
python -m scripts.run_experiment --models mock:demo --behaviour abstractive \
  --adjudicate --repeat-model mock:demo --repeats 3

# Error analysis over stored artifacts
python -m scripts.analyze_runs

# The real experiment, once a credential is set
python -m scripts.run_experiment --models anthropic:<model-id> --adjudicate
```
