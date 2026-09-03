# ClinEval AI

**An evaluation platform for clinical AI summarization systems — built to surface
the failures that aggregate scores hide.**

> ⚠️ **Research and evaluation use only.** ClinEval AI is not a medical device,
> has not been clinically validated, and is not intended for diagnosis,
> treatment, medical advice, or regulatory submission. All data in this
> repository is synthetic. See [Limitations](#limitations).

---

## The problem

A clinical summarizer can score well on standard summarization metrics and still
be unsafe to deploy. Consider two summaries of the same note:

| Source note says | Summary says | ROUGE impact | Clinical impact |
|---|---|---|---|
| "No history of diabetes." | "Patient has diabetes." | ~nil | Wrong diagnosis in the chart |
| "Creatinine increased from **1.4** to **1.9** mg/dL." | "Creatinine 1.4 mg/dL." | ~nil | Renal decline invisible |
| "Amlodipine was **discontinued**." | "Patient is on amlodipine." | ~nil | Duplicate antihypertensive therapy |

Lexical overlap is nearly blind to all three. Embedding similarity is barely
better — a negation flip barely moves a cosine score. The failures that decide
whether a system is deployable are precisely the ones a single number hides.

## The solution

ClinEval AI evaluates a clinical summary against a **gold-standard annotation
layer authored independently of the model under test**, across dimensions chosen
for clinical meaning rather than convenience:

clinical factuality · critical information retention · hallucination ·
contradiction · negation handling · temporal consistency · medication
consistency · repeatability across runs · subgroup performance

Its output is not a score. It is an evidenced finding:

```
HIGH · Critical information omitted            CASE_002 · fact CASE_002-F001

  Omitted:   Creatinine increased from 1.4 to 1.9 mg/dL over 3 months,
             consistent with significant decline in renal function
  Why it     Omitting the renal function decline could delay medication
  matters:   dose adjustment (e.g. metformin) and nephrology follow-up
```

Every score the platform reports traces back to records like this one.

## Status

**Phases 1–2 of 13 are complete.** The foundation and the benchmark are done:
schemas, the evaluator contract, a reproducible 100-case synthetic benchmark
built ground-truth-first, a twelve-check integrity validator, a live API, and
276 passing tests at 99% coverage.

The evaluator *implementations* are Phases 4–5 and are not written yet — the
interfaces are defined, deliberately, before anything depends on them. **No
model has been evaluated, and no evaluation results exist anywhere in this
repository.**

| | |
|---|---|
| ✅ Repository, architecture, data schemas | ✅ 100-case generated benchmark + 10 curated cases |
| ✅ Evaluation interfaces | ✅ Integrity audit with CI-gating exit code |
| ✅ Minimal runnable API | ✅ 276 passing tests, 99% coverage |
| ⬜ Evaluators, scoring, DB, dashboard, reports | |

See the [phase plan](docs/ARCHITECTURE.md#11-phase-plan) for what lands when.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  dashboard/            Streamlit UI (Phase 8)                │
│  app/api/              FastAPI HTTP surface                  │
├──────────────────────────────────────────────────────────────┤
│  app/reporting/        Evaluation report generation (P10)    │
│  app/scoring/          Weighted scoring + risk class. (P6)   │
├──────────────────────────────────────────────────────────────┤
│  app/evaluation/       Evaluator interfaces + impls (P4/P5)  │
├──────────────────────────────────────────────────────────────┤
│  app/benchmark/        Synthetic case generation             │
│  app/models/schemas.py Shared Pydantic contract  ◄─ everyone │
│  app/core/             Enums + env-driven configuration      │
├──────────────────────────────────────────────────────────────┤
│  app/database/         SQLite now, PostgreSQL-ready (P7)     │
└──────────────────────────────────────────────────────────────┘
```

One rule holds the design together: **the model under evaluation never
contributes to ground truth.** Gold-standard facts come only from the benchmark
case; the AI output is only ever the thing being judged. This is enforced in the
evaluator signature, not just documented.

Full detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## The benchmark

100 generated cases across 20 failure modes and 4 difficulty tiers, plus 10
hand-authored curated cases. Full methodology in
[docs/BENCHMARK.md](docs/BENCHMARK.md).

### Ground truth comes first

The structured case is the source of truth. The note, the reference summary,
and the annotations are three **independent** renderings of it:

```
                    StructuredCase
                  (source of truth)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    source note    reference summary   annotations
```

Nothing is written first and back-annotated. Two things follow:

- **The note and annotations cannot disagree** — neither was written against
  the other, so the mismatch hand annotation is prone to is structurally
  impossible.
- **The summary tests retention, not paraphrase** — generated from structure
  rather than from the note, so a model cannot score well by copying sentences.

### Difficulty tiers

| Tier | Share | Definition |
|---|---|---|
| **EASY** | 20% | Few facts, no traps — clean documentation of one problem |
| **MODERATE** | 35% | Several interacting facts: medications, labs, a temporal element |
| **HARD** | 30% | Interacting conditions and medications, a multi-day course, negations, abnormal values |
| **ADVERSARIAL** | 15% | Built to expose known LLM failures; always combines a negation, a temporal trap, and a medication trap |

The mix is exact, not probabilistic — largest-remainder allocation means a
100-case run contains exactly 20 easy and 15 adversarial cases.

### Reproducible

```bash
python -m scripts.generate_benchmark --cases 100 --seed 42
```

The same seed and case count produce byte-identical output, across processes
and machines. No LLM is involved anywhere in generation — ground truth produced
by a language model would be no more trustworthy than that model.

### Criticality is rule-based, not inferred

Ten enumerated rules (CR-01 … CR-10) assign importance, and every critical fact
records the id of the rule that marked it. "Why is this fact critical?" always
has a stated answer — a panic-range lab is CR-01, a major medication change is
CR-04, a reversed critical negative is CR-08. These are documented as
**benchmark design labels, not clinical standards**.

### Its integrity is machine-checked

Twelve checks verify every annotation traces to real structured content, that
the three renderings agree, that the summary invents nothing, and that no
identifiable information exists anywhere:

```bash
python -m scripts.audit_benchmark
```

Exits non-zero on any failure, so it gates CI. Each check is *proven to fire* —
the test suite corrupts a case in the specific way each check targets and
asserts it is caught, because a validator that never fails is worth nothing.

**These checks caught four real defects during development**: a fact mislabeled
critical, a medication annotated but absent from its note, two conflicting
sodium values at the same timepoint, and `.capitalize()` silently corrupting
"GERD" into "gerd".

## Installation

Requires Python 3.11+ (developed on 3.13).

```bash
git clone <repository-url> && cd "ClinEval AI"
```

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

On macOS/Linux use `.venv/bin/python` in place of `.venv/Scripts/python.exe`.

Then copy the example environment file. The app runs fully offline; no API key
is needed for anything implemented today.

```bash
cp .env.example .env
```

## Running

**Generate the benchmark** (writes `data/synthetic/clineval-benchmark-v0.1/`):

```bash
.venv/Scripts/python.exe -m scripts.generate_benchmark --cases 100 --seed 42
```

**Audit the benchmark** (exits non-zero if any integrity check fails):

```bash
.venv/Scripts/python.exe -m scripts.audit_benchmark
```

**Generate a targeted subset:**

```bash
.venv/Scripts/python.exe -m scripts.generate_benchmark --cases 40 --seed 7 --difficulty ADVERSARIAL
```

**Regenerate the curated 10-case seed set:**

```bash
.venv/Scripts/python.exe -m scripts.generate_benchmark --curated
```

**Run the API:**

```bash
.venv/Scripts/python.exe -m uvicorn app.api.main:app --reload --port 8000
```

Interactive API docs are then at `http://127.0.0.1:8000/docs`.

**Run the tests:**

```bash
.venv/Scripts/python.exe -m pytest -q
```

**Run the tests with coverage:**

```bash
.venv/Scripts/python.exe -m pytest --cov=app --cov-report=term-missing
```

## API (Phase 1)

| Endpoint | Returns |
|---|---|
| `GET /health` | Service status, app version, benchmark version |
| `GET /benchmarks` | Benchmark id, version, case count, case types |
| `GET /benchmarks/cases` | All cases with type and critical-fact count |
| `GET /benchmarks/cases/{case_id}` | One full case including the gold-standard layer |

Endpoints for running evaluations, comparing models, and generating reports
arrive in later phases. They are **not** stubbed out returning placeholder
numbers — an endpoint that invents evaluation results would violate the
project's rule against fabricated output, even as a placeholder.

## Evaluation methodology

| Dimension | Definition |
|---|---|
| **Factuality** | Each gold fact classified as retained, incorrectly represented, missing, contradicted, or newly introduced |
| **Critical retention** | `critical facts retained ÷ total critical facts`, reported with the omitted and contradicted facts named |
| **Hallucination** | Content in the output unsupported by the source, split into unsupported additions vs. explicit contradictions |
| **Negation** | Detects polarity reversal — a `NEGATED` source fact asserted as affirmed |
| **Temporal** | Detects confusion of historical vs. current values, reversed trends, clinically meaningful date omissions |
| **Medication** | Detects invented, omitted, wrong-drug, wrong-dose, and wrong-status medication errors |
| **Repeatability** | Agreement across repeated generations for stochastic models |

**On lexical and semantic metrics.** ROUGE, BLEU, and embedding similarity may be
reported as *supplementary* signals. They are not measures of clinical
correctness, and this platform exists largely because they are near-blind to the
negation and temporal failures above.

**On LLM-as-judge.** Permitted, never authoritative. Any LLM judge must emit
schema-validated JSON and record model version, prompt version, timestamp, and
configuration; results are labeled as model-based judgments and combined with
rule-based signals rather than replacing them.

**On the risk score.** The `LOW`/`MODERATE`/`HIGH` classification is a documented
research framework with configurable weights — not a medical or regulatory
standard.

## Dataset provenance

All data here is synthetic. No real or de-identified patient data is used,
required, or committed, and **no generative model is involved in producing any
case** — notes, summaries, and annotations all come from deterministic renderers
reading a structured object. Timing is always relative ("3 months ago"), never
a calendar date; patient ids are synthetic surrogates.

Restricted datasets such as MIMIC-IV-Note are **never** downloaded or
redistributed by this project; the architecture merely allows a credentialed
user to evaluate one locally under their own data use agreement. Full detail:
[data/README.md](data/README.md).

## Security

No secrets are committed. Configuration flows through environment variables via
`app/core/config.py`; `.env` is gitignored and `.env.example` is committed with
empty credential fields. With `LLM_API_KEY` unset the application runs entirely
offline, gated by a single `settings.is_llm_configured` check.

## Limitations

Stated plainly, because a project about honest evaluation should be honest about
itself:

- **The benchmark is synthetic, and structurally so.** Cases are composed from a
  curated library of 18 conditions and 31 medications. Real notes are messier,
  longer, more elliptical, and full of documentation patterns this generator
  does not model.
- **Template phrasing repeats.** Sentence structures recur across cases, so a
  model could learn surface regularities that would not transfer to real
  documentation.
- **Coverage is not representativeness.** 20 failure modes across 100 cases
  exercise 20 failure modes. They do not represent the distribution of real
  clinical documentation or of real summarization errors.
- **Criticality rules are benchmark judgments.** CR-01 … CR-10 are defensible
  but not validated against any external standard.
- **No clinician review.** Clinical content is plausible, not authoritative, and
  has not been reviewed by a licensed clinician.
- **Evaluator implementations do not exist yet.** Phases 1–2 ship the contract
  and the benchmark. No model has been evaluated, and no evaluation numbers
  appear anywhere in this repository.
- **Subgroup analysis is benchmark-level only.** Subgroup results describe this
  synthetic benchmark. They are **not** evidence about real-world population
  fairness and must never be presented as such.
- **The risk classification is a portfolio framework**, not a validated
  instrument, and carries no regulatory meaning.
- **Passing this benchmark would not make a system safe for clinical use.** No
  result produced here can support that claim.

## Roadmap

Phases 2–3 scale benchmark generation and annotation tooling. Phases 4–5 bring
the evaluator implementations and failure detection. Phase 6 adds configurable
scoring and risk classification, Phase 7 persistence and the full API, Phase 8
the Streamlit dashboard, Phase 9 multi-model comparison, Phase 10 report export,
and Phases 11–13 hardening, Docker, and packaging. See the
[phase plan](docs/ARCHITECTURE.md#11-phase-plan).

## License

MIT — see [LICENSE](LICENSE).
