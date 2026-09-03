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

**Phase 1 of 13 is complete.** The foundation runs end to end: schemas,
evaluator contract, a validated 10-case synthetic benchmark, a live API, and 67
passing tests. The evaluator *implementations* are Phase 4–5 and are not written
yet — the interfaces are defined, deliberately, before anything depends on them.

| | |
|---|---|
| ✅ Repository, architecture, data schemas | ✅ 10 synthetic benchmark cases |
| ✅ Evaluation interfaces | ✅ 67 passing tests |
| ✅ Minimal runnable API | ⬜ Evaluators, scoring, DB, dashboard, reports |

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

Ten hand-authored synthetic cases, one per stress-test dimension:

| Case | Type | What it exercises |
|---|---|---|
| CASE_001 | normal | Baseline, no expected risk flags |
| CASE_002 | critical omission | Creatinine 1.4 → 1.9 mg/dL; eGFR 52 → 38 |
| CASE_003 | hallucination | What must *not* be invented (pneumonia; antibiotics despite penicillin allergy) |
| CASE_004 | negation | Four critical negated diagnoses |
| CASE_005 | temporal | Prior vs. current furosemide dose; weight trend |
| CASE_006 | medication | All five medication statuses in one case |
| CASE_007 | multi-morbidity | Five conditions, four medications |
| CASE_008 | high complexity | 5-day admission; AKI that rises then resolves |
| CASE_009 | conflicting info | Later assessment supersedes earlier triage note |
| CASE_010 | subgroup | Pediatric case for subgroup coverage |

Every case carries a structured gold-standard layer — critical facts,
medication facts (with drug, dose, status), temporal facts (with earlier value,
later value, trend), negations, and expected risk flags.

The benchmark's own integrity is tested: fact ids unique and namespaced,
importance labels correct, negations actually negated, and every annotated drug
name and temporal value grounded in the note text. **These tests caught two real
annotation bugs during Phase 1** — a fact mislabeled critical, and a medication
annotated but never mentioned in its note.

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

**Generate the benchmark** (writes `data/synthetic/benchmark_v1.json`):

```bash
.venv/Scripts/python.exe scripts/generate_benchmark.py
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

All data here is synthetic and hand-authored, labeled `MANUAL_SYNTHETIC` on
every case. No real or de-identified patient data is used, required, or
committed. Restricted datasets such as MIMIC-IV-Note are **never** downloaded or
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

- **The benchmark is synthetic and small.** Ten cases exercise ten failure modes;
  they do not represent the distribution of real clinical documentation.
- **Evaluator implementations do not exist yet.** Phase 1 ships the contract and
  the benchmark. No model has been evaluated, and no evaluation numbers appear
  anywhere in this repository.
- **Subgroup analysis is benchmark-level only.** Subgroup results describe this
  synthetic benchmark. They are **not** evidence about real-world population
  fairness and must never be presented as such.
- **The risk classification is a portfolio framework**, not a validated
  instrument, and carries no regulatory meaning.
- **Clinical content is plausible, not authoritative.** It was written to
  exercise an evaluator, not to serve as medical reference material.
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
