# ClinEval AI — LLM generation and adjudication

> **Scope note.** LLM-based evaluation is itself fallible. It is not a
> substitute for clinical validation, and nothing produced by the adjudication
> layer establishes that a model is safe for clinical use. The judge is a
> component under test, not an authority.

---

## 1. The core principle

**The LLM is not ground truth.** The benchmark's structured case is.

```
                 Benchmark Ground Truth
                 (structured, Phase 2)
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      Deterministic              LLM-assisted
      Evaluation                 Adjudication
      (Phase 3)                  (only where
              │                   deterministic
              │                   was unsure)
              └────────────┬────────────┘
                           ▼
                  Evaluation Finding
              (both views always retained)
```

The LLM may **adjudicate ambiguous findings**. It may not **silently override
a deterministic result**. Where the two disagree in a way no documented rule
resolves, the outcome is `ADJUDICATION_REQUIRED` and both views are kept.

Two things the LLM is never asked to do:

- **Decide what is clinically critical.** Criticality comes from the
  benchmark's own rules (CR-01…CR-10). Letting a model reassign it would put
  ground truth back in the model's hands.
- **Evaluate a summary as a whole.** It answers one narrow, checkable
  question at a time: "is this one source fact conveyed by this summary?" A
  broad question invites an impressive paragraph that cannot be verified.

## 2. Provider abstraction

One interface, several backends:

```python
class ModelProvider(ABC):
    kind: ProviderKind
    is_remote: bool

    def generate(self, request: ModelRequest) -> ModelResponse: ...
```

| Provider | Backend | Notes |
|---|---|---|
| `mock` | none | Deterministic, offline. **All tests and CI use only this.** |
| `anthropic` | Anthropic SDK | Native system-prompt channel |
| `openai` | OpenAI SDK | |
| `gemini` | Google Generative AI SDK | |
| `local` | OpenAI-compatible server | Ollama, vLLM, LM Studio |

Every SDK is imported **inside** the call that needs it, so none is a hard
dependency. The application installs and runs with no provider SDK present;
using one that is not installed raises `ProviderNotConfigured` telling you what
to install, rather than failing at import time and taking the package down.

Model names come from configuration, never hard-coded — provider model
identifiers change, and a pinned name silently rots.

Adding a backend means implementing `generate` and registering it. No
evaluation code changes.

## 3. Credentials and what leaves the machine

Credentials live in the `CLINEVAL_` namespace, one per provider:

```
CLINEVAL_ANTHROPIC_API_KEY
CLINEVAL_OPENAI_API_KEY
CLINEVAL_GOOGLE_API_KEY
CLINEVAL_LOCAL_BASE_URL
```

Separate variables rather than one shared key, so several providers can be
configured at once and none can accidentally pick up another's credential.

**Nothing leaves the machine by default.** With no credential set, only the
mock provider can be constructed — `build_provider` refuses a remote provider
without one. The CLI prints a warning naming the provider and the number of
notes before sending anything.

A credential is held only on the provider instance. It never appears in
`ModelResponse`, `provider_config`, `describe()`, a run file, or a log line.
`provider_config` carries exactly `temperature`, `max_tokens`, and `base_url`,
and is written verbatim into runs.

## 4. Generation

```
BenchmarkCase ──> ClinicalSummaryRequest ──> prompt ──> provider
                                                          │
              ClinicalSummaryResponse <────────────────────┘
                          │
                     AIOutput ──> Phase 3 evaluators
```

`ClinicalSummaryResponse` records the model, model version, prompt name,
prompt version, prompt hash, temperature, latency, token usage, and
`is_real_model_output`. That last flag is `False` for every mock generation, so
**a synthetic output can never be mistaken for a real one** in a stored run.

Failure policy: a case whose generation fails is recorded with its error and
the batch continues. Failed generations are excluded from the outputs passed
to evaluation — a failed call is a harness problem, while an empty summary is
a model that produced nothing, and scoring them alike would misattribute
infrastructure failures to the model.

## 5. Prompts

Prompts live in `prompts/<name>_<version>.txt`, not in Python strings, because
a prompt is an experimental variable. A run that cannot say which prompt
produced it is not reproducible.

```
prompts/
  summarization_v1.txt   the task under evaluation
  adjudication_v1.txt    the judge
```

Each file carries YAML front matter (name, version, role) and `SYSTEM:` /
`USER:` sections. Every loaded prompt records a **content hash**, so a file
edited without a version bump is still detectable — the failure mode version
strings alone do not catch, and one that would silently invalidate every prior
result citing that version.

## 6. Structured output

The judge must return exactly this, and nothing else:

```json
{
  "judgment": "SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED" | "UNCERTAIN",
  "confidence": 0.94,
  "reason": "The output paraphrases the source fact without changing its meaning.",
  "evidence": [{"source_span": "...", "output_span": "..."}]
}
```

Validated by a Pydantic model with `extra="forbid"` — a judge inventing extra
keys has not followed the contract, which makes its other fields suspect too.

The parser tolerates *packaging* (a ```json fence, a sentence of preamble)
because that is a formatting habit. It will **not** infer a judgment from
prose: text saying "this looks supported to me" is a parse failure, not
`SUPPORTED`. Guessing there would fabricate a verdict the model never committed
to, in the one place the system is trying to measure whether the model can be
trusted.

`UNCERTAIN` exists so the judge can decline. A judge forced to choose between
SUPPORTED and CONTRADICTED will guess, and a guess recorded as a verdict is
worse than an admission of uncertainty.

Retry is bounded (2 attempts by default): a model that cannot produce the
schema twice will not produce it on the tenth, and the run should not stall
discovering that.

## 7. Deterministic-first routing

The LLM is called only where deterministic evaluation is genuinely weak. Every
call costs money, adds non-determinism, and the deterministic evaluator is
already validated at zero false positives on the oracle set.

```
deterministic finding
        │
        ├── high confidence, exact match ──> accept deterministic result
        │
        ├── weak match (partial/alias)   ──> LLM adjudication
        │
        └── low confidence               ──> LLM adjudication
```

Routed when **any** of these hold, and the finding type is eligible:

- deterministic confidence ≤ `confidence_threshold` (default 0.95)
- the match was `PARTIAL`, `ALIAS`, or `NONE`

Never routed:

| Excluded | Why |
|---|---|
| High-confidence exact matches | Nothing to adjudicate |
| `empty_output`, `malformed_output`, `missing_annotation` | Operational, not clinical questions |
| `medication_dose_error`, `value_error`, `negation_reversal`, `temporal_trend_reversal` | The answer does not turn on wording. 1.4 is not 1.9 regardless of phrasing, and asking a model to second-guess that invites it to talk the evaluator out of a correct finding |

All thresholds and the eligible-type list are configuration
(`config/evaluation.yaml`), and adjudication is **disabled by default** — an
evaluation must never start making paid external calls because a config file
was present.

## 8. Disagreement handling

Six rules, and only one of them ever removes a finding.

| Rule | Condition | Disposition | Finding |
|---|---|---|---|
| **R1** | LLM agrees (UNSUPPORTED / CONTRADICTED) | `AGREED` | kept |
| **R2** | Deterministic **omission** + LLM SUPPORTED, confident | `RESOLVED_BY_LLM` | **withdrawn** |
| **R3** | Deterministic **contradiction** + LLM SUPPORTED | `ADJUDICATION_REQUIRED` | kept |
| **R4** | LLM UNCERTAIN | `ADJUDICATION_REQUIRED` | kept |
| **R5** | LLM SUPPORTED below `min_judgment_confidence` | `ADJUDICATION_REQUIRED` | kept |
| **R6** | Call or parse failed | `FAILED` | kept |

**Why R2 and R3 differ.** R2 is the case the LLM is genuinely useful for: the
deterministic matcher could not find a fact that *was* there, expressed
differently. Withdrawing it removes a false positive. R3 is the opposite risk —
the evaluator found the summary asserting something incompatible with the
source, and a model saying "looks fine" is not grounds to erase a detected
contradiction. Contradictions carry the highest clinical cost if wrongly
dismissed, so they are never dismissed by a model, only escalated.

Withdrawn findings are **retained** in the run under
`adjudication.withdrawn_findings`, with the rule that withdrew them. An
evaluation that silently drops findings is not auditable.

**Metrics are not adjusted by adjudication.** The headline deterministic
numbers are computed before adjudication and stay that way
(`metrics_are_pre_adjudication: true`); the findings list is post-adjudication
(`findings_are_post_adjudication`). Letting a model's judgments move the
headline metrics would make them partly model-derived, which is what the
deterministic-first design exists to prevent.

## 9. Confidence is not calibration

The judge's stated confidence is stored as `model_reported_confidence` and
named that way deliberately. **It is not a calibrated probability.** It is a
number the model produced about itself, and this project has not measured
whether it corresponds to accuracy in any way.

It is used only as a threshold for R2/R5 — a conservative use, since a
miscalibrated high confidence can at worst withdraw an omission finding, never
erase a contradiction. Measuring calibration is future work.

## 10. Cost, tokens, and latency

Recorded per call where the provider reports them:

| Field | When unavailable |
|---|---|
| input / output / total tokens | `null` |
| `estimated_cost_usd` | `null` |
| `latency_ms` | `null` |

**Never fabricated.** An unpriced model, or a missing token count, yields
`None` rather than a guess — treating a missing count as zero would understate
cost, the one direction a cost estimate must not err in. The price table
(`app/providers/pricing.py`) records the date it was captured, and unknown
models return `None` rather than falling back to a default price.

Aggregates track *known-for* counts separately from totals, because "$0.00
across 10 calls" and "$0.00 across 0 priced calls" mean entirely different
things. The mock provider reports `None` cost for the same reason: reporting
0.0 would make aggregates look like real measurements.

Latency excludes cache hits, which have no model latency.

## 11. Caching

The cache key covers everything that determines the output:

```
case_id + provider + model + prompt name + prompt version + prompt hash
        + temperature + max_tokens + run_index
```

Two entries in that list are load-bearing. **Prompt hash** means editing a
prompt file without bumping its version still invalidates cached output.
**`run_index`** means deliberately repeated generations of one case do not
collapse into a single entry, which would make repeatability testing
meaningless.

Entries are JSON under `data/cache/` (gitignored runtime state). Reads never
raise: a corrupt entry is a miss, because a cache failure should cost a
regeneration, not the run. Failed generations are never cached.

## 12. Prompt-injection defense

**A clinical note is untrusted input.** It may contain, accidentally or
deliberately, text that reads as an instruction. Three layers, none sufficient
alone:

1. **Channel separation.** Evaluator instructions go in the system message,
   content in the user message. The strongest boundary available.
2. **Explicit delimiting.** Content is wrapped in `<<<X_BEGIN>>>` /
   `<<<X_END>>>`, and the prompt states that everything between them is data
   and that instruction-like text inside must never be followed.
3. **Delimiter neutralization.** Content containing those markers is escaped,
   so injected text cannot close the block early and appear to speak from
   outside it. Without this, layer 2 is trivially bypassed.

**Content is never censored.** Suspicious phrases are not stripped: "the
patient was instructed to ignore previous advice" is a legitimate clinical
sentence, and a filter that silently edits a note would break the ground-truth
guarantee the benchmark rests on. Content is *contained*, not removed.
`scan_for_injection` reports suspicious content for visibility and records it
on the response, without altering it.

Injection is tested through notes, summaries, and extracted facts. The
benchmark can generate adversarial cases:

```bash
python -m scripts.generate_benchmark --cases 10 --seed 7 \
  --difficulty ADVERSARIAL --categories prompt_injection
```

These are **opt-in**, not in any tier's default pool: they test the evaluation
harness rather than the summarizer, so including them by default would dilute
the summarization benchmark.

**Residual risk is real.** No prompt-level defense is complete. A sufficiently
capable injection may still influence a model. This is one more reason the LLM
never overrides a deterministic finding — the deterministic layer has no prompt
to inject.

## 13. Evaluating the judge

The judge is a component under test, scored against labels derived from
benchmark ground truth — never from a model.

| Scenario | Summary | Expected |
|---|---|---|
| `clearly_correct` | reference summary | SUPPORTED |
| `valid_paraphrase` | reworded, meaning preserved | SUPPORTED |
| `clearly_omitted` | the fact's sentence removed | UNSUPPORTED |
| `clearly_hallucinated` | invented diagnosis appended | UNSUPPORTED |
| `explicit_contradiction` | lab value altered | CONTRADICTED |
| `negation_reversal` | negative flipped to positive | CONTRADICTED |
| `temporal_reversal` | trend direction reversed | CONTRADICTED |
| `ambiguous` | a vague gesture at the fact | UNCERTAIN |

Reported: accuracy, per-class precision / recall / F1, macro-F1, and a
confusion matrix — plus adjudication rate, disagreement rate, and parse-failure
count.

`ambiguous` cases are excluded from accuracy by default. An accuracy figure
that includes questions with no single right answer is not a measure of
correctness; those cases exist to reveal whether a judge ever says "I cannot
tell", since one that never does is guessing somewhere.

Two properties keep the harness honest, both tested: an oracle judge scores
1.0, and a judge that always answers `SUPPORTED` scores badly. A harness that
cannot separate those two measures nothing.

**Undefined is undefined.** Precision with no predictions of a class is `None`,
not 0.0. Reporting 0.0 would make a judge that never predicts CONTRADICTED look
actively bad at it rather than silent about it.

## 14. Reproducibility

Every run records:

```
run_id, benchmark_version, case_count, model, model_version, provider,
prompt_name, prompt_version, prompt_hash, evaluator versions,
evaluation_version, temperature, max_tokens, seed, timestamp,
adjudication policy, cache statistics
```

With the mock provider, identical invocations produce identical findings and
metrics — asserted in the test suite. With a remote provider, reproducibility
is bounded by the provider's own determinism, which is why the cache exists:
re-evaluating a stored run reuses the exact generations it was based on.

## 15. Running it

```bash
# Fully offline: mock provider, no adjudication, nothing sent anywhere.
python -m scripts.evaluate_benchmark --cases 10

# Exercise a specific failure mode.
python -m scripts.evaluate_benchmark --cases 10 --behaviour negation_reversal

# The ambiguity adjudication exists for, with a mock judge.
python -m scripts.evaluate_benchmark --cases 10 --behaviour abstractive --adjudicate

# A real provider (requires CLINEVAL_ANTHROPIC_API_KEY).
python -m scripts.evaluate_benchmark --cases 10 \
  --provider anthropic --model <model-name> \
  --adjudicate --judge-provider anthropic --judge-model <model-name>
```

Mock behaviours: `correct`, `paraphrase`, `abstractive`, `empty`,
`verbatim_note`, `omission`, `hallucination`, `contradiction`,
`medication_error`, `dose_error`, `negation_reversal`, `temporal_reversal`,
`historical_confusion`.

The defect behaviours are built by running the Phase 3 injection harness over
the benchmark's own reference summaries, so a mock "hallucinating model"
produces exactly the corruption the evaluator was validated against.

## 16. Limitations and the risks of LLM-as-judge

**On the judge itself:**

- **An LLM judge is fallible in correlated ways.** It may share blind spots
  with the model it judges, particularly if both come from the same family.
  This is the central risk of LLM-as-judge and is not mitigated here beyond
  keeping the judge subordinate to deterministic evaluation.
- **Stated confidence is not calibrated.** See §9.
- **Judge results are measured on synthetic text only.** They do not
  generalize to real clinical documentation.
- **No judge has been evaluated against a real model's output.** Everything
  reported so far uses the mock provider.
- **Sycophancy is untested.** The prompt shows the deterministic result for
  context, which risks anchoring the judge to it. That the judge *disagrees*
  with the deterministic result in the abstractive scenario is weak evidence
  against strong anchoring, but it has not been measured properly.

**On this implementation:**

- Adjudication is per-finding, so a case with many findings makes many calls.
  `max_calls_per_run` caps this; a batched prompt would be cheaper and is not
  implemented.
- The eligible-type list is a judgment call. Excluding value and dose errors
  is defensible but means a genuinely wrong deterministic value finding cannot
  be corrected by adjudication.
- The reconciliation rules are conservative by design: only R2 removes a
  finding. This biases toward reporting too much rather than dropping a real
  clinical error, which trades precision for safety.
- Prompt-injection defense is containment, not prevention (§12).

**On the whole system:**

- No real patient data is used, ever. The benchmark is synthetic.
- No overall risk score, deployment classification, or clinical safety score is
  produced. That comes after the evaluation stack is validated, not before.
- **LLM-based evaluation is not a substitute for clinical validation.** Nothing
  here establishes that any model is safe for clinical use.
