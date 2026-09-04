# Changelog

Methodology changes are recorded here with the observation that prompted them,
because a change to how the evaluator works changes what past results mean.

Evaluator versions are stamped on every finding
(`EvaluationFinding.evaluator_version`), so a stored result stays interpretable
after the logic that produced it has moved on.

---

## Evaluation 0.5.0 — Phase 5 methodology validation

All five deterministic evaluators move to `0.5.0`, and `EVALUATION_VERSION` with
them. They share the support and normalization engine that changed, so all five
can behave differently even where only three were directly implicated — bumping
only the "obviously" affected ones would understate the blast radius.

Findings below came from the methodology challenge suite
(`app/experiments/challenges.py`), which evaluates summaries written in
realistic clinical registers rather than the benchmark's own phrasing. It found
**19 failures across 59 decidable challenges** on first run.

### P5-01 — Numeric values were compared as strings

**Observed.** A summary writing `12.50 mg` where the case says `12.5 mg` was
reported as stating a fabricated value, and a trend written `1.40 → 1.90` was
reported as information loss. Notation changed the verdict; the value did not.

**Changed.**
- `app/evaluation/support.py` — trend endpoints are matched by **numeric
  value**, not by string token. `_standalone_positions` now parses each numeric
  token and compares floats.
- `app/evaluation/evaluators/fact_support.py` — the unsupported-value check
  compares numerically against the values in the structured case.

**Regression tests.** `TestNumericEquivalence` in
`tests/evaluation/test_phase5_regressions.py`, including that a genuinely
invented value (`7.77`) is still caught, so the fix did not blunt hallucination
detection.

### P5-02 — Medication status cues were too narrow for clinical register

**Observed.** "has been held", "is no longer being taken", and "dose was
stepped up" all went undetected, so summaries that stated the medication status
perfectly clearly were flagged as status errors.

**Changed.** `app/evaluation/normalization.py` — `MEDICATION_STATUS_CUES`
widened for all five statuses. Shorter general cues (`no longer`, `step up`)
now cover families of phrasings rather than enumerating each one.

**Regression tests.** `TestMedicationRegister`, covering seven discontinuation
phrasings and four dose-change phrasings, plus a test that a stopped drug
reported as current is *still* an error.

### P5-03 — A defect in the challenge, not the evaluator

**Observed.** The compression challenge dropped current medications and
counted the resulting `medication_omission` findings as false positives.

**Assessed.** The evaluator was right. Removing the antibiotic from a
cellulitis summary is real information loss, not harmless concision. The
challenge encoded a wrong expectation.

**Changed.** `app/experiments/challenges.py` — compression preserves the full
medication list. No evaluator change.

**Regression test.** `TestChallengeConstruction`, recorded so the suite
remembers a time the system was right and the test was wrong.

### P5-04 — A truncating regex produced malformed text

**Observed.** The pattern `[^.]*\.` stopped at the decimal point in `4.1`,
truncating mid-sentence and leaving text like `"potassium is now deranged.1 to
5.7 mEq/L"`.

**Assessed.** Present in both the challenge suite and — introduced in Phase 4 —
`app/providers/mock.py`. The mock `abstractive` behaviour had therefore been
producing malformed summaries, so the Phase 4 adjudication demonstration ran
against slightly mangled text. It did not affect any deterministic evaluator or
any committed benchmark artifact.

**Changed.** Both sites now match to a genuine sentence end.

**Regression test.** `test_abstraction_regex_does_not_truncate_at_a_decimal_point`,
asserting against both call sites.

### Result

All 56 decidable challenges pass. The `abstraction` category (7 challenges)
remains **undecided by design**: whether "renal function has worsened"
adequately conveys "creatinine rose from 1.4 to 1.9 mg/dL" is a clinical
judgment, and encoding a contested opinion as a passing test would launder it
into a fact. The evaluator currently treats abstraction as information loss;
that behaviour is recorded, not asserted to be correct.

---

## Benchmark — Phase 5 additions

- `FailureModeCategory.PROMPT_INJECTION` added in Phase 4 remains **opt-in**
  and absent from every tier's default pool. The committed
  `clineval-benchmark-v0.1` artifact is unchanged.

---

## Evaluation 0.3.0 — Phase 3

Initial deterministic evaluation engine: five evaluators, configurable severity
policy, and self-validation by controlled failure injection (0 false positives,
9/9 injections detected). Phase 3 fixes are documented in the commit history
and covered by `tests/evaluation/test_regressions.py`.
