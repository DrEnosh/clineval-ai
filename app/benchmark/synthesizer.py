"""The benchmark generation pipeline.

Runs the ground-truth-first flow for each case:

    seed -> StructuredCase -> timeline
                           -> source note          (note_renderer)
                           -> reference summary    (summary_renderer)
                           -> annotations          (annotator)

Determinism: every case is seeded with the string "<seed>:<index>", so a case's
*content* depends only on the master seed, its index, and its difficulty tier.
Re-running with the same seed, case count, and configuration reproduces the
benchmark byte for byte.

The guarantee is scoped to the full triple, not the seed alone. Difficulty
tiers are allocated by largest remainder across the whole run (see
`BenchmarkConfig.allocate_difficulties`), so changing `case_count` reassigns
tiers and therefore changes cases - asking for 200 cases does *not* reproduce
the first 100 of a 100-case run. That is the cost of an exactly-matching
difficulty mix, which is worth more here than prefix stability: a benchmark
advertising 20% adversarial cases should contain exactly that, not 20% in
expectation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any

from app.benchmark.annotator import annotate
from app.benchmark.case_builder import GENERATOR_VERSION, CaseDraft, build_base_case
from app.benchmark.features import FEATURE_BUILDERS
from app.benchmark.note_renderer import render_note
from app.benchmark.summary_renderer import render_reference_summary
from app.core.benchmark_config import BenchmarkConfig, load_benchmark_config
from app.core.enums import (
    CaseType,
    DataProvenance,
    DifficultyTier,
    FailureModeCategory,
)
from app.models.schemas import BenchmarkCase
from app.models.structured import StructuredCase

# Which curated Phase 1 CaseType best describes a generated case. A generated
# case carries a list of categories; this picks the single most consequential
# one for continuity with the Phase 1 typology. First match wins, so the order
# encodes precedence.
_DOMINANT_TYPE_ORDER: tuple[tuple[FailureModeCategory, CaseType], ...] = (
    (FailureModeCategory.CONFLICTING_INFORMATION, CaseType.CONFLICTING_INFORMATION),
    (FailureModeCategory.CRITICAL_OMISSION, CaseType.CRITICAL_OMISSION),
    (FailureModeCategory.HALLUCINATION_OPPORTUNITY, CaseType.HALLUCINATION),
    (FailureModeCategory.COMBINED_CRITICAL, CaseType.HIGH_COMPLEXITY),
    (FailureModeCategory.HIGH_COMPLEXITY, CaseType.HIGH_COMPLEXITY),
    (FailureModeCategory.NEGATION, CaseType.NEGATION),
    (FailureModeCategory.MEDICATION_DISCONTINUATION, CaseType.MEDICATION),
    (FailureModeCategory.MEDICATION_DOSE_CHANGE, CaseType.MEDICATION),
    (FailureModeCategory.MEDICATION_CHANGE, CaseType.MEDICATION),
    (FailureModeCategory.TEMPORAL_PROGRESSION, CaseType.TEMPORAL),
    (FailureModeCategory.HISTORICAL_VS_CURRENT, CaseType.TEMPORAL),
    (FailureModeCategory.MULTIPLE_COMORBIDITIES, CaseType.MULTI_MORBIDITY),
)

# Feature application order. Features are applied in this fixed order rather
# than the order they were drawn, so a case's content does not depend on
# shuffle order - structural features (comorbidities) land before the features
# that read the case's conditions (negation traps, medication changes).
_FEATURE_ORDER: tuple[FailureModeCategory, ...] = (
    FailureModeCategory.MULTIPLE_COMORBIDITIES,
    FailureModeCategory.HISTORICAL_VS_CURRENT,
    FailureModeCategory.STRAIGHTFORWARD,
    FailureModeCategory.ABNORMAL_LABS,
    FailureModeCategory.ABNORMAL_VITALS,
    FailureModeCategory.TEMPORAL_PROGRESSION,
    FailureModeCategory.HIGH_COMPLEXITY,
    FailureModeCategory.CRITICAL_OMISSION,
    FailureModeCategory.MEDICATION_CHANGE,
    FailureModeCategory.MEDICATION_DOSE_CHANGE,
    FailureModeCategory.MEDICATION_DISCONTINUATION,
    FailureModeCategory.NEGATION,
    FailureModeCategory.COMBINED_CRITICAL,
    FailureModeCategory.RELEVANT_NEGATIVE_FINDINGS,
    FailureModeCategory.HALLUCINATION_OPPORTUNITY,
    FailureModeCategory.CONFLICTING_INFORMATION,
    FailureModeCategory.PROCEDURES_AND_FOLLOWUP,
    FailureModeCategory.DUPLICATE_DOCUMENTATION,
    FailureModeCategory.LONG_NOTE,
    FailureModeCategory.SHORT_NOTE,
)


def case_seed(seed: int | str, index: int) -> str:
    return f"{seed}:{index}"


def select_categories(
    difficulty: DifficultyTier, config: BenchmarkConfig, rng: Random
) -> list[FailureModeCategory]:
    """Choose this case's failure-mode categories from its tier's pool."""
    tier = config.tier_for(difficulty)
    categories = list(tier.guaranteed)

    low, high = tier.feature_count
    wanted = rng.randint(low, high)
    available = [c for c in tier.pool if c not in categories]
    chosen = rng.sample(available, k=min(wanted, len(available)))
    categories.extend(chosen)

    return [c for c in _FEATURE_ORDER if c in set(categories)]


def build_structured_case(
    case_id: str, seed: str, difficulty: DifficultyTier, config: BenchmarkConfig
) -> StructuredCase:
    """Build one structured case - the source of truth for everything else."""
    rng = Random(seed)
    draft = CaseDraft(case_id=case_id, seed=seed, rng=rng, difficulty=difficulty)
    build_base_case(draft, rng)

    for category in select_categories(difficulty, config, rng):
        draft.add_category(category)
        FEATURE_BUILDERS[category](draft, rng)

    return draft.build()


def _dominant_case_type(categories: list[FailureModeCategory]) -> CaseType:
    present = set(categories)
    for category, case_type in _DOMINANT_TYPE_ORDER:
        if category in present:
            return case_type
    if FailureModeCategory.SHORT_NOTE in present or FailureModeCategory.LONG_NOTE in present:
        return CaseType.NORMAL
    return CaseType.NORMAL


def to_benchmark_case(structured: StructuredCase, benchmark_version: str) -> BenchmarkCase:
    """Render a structured case into the `BenchmarkCase` the rest of the
    platform consumes, attaching the structured source for traceability."""
    rng = Random(f"{structured.seed}:render")
    note = render_note(structured, rng)
    summary = render_reference_summary(structured)
    annotations = annotate(structured)

    return BenchmarkCase(
        case_id=structured.case_id,
        case_type=_dominant_case_type(structured.categories),
        benchmark_version=benchmark_version,
        provenance=DataProvenance.SYNTHETIC_TEMPLATE,
        age_group=structured.patient.age_group,
        sex=structured.patient.sex,
        encounter_type=structured.encounter.encounter_type,
        conditions=[c.name for c in structured.conditions],
        medications=[m.describe() for m in structured.medications],
        allergies=[a.describe() for a in structured.allergies],
        procedures=[p.name for p in structured.procedures if p.performed],
        observations=[f.description for f in structured.findings if f.present],
        laboratory_values=[f"{lab.describe()} ({lab.timing_description})" for lab in structured.labs],
        vital_signs=[v.describe() for v in structured.vitals],
        clinical_timeline=[e.description for e in structured.timeline],
        diagnoses=[d.name for d in structured.diagnoses],
        clinical_findings=[f.description for f in structured.findings if f.present],
        relevant_negative_findings=[f.description for f in structured.relevant_negatives],
        follow_up=[
            f"{', '.join(p.actions) or 'clinical reassessment'}; follow up with {p.provider} in {p.interval}"
            for p in structured.follow_up
        ],
        clinical_note=note,
        reference_summary=summary,
        critical_facts=annotations.critical_facts,
        non_critical_facts=annotations.non_critical_facts,
        medication_facts=annotations.medication_facts,
        temporal_facts=annotations.temporal_facts,
        negations=annotations.negations,
        contradiction_facts=annotations.contradiction_facts,
        expected_risk_flags=annotations.risk_flags,
        structured_case=structured,
        difficulty=structured.difficulty,
        categories=structured.categories,
        expected_failure_opportunities=structured.failure_opportunities,
        generation_metadata={
            "seed": structured.seed,
            "generator_version": structured.generator_version,
        },
        subgroup={
            "age_group": structured.patient.age_group,
            "sex": structured.patient.sex,
            "encounter_setting": structured.encounter.setting,
            "difficulty": structured.difficulty.value,
        },
    )


def generate_benchmark(
    case_count: int | None = None,
    seed: int | str | None = None,
    config: BenchmarkConfig | None = None,
    difficulty: DifficultyTier | None = None,
    categories: list[FailureModeCategory] | None = None,
) -> list[BenchmarkCase]:
    """Generate a benchmark of `case_count` cases.

    `difficulty` forces every case to one tier; `categories` restricts the
    draw pool to a subset. Both are for targeted regeneration (e.g. "give me
    40 adversarial medication cases") and neither is used by the default run.
    """
    config = config or load_benchmark_config()
    case_count = case_count if case_count is not None else config.default_case_count
    seed = seed if seed is not None else config.default_seed

    if case_count < 1:
        raise ValueError(f"case_count must be at least 1, got {case_count}")

    effective_config = config
    if categories:
        allowed = set(categories)
        trimmed: dict[DifficultyTier, Any] = {}
        for tier_name, tier in config.tiers.items():
            pool = [c for c in tier.pool if c in allowed] or list(allowed)
            guaranteed = [c for c in tier.guaranteed if c in allowed]
            trimmed[tier_name] = tier.model_copy(update={"pool": pool, "guaranteed": guaranteed})
        effective_config = config.model_copy(update={"tiers": trimmed})

    if difficulty is not None:
        allocation = [difficulty] * case_count
    else:
        allocation = effective_config.allocate_difficulties(case_count)

    cases: list[BenchmarkCase] = []
    for index, tier in enumerate(allocation):
        case_id = f"SYN_{index + 1:04d}"
        structured = build_structured_case(
            case_id, case_seed(seed, index), tier, effective_config
        )
        cases.append(to_benchmark_case(structured, effective_config.version))
    return cases


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def build_metadata(
    cases: list[BenchmarkCase], seed: int | str, config: BenchmarkConfig
) -> dict[str, Any]:
    """Descriptive metadata for a generated benchmark.

    Everything needed to answer "what is in this benchmark?" and "how do I
    regenerate it?" without parsing the cases themselves.
    """
    by_difficulty: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_case_type: dict[str, int] = {}

    critical_facts = medication_facts = temporal_facts = negation_facts = 0
    contradiction_facts = risk_flags = opportunities = 0
    cases_with_negation = cases_with_medication_change = 0
    cases_with_temporal = cases_with_critical = cases_with_opportunity = 0
    note_lengths: list[int] = []
    summary_lengths: list[int] = []
    total_facts = 0

    for case in cases:
        if case.difficulty:
            by_difficulty[case.difficulty.value] = by_difficulty.get(case.difficulty.value, 0) + 1
        by_case_type[case.case_type.value] = by_case_type.get(case.case_type.value, 0) + 1
        for category in case.categories:
            by_category[category.value] = by_category.get(category.value, 0) + 1

        critical_facts += len(case.critical_facts)
        medication_facts += len(case.medication_facts)
        temporal_facts += len(case.temporal_facts)
        negation_facts += len(case.negations)
        contradiction_facts += len(case.contradiction_facts)
        risk_flags += len(case.expected_risk_flags)
        opportunities += len(case.expected_failure_opportunities)

        total_facts += (
            len(case.critical_facts) + len(case.non_critical_facts)
            + len(case.medication_facts) + len(case.temporal_facts) + len(case.negations)
        )

        if case.negations:
            cases_with_negation += 1
        if any(m.status.value in {"new", "discontinued", "changed_dose"} for m in case.medication_facts):
            cases_with_medication_change += 1
        if case.temporal_facts:
            cases_with_temporal += 1
        if case.critical_facts:
            cases_with_critical += 1
        if case.expected_failure_opportunities:
            cases_with_opportunity += 1

        note_lengths.append(len(case.clinical_note))
        summary_lengths.append(len(case.reference_summary))

    count = len(cases)
    return {
        "benchmark_version": config.version,
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "case_count": count,
        "provenance": DataProvenance.SYNTHETIC_TEMPLATE.value,
        "contains_real_patient_data": False,
        "distribution": {
            "by_difficulty": dict(sorted(by_difficulty.items())),
            "by_category": dict(sorted(by_category.items())),
            "by_dominant_case_type": dict(sorted(by_case_type.items())),
        },
        "counts": {
            "critical_facts": critical_facts,
            "medication_facts": medication_facts,
            "temporal_facts": temporal_facts,
            "negation_facts": negation_facts,
            "contradiction_facts": contradiction_facts,
            "expected_risk_flags": risk_flags,
            "failure_opportunities": opportunities,
            "total_annotated_facts": total_facts,
        },
        "cases_containing": {
            "negation": cases_with_negation,
            "medication_change": cases_with_medication_change,
            "temporal_trend": cases_with_temporal,
            "critical_facts": cases_with_critical,
            "deliberate_failure_opportunity": cases_with_opportunity,
            "high_complexity": by_category.get(FailureModeCategory.HIGH_COMPLEXITY.value, 0),
        },
        "averages": {
            "note_characters": round(sum(note_lengths) / count, 1) if count else 0.0,
            "summary_characters": round(sum(summary_lengths) / count, 1) if count else 0.0,
            "facts_per_case": round(total_facts / count, 2) if count else 0.0,
        },
    }


def benchmark_output_dir(base_dir: Path, version: str) -> Path:
    return Path(base_dir) / version


def save_generated_benchmark(
    cases: list[BenchmarkCase],
    metadata: dict[str, Any],
    base_dir: Path,
    force: bool = False,
) -> tuple[Path, Path]:
    """Write cases and metadata under `base_dir/<version>/`.

    Refuses to overwrite an existing benchmark unless `force` is set: a
    published benchmark version is a fixed artifact that results are reported
    against, and silently replacing one would invalidate any evaluation that
    cited it.
    """
    version = metadata["benchmark_version"]
    seed = metadata["seed"]
    count = metadata["case_count"]

    out_dir = benchmark_output_dir(base_dir, version)
    cases_path = out_dir / f"cases_seed{seed}_n{count}.json"
    metadata_path = out_dir / f"metadata_seed{seed}_n{count}.json"

    existing = [p for p in (cases_path, metadata_path) if p.exists()]
    if existing and not force:
        raise FileExistsError(
            f"Benchmark already exists at {existing[0]}. Pass force=True (or --force on the "
            f"CLI) to overwrite, or bump the benchmark version in config/benchmark.yaml to "
            f"publish a new one alongside it."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(
        json.dumps([case.model_dump(mode="json") for case in cases], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return cases_path, metadata_path


def load_generated_benchmark(path: Path | str) -> list[BenchmarkCase]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}. Run "
            f"`python -m scripts.generate_benchmark --cases 100 --seed 42` to generate one."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkCase.model_validate(item) for item in raw]
