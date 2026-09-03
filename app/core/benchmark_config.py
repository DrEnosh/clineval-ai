"""Benchmark generation configuration.

Case count, seed, difficulty mix, and the category pool for each tier are
configuration, not constants in the generator. Defaults live here so the
package runs with no config file present; `config/benchmark.yaml` overrides
them when it exists, and CLI flags override that.

Precedence: CLI flags > config file > these defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from app.core.config import BASE_DIR
from app.core.enums import DifficultyTier, FailureModeCategory

DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "benchmark.yaml"

# The dataset version stamped into generated benchmarks. Distinct from the
# curated Phase 1 seed set's version (app.benchmark.curated.BENCHMARK_VERSION),
# because the two are different artifacts with different provenance.
SYNTHETIC_BENCHMARK_VERSION = "clineval-benchmark-v0.1"


class TierConfig(BaseModel):
    """How one difficulty tier builds a case.

    `guaranteed` categories always apply; `feature_count` more are drawn from
    `pool` without replacement.
    """

    feature_count: tuple[int, int]
    pool: list[FailureModeCategory]
    guaranteed: list[FailureModeCategory] = Field(default_factory=list)

    @field_validator("feature_count")
    @classmethod
    def _valid_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        low, high = value
        if low < 1 or high < low:
            raise ValueError(f"feature_count must be a valid ascending range, got {value}")
        return value


class BenchmarkConfig(BaseModel):
    version: str = SYNTHETIC_BENCHMARK_VERSION
    default_case_count: int = 100
    default_seed: int = 42
    difficulty_distribution: dict[DifficultyTier, float] = Field(
        default_factory=lambda: {
            DifficultyTier.EASY: 0.20,
            DifficultyTier.MODERATE: 0.35,
            DifficultyTier.HARD: 0.30,
            DifficultyTier.ADVERSARIAL: 0.15,
        }
    )
    tiers: dict[DifficultyTier, TierConfig] = Field(default_factory=lambda: _default_tiers())

    @field_validator("difficulty_distribution")
    @classmethod
    def _weights_positive(cls, value: dict[DifficultyTier, float]) -> dict[DifficultyTier, float]:
        if not value:
            raise ValueError("difficulty_distribution must not be empty")
        if any(weight < 0 for weight in value.values()):
            raise ValueError("difficulty_distribution weights must be non-negative")
        if sum(value.values()) <= 0:
            raise ValueError("difficulty_distribution weights must sum to more than zero")
        return value

    def tier_for(self, difficulty: DifficultyTier) -> TierConfig:
        if difficulty not in self.tiers:
            raise KeyError(f"No tier configuration for difficulty {difficulty.value}")
        return self.tiers[difficulty]

    def allocate_difficulties(self, case_count: int) -> list[DifficultyTier]:
        """Deterministically allocate `case_count` cases across tiers.

        Uses largest-remainder allocation rather than random sampling, so the
        realized mix matches the configured mix exactly instead of only in
        expectation - a 100-case run configured for 20% EASY yields exactly 20
        easy cases, every time.
        """
        tiers = list(self.difficulty_distribution)
        total_weight = sum(self.difficulty_distribution.values())
        exact = {t: case_count * self.difficulty_distribution[t] / total_weight for t in tiers}
        counts = {t: int(exact[t]) for t in tiers}

        remainder = case_count - sum(counts.values())
        by_fraction = sorted(tiers, key=lambda t: (-(exact[t] - int(exact[t])), t.value))
        for index in range(remainder):
            counts[by_fraction[index % len(by_fraction)]] += 1

        allocation: list[DifficultyTier] = []
        for tier in tiers:
            allocation.extend([tier] * counts[tier])
        return allocation


def _default_tiers() -> dict[DifficultyTier, TierConfig]:
    C = FailureModeCategory
    return {
        DifficultyTier.EASY: TierConfig(
            feature_count=(1, 2),
            pool=[C.STRAIGHTFORWARD, C.SHORT_NOTE, C.RELEVANT_NEGATIVE_FINDINGS, C.PROCEDURES_AND_FOLLOWUP],
        ),
        DifficultyTier.MODERATE: TierConfig(
            feature_count=(2, 4),
            pool=[
                C.ABNORMAL_LABS, C.ABNORMAL_VITALS, C.MEDICATION_CHANGE, C.TEMPORAL_PROGRESSION,
                C.NEGATION, C.RELEVANT_NEGATIVE_FINDINGS, C.PROCEDURES_AND_FOLLOWUP,
                C.MULTIPLE_COMORBIDITIES,
            ],
        ),
        DifficultyTier.HARD: TierConfig(
            feature_count=(4, 6),
            guaranteed=[C.HIGH_COMPLEXITY],
            pool=[
                C.MULTIPLE_COMORBIDITIES, C.ABNORMAL_LABS, C.ABNORMAL_VITALS,
                C.TEMPORAL_PROGRESSION, C.MEDICATION_CHANGE, C.MEDICATION_DOSE_CHANGE,
                C.MEDICATION_DISCONTINUATION, C.NEGATION, C.LONG_NOTE,
                C.PROCEDURES_AND_FOLLOWUP, C.HISTORICAL_VS_CURRENT,
            ],
        ),
        DifficultyTier.ADVERSARIAL: TierConfig(
            feature_count=(4, 7),
            guaranteed=[C.COMBINED_CRITICAL],
            pool=[
                C.CRITICAL_OMISSION, C.HALLUCINATION_OPPORTUNITY, C.NEGATION,
                C.CONFLICTING_INFORMATION, C.HISTORICAL_VS_CURRENT, C.DUPLICATE_DOCUMENTATION,
                C.MEDICATION_DISCONTINUATION, C.MEDICATION_DOSE_CHANGE, C.TEMPORAL_PROGRESSION,
                C.LONG_NOTE, C.ABNORMAL_LABS,
            ],
        ),
    }


def load_benchmark_config(path: Path | str | None = None) -> BenchmarkConfig:
    """Load benchmark configuration, falling back to built-in defaults.

    A missing config file is not an error - the defaults are a complete,
    working configuration. A malformed one is an error, because silently
    generating a benchmark with a different mix than the operator intended
    would undermine reproducibility.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return BenchmarkConfig()

    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        return BenchmarkConfig()
    if not isinstance(raw, dict):
        raise ValueError(f"Benchmark config at {config_path} must be a mapping, got {type(raw).__name__}")

    section = raw.get("benchmark", raw)
    return BenchmarkConfig.model_validate(section)
