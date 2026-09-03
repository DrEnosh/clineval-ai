"""Tests for benchmark generation configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.benchmark_config import (
    DEFAULT_CONFIG_PATH,
    SYNTHETIC_BENCHMARK_VERSION,
    BenchmarkConfig,
    TierConfig,
    load_benchmark_config,
)
from app.core.enums import DifficultyTier, FailureModeCategory


class TestDefaults:
    def test_defaults_are_a_complete_working_configuration(self):
        config = BenchmarkConfig()
        assert config.version == SYNTHETIC_BENCHMARK_VERSION
        assert set(config.tiers) == set(DifficultyTier)
        assert sum(config.difficulty_distribution.values()) == pytest.approx(1.0)

    def test_every_tier_has_a_non_empty_pool(self):
        for tier, tier_config in BenchmarkConfig().tiers.items():
            assert tier_config.pool, tier

    def test_every_category_is_reachable_from_some_tier(self):
        config = BenchmarkConfig()
        reachable: set[FailureModeCategory] = set()
        for tier_config in config.tiers.values():
            reachable |= set(tier_config.pool) | set(tier_config.guaranteed)
        missing = set(FailureModeCategory) - reachable
        assert not missing, f"unreachable categories: {sorted(c.value for c in missing)}"

    def test_shipped_config_file_parses(self):
        assert DEFAULT_CONFIG_PATH.exists(), "config/benchmark.yaml should ship with the repo"
        config = load_benchmark_config(DEFAULT_CONFIG_PATH)
        assert set(config.tiers) == set(DifficultyTier)

    def test_shipped_config_matches_the_built_in_defaults(self):
        """The YAML is the defaults made explicit; drift between them would
        mean the docs describe one thing and the code does another."""
        assert load_benchmark_config(DEFAULT_CONFIG_PATH) == BenchmarkConfig()


class TestValidation:
    def test_rejects_empty_difficulty_distribution(self):
        with pytest.raises(ValidationError):
            BenchmarkConfig(difficulty_distribution={})

    def test_rejects_negative_weights(self):
        with pytest.raises(ValidationError):
            BenchmarkConfig(difficulty_distribution={DifficultyTier.EASY: -1.0})

    def test_rejects_all_zero_weights(self):
        with pytest.raises(ValidationError):
            BenchmarkConfig(
                difficulty_distribution={DifficultyTier.EASY: 0.0, DifficultyTier.HARD: 0.0}
            )

    def test_rejects_invalid_feature_count_range(self):
        with pytest.raises(ValidationError):
            TierConfig(feature_count=(5, 2), pool=[FailureModeCategory.NEGATION])
        with pytest.raises(ValidationError):
            TierConfig(feature_count=(0, 3), pool=[FailureModeCategory.NEGATION])

    def test_unknown_tier_lookup_is_an_explicit_error(self):
        config = BenchmarkConfig(tiers={})
        with pytest.raises(KeyError, match="No tier configuration"):
            config.tier_for(DifficultyTier.HARD)


class TestAllocation:
    def test_allocation_totals_the_requested_case_count(self):
        config = BenchmarkConfig()
        for count in (1, 7, 13, 100, 250):
            assert len(config.allocate_difficulties(count)) == count

    def test_allocation_matches_the_configured_mix_exactly(self):
        allocation = BenchmarkConfig().allocate_difficulties(100)
        counts = {tier: allocation.count(tier) for tier in set(allocation)}
        assert counts[DifficultyTier.EASY] == 20
        assert counts[DifficultyTier.MODERATE] == 35
        assert counts[DifficultyTier.HARD] == 30
        assert counts[DifficultyTier.ADVERSARIAL] == 15

    def test_allocation_is_deterministic(self):
        config = BenchmarkConfig()
        assert config.allocate_difficulties(37) == config.allocate_difficulties(37)

    def test_remainder_is_distributed_not_dropped(self):
        """A count that does not divide evenly must still allocate every case."""
        allocation = BenchmarkConfig().allocate_difficulties(7)
        assert len(allocation) == 7

    def test_single_tier_configuration_allocates_everything_to_it(self):
        config = BenchmarkConfig(difficulty_distribution={DifficultyTier.ADVERSARIAL: 1.0})
        assert config.allocate_difficulties(5) == [DifficultyTier.ADVERSARIAL] * 5


class TestLoading:
    def test_missing_file_falls_back_to_defaults(self, tmp_path):
        assert load_benchmark_config(tmp_path / "absent.yaml") == BenchmarkConfig()

    def test_empty_file_falls_back_to_defaults(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_benchmark_config(path) == BenchmarkConfig()

    def test_malformed_file_raises_rather_than_silently_defaulting(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_benchmark_config(path)

    def test_invalid_values_raise(self, tmp_path):
        path = tmp_path / "invalid.yaml"
        path.write_text("benchmark:\n  difficulty_distribution:\n    EASY: -3\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            load_benchmark_config(path)

    def test_file_overrides_defaults(self, tmp_path):
        path = tmp_path / "custom.yaml"
        path.write_text(
            "benchmark:\n"
            "  version: custom-v9\n"
            "  default_case_count: 5\n"
            "  difficulty_distribution:\n"
            "    HARD: 1.0\n",
            encoding="utf-8",
        )
        config = load_benchmark_config(path)
        assert config.version == "custom-v9"
        assert config.default_case_count == 5
        assert config.allocate_difficulties(3) == [DifficultyTier.HARD] * 3

    def test_accepts_a_config_without_the_benchmark_wrapper_key(self, tmp_path):
        path = tmp_path / "flat.yaml"
        path.write_text("version: flat-v1\n", encoding="utf-8")
        assert load_benchmark_config(path).version == "flat-v1"
