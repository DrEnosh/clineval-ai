"""Tests for the severity policy and the metric registry.

Severity and metrics are the two places where a number acquires meaning, so
both are tested for the properties that keep those numbers honest: severity is
configurable and applied outside the evaluators, and a metric with no
denominator is undefined rather than flattering.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.enums import FactImportance, FindingType, Severity
from app.evaluation import metrics
from app.evaluation.severity import (
    DEFAULT_CONFIG_PATH,
    SeverityPolicy,
    escalate,
    load_severity_policy,
)
from app.models.schemas import MetricValue


class TestSeverityLadder:
    def test_escalate_moves_one_step(self):
        assert escalate(Severity.LOW) is Severity.MODERATE
        assert escalate(Severity.MODERATE) is Severity.HIGH

    def test_escalate_saturates_at_critical(self):
        assert escalate(Severity.CRITICAL) is Severity.CRITICAL
        assert escalate(Severity.HIGH, steps=5) is Severity.CRITICAL

    def test_escalate_multiple_steps(self):
        assert escalate(Severity.INFO, steps=2) is Severity.MODERATE


class TestSeverityPolicy:
    @pytest.fixture
    def policy(self) -> SeverityPolicy:
        return SeverityPolicy()

    def test_every_finding_type_has_a_severity(self, policy):
        for finding_type in FindingType:
            assert isinstance(policy.assign(finding_type), Severity)

    def test_negation_reversal_is_high_before_escalation(self, policy):
        """Asserting a condition the source denies fabricates a diagnosis."""
        assert policy.assign(FindingType.NEGATION_REVERSAL) is Severity.HIGH

    def test_discontinued_drug_error_is_high(self, policy):
        assert policy.assign(FindingType.MEDICATION_STATUS_ERROR) is Severity.HIGH

    def test_value_approximation_is_informational(self, policy):
        """Rounding is not a clinical error."""
        assert policy.assign(FindingType.VALUE_APPROXIMATION) is Severity.INFO

    def test_empty_output_is_critical(self, policy):
        assert policy.assign(FindingType.EMPTY_OUTPUT) is Severity.CRITICAL

    def test_critical_fact_escalates(self, policy):
        base = policy.assign(FindingType.NON_CRITICAL_OMISSION)
        escalated = policy.assign(
            FindingType.NON_CRITICAL_OMISSION, fact_importance=FactImportance.CRITICAL
        )
        assert escalated is escalate(base)

    def test_approximation_does_not_escalate_on_critical_facts(self, policy):
        """A rounded value stays informational however important the fact."""
        assert (
            policy.assign(
                FindingType.VALUE_APPROXIMATION, fact_importance=FactImportance.CRITICAL
            )
            is Severity.INFO
        )

    def test_critical_value_escalates(self, policy):
        base = policy.assign(FindingType.CRITICAL_OMISSION)
        escalated = policy.assign(FindingType.CRITICAL_OMISSION, is_critical_value=True)
        assert escalated is escalate(base)

    def test_escalation_can_be_disabled(self):
        policy = SeverityPolicy(escalate_for_critical_facts=False)
        assert (
            policy.assign(
                FindingType.NON_CRITICAL_OMISSION, fact_importance=FactImportance.CRITICAL
            )
            is policy.assign(FindingType.NON_CRITICAL_OMISSION)
        )

    def test_overriding_a_base_severity(self):
        policy = SeverityPolicy(
            base_severities={**SeverityPolicy().base_severities,
                             FindingType.CRITICAL_OMISSION: Severity.LOW}
        )
        assert policy.assign(FindingType.CRITICAL_OMISSION) is Severity.LOW

    def test_unlisted_finding_type_falls_back_to_default(self):
        policy = SeverityPolicy(base_severities={}, default_severity=Severity.MODERATE)
        assert policy.assign(FindingType.OTHER) is Severity.MODERATE


class TestSeverityConfigLoading:
    def test_shipped_config_parses(self):
        assert DEFAULT_CONFIG_PATH.exists(), "config/evaluation.yaml should ship with the repo"
        assert isinstance(load_severity_policy(DEFAULT_CONFIG_PATH), SeverityPolicy)

    def test_shipped_config_matches_built_in_defaults(self):
        """The YAML is the defaults made explicit; drift would mean the docs
        describe one policy and the code applies another."""
        assert load_severity_policy(DEFAULT_CONFIG_PATH) == SeverityPolicy()

    def test_missing_file_falls_back_to_defaults(self, tmp_path):
        assert load_severity_policy(tmp_path / "absent.yaml") == SeverityPolicy()

    def test_empty_file_falls_back_to_defaults(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_severity_policy(path) == SeverityPolicy()

    def test_malformed_file_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_severity_policy(path)

    def test_malformed_severity_section_raises(self, tmp_path):
        path = tmp_path / "bad2.yaml"
        path.write_text("severity: notamapping\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_severity_policy(path)

    def test_config_overrides_are_applied(self, tmp_path):
        path = tmp_path / "custom.yaml"
        path.write_text(
            "severity:\n"
            "  escalate_for_critical_facts: false\n"
            "  base_severities:\n"
            "    critical_omission: LOW\n",
            encoding="utf-8",
        )
        policy = load_severity_policy(path)
        assert policy.assign(FindingType.CRITICAL_OMISSION) is Severity.LOW
        assert policy.escalate_for_critical_facts is False

    def test_unknown_finding_type_in_config_raises(self, tmp_path):
        path = tmp_path / "unknown.yaml"
        path.write_text(
            "severity:\n  base_severities:\n    not_a_finding_type: HIGH\n", encoding="utf-8"
        )
        with pytest.raises(ValueError):
            load_severity_policy(path)


class TestMetricDefinitions:
    def test_every_metric_documents_itself(self):
        for name, definition in metrics.METRIC_DEFINITIONS.items():
            assert definition.name == name
            assert definition.numerator.strip()
            assert definition.denominator.strip()
            assert definition.interpretation.strip()
            assert definition.edge_cases.strip()
            assert definition.limitations.strip()

    def test_rates_are_marked_lower_is_better(self):
        for name in (
            "critical_omission_rate", "critical_contradiction_rate",
            "negation_reversal_rate", "unsupported_fact_rate",
        ):
            assert metrics.get_definition(name).higher_is_better is False

    def test_unknown_metric_raises_an_actionable_error(self):
        with pytest.raises(KeyError, match="must be declared"):
            metrics.get_definition("invented_quality_score")

    def test_compute_returns_value_and_counts(self):
        metric = metrics.compute("critical_information_retention", 4, 5)
        assert metric.value == 0.8
        assert (metric.numerator, metric.denominator) == (4, 5)
        assert metric.definition

    def test_zero_denominator_is_undefined_not_zero(self):
        """A case with nothing to retain has no retention - reporting 100%
        would be flattering nonsense."""
        metric = metrics.compute("critical_information_retention", 0, 0)
        assert metric.value is None
        assert not metric.is_defined


class TestAggregation:
    def _metrics(self, numerator: float, denominator: float) -> dict[str, MetricValue]:
        return {"critical_information_retention": metrics.compute(
            "critical_information_retention", numerator, denominator
        )}

    def test_aggregation_pools_counts(self):
        """Micro-average: 1/1 and 1/9 pool to 2/10, not to the 0.55 that
        averaging the two rates would give."""
        aggregated = metrics.aggregate([self._metrics(1, 1), self._metrics(1, 9)])
        result = aggregated["critical_information_retention"]
        assert (result.numerator, result.denominator) == (2, 10)
        assert result.value == pytest.approx(0.2)

    def test_undefined_cases_do_not_contribute(self):
        aggregated = metrics.aggregate([self._metrics(4, 5), self._metrics(0, 0)])
        assert aggregated["critical_information_retention"].denominator == 5

    def test_aggregating_nothing_returns_nothing(self):
        assert metrics.aggregate([]) == {}

    def test_all_undefined_yields_no_metric(self):
        assert metrics.aggregate([self._metrics(0, 0)]) == {}

    def test_macro_average_is_the_unweighted_mean(self):
        macro = metrics.macro_average([self._metrics(1, 1), self._metrics(1, 9)])
        assert macro["critical_information_retention"] == pytest.approx((1.0 + 1 / 9) / 2)

    def test_macro_differs_from_micro_when_case_sizes_differ(self):
        """A large macro/micro gap means performance varies with how much is
        at stake in a case, which is worth surfacing."""
        case_metrics = [self._metrics(1, 1), self._metrics(1, 9)]
        micro = metrics.aggregate(case_metrics)["critical_information_retention"].value
        macro = metrics.macro_average(case_metrics)["critical_information_retention"]
        assert micro != pytest.approx(macro)

    def test_macro_ignores_undefined_values(self):
        macro = metrics.macro_average([self._metrics(4, 5), self._metrics(0, 0)])
        assert macro["critical_information_retention"] == pytest.approx(0.8)


class TestMetricValueSchema:
    def test_metric_value_requires_a_definition(self):
        with pytest.raises(ValidationError):
            MetricValue(name="x", value=1.0, numerator=1, denominator=1)
