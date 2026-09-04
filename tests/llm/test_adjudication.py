"""Tests for LLM adjudication: parsing, routing, reconciliation, and metrics.

Every test runs against the mock provider. The property under test throughout
is that the LLM is an assistant, not an authority: it can resolve ambiguity the
deterministic evaluator flagged, and it cannot erase a confident deterministic
finding.
"""

from __future__ import annotations

import json

import pytest

from app.core.enums import (
    AdjudicationJudgment,
    AdjudicationStatus,
    DisagreementType,
    FindingType,
    MatchLevel,
    ProviderKind,
)
from app.evaluation.llm import (
    AdjudicationPolicy,
    AdjudicationRecord,
    AdjudicationResponse,
    AdjudicationRouter,
    AdjudicationSummary,
    JudgeEvaluation,
    LLMAdjudicator,
    apply_records,
    classify_disagreement,
    load_adjudication_policy,
    parse_adjudication,
    parse_or_raise,
    reconcile,
)
from app.evaluation.llm.judge_metrics import ClassMetrics
from app.evaluation.llm.parser import ParseFailure
from app.models.schemas import EvaluationFinding
from app.providers import MockProvider, mock_judge
from app.providers.mock import judgment_json


def finding(
    finding_type: FindingType = FindingType.CRITICAL_OMISSION,
    match_level: MatchLevel = MatchLevel.NONE,
    confidence: float = 0.7,
    finding_id: str = "F1",
) -> EvaluationFinding:
    return EvaluationFinding(
        finding_id=finding_id, case_id="C1", evaluator_name="critical_retention",
        evaluator_version="0.3.0", finding_type=finding_type,
        explanation="Critical fact not retained.",
        source_evidence="Creatinine 1.9 mg/dL (today)",
        output_evidence=None, match_level=match_level, confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    def test_parses_a_clean_json_object(self):
        result = parse_adjudication(judgment_json("SUPPORTED", 0.91, "looks fine"))
        assert result.succeeded
        assert result.response.judgment is AdjudicationJudgment.SUPPORTED
        assert result.response.confidence == 0.91
        assert result.strategy == "direct"

    def test_parses_json_inside_a_code_fence(self):
        text = f"```json\n{judgment_json('CONTRADICTED')}\n```"
        result = parse_adjudication(text)
        assert result.succeeded
        assert result.response.judgment is AdjudicationJudgment.CONTRADICTED

    def test_parses_json_with_surrounding_prose(self):
        text = f"Here is my judgment:\n{judgment_json('UNSUPPORTED')}\nHope that helps."
        result = parse_adjudication(text)
        assert result.succeeded
        assert result.strategy == "brace_span"

    def test_parses_evidence_spans(self):
        result = parse_adjudication(
            judgment_json("SUPPORTED", source_span="Creatinine 1.9", output_span="Cr 1.9")
        )
        assert result.response.evidence[0].source_span == "Creatinine 1.9"

    @pytest.mark.parametrize(
        "text,reason",
        [
            ("", "empty"),
            ("   ", "empty"),
            ("not json at all", "JSON"),
            ("[1, 2, 3]", "object"),
            ('{"judgment": "MAYBE", "confidence": 0.5}', "judgment"),
            ('{"judgment": "SUPPORTED"}', "confidence"),
            ('{"judgment": "SUPPORTED", "confidence": 1.5}', "confidence"),
            ('{"judgment": "SUPPORTED", "confidence": -0.1}', "confidence"),
            ('{"judgment": "SUPPORTED", "confidence": "high"}', "confidence"),
        ],
    )
    def test_malformed_responses_fail_cleanly(self, text, reason):
        result = parse_adjudication(text)
        assert not result.succeeded
        assert reason.lower() in result.error.lower()

    def test_unknown_fields_are_rejected(self):
        """A judge inventing keys has not followed the contract, which makes
        its other fields suspect too."""
        text = json.dumps(
            {"judgment": "SUPPORTED", "confidence": 0.9, "secret_override": True}
        )
        assert not parse_adjudication(text).succeeded

    def test_prose_judgments_are_never_inferred(self):
        """Guessing a verdict the model never committed to would fabricate the
        very thing this system is trying to measure."""
        for text in (
            "I think this is supported.",
            "SUPPORTED",
            "The judgment is CONTRADICTED.",
        ):
            assert not parse_adjudication(text).succeeded

    def test_parse_or_raise_raises_on_failure(self):
        with pytest.raises(ParseFailure):
            parse_or_raise("not json")

    def test_parse_or_raise_returns_on_success(self):
        assert parse_or_raise(judgment_json("UNCERTAIN")).judgment is (
            AdjudicationJudgment.UNCERTAIN
        )

    def test_response_schema_trims_reason(self):
        response = AdjudicationResponse(
            judgment=AdjudicationJudgment.SUPPORTED, confidence=0.9, reason="  spaced  "
        )
        assert response.reason == "spaced"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestRouting:
    def test_disabled_policy_routes_nothing(self):
        router = AdjudicationRouter(AdjudicationPolicy(enabled=False))
        assert not router.decide(finding()).route

    def test_low_confidence_finding_is_routed(self):
        router = AdjudicationRouter(AdjudicationPolicy(enabled=True))
        decision = router.decide(finding(confidence=0.7))
        assert decision.route
        assert "confidence" in decision.reason

    def test_weak_match_is_routed(self):
        router = AdjudicationRouter(AdjudicationPolicy(enabled=True))
        decision = router.decide(finding(match_level=MatchLevel.PARTIAL, confidence=1.0))
        assert decision.route
        assert "match level" in decision.reason

    def test_high_confidence_exact_match_is_not_routed(self):
        """Deterministic-first: there is nothing to adjudicate, and a call
        would cost money and add non-determinism for no gain."""
        router = AdjudicationRouter(AdjudicationPolicy(enabled=True))
        decision = router.decide(finding(match_level=MatchLevel.EXACT, confidence=1.0))
        assert not decision.route
        assert "high confidence" in decision.reason

    @pytest.mark.parametrize(
        "finding_type",
        [
            FindingType.MEDICATION_DOSE_ERROR,
            FindingType.VALUE_ERROR,
            FindingType.NEGATION_REVERSAL,
            FindingType.TEMPORAL_TREND_REVERSAL,
        ],
    )
    def test_non_linguistic_findings_are_never_routed(self, finding_type):
        """1.4 is not 1.9 regardless of phrasing; asking a model invites it to
        talk the evaluator out of a correct finding."""
        router = AdjudicationRouter(AdjudicationPolicy(enabled=True))
        decision = router.decide(finding(finding_type, MatchLevel.PARTIAL, 0.7))
        assert not decision.route
        assert "not adjudicable" in decision.reason

    @pytest.mark.parametrize(
        "finding_type",
        [FindingType.EMPTY_OUTPUT, FindingType.MALFORMED_OUTPUT, FindingType.MISSING_ANNOTATION],
    )
    def test_operational_findings_are_never_routed(self, finding_type):
        router = AdjudicationRouter(AdjudicationPolicy(enabled=True))
        decision = router.decide(finding(finding_type, MatchLevel.NONE, 0.5))
        assert not decision.route
        assert "operational" in decision.reason

    def test_call_budget_is_enforced(self):
        router = AdjudicationRouter(
            AdjudicationPolicy(enabled=True, max_calls_per_run=2)
        )
        assert router.decide(finding()).route
        assert router.decide(finding()).route
        third = router.decide(finding())
        assert not third.route
        assert "budget" in third.reason

    def test_reset_clears_the_budget(self):
        router = AdjudicationRouter(AdjudicationPolicy(enabled=True, max_calls_per_run=1))
        router.decide(finding())
        router.reset()
        assert router.decide(finding()).route

    def test_threshold_is_configurable(self):
        strict = AdjudicationRouter(
            AdjudicationPolicy(enabled=True, confidence_threshold=0.5, route_weak_matches=False)
        )
        assert not strict.decide(finding(confidence=0.7)).route
        assert strict.decide(finding(confidence=0.4)).route


class TestPolicyLoading:
    def test_shipped_config_parses_and_defaults_to_disabled(self):
        """An evaluation must never start making paid calls because a config
        file was present."""
        assert load_adjudication_policy().enabled is False

    def test_missing_file_falls_back_to_defaults(self, tmp_path):
        assert load_adjudication_policy(tmp_path / "absent.yaml") == AdjudicationPolicy()

    def test_empty_file_falls_back_to_defaults(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_adjudication_policy(path) == AdjudicationPolicy()

    def test_missing_section_falls_back_to_defaults(self, tmp_path):
        path = tmp_path / "other.yaml"
        path.write_text("severity:\n  default_severity: LOW\n", encoding="utf-8")
        assert load_adjudication_policy(path) == AdjudicationPolicy()

    def test_malformed_file_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_adjudication_policy(path)

    def test_malformed_section_raises(self, tmp_path):
        path = tmp_path / "bad2.yaml"
        path.write_text("adjudication: notamapping\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_adjudication_policy(path)

    def test_overrides_are_applied(self, tmp_path):
        path = tmp_path / "custom.yaml"
        path.write_text(
            "adjudication:\n  enabled: true\n  confidence_threshold: 0.5\n"
            "  max_calls_per_run: 10\n",
            encoding="utf-8",
        )
        policy = load_adjudication_policy(path)
        assert policy.enabled is True
        assert policy.confidence_threshold == 0.5
        assert policy.max_calls_per_run == 10


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def record_for(finding_type: FindingType) -> AdjudicationRecord:
    return AdjudicationRecord(
        finding_id="F1", case_id="C1",
        deterministic_finding_type=finding_type.value,
        deterministic_confidence=0.7,
    )


def response_of(judgment: str, confidence: float = 0.95) -> AdjudicationResponse:
    return AdjudicationResponse(
        judgment=AdjudicationJudgment(judgment), confidence=confidence, reason="because"
    )


class TestReconciliationRules:
    def test_r1_agreement_keeps_the_finding(self):
        result = reconcile(
            record_for(FindingType.CRITICAL_OMISSION),
            response_of("UNSUPPORTED"),
            AdjudicationPolicy(),
        )
        assert result.status is AdjudicationStatus.AGREED
        assert result.resolution_rule == "R1"
        assert result.finding_withdrawn is False

    def test_r2_withdraws_a_false_omission(self):
        """The case the LLM is genuinely useful for: a fact that was present,
        expressed differently."""
        result = reconcile(
            record_for(FindingType.CRITICAL_OMISSION),
            response_of("SUPPORTED", 0.95),
            AdjudicationPolicy(),
        )
        assert result.status is AdjudicationStatus.RESOLVED_BY_LLM
        assert result.resolution_rule == "R2"
        assert result.finding_withdrawn is True

    @pytest.mark.parametrize(
        "finding_type",
        [
            FindingType.CONTRADICTION,
            FindingType.NEGATION_REVERSAL,
            FindingType.MEDICATION_STATUS_ERROR,
            FindingType.TEMPORAL_VALUE_CONFUSION,
            FindingType.CONFLICT_RESOLUTION_ERROR,
        ],
    )
    def test_r3_never_lets_a_model_erase_a_contradiction(self, finding_type):
        """Contradictions carry the highest clinical cost if wrongly dismissed,
        so a model's 'looks fine' escalates to review rather than deleting."""
        result = reconcile(
            record_for(finding_type), response_of("SUPPORTED", 0.99), AdjudicationPolicy()
        )
        assert result.status is AdjudicationStatus.ADJUDICATION_REQUIRED
        assert result.resolution_rule == "R3"
        assert result.finding_withdrawn is False

    def test_r4_uncertain_leaves_it_unresolved(self):
        result = reconcile(
            record_for(FindingType.CRITICAL_OMISSION),
            response_of("UNCERTAIN", 0.99),
            AdjudicationPolicy(),
        )
        assert result.status is AdjudicationStatus.ADJUDICATION_REQUIRED
        assert result.resolution_rule == "R4"
        assert result.finding_withdrawn is False

    def test_r5_low_confidence_cannot_overturn(self):
        result = reconcile(
            record_for(FindingType.CRITICAL_OMISSION),
            response_of("SUPPORTED", 0.5),
            AdjudicationPolicy(min_judgment_confidence=0.75),
        )
        assert result.status is AdjudicationStatus.ADJUDICATION_REQUIRED
        assert result.resolution_rule == "R5"
        assert result.finding_withdrawn is False

    def test_r6_failed_call_keeps_the_finding(self):
        result = reconcile(
            record_for(FindingType.CRITICAL_OMISSION), None, AdjudicationPolicy()
        )
        assert result.status is AdjudicationStatus.FAILED
        assert result.resolution_rule == "R6"
        assert result.finding_withdrawn is False

    def test_only_r2_ever_withdraws(self):
        """The system's bias is toward reporting too much rather than silently
        dropping a clinical error."""
        withdrawn = []
        for finding_type in FindingType:
            for judgment in ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "UNCERTAIN"):
                result = reconcile(
                    record_for(finding_type), response_of(judgment, 0.99), AdjudicationPolicy()
                )
                if result.finding_withdrawn:
                    withdrawn.append((finding_type, judgment, result.resolution_rule))
        assert withdrawn
        assert {rule for _, _, rule in withdrawn} == {"R2"}
        assert {judgment for _, judgment, _ in withdrawn} == {"SUPPORTED"}

    def test_both_views_are_always_retained(self):
        result = reconcile(
            record_for(FindingType.CONTRADICTION),
            response_of("SUPPORTED", 0.99),
            AdjudicationPolicy(),
        )
        assert result.deterministic_finding_type == "contradiction"
        assert result.judgment is AdjudicationJudgment.SUPPORTED
        assert result.model_reason == "because"

    def test_model_confidence_is_stored_as_self_reported(self):
        result = reconcile(
            record_for(FindingType.CRITICAL_OMISSION),
            response_of("SUPPORTED", 0.87),
            AdjudicationPolicy(),
        )
        assert result.model_reported_confidence == 0.87


class TestDisagreementClassification:
    def test_supported_against_a_finding_is_lenient(self):
        assert classify_disagreement(
            FindingType.CRITICAL_OMISSION, AdjudicationJudgment.SUPPORTED
        ) is DisagreementType.LLM_MORE_LENIENT

    def test_uncertain_is_its_own_category(self):
        assert classify_disagreement(
            FindingType.CRITICAL_OMISSION, AdjudicationJudgment.UNCERTAIN
        ) is DisagreementType.LLM_UNCERTAIN

    def test_contradicted_against_an_omission_is_stricter(self):
        assert classify_disagreement(
            FindingType.CRITICAL_OMISSION, AdjudicationJudgment.CONTRADICTED
        ) is DisagreementType.LLM_MORE_STRICT

    def test_agreement_is_no_disagreement(self):
        assert classify_disagreement(
            FindingType.CRITICAL_OMISSION, AdjudicationJudgment.UNSUPPORTED
        ) is DisagreementType.NONE


class TestApplyRecords:
    def test_withdrawn_findings_are_separated_not_deleted(self):
        """An evaluation that silently drops findings is not auditable."""
        findings = [finding(finding_id="F1"), finding(finding_id="F2")]
        records = [
            AdjudicationRecord(
                finding_id="F1", case_id="C1",
                deterministic_finding_type="critical_omission",
                deterministic_confidence=0.7, finding_withdrawn=True,
            )
        ]
        kept, withdrawn = apply_records(findings, records)
        assert [f.finding_id for f in kept] == ["F2"]
        assert [f.finding_id for f in withdrawn] == ["F1"]

    def test_no_withdrawals_keeps_everything(self):
        findings = [finding(finding_id="F1")]
        kept, withdrawn = apply_records(findings, [])
        assert kept == findings
        assert withdrawn == []


# ---------------------------------------------------------------------------
# Adjudicator
# ---------------------------------------------------------------------------


class TestAdjudicator:
    def test_not_routed_findings_make_no_call(self):
        provider = mock_judge("SUPPORTED")
        adjudicator = LLMAdjudicator(provider, "j", AdjudicationPolicy(enabled=True))
        record = adjudicator.adjudicate(
            finding(match_level=MatchLevel.EXACT, confidence=1.0), "summary"
        )
        assert record.status is AdjudicationStatus.NOT_REQUIRED
        assert record.call is None
        assert provider.call_count == 0

    def test_routed_finding_records_the_call(self):
        adjudicator = LLMAdjudicator(
            mock_judge("SUPPORTED", 0.95), "j", AdjudicationPolicy(enabled=True)
        )
        record = adjudicator.adjudicate(finding(), "summary")
        assert record.call is not None
        assert record.call.prompt_version == "v1"
        assert record.call.model == "j"
        assert record.call.succeeded

    def test_call_record_captures_usage(self):
        adjudicator = LLMAdjudicator(
            mock_judge("SUPPORTED"), "j", AdjudicationPolicy(enabled=True)
        )
        call = adjudicator.adjudicate(finding(), "summary").call
        assert call.total_tokens is not None
        assert call.latency_ms is not None

    def test_raw_response_is_off_by_default(self):
        """Raw payloads echo clinical content, so retaining them is opt-in."""
        adjudicator = LLMAdjudicator(
            mock_judge("SUPPORTED"), "j", AdjudicationPolicy(enabled=True)
        )
        assert adjudicator.adjudicate(finding(), "s").call.raw_response is None

    def test_raw_response_can_be_captured(self):
        adjudicator = LLMAdjudicator(
            mock_judge("SUPPORTED"), "j", AdjudicationPolicy(enabled=True), capture_raw=True
        )
        assert adjudicator.adjudicate(finding(), "s").call.raw_response is not None

    def test_unparseable_response_is_retried_then_failed(self):
        provider = MockProvider(default_response="I think it is fine, honestly.")
        adjudicator = LLMAdjudicator(
            provider, "j", AdjudicationPolicy(enabled=True), max_attempts=2
        )
        record = adjudicator.adjudicate(finding(), "summary")
        assert record.status is AdjudicationStatus.FAILED
        assert record.call.parse_failures == 2
        assert provider.call_count == 2
        assert record.finding_withdrawn is False

    def test_transport_failure_then_success(self):
        provider = MockProvider(default_response=judgment_json("SUPPORTED", 0.95), fail_times=1)
        adjudicator = LLMAdjudicator(
            provider, "j", AdjudicationPolicy(enabled=True), max_attempts=2
        )
        record = adjudicator.adjudicate(finding(), "summary")
        assert record.status is AdjudicationStatus.RESOLVED_BY_LLM
        assert record.call.attempts == 2

    def test_persistent_transport_failure_keeps_the_finding(self):
        adjudicator = LLMAdjudicator(
            MockProvider(default_response="x", fail_times=99), "j",
            AdjudicationPolicy(enabled=True), max_attempts=2,
        )
        record = adjudicator.adjudicate(finding(), "summary")
        assert record.status is AdjudicationStatus.FAILED
        assert record.finding_withdrawn is False

    def test_missing_summary_is_skipped_explicitly(self):
        """Adjudicating against an empty string would invite the judge to
        confirm an omission that was never actually generated."""
        adjudicator = LLMAdjudicator(
            mock_judge("SUPPORTED"), "j", AdjudicationPolicy(enabled=True)
        )
        records = adjudicator.adjudicate_findings([finding()], summaries={})
        assert records[0].status is AdjudicationStatus.SKIPPED
        assert records[0].call is None

    def test_batch_adjudication(self):
        adjudicator = LLMAdjudicator(
            mock_judge("SUPPORTED", 0.95), "j", AdjudicationPolicy(enabled=True)
        )
        findings = [finding(finding_id=f"F{i}") for i in range(3)]
        records = adjudicator.adjudicate_findings(findings, {"C1": "summary"})
        assert len(records) == 3
        assert all(record.status is AdjudicationStatus.RESOLVED_BY_LLM for record in records)

    def test_metadata_labels_judgments_as_model_based(self):
        adjudicator = LLMAdjudicator(mock_judge(), "j", AdjudicationPolicy(enabled=True))
        metadata = adjudicator.metadata()
        assert metadata["is_model_based_judgment"] is True
        assert metadata["prompt_version"] == "v1"
        assert "api_key" not in str(metadata).lower()

    def test_mock_adjudicator_is_identifiable(self):
        assert LLMAdjudicator(mock_judge(), "j").is_mock is True

    def test_prompt_carries_the_finding_context(self):
        provider = mock_judge("SUPPORTED")
        adjudicator = LLMAdjudicator(provider, "j", AdjudicationPolicy(enabled=True))
        adjudicator.adjudicate(finding(), "the model summary text")
        sent = provider.calls[0]
        assert "Creatinine 1.9 mg/dL (today)" in sent.user
        assert "the model summary text" in sent.user
        assert "adjudicator" in sent.system.lower()


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------


class TestAdjudicationSummary:
    def test_counts_by_status(self):
        summary = AdjudicationSummary()
        for status in (
            AdjudicationStatus.NOT_REQUIRED,
            AdjudicationStatus.AGREED,
            AdjudicationStatus.RESOLVED_BY_LLM,
            AdjudicationStatus.ADJUDICATION_REQUIRED,
            AdjudicationStatus.FAILED,
        ):
            summary.record(
                AdjudicationRecord(
                    finding_id="F", case_id="C",
                    deterministic_finding_type="critical_omission",
                    deterministic_confidence=0.7, status=status,
                )
            )
        report = summary.report()
        assert report["skipped_high_confidence"] == 1
        assert report["agreed"] == 1
        assert report["resolved_by_llm"] == 1
        assert report["adjudication_required"] == 1
        assert report["failed"] == 1

    def test_rates_are_undefined_with_no_data(self):
        summary = AdjudicationSummary()
        assert summary.adjudication_rate is None
        assert summary.disagreement_rate is None
        assert summary.average_latency_ms is None

    def test_unknown_cost_reports_none(self):
        adjudicator = LLMAdjudicator(
            mock_judge("SUPPORTED", 0.95), "j", AdjudicationPolicy(enabled=True)
        )
        adjudicator.adjudicate(finding(), "summary")
        assert adjudicator.summary.report()["estimated_cost_usd"] is None

    def test_adjudication_rate_reflects_routing(self):
        adjudicator = LLMAdjudicator(
            mock_judge("SUPPORTED", 0.95), "j", AdjudicationPolicy(enabled=True)
        )
        adjudicator.adjudicate(finding(finding_id="F1"), "s")
        adjudicator.adjudicate(
            finding(finding_id="F2", match_level=MatchLevel.EXACT, confidence=1.0), "s"
        )
        assert adjudicator.summary.adjudication_rate == 0.5


# ---------------------------------------------------------------------------
# Judge evaluation metrics
# ---------------------------------------------------------------------------


class TestJudgeMetrics:
    def test_perfect_judge_scores_one(self):
        evaluation = JudgeEvaluation()
        for judgment in (
            AdjudicationJudgment.SUPPORTED,
            AdjudicationJudgment.CONTRADICTED,
            AdjudicationJudgment.UNSUPPORTED,
        ):
            evaluation.record(judgment, judgment)
        assert evaluation.accuracy == 1.0
        assert evaluation.macro_f1() == 1.0

    def test_wrong_judge_scores_zero(self):
        evaluation = JudgeEvaluation()
        evaluation.record(AdjudicationJudgment.SUPPORTED, AdjudicationJudgment.CONTRADICTED)
        assert evaluation.accuracy == 0.0

    def test_unparseable_counts_as_an_error_not_a_drop(self):
        evaluation = JudgeEvaluation()
        evaluation.record(AdjudicationJudgment.SUPPORTED, None)
        assert evaluation.total == 1
        assert evaluation.unparseable == 1
        assert evaluation.accuracy == 0.0

    def test_confusion_matrix_shape(self):
        evaluation = JudgeEvaluation()
        evaluation.record(AdjudicationJudgment.SUPPORTED, AdjudicationJudgment.SUPPORTED)
        evaluation.record(AdjudicationJudgment.SUPPORTED, AdjudicationJudgment.CONTRADICTED)
        matrix = evaluation.confusion_matrix()
        assert matrix["SUPPORTED"]["SUPPORTED"] == 1
        assert matrix["SUPPORTED"]["CONTRADICTED"] == 1

    def test_precision_recall_f1_per_class(self):
        evaluation = JudgeEvaluation()
        evaluation.record(AdjudicationJudgment.SUPPORTED, AdjudicationJudgment.SUPPORTED)
        evaluation.record(AdjudicationJudgment.UNSUPPORTED, AdjudicationJudgment.SUPPORTED)
        supported = evaluation.per_class["SUPPORTED"]
        assert supported.true_positives == 1
        assert supported.false_positives == 1
        assert supported.precision == 0.5
        assert supported.recall == 1.0
        assert supported.f1 == pytest.approx(0.6667, abs=1e-3)

    def test_undefined_precision_is_none_not_zero(self):
        """A judge that never predicts a class is silent about it, not bad at
        it - and those are different failures."""
        metrics = ClassMetrics(label="CONTRADICTED")
        assert metrics.precision is None
        assert metrics.recall is None
        assert metrics.f1 is None

    def test_empty_evaluation_has_no_accuracy(self):
        assert JudgeEvaluation().accuracy is None
        assert JudgeEvaluation().macro_f1() is None

    def test_report_states_its_scope(self):
        evaluation = JudgeEvaluation()
        evaluation.record(AdjudicationJudgment.SUPPORTED, AdjudicationJudgment.SUPPORTED)
        assert "do not generalize" in evaluation.report()["scope_note"]
