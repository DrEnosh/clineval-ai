"""LLM-assisted adjudication.

The LLM is never ground truth. It adjudicates findings the deterministic
evaluator was unsure about, and it cannot silently override a deterministic
result - see app/evaluation/llm/reconciler.py for the documented rules.
"""

from app.evaluation.llm.adjudicator import LLMAdjudicator
from app.evaluation.llm.judge_benchmark import (
    JudgeBenchmark,
    JudgeCase,
    build_judge_benchmark,
    evaluate_judge,
)
from app.evaluation.llm.judge_metrics import ClassMetrics, JudgeEvaluation
from app.evaluation.llm.parser import ParseResult, parse_adjudication, parse_or_raise
from app.evaluation.llm.reconciler import apply_records, classify_disagreement, reconcile
from app.evaluation.llm.router import (
    AdjudicationPolicy,
    AdjudicationRouter,
    load_adjudication_policy,
)
from app.evaluation.llm.schemas import (
    AdjudicationRecord,
    AdjudicationResponse,
    AdjudicationSummary,
    EvidenceSpan,
    LLMCallRecord,
)

__all__ = [
    "AdjudicationPolicy",
    "AdjudicationRecord",
    "AdjudicationResponse",
    "AdjudicationRouter",
    "AdjudicationSummary",
    "ClassMetrics",
    "EvidenceSpan",
    "JudgeBenchmark",
    "JudgeCase",
    "JudgeEvaluation",
    "LLMAdjudicator",
    "LLMCallRecord",
    "ParseResult",
    "apply_records",
    "build_judge_benchmark",
    "classify_disagreement",
    "evaluate_judge",
    "load_adjudication_policy",
    "parse_adjudication",
    "parse_or_raise",
    "reconcile",
]
