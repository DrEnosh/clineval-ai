"""Evaluation engine.

Layered so deterministic and model-based evaluators stay independently
testable (see docs/EVALUATION.md):

    normalization -> matching / negation / numeric -> support -> evaluators
                                                                     |
                                       severity <- runner <- registry
"""

from app.evaluation.base import (
    EVALUATION_VERSION,
    BaseEvaluator,
    DeterministicEvaluator,
    EvaluatorResult,
    LLMEvaluator,
)
from app.evaluation.evaluators import (
    CriticalRetentionEvaluator,
    FactSupportEvaluator,
    MedicationConsistencyEvaluator,
    NegationConsistencyEvaluator,
    TemporalConsistencyEvaluator,
)
from app.evaluation.matching import FactMatcher, MatchResult
from app.evaluation.registry import available, create, create_all, register, register_defaults
from app.evaluation.runner import EvaluationRunner, summarize_findings
from app.evaluation.severity import SeverityPolicy, load_severity_policy
from app.evaluation.support import SupportResult, analyze_entity

__all__ = [
    "EVALUATION_VERSION",
    "BaseEvaluator",
    "CriticalRetentionEvaluator",
    "DeterministicEvaluator",
    "EvaluationRunner",
    "EvaluatorResult",
    "FactMatcher",
    "FactSupportEvaluator",
    "LLMEvaluator",
    "MatchResult",
    "MedicationConsistencyEvaluator",
    "NegationConsistencyEvaluator",
    "SeverityPolicy",
    "SupportResult",
    "TemporalConsistencyEvaluator",
    "analyze_entity",
    "available",
    "create",
    "create_all",
    "load_severity_policy",
    "register",
    "register_defaults",
    "summarize_findings",
]
