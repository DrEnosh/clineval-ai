"""Deterministic evaluators.

Each covers one clinically meaningful dimension and is independently testable.
They share the entity support engine (app/evaluation/support.py) rather than
duplicating matching logic, so a fix to negation scope or numeric comparison
improves all of them at once.
"""

from app.evaluation.evaluators.critical_retention import CriticalRetentionEvaluator
from app.evaluation.evaluators.fact_support import FactSupportEvaluator
from app.evaluation.evaluators.medication_consistency import MedicationConsistencyEvaluator
from app.evaluation.evaluators.negation_consistency import NegationConsistencyEvaluator
from app.evaluation.evaluators.temporal_consistency import TemporalConsistencyEvaluator

__all__ = [
    "CriticalRetentionEvaluator",
    "FactSupportEvaluator",
    "MedicationConsistencyEvaluator",
    "NegationConsistencyEvaluator",
    "TemporalConsistencyEvaluator",
]
