"""Evaluator registry.

A name -> factory table so evaluators are selected by configuration rather
than by import. Adding an evaluator means registering it; no existing
evaluator, and no part of the runner, changes.

Registration is explicit rather than auto-discovered by scanning the package.
Auto-discovery would make the set of evaluators that ran depend on import
side effects, which for a system whose output is meant to be reproducible is
a bad trade for the small convenience.
"""

from __future__ import annotations

from typing import Callable, Iterable

from app.evaluation.base import BaseEvaluator

EvaluatorFactory = Callable[[], BaseEvaluator]

_REGISTRY: dict[str, EvaluatorFactory] = {}


def register(name: str, factory: EvaluatorFactory, *, replace: bool = False) -> None:
    """Register an evaluator factory under `name`.

    Refuses to shadow an existing name unless `replace` is set, so two
    evaluators cannot silently claim the same slot and leave which one runs
    dependent on import order.
    """
    if name in _REGISTRY and not replace:
        raise ValueError(
            f"An evaluator named '{name}' is already registered. Pass replace=True to "
            f"override it deliberately."
        )
    _REGISTRY[name] = factory


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def available() -> list[str]:
    return sorted(_REGISTRY)


def create(name: str) -> BaseEvaluator:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown evaluator '{name}'. Available: {', '.join(available()) or 'none'}."
        )
    return _REGISTRY[name]()


def create_all(names: Iterable[str] | None = None) -> list[BaseEvaluator]:
    """Instantiate the named evaluators, or every registered one."""
    return [create(name) for name in (list(names) if names is not None else available())]


def register_defaults() -> None:
    """Register the Phase 3 deterministic evaluators.

    Imported here rather than at module scope to keep the registry module free
    of dependencies on the evaluators it registers.
    """
    from app.evaluation.evaluators import (
        CriticalRetentionEvaluator,
        FactSupportEvaluator,
        MedicationConsistencyEvaluator,
        NegationConsistencyEvaluator,
        TemporalConsistencyEvaluator,
    )

    for evaluator_class in (
        CriticalRetentionEvaluator,
        MedicationConsistencyEvaluator,
        NegationConsistencyEvaluator,
        TemporalConsistencyEvaluator,
        FactSupportEvaluator,
    ):
        register(evaluator_class.name, evaluator_class, replace=True)
