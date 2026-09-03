"""Transparent, configuration-driven scoring and risk-classification layer.

Combines per-dimension `EvaluationDimensionResult`s (see app.evaluation.base)
into normalized scores and a `RiskLevel` using weights read from
configuration, not hard-coded. Not yet implemented - see
docs/ARCHITECTURE.md, Phase 6.
"""
