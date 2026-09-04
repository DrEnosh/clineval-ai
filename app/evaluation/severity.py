"""Severity classification, separated from detection.

Evaluators detect; this layer classifies. An evaluator emits
`finding_type=critical_omission` and says nothing about how bad that is. The
policy here turns that into a `Severity`.

The separation is not tidiness for its own sake. Severity is the most
contestable judgment in the system - an institution deploying this would want
to set its own thresholds - and burying it inside each evaluator would mean
changing detection code to change a policy opinion, and would let two
evaluators disagree about how bad the same class of error is. Here it is one
configurable table.

Escalation rules layer on top of the base table:

- A finding against a fact the benchmark marked CRITICAL escalates one step.
- A finding whose match rested on a weak level, or with low confidence, is
  *not* de-escalated - a probable serious error stays serious, and confidence
  is reported separately so a reviewer can weigh it.

As everywhere else in this project: these are research-framework severity
labels, not clinical or regulatory classifications.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.core.config import BASE_DIR
from app.core.enums import FactImportance, FindingType, Severity

DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "evaluation.yaml"

_LADDER: tuple[Severity, ...] = (
    Severity.INFO, Severity.LOW, Severity.MODERATE, Severity.HIGH, Severity.CRITICAL,
)


def escalate(severity: Severity, steps: int = 1) -> Severity:
    """Move `steps` up the severity ladder, saturating at CRITICAL."""
    index = min(_LADDER.index(severity) + steps, len(_LADDER) - 1)
    return _LADDER[index]


def _default_severities() -> dict[FindingType, Severity]:
    """Base severity per finding type, before escalation.

    Rationale for the notable ones:

    - `negation_reversal` starts at HIGH: asserting a condition the source
      denies is a fabricated diagnosis, not a wording slip.
    - `medication_status_error` starts at HIGH: reporting a discontinued drug
      as current risks duplicate or contraindicated therapy.
    - `temporal_value_confusion` starts at HIGH but
      `temporal_information_loss` at MODERATE: reporting a stale value as
      current actively misinforms, whereas dropping the earlier endpoint
      merely loses context.
    - `value_approximation` is INFO: rounding is not a clinical error.
    - `unsupported_addition` starts at MODERATE and escalates via the rules
      below, because unsupported content ranges from harmless padding to a
      fabricated diagnosis.
    """
    return {
        FindingType.CRITICAL_OMISSION: Severity.HIGH,
        FindingType.NON_CRITICAL_OMISSION: Severity.LOW,
        FindingType.CONTRADICTION: Severity.HIGH,
        FindingType.NEGATION_REVERSAL: Severity.HIGH,
        FindingType.NEGATION_OMISSION: Severity.MODERATE,
        FindingType.MEDICATION_OMISSION: Severity.MODERATE,
        FindingType.MEDICATION_HALLUCINATION: Severity.HIGH,
        FindingType.MEDICATION_STATUS_ERROR: Severity.HIGH,
        FindingType.MEDICATION_DOSE_ERROR: Severity.HIGH,
        FindingType.TEMPORAL_INFORMATION_LOSS: Severity.MODERATE,
        FindingType.TEMPORAL_TREND_REVERSAL: Severity.HIGH,
        FindingType.TEMPORAL_VALUE_CONFUSION: Severity.HIGH,
        FindingType.HISTORICAL_CURRENT_CONFUSION: Severity.MODERATE,
        FindingType.VALUE_ERROR: Severity.MODERATE,
        FindingType.VALUE_APPROXIMATION: Severity.INFO,
        FindingType.UNSUPPORTED_ADDITION: Severity.MODERATE,
        FindingType.CONFLICT_RESOLUTION_ERROR: Severity.HIGH,
        FindingType.EMPTY_OUTPUT: Severity.CRITICAL,
        FindingType.MALFORMED_OUTPUT: Severity.HIGH,
        FindingType.MISSING_ANNOTATION: Severity.INFO,
        FindingType.INCONSISTENT_ACROSS_RUNS: Severity.MODERATE,
        FindingType.OTHER: Severity.LOW,
    }


class SeverityPolicy(BaseModel):
    """A configurable finding-type -> severity table with escalation rules."""

    base_severities: dict[FindingType, Severity] = Field(default_factory=_default_severities)
    escalate_for_critical_facts: bool = True
    escalate_for_critical_value: bool = True
    default_severity: Severity = Severity.LOW

    def assign(
        self,
        finding_type: FindingType,
        fact_importance: FactImportance | None = None,
        is_critical_value: bool = False,
    ) -> Severity:
        """Classify one finding.

        `fact_importance` is the benchmark's own importance label for the fact
        the finding is about; `is_critical_value` marks a panic-range result.
        Both escalate, and escalation saturates rather than wrapping.
        """
        severity = self.base_severities.get(finding_type, self.default_severity)

        if (
            self.escalate_for_critical_facts
            and fact_importance is FactImportance.CRITICAL
            and finding_type is not FindingType.VALUE_APPROXIMATION
        ):
            severity = escalate(severity)

        if self.escalate_for_critical_value and is_critical_value:
            severity = escalate(severity)

        return severity


def load_severity_policy(path: Path | str | None = None) -> SeverityPolicy:
    """Load the severity policy, falling back to built-in defaults.

    A missing file is fine - the defaults are a complete policy. A malformed
    one raises, because silently evaluating under a different severity policy
    than the operator configured would make results incomparable.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return SeverityPolicy()

    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        return SeverityPolicy()
    if not isinstance(raw, dict):
        raise ValueError(
            f"Evaluation config at {config_path} must be a mapping, got {type(raw).__name__}"
        )

    section = raw.get("severity", {})
    if not isinstance(section, dict):
        raise ValueError(f"'severity' section in {config_path} must be a mapping")

    policy = SeverityPolicy()
    overrides = section.get("base_severities") or {}
    merged = dict(policy.base_severities)
    for key, value in overrides.items():
        merged[FindingType(key)] = Severity(value)

    return SeverityPolicy(
        base_severities=merged,
        escalate_for_critical_facts=section.get(
            "escalate_for_critical_facts", policy.escalate_for_critical_facts
        ),
        escalate_for_critical_value=section.get(
            "escalate_for_critical_value", policy.escalate_for_critical_value
        ),
        default_severity=Severity(section.get("default_severity", policy.default_severity.value)),
    )
