"""Adjudication routing: deciding which findings an LLM should see.

Deterministic-first. The LLM is called only where deterministic evaluation is
genuinely weak, for three reasons: every call costs money, every call adds
non-determinism to a system whose value rests on reproducibility, and the
deterministic evaluator is already validated at zero false positives on the
oracle set.

The router sends a finding to the LLM when *any* of these hold:

- **Low match confidence.** The deterministic conclusion rested on a PARTIAL
  or ALIAS match — the semantic-paraphrase case the LLM is genuinely better at.
- **An eligible finding type.** Omission and support findings turn on whether
  a reworded statement conveys the same fact. Value and dose errors do not:
  1.4 is not 1.9 regardless of phrasing, and asking a model to second-guess
  arithmetic invites it to talk the evaluator out of a correct finding.
- **Explicit ambiguity.** The support analysis returned PARTIAL.

It does **not** route:

- High-confidence exact-match findings. There is nothing to adjudicate.
- Operational findings (empty output, missing annotation). Not clinical
  questions.
- Anything, when adjudication is disabled or no judge is configured.

Thresholds and the eligible-type list are configuration
(`config/evaluation.yaml`), not constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.core.config import BASE_DIR
from app.core.enums import FindingType, MatchLevel
from app.models.schemas import EvaluationFinding

DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "evaluation.yaml"

#: Finding types where wording can decide the answer, so a model can help.
DEFAULT_ELIGIBLE_TYPES: tuple[FindingType, ...] = (
    FindingType.CRITICAL_OMISSION,
    FindingType.NON_CRITICAL_OMISSION,
    FindingType.CONTRADICTION,
    FindingType.MEDICATION_OMISSION,
    FindingType.NEGATION_OMISSION,
    FindingType.TEMPORAL_INFORMATION_LOSS,
    FindingType.UNSUPPORTED_ADDITION,
)

#: Match levels that indicate the deterministic match was not certain.
_WEAK_MATCH_LEVELS = {MatchLevel.PARTIAL, MatchLevel.ALIAS, MatchLevel.NONE}

#: Never routed: these are not judgments about clinical language.
_OPERATIONAL_TYPES = {
    FindingType.EMPTY_OUTPUT,
    FindingType.MALFORMED_OUTPUT,
    FindingType.MISSING_ANNOTATION,
}


@dataclass(frozen=True)
class RoutingDecision:
    route: bool
    reason: str


class AdjudicationPolicy(BaseModel):
    """Configurable routing thresholds."""

    enabled: bool = False
    #: Deterministic confidence at or below which a finding is adjudicated.
    confidence_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    #: Below this, an LLM judgment is not strong enough to withdraw a
    #: deterministic finding; the result stays ADJUDICATION_REQUIRED.
    min_judgment_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    eligible_finding_types: list[FindingType] = Field(
        default_factory=lambda: list(DEFAULT_ELIGIBLE_TYPES)
    )
    route_weak_matches: bool = True
    #: Cap on judge calls per run. 0 means unlimited.
    max_calls_per_run: int = 0

    def is_eligible(self, finding_type: FindingType) -> bool:
        return finding_type in set(self.eligible_finding_types)


class AdjudicationRouter:
    """Decides which findings go to the LLM."""

    def __init__(self, policy: AdjudicationPolicy | None = None):
        self.policy = policy or AdjudicationPolicy()
        self.calls_routed = 0

    def decide(self, finding: EvaluationFinding) -> RoutingDecision:
        policy = self.policy

        if not policy.enabled:
            return RoutingDecision(False, "adjudication disabled")

        if finding.finding_type in _OPERATIONAL_TYPES:
            return RoutingDecision(
                False, f"{finding.finding_type.value} is operational, not a clinical judgment"
            )

        if not policy.is_eligible(finding.finding_type):
            return RoutingDecision(
                False,
                f"{finding.finding_type.value} is not adjudicable: the answer does not "
                f"turn on wording",
            )

        if policy.max_calls_per_run and self.calls_routed >= policy.max_calls_per_run:
            return RoutingDecision(
                False, f"per-run adjudication budget of {policy.max_calls_per_run} reached"
            )

        weak_match = (
            policy.route_weak_matches
            and finding.match_level is not None
            and finding.match_level in _WEAK_MATCH_LEVELS
        )
        low_confidence = finding.confidence <= policy.confidence_threshold

        if not (weak_match or low_confidence):
            return RoutingDecision(
                False,
                f"deterministic result is high confidence "
                f"({finding.confidence:.2f}, match {finding.match_level.value if finding.match_level else 'n/a'})",
            )

        self.calls_routed += 1
        reasons = []
        if low_confidence:
            reasons.append(f"confidence {finding.confidence:.2f} <= {policy.confidence_threshold}")
        if weak_match:
            reasons.append(f"match level {finding.match_level.value}")
        return RoutingDecision(True, "; ".join(reasons))

    def reset(self) -> None:
        self.calls_routed = 0


def load_adjudication_policy(path: Path | str | None = None) -> AdjudicationPolicy:
    """Load the routing policy, falling back to built-in defaults.

    Defaults keep adjudication **disabled**: an evaluation must not start
    making paid external calls because a config file was absent.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return AdjudicationPolicy()

    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        return AdjudicationPolicy()
    if not isinstance(raw, dict):
        raise ValueError(
            f"Evaluation config at {config_path} must be a mapping, got {type(raw).__name__}"
        )

    section = raw.get("adjudication")
    if section is None:
        return AdjudicationPolicy()
    if not isinstance(section, dict):
        raise ValueError(f"'adjudication' section in {config_path} must be a mapping")

    return AdjudicationPolicy.model_validate(section)
