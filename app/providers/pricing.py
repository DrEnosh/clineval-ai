"""Token cost estimation.

A small, explicitly-versioned price table. Prices change, and a stale table
that silently reports wrong costs is worse than no table, so:

- Every entry records the date it was recorded.
- An unknown model returns **None**, never a guess. `estimated_cost_usd` stays
  unset rather than being fabricated.
- Costs are labelled *estimated* everywhere they surface.

Prices are per million tokens, in USD. Update `PRICE_TABLE_RECORDED` whenever
an entry changes.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICE_TABLE_RECORDED = "2026-09"


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float


# Keyed by a prefix of the model identifier, matched longest-first, so a
# pinned dated model ("anthropic-sonnet-5-20260101") resolves through its family
# prefix without needing its own entry.
PRICE_TABLE: dict[str, ModelPricing] = {
    "anthropic-opus-4": ModelPricing(15.0, 75.0),
    "anthropic-sonnet-4": ModelPricing(3.0, 15.0),
    "anthropic-haiku-4": ModelPricing(1.0, 5.0),
    "anthropic-3-5-haiku": ModelPricing(0.8, 4.0),
    "anthropic-3-5-sonnet": ModelPricing(3.0, 15.0),
    "gpt-4o-mini": ModelPricing(0.15, 0.6),
    "gpt-4o": ModelPricing(2.5, 10.0),
    "gpt-4.1-mini": ModelPricing(0.4, 1.6),
    "gpt-4.1": ModelPricing(2.0, 8.0),
    "gemini-1.5-flash": ModelPricing(0.075, 0.3),
    "gemini-1.5-pro": ModelPricing(1.25, 5.0),
    "gemini-2.0-flash": ModelPricing(0.1, 0.4),
}


def lookup(model: str) -> ModelPricing | None:
    """Pricing for `model`, or None when it is not in the table."""
    if not model:
        return None
    lowered = model.lower()
    matches = [prefix for prefix in PRICE_TABLE if lowered.startswith(prefix)]
    if not matches:
        return None
    return PRICE_TABLE[max(matches, key=len)]


def estimate_cost(
    model: str, input_tokens: int | None, output_tokens: int | None
) -> float | None:
    """Estimated USD cost, or None when it cannot be computed honestly.

    Returns None if the model is unpriced or either token count is missing.
    Treating a missing count as zero would understate cost, which is exactly
    the direction a cost estimate must not err in.
    """
    pricing = lookup(model)
    if pricing is None or input_tokens is None or output_tokens is None:
        return None
    cost = (
        input_tokens * pricing.input_per_million
        + output_tokens * pricing.output_per_million
    ) / 1_000_000
    return round(cost, 6)


def is_priced(model: str) -> bool:
    return lookup(model) is not None
