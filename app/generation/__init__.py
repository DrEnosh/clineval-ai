"""Model output generation: request/response shapes, caching, and the generator."""

from app.generation.cache import OutputCache
from app.generation.generator import SummaryGenerator, to_ai_outputs
from app.generation.schemas import (
    ClinicalSummaryRequest,
    ClinicalSummaryResponse,
    GenerationSummary,
)

__all__ = [
    "ClinicalSummaryRequest",
    "ClinicalSummaryResponse",
    "GenerationSummary",
    "OutputCache",
    "SummaryGenerator",
    "to_ai_outputs",
]
