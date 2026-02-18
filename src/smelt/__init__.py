"""Smelt — LLM-powered structured data transformation.

Provides a simple API for transforming structured data through LLMs
with strict Pydantic validation, concurrent batch processing, and
automatic retry logic.

Example:
    >>> from smelt import Model, Job
    >>> from pydantic import BaseModel
    >>>
    >>> class Classification(BaseModel):
    ...     category: str
    ...     confidence: float
    >>>
    >>> model = Model(provider="openai", name="gpt-4o")
    >>> job = Job(
    ...     prompt="Classify each company by industry sector",
    ...     output_model=Classification,
    ... )
    >>> result = job.run(model, data=[{"name": "Apple", "description": "Tech company"}])
"""

from smelt.errors import (
    SmeltAPIError,
    SmeltConfigError,
    SmeltError,
    SmeltExhaustionError,
    SmeltValidationError,
)
from smelt.job import Job
from smelt.model import Model
from smelt.types import BatchError, SmeltMetrics, SmeltResult

__version__: str = "0.1.1"

__all__: list[str] = [
    "Model",
    "Job",
    "SmeltResult",
    "SmeltMetrics",
    "BatchError",
    "SmeltError",
    "SmeltConfigError",
    "SmeltValidationError",
    "SmeltAPIError",
    "SmeltExhaustionError",
]
