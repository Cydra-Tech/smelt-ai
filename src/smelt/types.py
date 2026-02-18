"""Core data types for smelt.

Defines result containers, metrics, error records, and internal types
used throughout the batch processing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class BatchError:
    """Record of a failed batch after all retries were exhausted.

    Attributes:
        batch_index: Zero-based index of the batch in the overall run.
        row_ids: The row IDs that were part of this batch.
        error_type: Short classification of the error (e.g. "validation", "api", "parse").
        message: Human-readable error description.
        attempts: Total number of attempts made (including the initial try).
        raw_response: The raw LLM response string, if available.
    """

    batch_index: int
    row_ids: tuple[int, ...]
    error_type: str
    message: str
    attempts: int
    raw_response: str | None = None


@dataclass
class SmeltMetrics:
    """Aggregated metrics for a completed smelt run.

    Attributes:
        total_rows: Total number of input rows.
        successful_rows: Number of rows that produced valid output.
        failed_rows: Number of rows in failed batches.
        total_batches: Total number of batches processed.
        successful_batches: Number of batches that succeeded.
        failed_batches: Number of batches that exhausted retries.
        total_retries: Cumulative retry count across all batches.
        input_tokens: Total input tokens consumed (from LLM usage metadata).
        output_tokens: Total output tokens consumed (from LLM usage metadata).
        wall_time_seconds: Wall-clock duration of the entire run.
    """

    total_rows: int = 0
    successful_rows: int = 0
    failed_rows: int = 0
    total_batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    total_retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_seconds: float = 0.0


@dataclass
class SmeltResult(Generic[T]):
    """Result container for a smelt run.

    Generic over ``T``, the user's Pydantic output model.

    Attributes:
        data: Successfully transformed rows, in original input order.
        errors: Records for each batch that failed after all retries.
        metrics: Aggregated run metrics (tokens, timing, retries).
    """

    data: list[T] = field(default_factory=list)
    errors: list[BatchError] = field(default_factory=list)
    metrics: SmeltMetrics = field(default_factory=SmeltMetrics)

    @property
    def success(self) -> bool:
        """Whether the run completed with zero batch errors."""
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# Internal types — not part of the public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TaggedRow:
    """A single input row tagged with a positional row ID.

    Attributes:
        row_id: Zero-based position in the original input list.
        data: The raw input dictionary for this row.
    """

    row_id: int
    data: dict[str, Any]


@dataclass
class _BatchResult:
    """Result of processing a single batch.

    Attributes:
        batch_index: Zero-based index of this batch.
        rows: Successfully parsed and validated output rows (with row_id still attached).
        error: A ``BatchError`` if this batch ultimately failed, else ``None``.
        retries: Number of retries consumed by this batch.
        input_tokens: Input tokens used across all attempts for this batch.
        output_tokens: Output tokens used across all attempts for this batch.
    """

    batch_index: int
    rows: list[Any] = field(default_factory=list)
    error: BatchError | None = None
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
