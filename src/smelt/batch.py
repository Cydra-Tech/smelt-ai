"""Async batch processing engine for smelt.

Orchestrates concurrent LLM calls with retry logic, exponential backoff,
cooperative cancellation, and result reassembly in original row order.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Type, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from smelt.errors import SmeltExhaustionError, SmeltValidationError
from smelt.prompt import build_human_message, build_system_message, describe_output_schema
from smelt.types import BatchError, SmeltMetrics, SmeltResult, _BatchResult, _TaggedRow
from smelt.validation import (
    create_batch_wrapper,
    create_internal_model,
    strip_row_id,
    validate_batch_response,
)

T = TypeVar("T", bound=BaseModel)

_RETRIABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_BASE_BACKOFF_SECONDS: float = 1.0
_MAX_BACKOFF_SECONDS: float = 60.0


def _is_retriable_error(exc: Exception) -> bool:
    """Determine whether an exception is worth retrying.

    Retriable errors include rate limits, server errors, timeouts,
    and connection failures. Client errors (400, 401, 403) are not retriable.

    Args:
        exc: The exception raised during an LLM call.

    Returns:
        ``True`` if the error should be retried, ``False`` otherwise.
    """
    status_code: int | None = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code in _RETRIABLE_STATUS_CODES

    exc_name: str = type(exc).__name__.lower()
    exc_msg: str = str(exc).lower()
    retriable_keywords: list[str] = ["timeout", "connection", "rate", "overloaded"]
    return any(kw in exc_name or kw in exc_msg for kw in retriable_keywords)


def _compute_backoff(attempt: int) -> float:
    """Compute exponential backoff delay with jitter.

    Uses the formula: ``min(base * 2^attempt + jitter, max_backoff)``.

    Args:
        attempt: Zero-based attempt index (0 = first retry).

    Returns:
        Delay in seconds before the next retry attempt.
    """
    delay: float = _BASE_BACKOFF_SECONDS * (2 ** attempt)
    jitter: float = random.uniform(0, delay * 0.1)
    return min(delay + jitter, _MAX_BACKOFF_SECONDS)


async def _process_batch(
    structured_model: Runnable[Any, Any],
    system_message: SystemMessage,
    tagged_rows: list[_TaggedRow],
    batch_index: int,
    max_retries: int,
    cancel_event: asyncio.Event,
) -> _BatchResult:
    """Process a single batch with retry logic.

    Invokes the structured LLM model, validates the response, and retries
    on transient failures with exponential backoff.

    Args:
        structured_model: The LangChain runnable with structured output configured.
        system_message: The system message to prepend to each request.
        tagged_rows: The input rows for this batch, each tagged with a ``row_id``.
        batch_index: Zero-based index of this batch in the overall run.
        max_retries: Maximum number of retry attempts before giving up.
        cancel_event: Cooperative cancellation event; if set, the batch
            short-circuits and returns a cancellation error.

    Returns:
        A ``_BatchResult`` containing either the validated rows or an error record.
    """
    expected_row_ids: list[int] = [row.row_id for row in tagged_rows]
    human_message = build_human_message(tagged_rows)
    result = _BatchResult(batch_index=batch_index)

    for attempt in range(max_retries + 1):
        if cancel_event.is_set():
            result.error = BatchError(
                batch_index=batch_index,
                row_ids=tuple(expected_row_ids),
                error_type="cancelled",
                message="Batch cancelled due to stop_on_exhaustion.",
                attempts=attempt,
            )
            return result

        try:
            response: dict[str, Any] = await structured_model.ainvoke(
                [system_message, human_message]
            )

            raw_message = response.get("raw")
            if raw_message is not None:
                usage = getattr(raw_message, "usage_metadata", None)
                if usage is not None:
                    result.input_tokens += usage.get("input_tokens", 0)
                    result.output_tokens += usage.get("output_tokens", 0)

            parsing_error = response.get("parsing_error")
            if parsing_error is not None:
                raise SmeltValidationError(
                    f"LLM output parsing failed: {parsing_error}",
                    raw_response=response.get("raw"),
                )

            parsed: BaseModel = response["parsed"]
            validated_rows: list[BaseModel] = validate_batch_response(parsed, expected_row_ids)
            result.rows = validated_rows
            return result

        except SmeltValidationError:
            if attempt < max_retries:
                result.retries += 1
                backoff: float = _compute_backoff(attempt)
                await asyncio.sleep(backoff)
                continue

            result.error = BatchError(
                batch_index=batch_index,
                row_ids=tuple(expected_row_ids),
                error_type="validation",
                message=f"Validation failed after {attempt + 1} attempts.",
                attempts=attempt + 1,
            )
            return result

        except Exception as exc:
            if _is_retriable_error(exc):
                if attempt < max_retries:
                    result.retries += 1
                    backoff = _compute_backoff(attempt)
                    await asyncio.sleep(backoff)
                    continue

                result.error = BatchError(
                    batch_index=batch_index,
                    row_ids=tuple(expected_row_ids),
                    error_type="api",
                    message=f"API error after {attempt + 1} attempts: {exc}",
                    attempts=attempt + 1,
                )
                return result

            result.error = BatchError(
                batch_index=batch_index,
                row_ids=tuple(expected_row_ids),
                error_type="api",
                message=f"Non-retriable error: {exc}",
                attempts=attempt + 1,
            )
            return result

    return result


async def execute_batches(
    chat_model: BaseChatModel,
    user_prompt: str,
    output_model: Type[T],
    data: list[dict[str, Any]],
    batch_size: int,
    concurrency: int,
    max_retries: int,
    stop_on_exhaustion: bool,
) -> SmeltResult[T]:
    """Execute the full batch processing pipeline.

    Tags rows, creates internal models, splits into batches, runs them
    concurrently through the LLM, and reassembles results in original order.

    Args:
        chat_model: The initialized LangChain chat model.
        user_prompt: The user's transformation instruction.
        output_model: The user's Pydantic model for output rows.
        data: The input rows as a list of dictionaries.
        batch_size: Number of rows per batch.
        concurrency: Maximum number of concurrent batch requests.
        max_retries: Maximum retry attempts per batch.
        stop_on_exhaustion: If ``True``, raises ``SmeltExhaustionError``
            when any batch exhausts its retries.

    Returns:
        A ``SmeltResult[T]`` containing validated data, errors, and metrics.

    Raises:
        SmeltExhaustionError: If ``stop_on_exhaustion`` is ``True`` and a batch
            fails after all retries.
    """
    start_time: float = time.monotonic()

    tagged_rows: list[_TaggedRow] = [
        _TaggedRow(row_id=i, data=row) for i, row in enumerate(data)
    ]

    internal_model: Type[BaseModel] = create_internal_model(output_model)
    batch_wrapper: Type[BaseModel] = create_batch_wrapper(internal_model)

    structured_model: Runnable[Any, Any] = chat_model.with_structured_output(
        batch_wrapper, include_raw=True
    )

    schema_description: str = describe_output_schema(internal_model)
    system_message = build_system_message(user_prompt, schema_description)

    batches: list[list[_TaggedRow]] = [
        tagged_rows[i : i + batch_size]
        for i in range(0, len(tagged_rows), batch_size)
    ]

    semaphore = asyncio.Semaphore(concurrency)
    cancel_event = asyncio.Event()

    async def _run_with_semaphore(batch: list[_TaggedRow], idx: int) -> _BatchResult:
        async with semaphore:
            return await _process_batch(
                structured_model=structured_model,
                system_message=system_message,
                tagged_rows=batch,
                batch_index=idx,
                max_retries=max_retries,
                cancel_event=cancel_event,
            )

    tasks: list[asyncio.Task[_BatchResult]] = [
        asyncio.create_task(_run_with_semaphore(batch, idx))
        for idx, batch in enumerate(batches)
    ]

    batch_results: list[_BatchResult] = []
    errors: list[BatchError] = []
    total_retries: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    for coro in asyncio.as_completed(tasks):
        batch_result: _BatchResult = await coro
        batch_results.append(batch_result)

        total_retries += batch_result.retries
        total_input_tokens += batch_result.input_tokens
        total_output_tokens += batch_result.output_tokens

        if batch_result.error is not None:
            errors.append(batch_result.error)
            if stop_on_exhaustion:
                cancel_event.set()

    batch_results.sort(key=lambda br: br.batch_index)

    all_rows: list[tuple[int, BaseModel]] = []
    for br in batch_results:
        for row in br.rows:
            row_id: int = row.row_id  # type: ignore[attr-defined]
            all_rows.append((row_id, row))

    all_rows.sort(key=lambda pair: pair[0])
    ordered_data: list[T] = [strip_row_id(row, output_model) for _, row in all_rows]

    wall_time: float = time.monotonic() - start_time
    failed_row_count: int = sum(len(e.row_ids) for e in errors)

    metrics = SmeltMetrics(
        total_rows=len(data),
        successful_rows=len(ordered_data),
        failed_rows=failed_row_count,
        total_batches=len(batches),
        successful_batches=len(batches) - len(errors),
        failed_batches=len(errors),
        total_retries=total_retries,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        wall_time_seconds=round(wall_time, 3),
    )

    smelt_result: SmeltResult[T] = SmeltResult(
        data=ordered_data,
        errors=errors,
        metrics=metrics,
    )

    if stop_on_exhaustion and errors:
        raise SmeltExhaustionError(
            f"{len(errors)} batch(es) failed after exhausting retries.",
            partial_result=smelt_result,
        )

    return smelt_result
