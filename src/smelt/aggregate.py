"""Tree-reduction aggregate engine for smelt.

Processes input data through a map-reduce tree: first maps each batch
to a single output in parallel, then merges outputs pairwise until
one final result remains. Supports both structured and free-text modes.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any, Type, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from smelt.batch import _compute_backoff, _is_retriable_error
from smelt.errors import SmeltExhaustionError, SmeltValidationError
from smelt.prompt import (
    build_aggregate_human_message,
    build_aggregate_system_message,
    describe_output_schema,
)
from smelt.types import BatchError, SmeltMetrics, SmeltResult

T = TypeVar("T", bound=BaseModel)


class _AggregateTextOutput(BaseModel):
    """Internal model for free-text aggregate output.

    Used when ``output_model=None`` to get a structured response
    containing a single ``text`` field from the LLM.

    Attributes:
        text: The free-text aggregation result.
    """

    text: str


async def _process_aggregate_step(
    structured_model: Runnable[Any, Any],
    system_message: SystemMessage,
    human_message: Any,
    step_index: int,
    max_retries: int,
    step_row_ids: tuple[int, ...],
) -> tuple[BaseModel | None, BatchError | None, int, int, int]:
    """Process a single aggregate step (map or merge) with retry logic.

    Args:
        structured_model: The LangChain runnable with structured output.
        system_message: The system message for this step.
        human_message: The human message containing data or partial results.
        step_index: Index of this step for error reporting.
        max_retries: Maximum retry attempts.
        step_row_ids: Row indices associated with this step (for error reporting).

    Returns:
        A tuple of ``(parsed_output, error, retries, input_tokens, output_tokens)``.
        On success, ``parsed_output`` is the model instance and ``error`` is ``None``.
        On failure, ``parsed_output`` is ``None`` and ``error`` is a ``BatchError``.
    """
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    for attempt in range(max_retries + 1):
        try:
            response: dict[str, Any] = await structured_model.ainvoke(
                [system_message, human_message]
            )

            raw_message = response.get("raw")
            if raw_message is not None:
                usage = getattr(raw_message, "usage_metadata", None)
                if usage is not None:
                    input_tokens += usage.get("input_tokens", 0)
                    output_tokens += usage.get("output_tokens", 0)

            parsing_error = response.get("parsing_error")
            if parsing_error is not None:
                raise SmeltValidationError(
                    f"LLM output parsing failed: {parsing_error}",
                    raw_response=response.get("raw"),
                )

            parsed: BaseModel = response["parsed"]
            return parsed, None, retries, input_tokens, output_tokens

        except SmeltValidationError:
            if attempt < max_retries:
                retries += 1
                backoff: float = _compute_backoff(attempt)
                await asyncio.sleep(backoff)
                continue

            error = BatchError(
                batch_index=step_index,
                row_ids=step_row_ids,
                error_type="validation",
                message=f"Validation failed after {attempt + 1} attempts.",
                attempts=attempt + 1,
            )
            return None, error, retries, input_tokens, output_tokens

        except Exception as exc:
            if _is_retriable_error(exc):
                if attempt < max_retries:
                    retries += 1
                    backoff = _compute_backoff(attempt)
                    await asyncio.sleep(backoff)
                    continue

                error = BatchError(
                    batch_index=step_index,
                    row_ids=step_row_ids,
                    error_type="api",
                    message=f"API error after {attempt + 1} attempts: {exc}",
                    attempts=attempt + 1,
                )
                return None, error, retries, input_tokens, output_tokens

            error = BatchError(
                batch_index=step_index,
                row_ids=step_row_ids,
                error_type="api",
                message=f"Non-retriable error: {exc}",
                attempts=attempt + 1,
            )
            return None, error, retries, input_tokens, output_tokens

    return None, None, retries, input_tokens, output_tokens


def _serialize_output(output: BaseModel, text_mode: bool) -> str:
    """Serialize a parsed output to a string for use in merge steps.

    Args:
        output: The parsed Pydantic model instance.
        text_mode: Whether to extract the ``text`` field (free-text mode)
            or dump the full model as JSON.

    Returns:
        A string representation of the output.
    """
    if text_mode:
        return output.text  # type: ignore[attr-defined]
    return json.dumps(output.model_dump(), indent=2)


def _extract_final_output(output: BaseModel, text_mode: bool) -> Any:
    """Extract the final user-facing result from a parsed output.

    Args:
        output: The parsed Pydantic model instance.
        text_mode: Whether to extract the ``text`` field as a plain string.

    Returns:
        The model instance directly (structured mode) or a plain string (text mode).
    """
    if text_mode:
        return output.text  # type: ignore[attr-defined]
    return output


async def execute_aggregate(
    chat_model: BaseChatModel,
    user_prompt: str,
    output_model: Type[T] | None,
    data: list[dict[str, Any]],
    batch_size: int,
    concurrency: int,
    max_retries: int,
    stop_on_exhaustion: bool,
) -> SmeltResult[Any]:
    """Execute the tree-reduction aggregate pipeline.

    Splits data into batches, maps each batch to a single output in parallel,
    then merges outputs pairwise in a tree until one final result remains.

    Args:
        chat_model: The initialized LangChain chat model.
        user_prompt: The user's aggregation instruction.
        output_model: The user's Pydantic model for the aggregate output,
            or ``None`` for free-text mode.
        data: The input rows as a list of dictionaries.
        batch_size: Number of rows per map step.
        concurrency: Maximum number of concurrent map/merge steps.
        max_retries: Maximum retry attempts per step.
        stop_on_exhaustion: If ``True``, raises ``SmeltExhaustionError``
            when a step exhausts its retries.

    Returns:
        A ``SmeltResult`` with a single-element ``data`` list containing
        the aggregate output (structured model or string).

    Raises:
        SmeltExhaustionError: If ``stop_on_exhaustion`` is ``True`` and
            any step fails after all retries.
    """
    start_time: float = time.monotonic()

    text_mode: bool = output_model is None
    model_for_output: Type[BaseModel] = _AggregateTextOutput if text_mode else output_model

    structured_model: Runnable[Any, Any] = chat_model.with_structured_output(
        model_for_output, include_raw=True
    )

    schema_description: str = "" if text_mode else describe_output_schema(model_for_output)

    map_system_message: SystemMessage = build_aggregate_system_message(
        user_prompt, schema_description, text_mode=text_mode, is_merge=False,
    )
    merge_system_message: SystemMessage = build_aggregate_system_message(
        user_prompt, schema_description, text_mode=text_mode, is_merge=True,
    )

    # Split data into batches
    batches: list[list[dict[str, Any]]] = [
        data[i : i + batch_size]
        for i in range(0, len(data), batch_size)
    ]

    if not batches:
        wall_time: float = time.monotonic() - start_time
        return SmeltResult(
            data=[],
            errors=[],
            metrics=SmeltMetrics(
                total_rows=0,
                wall_time_seconds=round(wall_time, 3),
            ),
        )

    errors: list[BatchError] = []
    total_retries: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_steps: int = 0
    semaphore: asyncio.Semaphore = asyncio.Semaphore(concurrency)

    # --- Phase 1: Map (parallel) ---
    async def _map_step(
        batch: list[dict[str, Any]], idx: int, row_offset: int,
    ) -> tuple[int, BaseModel | None, BatchError | None]:
        async with semaphore:
            nonlocal total_retries, total_input_tokens, total_output_tokens, total_steps
            total_steps += 1

            row_ids = tuple(range(row_offset, row_offset + len(batch)))
            human_msg = build_aggregate_human_message(rows=batch)

            parsed, error, retries, in_tok, out_tok = await _process_aggregate_step(
                structured_model=structured_model,
                system_message=map_system_message,
                human_message=human_msg,
                step_index=idx,
                max_retries=max_retries,
                step_row_ids=row_ids,
            )

            total_retries += retries
            total_input_tokens += in_tok
            total_output_tokens += out_tok

            return idx, parsed, error

    map_tasks: list[asyncio.Task[tuple[int, BaseModel | None, BatchError | None]]] = []
    row_offset: int = 0
    for idx, batch in enumerate(batches):
        task = asyncio.create_task(_map_step(batch, idx, row_offset))
        map_tasks.append(task)
        row_offset += len(batch)

    map_results: list[tuple[int, BaseModel | None, BatchError | None]] = []
    for coro in asyncio.as_completed(map_tasks):
        result = await coro
        map_results.append(result)

    map_results.sort(key=lambda r: r[0])

    # Collect successful map outputs and any errors
    map_outputs: list[BaseModel] = []
    for idx, parsed, error in map_results:
        if error is not None:
            errors.append(error)
            if stop_on_exhaustion:
                break
        else:
            map_outputs.append(parsed)  # type: ignore[arg-type]

    # If any map step failed and stop_on_exhaustion, abort
    if errors and stop_on_exhaustion:
        wall_time = time.monotonic() - start_time
        smelt_result: SmeltResult[Any] = SmeltResult(
            data=[],
            errors=errors,
            metrics=SmeltMetrics(
                total_rows=len(data),
                failed_rows=len(data),
                total_batches=total_steps,
                failed_batches=len(errors),
                total_retries=total_retries,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                wall_time_seconds=round(wall_time, 3),
            ),
        )
        raise SmeltExhaustionError(
            f"{len(errors)} step(s) failed after exhausting retries.",
            partial_result=smelt_result,
        )

    # If some map steps failed (not stop_on_exhaustion), we can't produce a valid aggregate
    if len(map_outputs) < len(batches):
        wall_time = time.monotonic() - start_time
        return SmeltResult(
            data=[],
            errors=errors,
            metrics=SmeltMetrics(
                total_rows=len(data),
                failed_rows=len(data),
                total_batches=total_steps,
                successful_batches=len(map_outputs),
                failed_batches=len(errors),
                total_retries=total_retries,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                wall_time_seconds=round(wall_time, 3),
            ),
        )

    # Single batch — no merge needed
    if len(map_outputs) == 1:
        final_output: Any = _extract_final_output(map_outputs[0], text_mode)
        wall_time = time.monotonic() - start_time
        return SmeltResult(
            data=[final_output],
            errors=[],
            metrics=SmeltMetrics(
                total_rows=len(data),
                successful_rows=len(data),
                total_batches=total_steps,
                successful_batches=total_steps,
                total_retries=total_retries,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                wall_time_seconds=round(wall_time, 3),
            ),
        )

    # --- Phase 2: Merge (tree reduction, pairwise parallel) ---
    current_level: list[BaseModel] = map_outputs
    merge_step_idx: int = len(batches)

    while len(current_level) > 1:
        pairs: list[tuple[BaseModel, BaseModel | None]] = []
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                pairs.append((current_level[i], current_level[i + 1]))
            else:
                pairs.append((current_level[i], None))

        async def _merge_step(
            left: BaseModel, right: BaseModel | None, step_idx: int,
        ) -> tuple[int, BaseModel | None, BatchError | None]:
            async with semaphore:
                nonlocal total_retries, total_input_tokens, total_output_tokens, total_steps
                total_steps += 1

                if right is None:
                    return step_idx, left, None

                left_str: str = _serialize_output(left, text_mode)
                right_str: str = _serialize_output(right, text_mode)

                human_msg = build_aggregate_human_message(
                    previous_result=left_str,
                    second_result=right_str,
                )

                parsed, error, retries, in_tok, out_tok = await _process_aggregate_step(
                    structured_model=structured_model,
                    system_message=merge_system_message,
                    human_message=human_msg,
                    step_index=step_idx,
                    max_retries=max_retries,
                    step_row_ids=(),
                )

                total_retries += retries
                total_input_tokens += in_tok
                total_output_tokens += out_tok

                return step_idx, parsed, error

        merge_tasks: list[asyncio.Task[tuple[int, BaseModel | None, BatchError | None]]] = []
        for left, right in pairs:
            task = asyncio.create_task(_merge_step(left, right, merge_step_idx))
            merge_tasks.append(task)
            merge_step_idx += 1

        merge_results: list[tuple[int, BaseModel | None, BatchError | None]] = []
        for coro in asyncio.as_completed(merge_tasks):
            merge_result = await coro
            merge_results.append(merge_result)

        merge_results.sort(key=lambda r: r[0])

        next_level: list[BaseModel] = []
        merge_failed: bool = False
        for step_idx, parsed, error in merge_results:
            if error is not None:
                errors.append(error)
                merge_failed = True
                if stop_on_exhaustion:
                    break
            else:
                next_level.append(parsed)  # type: ignore[arg-type]

        if merge_failed:
            wall_time = time.monotonic() - start_time
            result_data: list[Any] = []
            smelt_result = SmeltResult(
                data=result_data,
                errors=errors,
                metrics=SmeltMetrics(
                    total_rows=len(data),
                    failed_rows=len(data),
                    total_batches=total_steps,
                    successful_batches=total_steps - len(errors),
                    failed_batches=len(errors),
                    total_retries=total_retries,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    wall_time_seconds=round(wall_time, 3),
                ),
            )
            if stop_on_exhaustion:
                raise SmeltExhaustionError(
                    f"{len(errors)} step(s) failed after exhausting retries.",
                    partial_result=smelt_result,
                )
            return smelt_result

        current_level = next_level

    # Final result
    final_output = _extract_final_output(current_level[0], text_mode)
    wall_time = time.monotonic() - start_time

    num_levels: int = 1 + (math.ceil(math.log2(len(batches))) if len(batches) > 1 else 0)

    return SmeltResult(
        data=[final_output],
        errors=[],
        metrics=SmeltMetrics(
            total_rows=len(data),
            successful_rows=len(data),
            total_batches=total_steps,
            successful_batches=total_steps,
            total_retries=total_retries,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            wall_time_seconds=round(wall_time, 3),
        ),
    )
