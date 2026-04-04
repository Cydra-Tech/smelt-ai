"""Prompt construction for smelt batch processing.

Builds the system and human messages sent to the LLM for each batch,
including row ID tracking instructions and output format guidance.
Supports multimodal content blocks when PIL images are present.
"""

from __future__ import annotations

import json
from typing import Any, Type

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from smelt.image import _ImageRef, batch_has_images, extract_images_from_rows
from smelt.types import _TaggedRow

_SYSTEM_TEMPLATE = """You are a structured data transformation assistant.

## Task
{user_prompt}

## Rules
- You will receive a JSON array of objects. Each object has a "row_id" field.
- For EACH input object, produce exactly one output object.
- Every output object MUST include the same "row_id" as the corresponding input object.
- Do NOT skip, merge, duplicate, or reorder rows.
- Return ALL rows — the count of output rows must equal the count of input rows.

## Output Schema
{schema_description}

Return your response as a JSON object with a single "rows" key containing the array of output objects."""

_TEXT_SYSTEM_TEMPLATE = """You are a text generation assistant.

## Task
{user_prompt}

## Rules
- You will receive a JSON array of objects. Each object has a "row_id" field.
- For EACH input object, produce exactly one output object with a "row_id" and a "text" field.
- The "text" field should contain your free-text response for that row.
- Every output object MUST include the same "row_id" as the corresponding input object.
- Do NOT skip, merge, duplicate, or reorder rows.
- Return ALL rows — the count of output rows must equal the count of input rows.

Return your response as a JSON object with a single "rows" key containing the array of output objects.

Example output format:
{{"rows": [{{"row_id": 0, "text": "Your response for row 0"}}, {{"row_id": 1, "text": "Your response for row 1"}}]}}"""

_IMAGE_SYSTEM_ADDENDUM = """

## Images
- Some input fields contain image placeholders like "[image: field_name]".
- The actual images are provided as image blocks in the user message, labeled with row ID and field name.
- Analyze each image and use your observations to produce the output fields."""

_SCHEMA_DESCRIPTION_TEMPLATE = """Each output row must conform to this schema:
{fields}"""


def describe_output_schema(model: Type[BaseModel]) -> str:
    """Generate a human-readable description of a Pydantic model's fields.

    Args:
        model: The Pydantic model class to describe.

    Returns:
        A formatted string listing each field name, type annotation, and
        whether it is required or optional.
    """
    lines: list[str] = []
    for name, field_info in model.model_fields.items():
        raw_annotation = field_info.annotation
        annotation: str = (
            raw_annotation.__name__
            if hasattr(raw_annotation, "__name__")
            else str(raw_annotation)
        )
        required: str = "required" if field_info.is_required() else "optional"
        description: str = f" — {field_info.description}" if field_info.description else ""
        lines.append(f"- {name} ({annotation}, {required}){description}")

    fields_text: str = "\n".join(lines)
    return _SCHEMA_DESCRIPTION_TEMPLATE.format(fields=fields_text)


def build_system_message(
    user_prompt: str,
    schema_description: str = "",
    has_images: bool = False,
    text_mode: bool = False,
) -> SystemMessage:
    """Build the system message for a smelt batch request.

    Combines the user's transformation instructions with row-tracking rules
    and the output schema description. When ``has_images`` is ``True``, an
    addendum explaining image placeholders and blocks is appended.

    In text mode (``output_model=None``), uses a simplified template that
    instructs the LLM to return free-text responses per row.

    Args:
        user_prompt: The user-provided transformation instruction.
        schema_description: The formatted schema description from
            :func:`describe_output_schema`. Ignored in text mode.
        has_images: Whether the batch data contains PIL images.
        text_mode: Whether to use the free-text output template.

    Returns:
        A LangChain ``SystemMessage`` ready for inclusion in the prompt.
    """
    if text_mode:
        content: str = _TEXT_SYSTEM_TEMPLATE.format(user_prompt=user_prompt)
    else:
        content = _SYSTEM_TEMPLATE.format(
            user_prompt=user_prompt,
            schema_description=schema_description,
        )
    if has_images:
        content += _IMAGE_SYSTEM_ADDENDUM
    return SystemMessage(content=content)


def build_human_message(tagged_rows: list[_TaggedRow]) -> HumanMessage:
    """Build the human message containing the batch data payload.

    Serializes the tagged rows to a JSON array for the LLM to process.
    When PIL images are detected in the row data, builds multimodal
    content blocks with base64-encoded image data.

    Args:
        tagged_rows: The input rows, each tagged with a positional ``row_id``.

    Returns:
        A LangChain ``HumanMessage`` with text content (text-only data) or
        a list of multimodal content blocks (when images are present).
    """
    raw_data: list[dict[str, Any]] = [row.data for row in tagged_rows]
    if not batch_has_images(raw_data):
        payload: list[dict[str, object]] = [
            {"row_id": row.row_id, **row.data} for row in tagged_rows
        ]
        content: str = json.dumps(payload, indent=2)
        return HumanMessage(content=content)

    cleaned_rows, image_refs = extract_images_from_rows(tagged_rows)

    payload = [
        {"row_id": row.row_id, **row.data} for row in cleaned_rows
    ]
    text_json: str = json.dumps(payload, indent=2)

    content_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": text_json},
    ]

    for ref in image_refs:
        content_blocks.append(
            {"type": "text", "text": f"Row {ref.row_id}, field '{ref.field_name}':"}
        )
        content_blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{ref.mime_type};base64,{ref.base64_data}"},
            }
        )

    return HumanMessage(content=content_blocks)


# ---------------------------------------------------------------------------
# Aggregate prompts
# ---------------------------------------------------------------------------

_AGGREGATE_SYSTEM_TEMPLATE = """You are a data aggregation assistant.

## Task
{user_prompt}

## Context
You are processing a subset of a larger dataset. Other subsets are being processed in parallel. Your output will be merged with outputs from other subsets.

## Output Schema
{schema_description}"""

_AGGREGATE_TEXT_SYSTEM_TEMPLATE = """You are a data aggregation assistant.

## Task
{user_prompt}

## Context
You are processing a subset of a larger dataset. Other subsets are being processed in parallel. Your output will be merged with outputs from other subsets."""

_AGGREGATE_MERGE_SYSTEM_TEMPLATE = """You are a data aggregation assistant.

## Task
{user_prompt}

## Context
You are merging two partial results produced from different subsets of the original data. Your output may be merged further with other partial results.

## Output Schema
{schema_description}"""

_AGGREGATE_MERGE_TEXT_SYSTEM_TEMPLATE = """You are a data aggregation assistant.

## Task
{user_prompt}

## Context
You are merging two partial results produced from different subsets of the original data. Your output may be merged further with other partial results."""


_AGGREGATE_SEQ_SYSTEM_TEMPLATE = """You are a data aggregation assistant.

## Task
{user_prompt}

## Context
You are processing data in sequential steps. You may receive a previous result representing your accumulated output so far. Incorporate the new data into the previous result to produce an updated output.

## Output Schema
{schema_description}"""

_AGGREGATE_SEQ_TEXT_SYSTEM_TEMPLATE = """You are a data aggregation assistant.

## Task
{user_prompt}

## Context
You are processing data in sequential steps. You may receive a previous result representing your accumulated output so far. Incorporate the new data into the previous result to produce an updated output."""


def build_aggregate_system_message(
    user_prompt: str,
    schema_description: str = "",
    text_mode: bool = False,
    is_merge: bool = False,
    is_sequential: bool = False,
) -> SystemMessage:
    """Build the system message for an aggregate step.

    Args:
        user_prompt: The user-provided aggregation instruction.
        schema_description: The formatted schema description. Ignored in text mode.
        text_mode: Whether to use the free-text template.
        is_merge: Whether this is a merge step (combining two partial results)
            rather than a map step (processing raw data rows).
        is_sequential: Whether to use the sequential fold template.

    Returns:
        A LangChain ``SystemMessage`` ready for inclusion in the prompt.
    """
    if is_sequential:
        if text_mode:
            content: str = _AGGREGATE_SEQ_TEXT_SYSTEM_TEMPLATE.format(
                user_prompt=user_prompt,
            )
        else:
            content = _AGGREGATE_SEQ_SYSTEM_TEMPLATE.format(
                user_prompt=user_prompt,
                schema_description=schema_description,
            )
    elif is_merge:
        if text_mode:
            content = _AGGREGATE_MERGE_TEXT_SYSTEM_TEMPLATE.format(
                user_prompt=user_prompt,
            )
        else:
            content = _AGGREGATE_MERGE_SYSTEM_TEMPLATE.format(
                user_prompt=user_prompt,
                schema_description=schema_description,
            )
    else:
        if text_mode:
            content = _AGGREGATE_TEXT_SYSTEM_TEMPLATE.format(
                user_prompt=user_prompt,
            )
        else:
            content = _AGGREGATE_SYSTEM_TEMPLATE.format(
                user_prompt=user_prompt,
                schema_description=schema_description,
            )
    return SystemMessage(content=content)


def build_aggregate_human_message(
    rows: list[dict[str, Any]] | None = None,
    previous_result: str | None = None,
    second_result: str | None = None,
) -> HumanMessage:
    """Build the human message for an aggregate step.

    Supports three modes:

    - **Map step**: ``rows`` only — serializes rows as JSON.
    - **Merge step**: ``previous_result`` + ``second_result`` — two partial results.
    - **Sequential step**: ``rows`` + ``previous_result`` — new data plus accumulated output.

    Args:
        rows: Raw data rows as dictionaries. Used for map and sequential steps.
        previous_result: Accumulated or first partial result as a string.
        second_result: Second partial result (merge step only).

    Returns:
        A LangChain ``HumanMessage`` with the aggregate data payload.
    """
    # Merge step: two partial results
    if second_result is not None:
        content: str = (
            f"Partial result 1:\n{previous_result}\n\n"
            f"Partial result 2:\n{second_result}"
        )
        return HumanMessage(content=content)

    # Sequential step: previous result + new rows
    if rows is not None and previous_result is not None:
        rows_json: str = json.dumps(rows, indent=2)
        content = (
            f"Previous result:\n{previous_result}\n\n"
            f"New data to incorporate:\n{rows_json}"
        )
        return HumanMessage(content=content)

    # Map step: rows only
    content = json.dumps(rows, indent=2)
    return HumanMessage(content=content)
