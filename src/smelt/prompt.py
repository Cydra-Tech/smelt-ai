"""Prompt construction for smelt batch processing.

Builds the system and human messages sent to the LLM for each batch,
including row ID tracking instructions and output format guidance.
"""

from __future__ import annotations

import json
from typing import Type

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

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


def build_system_message(user_prompt: str, schema_description: str) -> SystemMessage:
    """Build the system message for a smelt batch request.

    Combines the user's transformation instructions with row-tracking rules
    and the output schema description.

    Args:
        user_prompt: The user-provided transformation instruction.
        schema_description: The formatted schema description from
            :func:`describe_output_schema`.

    Returns:
        A LangChain ``SystemMessage`` ready for inclusion in the prompt.
    """
    content: str = _SYSTEM_TEMPLATE.format(
        user_prompt=user_prompt,
        schema_description=schema_description,
    )
    return SystemMessage(content=content)


def build_human_message(tagged_rows: list[_TaggedRow]) -> HumanMessage:
    """Build the human message containing the batch data payload.

    Serializes the tagged rows to a JSON array for the LLM to process.

    Args:
        tagged_rows: The input rows, each tagged with a positional ``row_id``.

    Returns:
        A LangChain ``HumanMessage`` with the JSON-serialized row data.
    """
    payload: list[dict[str, object]] = [
        {"row_id": row.row_id, **row.data} for row in tagged_rows
    ]
    content: str = json.dumps(payload, indent=2)
    return HumanMessage(content=content)
