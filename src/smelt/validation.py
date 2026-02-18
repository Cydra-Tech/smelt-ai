"""Dynamic Pydantic model creation and response validation for smelt.

Handles injection of ``row_id`` into user models, creation of batch wrapper
models for structured output, and post-response validation of row IDs.
"""

from __future__ import annotations

from typing import Any, Type, TypeVar

from pydantic import BaseModel, Field, create_model

from smelt.errors import SmeltConfigError, SmeltValidationError

T = TypeVar("T", bound=BaseModel)


def create_internal_model(user_model: Type[T]) -> Type[BaseModel]:
    """Create an internal model that extends the user's model with a ``row_id`` field.

    The ``row_id`` field is used to track which input row each output row corresponds
    to, enabling correct reordering after concurrent batch processing.

    Args:
        user_model: The user-defined Pydantic model for output rows.

    Returns:
        A new Pydantic model class with all fields from ``user_model`` plus ``row_id: int``.

    Raises:
        SmeltConfigError: If the user model already defines a ``row_id`` field.
    """
    if "row_id" in user_model.model_fields:
        raise SmeltConfigError(
            f"Output model '{user_model.__name__}' already has a 'row_id' field. "
            "Smelt reserves 'row_id' for internal row tracking. "
            "Please rename your field."
        )

    internal_model: Type[BaseModel] = create_model(
        f"_Smelt{user_model.__name__}",
        __base__=user_model,
        row_id=(int, Field(...)),
    )
    return internal_model


def create_batch_wrapper(internal_model: Type[BaseModel]) -> Type[BaseModel]:
    """Create a batch wrapper model containing a list of internal model rows.

    LangChain's ``with_structured_output`` requires a single Pydantic model,
    not a bare list. This wrapper provides the ``rows`` field needed to hold
    a batch of output rows.

    Args:
        internal_model: The internal model (user model + ``row_id``).

    Returns:
        A Pydantic model with a single field ``rows: list[internal_model]``.
    """
    batch_wrapper: Type[BaseModel] = create_model(
        "_SmeltBatch",
        rows=(list[internal_model], Field(...)),  # type: ignore[valid-type]
    )
    return batch_wrapper


def validate_batch_response(
    parsed: BaseModel,
    expected_row_ids: list[int],
) -> list[BaseModel]:
    """Validate that a parsed batch response contains exactly the expected rows.

    Checks that:
    - The number of returned rows matches the expected count.
    - All expected row IDs are present.
    - No duplicate row IDs exist.
    - No unexpected row IDs are included.

    Args:
        parsed: The parsed ``_SmeltBatch`` instance from the LLM response.
        expected_row_ids: The row IDs that should be present in this batch.

    Returns:
        The list of validated row model instances.

    Raises:
        SmeltValidationError: If any of the row ID invariants are violated.
    """
    rows: list[BaseModel] = parsed.rows  # type: ignore[attr-defined]
    returned_ids: list[int] = [row.row_id for row in rows]  # type: ignore[attr-defined]
    expected_set: set[int] = set(expected_row_ids)
    returned_set: set[int] = set(returned_ids)

    if len(returned_ids) != len(expected_row_ids):
        raise SmeltValidationError(
            f"Expected {len(expected_row_ids)} rows but got {len(returned_ids)}.",
            raw_response=parsed,
        )

    if len(returned_ids) != len(returned_set):
        duplicates: list[int] = [
            rid for rid in returned_ids if returned_ids.count(rid) > 1
        ]
        raise SmeltValidationError(
            f"Duplicate row IDs in response: {sorted(set(duplicates))}.",
            raw_response=parsed,
        )

    missing: set[int] = expected_set - returned_set
    if missing:
        raise SmeltValidationError(
            f"Missing row IDs in response: {sorted(missing)}.",
            raw_response=parsed,
        )

    unexpected: set[int] = returned_set - expected_set
    if unexpected:
        raise SmeltValidationError(
            f"Unexpected row IDs in response: {sorted(unexpected)}.",
            raw_response=parsed,
        )

    return rows


def strip_row_id(row: BaseModel, user_model: Type[T]) -> T:
    """Remove the ``row_id`` field and return a clean instance of the user's model.

    Args:
        row: An internal model instance (with ``row_id``).
        user_model: The original user-defined Pydantic model class.

    Returns:
        A new instance of ``user_model`` with only the user-defined fields.
    """
    data: dict[str, Any] = row.model_dump(exclude={"row_id"})
    return user_model.model_validate(data)
