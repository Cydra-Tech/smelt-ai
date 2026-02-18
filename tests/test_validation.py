"""Tests for smelt.validation module."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from smelt.errors import SmeltConfigError, SmeltValidationError
from smelt.validation import (
    create_batch_wrapper,
    create_internal_model,
    strip_row_id,
    validate_batch_response,
)


class SimpleModel(BaseModel):
    """Simple model for testing."""

    name: str
    score: float


class ModelWithRowId(BaseModel):
    """Model that conflicts with smelt's internal row_id field."""

    row_id: int
    value: str


class TestCreateInternalModel:
    """Tests for create_internal_model."""

    def test_adds_row_id_field(self) -> None:
        """Internal model should have row_id plus all user fields."""
        internal = create_internal_model(SimpleModel)
        assert "row_id" in internal.model_fields
        assert "name" in internal.model_fields
        assert "score" in internal.model_fields

    def test_row_id_is_required_int(self) -> None:
        """The row_id field should be a required integer."""
        internal = create_internal_model(SimpleModel)
        field = internal.model_fields["row_id"]
        assert field.annotation is int
        assert field.is_required()

    def test_raises_on_existing_row_id(self) -> None:
        """Should raise SmeltConfigError if user model already has row_id."""
        with pytest.raises(SmeltConfigError, match="already has a 'row_id' field"):
            create_internal_model(ModelWithRowId)

    def test_internal_model_validates_correctly(self) -> None:
        """Internal model should accept valid data with row_id."""
        internal = create_internal_model(SimpleModel)
        instance = internal(row_id=0, name="test", score=0.9)
        assert instance.row_id == 0  # type: ignore[attr-defined]
        assert instance.name == "test"  # type: ignore[attr-defined]

    def test_inherits_user_validators(self) -> None:
        """Internal model should preserve user model validation."""
        internal = create_internal_model(SimpleModel)
        with pytest.raises(Exception):
            internal(row_id=0, name="test", score="not_a_float_or_numeric")  # type: ignore[arg-type]


class TestCreateBatchWrapper:
    """Tests for create_batch_wrapper."""

    def test_has_rows_field(self) -> None:
        """Batch wrapper should have a 'rows' field."""
        internal = create_internal_model(SimpleModel)
        wrapper = create_batch_wrapper(internal)
        assert "rows" in wrapper.model_fields

    def test_accepts_list_of_internal_models(self) -> None:
        """Batch wrapper should accept a list of internal model instances."""
        internal = create_internal_model(SimpleModel)
        wrapper = create_batch_wrapper(internal)
        instance = wrapper(rows=[{"row_id": 0, "name": "a", "score": 1.0}])
        assert len(instance.rows) == 1  # type: ignore[attr-defined]


class TestValidateBatchResponse:
    """Tests for validate_batch_response."""

    def _make_parsed(self, rows_data: list[dict]) -> BaseModel:
        """Helper to create a parsed batch response."""
        internal = create_internal_model(SimpleModel)
        wrapper = create_batch_wrapper(internal)
        return wrapper(rows=rows_data)

    def test_valid_response(self) -> None:
        """Should return rows when response is valid."""
        parsed = self._make_parsed([
            {"row_id": 0, "name": "a", "score": 1.0},
            {"row_id": 1, "name": "b", "score": 0.5},
        ])
        rows = validate_batch_response(parsed, [0, 1])
        assert len(rows) == 2

    def test_wrong_count(self) -> None:
        """Should raise SmeltValidationError when row count doesn't match."""
        parsed = self._make_parsed([
            {"row_id": 0, "name": "a", "score": 1.0},
        ])
        with pytest.raises(SmeltValidationError, match="Expected 2 rows but got 1"):
            validate_batch_response(parsed, [0, 1])

    def test_duplicate_ids(self) -> None:
        """Should raise SmeltValidationError on duplicate row IDs."""
        parsed = self._make_parsed([
            {"row_id": 0, "name": "a", "score": 1.0},
            {"row_id": 0, "name": "b", "score": 0.5},
        ])
        with pytest.raises(SmeltValidationError, match="Duplicate row IDs"):
            validate_batch_response(parsed, [0, 1])

    def test_missing_ids(self) -> None:
        """Should raise SmeltValidationError when expected IDs are missing."""
        parsed = self._make_parsed([
            {"row_id": 0, "name": "a", "score": 1.0},
            {"row_id": 2, "name": "b", "score": 0.5},
        ])
        with pytest.raises(SmeltValidationError, match="Missing row IDs"):
            validate_batch_response(parsed, [0, 1])

    def test_unexpected_ids(self) -> None:
        """Should raise SmeltValidationError when IDs don't match expected set.

        When count matches but IDs differ, the missing-ID check fires first
        since missing and unexpected IDs are complementary.
        """
        parsed = self._make_parsed([
            {"row_id": 0, "name": "a", "score": 1.0},
            {"row_id": 5, "name": "b", "score": 0.5},
        ])
        with pytest.raises(SmeltValidationError, match="Missing row IDs"):
            validate_batch_response(parsed, [0, 1])


class TestStripRowId:
    """Tests for strip_row_id."""

    def test_returns_user_model_instance(self) -> None:
        """Should return an instance of the user's model without row_id."""
        internal = create_internal_model(SimpleModel)
        row = internal(row_id=42, name="test", score=0.8)
        result = strip_row_id(row, SimpleModel)

        assert isinstance(result, SimpleModel)
        assert result.name == "test"
        assert result.score == 0.8
        assert not hasattr(result, "row_id") or "row_id" not in result.model_fields
