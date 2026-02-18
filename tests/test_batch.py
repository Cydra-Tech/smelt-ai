"""Tests for smelt.batch module."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from smelt.batch import _compute_backoff, _is_retriable_error, _process_batch, execute_batches
from smelt.types import _TaggedRow
from smelt.validation import create_batch_wrapper, create_internal_model
from tests.conftest import SampleOutput, create_mock_structured_model


class TestIsRetriableError:
    """Tests for _is_retriable_error."""

    def test_retriable_status_codes(self) -> None:
        """429, 500, 502, 503, 504 should be retriable."""
        for code in [429, 500, 502, 503, 504]:
            exc = Exception("error")
            exc.status_code = code  # type: ignore[attr-defined]
            assert _is_retriable_error(exc) is True

    def test_non_retriable_status_codes(self) -> None:
        """400, 401, 403 should not be retriable."""
        for code in [400, 401, 403]:
            exc = Exception("error")
            exc.status_code = code  # type: ignore[attr-defined]
            assert _is_retriable_error(exc) is False

    def test_timeout_error_by_name(self) -> None:
        """Timeout-related exceptions should be retriable."""

        class TimeoutError(Exception):
            pass

        assert _is_retriable_error(TimeoutError("timed out")) is True

    def test_connection_error_by_message(self) -> None:
        """Connection errors in message should be retriable."""
        assert _is_retriable_error(Exception("connection refused")) is True

    def test_generic_error_not_retriable(self) -> None:
        """Generic exceptions should not be retriable."""
        assert _is_retriable_error(Exception("something broke")) is False


class TestComputeBackoff:
    """Tests for _compute_backoff."""

    def test_increases_exponentially(self) -> None:
        """Backoff should increase with each attempt."""
        b0 = _compute_backoff(0)
        b1 = _compute_backoff(1)
        b2 = _compute_backoff(2)
        assert b0 < b1 < b2

    def test_capped_at_max(self) -> None:
        """Backoff should never exceed 60 seconds."""
        result = _compute_backoff(100)
        assert result <= 60.0

    def test_non_negative(self) -> None:
        """Backoff should always be positive."""
        assert _compute_backoff(0) > 0


class TestProcessBatch:
    """Tests for _process_batch."""

    @pytest.fixture
    def internal_model(self) -> type[BaseModel]:
        """Create the internal model for SampleOutput."""
        return create_internal_model(SampleOutput)

    @pytest.fixture
    def batch_wrapper(self, internal_model: type[BaseModel]) -> type[BaseModel]:
        """Create the batch wrapper model."""
        return create_batch_wrapper(internal_model)

    @pytest.fixture
    def tagged_rows(self) -> list[_TaggedRow]:
        """Create sample tagged rows."""
        return [
            _TaggedRow(row_id=0, data={"name": "Apple"}),
            _TaggedRow(row_id=1, data={"name": "Google"}),
        ]

    @pytest.mark.asyncio
    async def test_successful_batch(
        self,
        batch_wrapper: type[BaseModel],
        tagged_rows: list[_TaggedRow],
    ) -> None:
        """Should return rows on successful LLM response."""
        parsed = batch_wrapper(rows=[
            {"row_id": 0, "category": "tech", "confidence": 0.9},
            {"row_id": 1, "category": "tech", "confidence": 0.85},
        ])
        mock_model = create_mock_structured_model([parsed])
        cancel = asyncio.Event()

        result = await _process_batch(
            structured_model=mock_model,
            system_message="sys",
            tagged_rows=tagged_rows,
            batch_index=0,
            max_retries=3,
            cancel_event=cancel,
        )

        assert result.error is None
        assert len(result.rows) == 2

    @pytest.mark.asyncio
    async def test_retry_then_succeed(
        self,
        batch_wrapper: type[BaseModel],
        tagged_rows: list[_TaggedRow],
    ) -> None:
        """Should retry on failure and succeed if subsequent attempt works."""
        parsed = batch_wrapper(rows=[
            {"row_id": 0, "category": "tech", "confidence": 0.9},
            {"row_id": 1, "category": "tech", "confidence": 0.85},
        ])
        error = Exception("rate limit")
        error.status_code = 429  # type: ignore[attr-defined]

        mock_model = create_mock_structured_model(
            responses=[None, parsed],  # type: ignore[list-item]
            errors=[error, None],
        )
        cancel = asyncio.Event()

        with patch("smelt.batch._compute_backoff", return_value=0.0):
            result = await _process_batch(
                structured_model=mock_model,
                system_message="sys",
                tagged_rows=tagged_rows,
                batch_index=0,
                max_retries=3,
                cancel_event=cancel,
            )

        assert result.error is None
        assert result.retries == 1

    @pytest.mark.asyncio
    async def test_exhaustion_returns_error(
        self,
        batch_wrapper: type[BaseModel],
        tagged_rows: list[_TaggedRow],
    ) -> None:
        """Should return BatchError after exhausting retries."""
        error = Exception("rate limit")
        error.status_code = 429  # type: ignore[attr-defined]

        mock_model = create_mock_structured_model(
            responses=[],
            errors=[error, error, error, error],
        )
        cancel = asyncio.Event()

        with patch("smelt.batch._compute_backoff", return_value=0.0):
            result = await _process_batch(
                structured_model=mock_model,
                system_message="sys",
                tagged_rows=tagged_rows,
                batch_index=0,
                max_retries=3,
                cancel_event=cancel,
            )

        assert result.error is not None
        assert result.error.error_type == "api"
        assert result.error.attempts == 4

    @pytest.mark.asyncio
    async def test_non_retriable_error_fails_immediately(
        self,
        batch_wrapper: type[BaseModel],
        tagged_rows: list[_TaggedRow],
    ) -> None:
        """Non-retriable errors should fail without retrying."""
        error = Exception("unauthorized")
        error.status_code = 401  # type: ignore[attr-defined]

        mock_model = create_mock_structured_model(
            responses=[],
            errors=[error],
        )
        cancel = asyncio.Event()

        result = await _process_batch(
            structured_model=mock_model,
            system_message="sys",
            tagged_rows=tagged_rows,
            batch_index=0,
            max_retries=3,
            cancel_event=cancel,
        )

        assert result.error is not None
        assert result.error.attempts == 1

    @pytest.mark.asyncio
    async def test_cancellation_short_circuits(
        self,
        batch_wrapper: type[BaseModel],
        tagged_rows: list[_TaggedRow],
    ) -> None:
        """Should return cancelled error when cancel event is set."""
        mock_model = create_mock_structured_model([])
        cancel = asyncio.Event()
        cancel.set()

        result = await _process_batch(
            structured_model=mock_model,
            system_message="sys",
            tagged_rows=tagged_rows,
            batch_index=0,
            max_retries=3,
            cancel_event=cancel,
        )

        assert result.error is not None
        assert result.error.error_type == "cancelled"


class TestExecuteBatches:
    """Tests for execute_batches."""

    def _make_mock_chat_model(
        self,
        responses: list[Any],
        errors: list[Exception | None] | None = None,
    ) -> MagicMock:
        """Create a mock chat model with with_structured_output support."""
        mock_structured = create_mock_structured_model(responses, errors)
        mock_chat = MagicMock()
        mock_chat.with_structured_output.return_value = mock_structured
        return mock_chat

    @pytest.mark.asyncio
    async def test_basic_execution(self) -> None:
        """Should process all rows and return ordered results."""
        internal = create_internal_model(SampleOutput)
        wrapper = create_batch_wrapper(internal)

        parsed = wrapper(rows=[
            {"row_id": 0, "category": "tech", "confidence": 0.9},
            {"row_id": 1, "category": "finance", "confidence": 0.8},
            {"row_id": 2, "category": "health", "confidence": 0.95},
        ])

        mock_chat = self._make_mock_chat_model([parsed])

        result = await execute_batches(
            chat_model=mock_chat,
            user_prompt="classify",
            output_model=SampleOutput,
            data=[
                {"name": "Apple"},
                {"name": "JPMorgan"},
                {"name": "Pfizer"},
            ],
            batch_size=10,
            concurrency=1,
            max_retries=3,
            shuffle=False,
            stop_on_exhaustion=False,
        )

        assert result.success
        assert len(result.data) == 3
        assert all(isinstance(r, SampleOutput) for r in result.data)
        assert result.data[0].category == "tech"
        assert result.data[1].category == "finance"
        assert result.data[2].category == "health"

    @pytest.mark.asyncio
    async def test_multiple_batches_ordered(self) -> None:
        """Results should be in original row order regardless of batch completion order."""
        internal = create_internal_model(SampleOutput)
        wrapper = create_batch_wrapper(internal)

        batch_0 = wrapper(rows=[
            {"row_id": 0, "category": "a", "confidence": 0.1},
            {"row_id": 1, "category": "b", "confidence": 0.2},
        ])
        batch_1 = wrapper(rows=[
            {"row_id": 2, "category": "c", "confidence": 0.3},
            {"row_id": 3, "category": "d", "confidence": 0.4},
        ])

        mock_chat = self._make_mock_chat_model([batch_0, batch_1])

        result = await execute_batches(
            chat_model=mock_chat,
            user_prompt="classify",
            output_model=SampleOutput,
            data=[{"n": "a"}, {"n": "b"}, {"n": "c"}, {"n": "d"}],
            batch_size=2,
            concurrency=2,
            max_retries=0,
            shuffle=False,
            stop_on_exhaustion=False,
        )

        assert result.success
        assert len(result.data) == 4
        assert [r.category for r in result.data] == ["a", "b", "c", "d"]

    @pytest.mark.asyncio
    async def test_metrics_populated(self) -> None:
        """Metrics should reflect the run accurately."""
        internal = create_internal_model(SampleOutput)
        wrapper = create_batch_wrapper(internal)

        parsed = wrapper(rows=[
            {"row_id": 0, "category": "tech", "confidence": 0.9},
        ])

        mock_chat = self._make_mock_chat_model([parsed])

        result = await execute_batches(
            chat_model=mock_chat,
            user_prompt="classify",
            output_model=SampleOutput,
            data=[{"name": "Apple"}],
            batch_size=10,
            concurrency=1,
            max_retries=0,
            shuffle=False,
            stop_on_exhaustion=False,
        )

        assert result.metrics.total_rows == 1
        assert result.metrics.successful_rows == 1
        assert result.metrics.total_batches == 1
        assert result.metrics.successful_batches == 1
        assert result.metrics.input_tokens == 100
        assert result.metrics.output_tokens == 50
        assert result.metrics.wall_time_seconds > 0
