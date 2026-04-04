"""Tests for smelt.aggregate module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from smelt.aggregate import (
    _AggregateTextOutput,
    _extract_final_output,
    _serialize_output,
    execute_aggregate,
)
from smelt.errors import SmeltExhaustionError
from tests.conftest import create_mock_structured_model


# ---------------------------------------------------------------------------
# Test output models
# ---------------------------------------------------------------------------


class SummaryModel(BaseModel):
    """Simple aggregate output model for tests."""

    total_items: int = Field(description="Total number of items")
    categories: list[str] = Field(description="List of unique categories")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_chat_model(
    responses: list[Any],
    errors: list[Exception | None] | None = None,
) -> MagicMock:
    """Create a mock chat model with with_structured_output support."""
    mock_structured = create_mock_structured_model(responses, errors)
    mock_chat = MagicMock()
    mock_chat.with_structured_output.return_value = mock_structured
    return mock_chat


# ---------------------------------------------------------------------------
# TestSerializeOutput
# ---------------------------------------------------------------------------


class TestSerializeOutput:
    """Tests for _serialize_output."""

    def test_structured_mode_returns_json(self) -> None:
        """Structured mode should return JSON string."""
        output = SummaryModel(total_items=5, categories=["tech", "finance"])
        result = _serialize_output(output, text_mode=False)
        assert '"total_items": 5' in result
        assert '"tech"' in result

    def test_text_mode_returns_text(self) -> None:
        """Text mode should return the text field."""
        output = _AggregateTextOutput(text="Hello world")
        result = _serialize_output(output, text_mode=True)
        assert result == "Hello world"


class TestExtractFinalOutput:
    """Tests for _extract_final_output."""

    def test_structured_mode_returns_model(self) -> None:
        """Structured mode should return the model instance directly."""
        output = SummaryModel(total_items=5, categories=["tech"])
        result = _extract_final_output(output, text_mode=False)
        assert isinstance(result, SummaryModel)
        assert result.total_items == 5

    def test_text_mode_returns_string(self) -> None:
        """Text mode should return the text field as a string."""
        output = _AggregateTextOutput(text="Summary text")
        result = _extract_final_output(output, text_mode=True)
        assert result == "Summary text"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestExecuteAggregate
# ---------------------------------------------------------------------------


class TestExecuteAggregate:
    """Tests for execute_aggregate."""

    @pytest.mark.asyncio
    async def test_single_batch_structured(self) -> None:
        """Single batch should return one structured result without merging."""
        parsed = SummaryModel(total_items=3, categories=["tech", "finance"])
        mock_chat = _make_mock_chat_model([parsed])

        result = await execute_aggregate(
            chat_model=mock_chat,
            user_prompt="Summarize",
            output_model=SummaryModel,
            data=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            batch_size=10,
            concurrency=1,
            max_retries=0,
            stop_on_exhaustion=False,
        )

        assert result.success
        assert len(result.data) == 1
        assert isinstance(result.data[0], SummaryModel)
        assert result.data[0].total_items == 3
        assert result.metrics.total_rows == 3

    @pytest.mark.asyncio
    async def test_single_batch_text_mode(self) -> None:
        """Single batch in text mode should return one string."""
        parsed = _AggregateTextOutput(text="All three are companies.")
        mock_chat = _make_mock_chat_model([parsed])

        result = await execute_aggregate(
            chat_model=mock_chat,
            user_prompt="Summarize",
            output_model=None,
            data=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            batch_size=10,
            concurrency=1,
            max_retries=0,
            stop_on_exhaustion=False,
        )

        assert result.success
        assert len(result.data) == 1
        assert result.data[0] == "All three are companies."

    @pytest.mark.asyncio
    async def test_multi_batch_with_merge(self) -> None:
        """Multiple batches should map then merge into one result."""
        # 4 rows, batch_size=2 → 2 map outputs → 1 merge
        map_out_1 = SummaryModel(total_items=2, categories=["tech"])
        map_out_2 = SummaryModel(total_items=2, categories=["finance"])
        merge_out = SummaryModel(total_items=4, categories=["tech", "finance"])

        mock_chat = _make_mock_chat_model([map_out_1, map_out_2, merge_out])

        result = await execute_aggregate(
            chat_model=mock_chat,
            user_prompt="Summarize",
            output_model=SummaryModel,
            data=[{"n": "a"}, {"n": "b"}, {"n": "c"}, {"n": "d"}],
            batch_size=2,
            concurrency=2,
            max_retries=0,
            stop_on_exhaustion=False,
        )

        assert result.success
        assert len(result.data) == 1
        assert result.data[0].total_items == 4
        assert result.data[0].categories == ["tech", "finance"]

    @pytest.mark.asyncio
    async def test_three_batches_with_odd_merge(self) -> None:
        """3 batches: 2 merge first, odd one passes through, then final merge."""
        map_1 = SummaryModel(total_items=2, categories=["a"])
        map_2 = SummaryModel(total_items=2, categories=["b"])
        map_3 = SummaryModel(total_items=1, categories=["c"])
        merge_12 = SummaryModel(total_items=4, categories=["a", "b"])
        merge_final = SummaryModel(total_items=5, categories=["a", "b", "c"])

        mock_chat = _make_mock_chat_model([map_1, map_2, map_3, merge_12, merge_final])

        result = await execute_aggregate(
            chat_model=mock_chat,
            user_prompt="Summarize",
            output_model=SummaryModel,
            data=[{"n": i} for i in range(5)],
            batch_size=2,
            concurrency=3,
            max_retries=0,
            stop_on_exhaustion=False,
        )

        assert result.success
        assert len(result.data) == 1
        assert result.data[0].total_items == 5

    @pytest.mark.asyncio
    async def test_empty_data(self) -> None:
        """Empty data should return empty result."""
        mock_chat = MagicMock()
        mock_chat.with_structured_output.return_value = MagicMock()

        result = await execute_aggregate(
            chat_model=mock_chat,
            user_prompt="Summarize",
            output_model=SummaryModel,
            data=[],
            batch_size=10,
            concurrency=1,
            max_retries=0,
            stop_on_exhaustion=False,
        )

        assert result.success
        assert result.data == []
        assert result.metrics.total_rows == 0

    @pytest.mark.asyncio
    async def test_metrics_populated(self) -> None:
        """Metrics should track tokens, steps, and timing."""
        parsed = SummaryModel(total_items=1, categories=["tech"])
        mock_chat = _make_mock_chat_model([parsed])

        result = await execute_aggregate(
            chat_model=mock_chat,
            user_prompt="Summarize",
            output_model=SummaryModel,
            data=[{"name": "Apple"}],
            batch_size=10,
            concurrency=1,
            max_retries=0,
            stop_on_exhaustion=False,
        )

        assert result.metrics.total_rows == 1
        assert result.metrics.successful_rows == 1
        assert result.metrics.input_tokens == 100
        assert result.metrics.output_tokens == 50
        assert result.metrics.wall_time_seconds >= 0

    @pytest.mark.asyncio
    async def test_map_error_stop_on_exhaustion(self) -> None:
        """Map step failure with stop_on_exhaustion should raise."""
        error = Exception("server error")
        error.status_code = 500  # type: ignore[attr-defined]

        mock_chat = _make_mock_chat_model([], errors=[error, error])

        with pytest.raises(SmeltExhaustionError):
            with patch("smelt.aggregate._compute_backoff", return_value=0.0):
                await execute_aggregate(
                    chat_model=mock_chat,
                    user_prompt="Summarize",
                    output_model=SummaryModel,
                    data=[{"n": "a"}],
                    batch_size=10,
                    concurrency=1,
                    max_retries=1,
                    stop_on_exhaustion=True,
                )

    @pytest.mark.asyncio
    async def test_map_error_no_stop(self) -> None:
        """Map step failure without stop_on_exhaustion should return errors."""
        error = Exception("server error")
        error.status_code = 500  # type: ignore[attr-defined]

        mock_chat = _make_mock_chat_model([], errors=[error, error])

        with patch("smelt.aggregate._compute_backoff", return_value=0.0):
            result = await execute_aggregate(
                chat_model=mock_chat,
                user_prompt="Summarize",
                output_model=SummaryModel,
                data=[{"n": "a"}],
                batch_size=10,
                concurrency=1,
                max_retries=1,
                stop_on_exhaustion=False,
            )

        assert not result.success
        assert len(result.errors) == 1
        assert result.data == []

    @pytest.mark.asyncio
    async def test_non_retriable_error(self) -> None:
        """Non-retriable error should fail immediately."""
        error = Exception("unauthorized")
        error.status_code = 401  # type: ignore[attr-defined]

        mock_chat = _make_mock_chat_model([], errors=[error])

        result = await execute_aggregate(
            chat_model=mock_chat,
            user_prompt="Summarize",
            output_model=SummaryModel,
            data=[{"n": "a"}],
            batch_size=10,
            concurrency=1,
            max_retries=3,
            stop_on_exhaustion=False,
        )

        assert not result.success
        assert result.errors[0].attempts == 1

    @pytest.mark.asyncio
    async def test_text_mode_multi_batch_merge(self) -> None:
        """Text mode with multiple batches should merge into one string."""
        map_1 = _AggregateTextOutput(text="Tech companies are dominant.")
        map_2 = _AggregateTextOutput(text="Finance sector is stable.")
        merge_out = _AggregateTextOutput(text="Tech dominates; finance is stable.")

        mock_chat = _make_mock_chat_model([map_1, map_2, merge_out])

        result = await execute_aggregate(
            chat_model=mock_chat,
            user_prompt="Summarize",
            output_model=None,
            data=[{"n": "a"}, {"n": "b"}, {"n": "c"}, {"n": "d"}],
            batch_size=2,
            concurrency=2,
            max_retries=0,
            stop_on_exhaustion=False,
        )

        assert result.success
        assert len(result.data) == 1
        assert isinstance(result.data[0], str)
        assert "Tech" in result.data[0]

    @pytest.mark.asyncio
    async def test_uses_output_model_directly(self) -> None:
        """Should call with_structured_output with user model directly (no wrapper)."""
        parsed = SummaryModel(total_items=1, categories=[])
        mock_chat = _make_mock_chat_model([parsed])

        await execute_aggregate(
            chat_model=mock_chat,
            user_prompt="Summarize",
            output_model=SummaryModel,
            data=[{"n": "a"}],
            batch_size=10,
            concurrency=1,
            max_retries=0,
            stop_on_exhaustion=False,
        )

        mock_chat.with_structured_output.assert_called_once()
        call_args = mock_chat.with_structured_output.call_args
        assert call_args[0][0] is SummaryModel

    @pytest.mark.asyncio
    async def test_merge_phase_error_stop_on_exhaustion(self) -> None:
        """Merge step failure with stop_on_exhaustion should raise."""
        # 4 rows, batch_size=2 → 2 map outputs → 1 merge (which fails)
        map_1 = SummaryModel(total_items=2, categories=["a"])
        map_2 = SummaryModel(total_items=2, categories=["b"])
        merge_error = Exception("server error")
        merge_error.status_code = 500  # type: ignore[attr-defined]

        mock_chat = _make_mock_chat_model(
            [map_1, map_2],
            errors=[None, None, merge_error, merge_error],
        )

        with pytest.raises(SmeltExhaustionError):
            with patch("smelt.aggregate._compute_backoff", return_value=0.0):
                await execute_aggregate(
                    chat_model=mock_chat,
                    user_prompt="Summarize",
                    output_model=SummaryModel,
                    data=[{"n": "a"}, {"n": "b"}, {"n": "c"}, {"n": "d"}],
                    batch_size=2,
                    concurrency=2,
                    max_retries=1,
                    stop_on_exhaustion=True,
                )

    @pytest.mark.asyncio
    async def test_merge_phase_error_no_stop(self) -> None:
        """Merge step failure without stop_on_exhaustion should return errors."""
        map_1 = SummaryModel(total_items=2, categories=["a"])
        map_2 = SummaryModel(total_items=2, categories=["b"])
        merge_error = Exception("server error")
        merge_error.status_code = 500  # type: ignore[attr-defined]

        mock_chat = _make_mock_chat_model(
            [map_1, map_2],
            errors=[None, None, merge_error, merge_error],
        )

        with patch("smelt.aggregate._compute_backoff", return_value=0.0):
            result = await execute_aggregate(
                chat_model=mock_chat,
                user_prompt="Summarize",
                output_model=SummaryModel,
                data=[{"n": "a"}, {"n": "b"}, {"n": "c"}, {"n": "d"}],
                batch_size=2,
                concurrency=2,
                max_retries=1,
                stop_on_exhaustion=False,
            )

        assert not result.success
        assert len(result.errors) == 1
        assert result.data == []

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self) -> None:
        """Should retry on transient error and succeed."""
        retriable_error = Exception("rate limit")
        retriable_error.status_code = 429  # type: ignore[attr-defined]
        success = SummaryModel(total_items=1, categories=["tech"])

        mock_chat = _make_mock_chat_model(
            [None, success],  # type: ignore[list-item]
            errors=[retriable_error, None],
        )

        with patch("smelt.aggregate._compute_backoff", return_value=0.0):
            result = await execute_aggregate(
                chat_model=mock_chat,
                user_prompt="Summarize",
                output_model=SummaryModel,
                data=[{"n": "a"}],
                batch_size=10,
                concurrency=1,
                max_retries=3,
                stop_on_exhaustion=False,
            )

        assert result.success
        assert result.data[0].total_items == 1
        assert result.metrics.total_retries == 1

    @pytest.mark.asyncio
    async def test_multi_batch_metrics(self) -> None:
        """Metrics should accumulate tokens from both map and merge phases."""
        map_1 = SummaryModel(total_items=2, categories=["a"])
        map_2 = SummaryModel(total_items=2, categories=["b"])
        merge_out = SummaryModel(total_items=4, categories=["a", "b"])

        mock_chat = _make_mock_chat_model([map_1, map_2, merge_out])

        result = await execute_aggregate(
            chat_model=mock_chat,
            user_prompt="Summarize",
            output_model=SummaryModel,
            data=[{"n": "a"}, {"n": "b"}, {"n": "c"}, {"n": "d"}],
            batch_size=2,
            concurrency=2,
            max_retries=0,
            stop_on_exhaustion=False,
        )

        assert result.success
        # 2 map steps + 1 merge step = 3 total steps
        assert result.metrics.total_batches == 3
        assert result.metrics.successful_batches == 3
        # Each step uses 100 input + 50 output tokens (from mock)
        assert result.metrics.input_tokens == 300
        assert result.metrics.output_tokens == 150

    @pytest.mark.asyncio
    async def test_partial_map_failure(self) -> None:
        """When some map steps succeed and some fail, should return empty data."""
        success = SummaryModel(total_items=2, categories=["a"])
        error = Exception("server error")
        error.status_code = 500  # type: ignore[attr-defined]

        mock_chat = _make_mock_chat_model(
            [success],
            errors=[None, error, error],
        )

        with patch("smelt.aggregate._compute_backoff", return_value=0.0):
            result = await execute_aggregate(
                chat_model=mock_chat,
                user_prompt="Summarize",
                output_model=SummaryModel,
                data=[{"n": "a"}, {"n": "b"}, {"n": "c"}, {"n": "d"}],
                batch_size=2,
                concurrency=1,
                max_retries=1,
                stop_on_exhaustion=False,
            )

        assert not result.success
        assert result.data == []
        assert len(result.errors) == 1
