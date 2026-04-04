"""Tests for smelt.aggregate_job module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from smelt.aggregate_job import AggregateJob
from smelt.errors import SmeltConfigError
from smelt.model import Model
from smelt.types import SmeltMetrics, SmeltResult


class SummaryModel(BaseModel):
    """Simple aggregate output model for tests."""

    total_items: int
    categories: list[str]


class TestAggregateJobValidation:
    """Tests for AggregateJob.__post_init__ validation."""

    def test_valid_structured_job(self) -> None:
        """Should create an aggregate job with valid parameters."""
        job = AggregateJob(prompt="summarize", output_model=SummaryModel)
        assert job.prompt == "summarize"
        assert job.output_model is SummaryModel

    def test_valid_text_job(self) -> None:
        """Should create an aggregate job with output_model=None."""
        job = AggregateJob(prompt="summarize")
        assert job.output_model is None

    def test_default_output_model_is_none(self) -> None:
        """output_model should default to None."""
        job = AggregateJob(prompt="summarize")
        assert job.output_model is None

    def test_default_batch_size_is_50(self) -> None:
        """Default batch_size should be 50 for aggregate."""
        job = AggregateJob(prompt="summarize")
        assert job.batch_size == 50

    def test_empty_prompt_raises(self) -> None:
        """Should reject empty prompt."""
        with pytest.raises(SmeltConfigError, match="non-empty string"):
            AggregateJob(prompt="")

    def test_non_model_output_raises(self) -> None:
        """Should reject output_model that isn't BaseModel or None."""
        with pytest.raises(SmeltConfigError, match="BaseModel subclass or None"):
            AggregateJob(prompt="summarize", output_model=dict)  # type: ignore[arg-type]

    def test_zero_batch_size_raises(self) -> None:
        """Should reject batch_size < 1."""
        with pytest.raises(SmeltConfigError, match="batch_size must be >= 1"):
            AggregateJob(prompt="summarize", batch_size=0)

    def test_negative_concurrency_raises(self) -> None:
        """Should reject concurrency < 1."""
        with pytest.raises(SmeltConfigError, match="concurrency must be >= 1"):
            AggregateJob(prompt="summarize", concurrency=-1)

    def test_negative_retries_raises(self) -> None:
        """Should reject max_retries < 0."""
        with pytest.raises(SmeltConfigError, match="max_retries must be >= 0"):
            AggregateJob(prompt="summarize", max_retries=-1)


class TestAggregateJobRun:
    """Tests for AggregateJob.run and AggregateJob.arun."""

    @pytest.mark.asyncio
    async def test_arun_delegates_to_execute_aggregate(self) -> None:
        """arun should delegate to execute_aggregate with correct arguments."""
        job = AggregateJob(
            prompt="summarize",
            output_model=SummaryModel,
            batch_size=20,
            concurrency=2,
            max_retries=1,
            stop_on_exhaustion=False,
        )

        mock_result: SmeltResult[SummaryModel] = SmeltResult(
            data=[SummaryModel(total_items=5, categories=["tech"])],
            errors=[],
            metrics=SmeltMetrics(total_rows=5, successful_rows=5),
        )

        mock_model = MagicMock(spec=Model)
        mock_chat_model = MagicMock()
        mock_model.get_chat_model.return_value = mock_chat_model

        with patch("smelt.aggregate_job.execute_aggregate", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_result
            result = await job.arun(mock_model, data=[{"n": "a"}] * 5)

            mock_exec.assert_called_once_with(
                chat_model=mock_chat_model,
                user_prompt="summarize",
                output_model=SummaryModel,
                data=[{"n": "a"}] * 5,
                batch_size=20,
                concurrency=2,
                max_retries=1,
                stop_on_exhaustion=False,
            )
            assert result.success

    def test_run_sync_wrapper(self) -> None:
        """run should call arun via asyncio.run."""
        job = AggregateJob(prompt="summarize", output_model=SummaryModel)

        mock_result: SmeltResult[SummaryModel] = SmeltResult(
            data=[SummaryModel(total_items=1, categories=[])],
        )

        mock_model = MagicMock(spec=Model)
        mock_chat_model = MagicMock()
        mock_model.get_chat_model.return_value = mock_chat_model

        with patch("smelt.aggregate_job.execute_aggregate", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_result
            result = job.run(mock_model, data=[{"n": "a"}])

            assert result.success
            assert len(result.data) == 1

    @pytest.mark.asyncio
    async def test_run_from_async_raises(self) -> None:
        """run() should raise RuntimeError when called from async context."""
        job = AggregateJob(prompt="summarize")
        mock_model = MagicMock(spec=Model)

        with pytest.raises(RuntimeError, match="cannot be called from an async context"):
            job.run(mock_model, data=[])

    @pytest.mark.asyncio
    async def test_arun_text_mode(self) -> None:
        """arun with output_model=None should pass None to execute_aggregate."""
        job = AggregateJob(prompt="summarize")

        mock_result: SmeltResult[str] = SmeltResult(
            data=["Summary text"],
            errors=[],
            metrics=SmeltMetrics(total_rows=1, successful_rows=1),
        )

        mock_model = MagicMock(spec=Model)
        mock_chat_model = MagicMock()
        mock_model.get_chat_model.return_value = mock_chat_model

        with patch("smelt.aggregate_job.execute_aggregate", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_result
            result = await job.arun(mock_model, data=[{"n": "a"}])

            assert mock_exec.call_args.kwargs["output_model"] is None
            assert result.data == ["Summary text"]
