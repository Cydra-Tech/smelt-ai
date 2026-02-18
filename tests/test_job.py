"""Tests for smelt.job module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smelt.errors import SmeltConfigError
from smelt.job import Job
from smelt.model import Model
from smelt.types import SmeltMetrics, SmeltResult
from tests.conftest import SampleOutput


class TestJobValidation:
    """Tests for Job.__post_init__ validation."""

    def test_valid_job(self) -> None:
        """Should create a job with valid parameters."""
        job = Job(prompt="classify", output_model=SampleOutput)
        assert job.prompt == "classify"
        assert job.output_model is SampleOutput

    def test_empty_prompt_raises(self) -> None:
        """Should reject empty prompt."""
        with pytest.raises(SmeltConfigError, match="non-empty string"):
            Job(prompt="", output_model=SampleOutput)

    def test_whitespace_prompt_raises(self) -> None:
        """Should reject whitespace-only prompt."""
        with pytest.raises(SmeltConfigError, match="non-empty string"):
            Job(prompt="   ", output_model=SampleOutput)

    def test_non_model_output_raises(self) -> None:
        """Should reject output_model that isn't a BaseModel subclass."""
        with pytest.raises(SmeltConfigError, match="BaseModel subclass"):
            Job(prompt="classify", output_model=dict)  # type: ignore[arg-type]

    def test_zero_batch_size_raises(self) -> None:
        """Should reject batch_size < 1."""
        with pytest.raises(SmeltConfigError, match="batch_size must be >= 1"):
            Job(prompt="classify", output_model=SampleOutput, batch_size=0)

    def test_negative_concurrency_raises(self) -> None:
        """Should reject concurrency < 1."""
        with pytest.raises(SmeltConfigError, match="concurrency must be >= 1"):
            Job(prompt="classify", output_model=SampleOutput, concurrency=-1)

    def test_negative_retries_raises(self) -> None:
        """Should reject max_retries < 0."""
        with pytest.raises(SmeltConfigError, match="max_retries must be >= 0"):
            Job(prompt="classify", output_model=SampleOutput, max_retries=-1)


class TestJobRun:
    """Tests for Job.run and Job.arun."""

    @pytest.mark.asyncio
    async def test_arun_calls_execute_batches(self) -> None:
        """arun should delegate to execute_batches with correct arguments."""
        job = Job(
            prompt="classify",
            output_model=SampleOutput,
            batch_size=5,
            concurrency=2,
            max_retries=1,
            stop_on_exhaustion=False,
        )

        mock_result: SmeltResult[SampleOutput] = SmeltResult(
            data=[SampleOutput(category="tech", confidence=0.9)],
            errors=[],
            metrics=SmeltMetrics(total_rows=1, successful_rows=1),
        )

        mock_model = MagicMock(spec=Model)
        mock_chat_model = MagicMock()
        mock_model.get_chat_model.return_value = mock_chat_model

        with patch("smelt.job.execute_batches", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_result
            result = await job.arun(mock_model, data=[{"name": "Apple"}])

            mock_exec.assert_called_once_with(
                chat_model=mock_chat_model,
                user_prompt="classify",
                output_model=SampleOutput,
                data=[{"name": "Apple"}],
                batch_size=5,
                concurrency=2,
                max_retries=1,
                shuffle=False,
                stop_on_exhaustion=False,
            )
            assert result.success
            assert len(result.data) == 1

    def test_run_sync_wrapper(self) -> None:
        """run should call arun via asyncio.run."""
        job = Job(prompt="classify", output_model=SampleOutput)

        mock_result: SmeltResult[SampleOutput] = SmeltResult(
            data=[SampleOutput(category="tech", confidence=0.9)],
        )

        mock_model = MagicMock(spec=Model)
        mock_chat_model = MagicMock()
        mock_model.get_chat_model.return_value = mock_chat_model

        with patch("smelt.job.execute_batches", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_result
            result = job.run(mock_model, data=[{"name": "Apple"}])

            assert result.success
            assert len(result.data) == 1

    @pytest.mark.asyncio
    async def test_run_from_async_raises(self) -> None:
        """run() should raise RuntimeError when called from async context."""
        job = Job(prompt="classify", output_model=SampleOutput)
        mock_model = MagicMock(spec=Model)

        with pytest.raises(RuntimeError, match="cannot be called from an async context"):
            job.run(mock_model, data=[])


class TestJobTest:
    """Tests for Job.test and Job.atest."""

    @pytest.mark.asyncio
    async def test_atest_uses_first_row_only(self) -> None:
        """atest should only send the first row to execute_batches."""
        job = Job(
            prompt="classify",
            output_model=SampleOutput,
            batch_size=20,
            concurrency=5,
            shuffle=True,
        )

        mock_result: SmeltResult[SampleOutput] = SmeltResult(
            data=[SampleOutput(category="tech", confidence=0.9)],
            errors=[],
            metrics=SmeltMetrics(total_rows=1, successful_rows=1),
        )

        mock_model = MagicMock(spec=Model)
        mock_chat_model = MagicMock()
        mock_model.get_chat_model.return_value = mock_chat_model

        with patch("smelt.job.execute_batches", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_result
            result = await job.atest(
                mock_model,
                data=[{"name": "Apple"}, {"name": "Google"}, {"name": "Pfizer"}],
            )

            mock_exec.assert_called_once_with(
                chat_model=mock_chat_model,
                user_prompt="classify",
                output_model=SampleOutput,
                data=[{"name": "Apple"}],
                batch_size=1,
                concurrency=1,
                max_retries=3,
                shuffle=False,
                stop_on_exhaustion=True,
            )
            assert result.success
            assert len(result.data) == 1

    @pytest.mark.asyncio
    async def test_atest_empty_data_raises(self) -> None:
        """atest should raise SmeltConfigError if data is empty."""
        job = Job(prompt="classify", output_model=SampleOutput)
        mock_model = MagicMock(spec=Model)

        with pytest.raises(SmeltConfigError, match="at least one row"):
            await job.atest(mock_model, data=[])

    def test_test_sync_wrapper(self) -> None:
        """test should call atest via asyncio.run."""
        job = Job(prompt="classify", output_model=SampleOutput)

        mock_result: SmeltResult[SampleOutput] = SmeltResult(
            data=[SampleOutput(category="tech", confidence=0.9)],
        )

        mock_model = MagicMock(spec=Model)
        mock_chat_model = MagicMock()
        mock_model.get_chat_model.return_value = mock_chat_model

        with patch("smelt.job.execute_batches", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_result
            result = job.test(
                mock_model,
                data=[{"name": "Apple"}, {"name": "Google"}],
            )

            # Should only pass first row
            call_data = mock_exec.call_args.kwargs["data"]
            assert call_data == [{"name": "Apple"}]
            assert result.success

    @pytest.mark.asyncio
    async def test_test_from_async_raises(self) -> None:
        """test() should raise RuntimeError when called from async context."""
        job = Job(prompt="classify", output_model=SampleOutput)
        mock_model = MagicMock(spec=Model)

        with pytest.raises(RuntimeError, match="cannot be called from an async context"):
            job.test(mock_model, data=[{"name": "Apple"}])
