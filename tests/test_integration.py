"""Integration tests for smelt end-to-end pipeline.

Tests the full flow from Job → batch engine → validation → result
using mocked LLM responses.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smelt import Job, Model, SmeltResult
from smelt.errors import SmeltExhaustionError
from smelt.validation import create_batch_wrapper, create_internal_model
from tests.conftest import SampleOutput, create_mock_structured_model


class TestEndToEnd:
    """End-to-end integration tests with mocked LLM."""

    def _setup_mock_model(
        self,
        responses: list[Any],
        errors: list[Exception | None] | None = None,
    ) -> MagicMock:
        """Create a mock Model whose chat model returns structured responses."""
        mock_structured = create_mock_structured_model(responses, errors)
        mock_chat_model = MagicMock()
        mock_chat_model.with_structured_output.return_value = mock_structured

        mock_model = MagicMock(spec=Model)
        mock_model.get_chat_model.return_value = mock_chat_model
        return mock_model

    def test_single_batch_sync(self) -> None:
        """Sync run with a single batch should return all rows."""
        internal = create_internal_model(SampleOutput)
        wrapper = create_batch_wrapper(internal)

        parsed = wrapper(rows=[
            {"row_id": 0, "category": "tech", "confidence": 0.95},
            {"row_id": 1, "category": "finance", "confidence": 0.88},
        ])

        mock_model = self._setup_mock_model([parsed])

        job = Job(
            prompt="Classify by industry",
            output_model=SampleOutput,
            batch_size=10,
            stop_on_exhaustion=False,
        )

        result = job.run(
            mock_model,
            data=[
                {"name": "Apple", "desc": "Tech company"},
                {"name": "JPMorgan", "desc": "Bank"},
            ],
        )

        assert result.success
        assert len(result.data) == 2
        assert result.data[0].category == "tech"
        assert result.data[1].category == "finance"
        assert result.metrics.total_rows == 2
        assert result.metrics.successful_rows == 2
        assert result.metrics.failed_rows == 0

    @pytest.mark.asyncio
    async def test_multi_batch_async(self) -> None:
        """Async run with multiple batches should reassemble in order."""
        internal = create_internal_model(SampleOutput)
        wrapper = create_batch_wrapper(internal)

        batch_0 = wrapper(rows=[
            {"row_id": 0, "category": "a", "confidence": 0.1},
        ])
        batch_1 = wrapper(rows=[
            {"row_id": 1, "category": "b", "confidence": 0.2},
        ])
        batch_2 = wrapper(rows=[
            {"row_id": 2, "category": "c", "confidence": 0.3},
        ])

        mock_model = self._setup_mock_model([batch_0, batch_1, batch_2])

        job = Job(
            prompt="Classify",
            output_model=SampleOutput,
            batch_size=1,
            concurrency=3,
            stop_on_exhaustion=False,
        )

        result = await job.arun(
            mock_model,
            data=[{"n": "x"}, {"n": "y"}, {"n": "z"}],
        )

        assert result.success
        assert len(result.data) == 3
        categories: list[str] = [r.category for r in result.data]
        assert categories == ["a", "b", "c"]

    def test_stop_on_exhaustion(self) -> None:
        """Should raise SmeltExhaustionError with partial results."""
        error = Exception("server error")
        error.status_code = 500  # type: ignore[attr-defined]

        mock_structured = AsyncMock()

        async def always_fail(messages: Any, **kwargs: Any) -> dict[str, Any]:
            raise error

        mock_structured.ainvoke = always_fail

        mock_chat_model = MagicMock()
        mock_chat_model.with_structured_output.return_value = mock_structured

        mock_model = MagicMock(spec=Model)
        mock_model.get_chat_model.return_value = mock_chat_model

        job = Job(
            prompt="Classify",
            output_model=SampleOutput,
            batch_size=10,
            max_retries=1,
            stop_on_exhaustion=True,
        )

        with pytest.raises(SmeltExhaustionError) as exc_info:
            with patch("smelt.batch._compute_backoff", return_value=0.0):
                job.run(mock_model, data=[{"name": "Apple"}])

        assert exc_info.value.partial_result is not None
        assert len(exc_info.value.partial_result.errors) > 0

    def test_empty_data(self) -> None:
        """Should handle empty input gracefully."""
        mock_chat_model = MagicMock()
        mock_chat_model.with_structured_output.return_value = MagicMock()

        mock_model = MagicMock(spec=Model)
        mock_model.get_chat_model.return_value = mock_chat_model

        job = Job(
            prompt="Classify",
            output_model=SampleOutput,
            stop_on_exhaustion=False,
        )

        result = job.run(mock_model, data=[])
        assert result.success
        assert len(result.data) == 0
        assert result.metrics.total_rows == 0

    def test_result_success_property(self) -> None:
        """SmeltResult.success should reflect error state."""
        from smelt.types import BatchError

        result_ok: SmeltResult[SampleOutput] = SmeltResult(
            data=[SampleOutput(category="x", confidence=0.5)],
        )
        assert result_ok.success is True

        result_bad: SmeltResult[SampleOutput] = SmeltResult(
            data=[],
            errors=[
                BatchError(
                    batch_index=0,
                    row_ids=(0,),
                    error_type="api",
                    message="fail",
                    attempts=3,
                )
            ],
        )
        assert result_bad.success is False
