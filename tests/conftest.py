"""Shared test fixtures and mock LLM for smelt tests."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Type
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel


class SampleOutput(BaseModel):
    """Sample output model used across tests."""

    category: str
    confidence: float


class MockChatModel(BaseChatModel):
    """A mock chat model that returns pre-defined structured responses.

    Configure ``responses`` as a list of dicts that will be returned
    sequentially on each call. Each dict should match the batch wrapper schema.

    Set ``errors`` to a list of exceptions to raise on specific calls (use
    ``None`` for calls that should succeed).
    """

    responses: list[dict[str, Any]] = []
    errors: list[Exception | None] = []
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("Use async methods for testing.")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx: int = self.call_count
        self.call_count += 1

        if idx < len(self.errors) and self.errors[idx] is not None:
            raise self.errors[idx]  # type: ignore[misc]

        message = AIMessage(
            content="mock response",
            usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def create_mock_structured_model(
    responses: list[Any],
    errors: list[Exception | None] | None = None,
) -> AsyncMock:
    """Create a mock for ``chat_model.with_structured_output().ainvoke()``.

    Args:
        responses: List of parsed Pydantic model instances to return.
        errors: Optional list of exceptions; ``None`` entries mean success.

    Returns:
        An ``AsyncMock`` configured to return structured output dicts.
    """
    call_idx: int = 0

    async def mock_ainvoke(messages: Sequence[BaseMessage], **kwargs: Any) -> dict[str, Any]:
        nonlocal call_idx
        idx: int = call_idx
        call_idx += 1

        if errors and idx < len(errors) and errors[idx] is not None:
            raise errors[idx]  # type: ignore[misc]

        parsed = responses[idx] if idx < len(responses) else responses[-1]
        raw_message = AIMessage(
            content="mock",
            usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )
        return {"raw": raw_message, "parsed": parsed, "parsing_error": None}

    mock = AsyncMock()
    mock.ainvoke = mock_ainvoke  # type: ignore[assignment]
    return mock


@pytest.fixture
def sample_data() -> list[dict[str, Any]]:
    """Sample input data for testing."""
    return [
        {"name": "Apple", "description": "Technology company"},
        {"name": "JPMorgan", "description": "Banking and financial services"},
        {"name": "Pfizer", "description": "Pharmaceutical company"},
    ]


@pytest.fixture
def sample_output_model() -> Type[SampleOutput]:
    """Sample Pydantic output model for testing."""
    return SampleOutput
