"""Job definition and entry points for smelt.

A ``Job`` encapsulates the transformation configuration — prompt, output model,
batching parameters, and retry policy — and exposes sync and async run methods.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from smelt.batch import execute_batches
from smelt.errors import SmeltConfigError
from smelt.model import Model
from smelt.types import SmeltResult

T = TypeVar("T", bound=BaseModel)


@dataclass
class Job:
    """A smelt transformation job.

    Defines what transformation to apply, how to validate output, and
    how to manage batching and retries.

    Attributes:
        prompt: The transformation instruction sent to the LLM.
        output_model: A Pydantic ``BaseModel`` subclass defining the expected
            output schema for each row.
        batch_size: Number of input rows per LLM request. Defaults to 10.
        concurrency: Maximum number of concurrent batch requests. Defaults to 3.
        max_retries: Maximum retry attempts per failed batch. Defaults to 3.
        stop_on_exhaustion: If ``True`` (default), raises ``SmeltExhaustionError``
            when any batch exhausts its retries. If ``False``, continues processing
            remaining batches and collects errors.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Classification(BaseModel):
        ...     category: str
        ...     confidence: float
        >>> job = Job(
        ...     prompt="Classify each company by industry sector",
        ...     output_model=Classification,
        ...     batch_size=20,
        ...     concurrency=3,
        ... )
    """

    prompt: str
    output_model: Type[BaseModel]
    batch_size: int = 10
    concurrency: int = 3
    max_retries: int = 3
    stop_on_exhaustion: bool = True

    def __post_init__(self) -> None:
        """Validate job configuration on initialization.

        Raises:
            SmeltConfigError: If any configuration value is invalid.
        """
        if not self.prompt or not self.prompt.strip():
            raise SmeltConfigError("Job prompt must be a non-empty string.")

        if not isinstance(self.output_model, type) or not issubclass(
            self.output_model, BaseModel
        ):
            raise SmeltConfigError(
                f"output_model must be a Pydantic BaseModel subclass, "
                f"got {type(self.output_model)!r}."
            )

        if self.batch_size < 1:
            raise SmeltConfigError(
                f"batch_size must be >= 1, got {self.batch_size}."
            )

        if self.concurrency < 1:
            raise SmeltConfigError(
                f"concurrency must be >= 1, got {self.concurrency}."
            )

        if self.max_retries < 0:
            raise SmeltConfigError(
                f"max_retries must be >= 0, got {self.max_retries}."
            )

        self._validated = True

    async def arun(self, model: Model, *, data: list[dict[str, Any]]) -> SmeltResult[Any]:
        """Run the transformation job asynchronously.

        Args:
            model: The :class:`~smelt.model.Model` configuration for the LLM.
            data: Input rows as a list of dictionaries.

        Returns:
            A :class:`~smelt.types.SmeltResult` containing transformed data,
            any errors, and run metrics.

        Raises:
            SmeltConfigError: If the model cannot be initialized.
            SmeltExhaustionError: If ``stop_on_exhaustion`` is ``True`` and
                a batch exhausts all retries.
        """
        chat_model = model.get_chat_model()

        return await execute_batches(
            chat_model=chat_model,
            user_prompt=self.prompt,
            output_model=self.output_model,
            data=data,
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            max_retries=self.max_retries,
            stop_on_exhaustion=self.stop_on_exhaustion,
        )

    def run(self, model: Model, *, data: list[dict[str, Any]]) -> SmeltResult[Any]:
        """Run the transformation job synchronously.

        Convenience wrapper around :meth:`arun` that creates an event loop
        if one is not already running.

        Args:
            model: The :class:`~smelt.model.Model` configuration for the LLM.
            data: Input rows as a list of dictionaries.

        Returns:
            A :class:`~smelt.types.SmeltResult` containing transformed data,
            any errors, and run metrics.

        Raises:
            RuntimeError: If called from within an already-running event loop.
                Use :meth:`arun` instead in async contexts.
            SmeltConfigError: If the model cannot be initialized.
            SmeltExhaustionError: If ``stop_on_exhaustion`` is ``True`` and
                a batch exhausts all retries.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            raise RuntimeError(
                "job.run() cannot be called from an async context. "
                "Use 'await job.arun(...)' instead."
            )

        return asyncio.run(self.arun(model, data=data))
