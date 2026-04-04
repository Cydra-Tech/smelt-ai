"""AggregateJob definition and entry points for smelt.

An ``AggregateJob`` processes all input rows into a single output via
tree-reduction: map each batch in parallel, then merge pairwise until
one result remains.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Type

from pydantic import BaseModel

from smelt.aggregate import execute_aggregate
from smelt.errors import SmeltConfigError
from smelt.model import Model
from smelt.types import SmeltResult


@dataclass
class AggregateJob:
    """A smelt aggregation job.

    Processes all input rows into a single aggregated output using
    tree-parallel reduction. Each batch is mapped to one output in parallel,
    then outputs are merged pairwise until one final result remains.

    Attributes:
        prompt: The aggregation instruction sent to the LLM.
        output_model: A Pydantic ``BaseModel`` subclass defining the expected
            aggregate output schema. When ``None``, operates in free-text
            mode and returns ``SmeltResult[str]`` with a single string.
        batch_size: Number of input rows per map step. Defaults to 50.
        concurrency: Maximum number of concurrent map/merge steps. Defaults to 3.
        max_retries: Maximum retry attempts per failed step. Defaults to 3.
        stop_on_exhaustion: If ``True`` (default), raises ``SmeltExhaustionError``
            when any step exhausts its retries.

    Examples:
        Structured aggregation:

        >>> from pydantic import BaseModel, Field
        >>> class Summary(BaseModel):
        ...     total_items: int
        ...     categories: list[str]
        >>> job = AggregateJob(
        ...     prompt="Summarize the dataset",
        ...     output_model=Summary,
        ...     batch_size=20,
        ... )
        >>> result = job.run(model, data=large_dataset)
        >>> result.data[0]  # Summary(total_items=500, categories=[...])

        Free-text aggregation:

        >>> job = AggregateJob(
        ...     prompt="Write an executive summary of all companies",
        ... )
        >>> result = job.run(model, data=companies)
        >>> result.data[0]  # "The portfolio consists of..."
    """

    prompt: str
    output_model: Type[BaseModel] | None = field(default=None)
    batch_size: int = 50
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

        if self.output_model is not None and (
            not isinstance(self.output_model, type)
            or not issubclass(self.output_model, BaseModel)
        ):
            raise SmeltConfigError(
                f"output_model must be a Pydantic BaseModel subclass or None, "
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

    async def arun(self, model: Model, *, data: list[dict[str, Any]]) -> SmeltResult[Any]:
        """Run the aggregation job asynchronously.

        Args:
            model: The :class:`~smelt.model.Model` configuration for the LLM.
            data: Input rows as a list of dictionaries.

        Returns:
            A :class:`~smelt.types.SmeltResult` with a single-element ``data``
            list containing the aggregate output.

        Raises:
            SmeltConfigError: If the model cannot be initialized.
            SmeltExhaustionError: If ``stop_on_exhaustion`` is ``True`` and
                any step exhausts all retries.
        """
        chat_model = model.get_chat_model()

        return await execute_aggregate(
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
        """Run the aggregation job synchronously.

        Convenience wrapper around :meth:`arun`.

        Args:
            model: The :class:`~smelt.model.Model` configuration for the LLM.
            data: Input rows as a list of dictionaries.

        Returns:
            A :class:`~smelt.types.SmeltResult` with a single-element ``data``
            list containing the aggregate output.

        Raises:
            RuntimeError: If called from within an already-running event loop.
                Use :meth:`arun` instead in async contexts.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            raise RuntimeError(
                "aggregate_job.run() cannot be called from an async context. "
                "Use 'await aggregate_job.arun(...)' instead."
            )

        return asyncio.run(self.arun(model, data=data))
