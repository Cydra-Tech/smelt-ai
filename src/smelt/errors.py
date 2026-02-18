"""Exception hierarchy for smelt.

All smelt-specific exceptions inherit from :class:`SmeltError`,
making it easy to catch any smelt failure with a single ``except`` clause.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from smelt.types import SmeltResult


class SmeltError(Exception):
    """Base exception for all smelt errors."""


class SmeltConfigError(SmeltError):
    """Raised when smelt configuration is invalid.

    Examples include an unresolvable model provider, invalid batch size,
    or a missing API key.
    """


class SmeltValidationError(SmeltError):
    """Raised when LLM output fails Pydantic validation.

    Attributes:
        raw_response: The raw LLM response that could not be validated.
    """

    def __init__(self, message: str, raw_response: Any = None) -> None:
        super().__init__(message)
        self.raw_response: Any = raw_response


class SmeltAPIError(SmeltError):
    """Raised when the LLM API returns a non-retriable error.

    Attributes:
        status_code: HTTP status code returned by the API, if available.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code: int | None = status_code


class SmeltExhaustionError(SmeltError):
    """Raised when a batch exhausts all retries and ``stop_on_exhaustion`` is enabled.

    Attributes:
        partial_result: The :class:`~smelt.types.SmeltResult` accumulated before
            the run was halted, including any successfully processed batches.
    """

    def __init__(self, message: str, partial_result: SmeltResult[Any]) -> None:
        super().__init__(message)
        self.partial_result: SmeltResult[Any] = partial_result
