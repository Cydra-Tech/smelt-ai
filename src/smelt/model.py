"""LLM model configuration for smelt.

Wraps LangChain's ``init_chat_model`` to provide a simple, validated
configuration interface for connecting to LLM providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from smelt.errors import SmeltConfigError


@dataclass
class Model:
    """Configuration for an LLM provider used by smelt.

    Lazily initializes the underlying LangChain chat model on first use
    and caches it for subsequent calls.

    Attributes:
        provider: The LangChain model provider identifier (e.g. "openai", "anthropic").
        name: The model name (e.g. "gpt-4o", "claude-sonnet-4-20250514").
        api_key: Optional API key. If not provided, the provider's default
            environment variable will be used.
        params: Additional keyword arguments forwarded to the chat model
            constructor (e.g. ``{"temperature": 0}``).

    Examples:
        >>> model = Model(provider="openai", name="gpt-4o", params={"temperature": 0})
        >>> chat_model = model.get_chat_model()
    """

    provider: str
    name: str
    api_key: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    _chat_model: BaseChatModel | None = field(default=None, init=False, repr=False, compare=False)

    def get_chat_model(self) -> BaseChatModel:
        """Return the LangChain chat model, initializing it on first call.

        Returns:
            The initialized ``BaseChatModel`` instance.

        Raises:
            SmeltConfigError: If the model provider cannot be initialized
                (e.g. missing provider package, invalid credentials).
        """
        if self._chat_model is not None:
            return self._chat_model

        try:
            kwargs: dict[str, Any] = {**self.params}
            if self.api_key is not None:
                kwargs["api_key"] = self.api_key

            self._chat_model = init_chat_model(
                model=self.name,
                model_provider=self.provider,
                **kwargs,
            )
        except Exception as exc:
            raise SmeltConfigError(
                f"Failed to initialize model '{self.name}' with provider '{self.provider}': {exc}"
            ) from exc

        return self._chat_model
