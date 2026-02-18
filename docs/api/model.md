# Model

The `Model` class configures which LLM provider and model to use. It wraps LangChain's `init_chat_model` with lazy initialization and caching.

## Quick reference

```python
from smelt import Model

# Minimal
model = Model(provider="openai", name="gpt-4.1-mini")

# With API key
model = Model(provider="openai", name="gpt-4.1-mini", api_key="sk-...")

# With parameters
model = Model(
    provider="openai",
    name="gpt-4.1-mini",
    params={"temperature": 0, "max_tokens": 4096},
)
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | *required* | LangChain provider identifier (e.g. `"openai"`, `"anthropic"`, `"google_genai"`) |
| `name` | `str` | *required* | Model name (e.g. `"gpt-4.1-mini"`, `"claude-sonnet-4-6"`) |
| `api_key` | `str \| None` | `None` | API key. Falls back to provider's environment variable if not set |
| `params` | `dict[str, Any]` | `{}` | Additional kwargs forwarded to the LangChain model constructor |

## Examples

### OpenAI

```python
model = Model(provider="openai", name="gpt-4.1-mini")
model = Model(provider="openai", name="gpt-5.2", params={"temperature": 0})
```

### Anthropic

```python
model = Model(provider="anthropic", name="claude-sonnet-4-6")
model = Model(provider="anthropic", name="claude-opus-4-6", params={"max_tokens": 4096})
```

### Google Gemini

```python
model = Model(provider="google_genai", name="gemini-3-flash-preview")
```

### Azure OpenAI

```python
model = Model(
    provider="azure_openai",
    name="my-deployment",
    params={
        "azure_endpoint": "https://my-resource.openai.azure.com/",
        "api_version": "2024-02-15-preview",
    },
)
```

## Methods

### `get_chat_model()`

Returns the initialized LangChain `BaseChatModel`. The model is lazily initialized on first call and cached for subsequent calls.

```python
model = Model(provider="openai", name="gpt-4.1-mini")
chat_model = model.get_chat_model()  # Initializes on first call
chat_model = model.get_chat_model()  # Returns cached instance
```

**Raises:** `SmeltConfigError` if the provider or model cannot be initialized (e.g. missing package, invalid credentials).

## Source

::: smelt.model.Model
    options:
      show_source: true
      members:
        - get_chat_model
