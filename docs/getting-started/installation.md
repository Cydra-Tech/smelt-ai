# Installation

## Install with your provider

Smelt uses LangChain under the hood, so you need the provider-specific LangChain package. Install smelt with the extra for your provider:

=== "OpenAI"

    ```bash
    pip install smelt-ai[openai]
    ```

    Set your API key:
    ```bash
    export OPENAI_API_KEY="sk-..."
    ```

    Or pass it directly:
    ```python
    model = Model(provider="openai", name="gpt-4.1-mini", api_key="sk-...")
    ```

=== "Anthropic"

    ```bash
    pip install smelt-ai[anthropic]
    ```

    Set your API key:
    ```bash
    export ANTHROPIC_API_KEY="sk-ant-..."
    ```

    Or pass it directly:
    ```python
    model = Model(provider="anthropic", name="claude-sonnet-4-6", api_key="sk-ant-...")
    ```

=== "Google Gemini"

    ```bash
    pip install smelt-ai[google]
    ```

    Set your API key:
    ```bash
    export GOOGLE_API_KEY="..."
    ```

    Or pass it directly:
    ```python
    model = Model(provider="google_genai", name="gemini-3-flash-preview", api_key="...")
    ```

## Multiple providers

Install multiple extras at once:

```bash
pip install smelt-ai[openai,anthropic,google]
```

## Using uv

If you use [uv](https://github.com/astral-sh/uv) as your package manager:

```bash
uv add smelt-ai[openai]
```

## Requirements

- Python 3.10 or higher
- `pydantic >= 2.0`
- `langchain >= 0.3`
- `langchain-core >= 0.3`

## Verify installation

```python
from smelt import Model, Job
print("smelt installed successfully")
```

Test that your provider is configured correctly:

```python
from pydantic import BaseModel
from smelt import Model, Job

class TestOutput(BaseModel):
    message: str

model = Model(provider="openai", name="gpt-4.1-mini")
job = Job(prompt="Return a greeting message", output_model=TestOutput)
result = job.test(model, data=[{"input": "hello"}])
print(result.data[0])  # TestOutput(message='Hello! How can I help you today?')
```

## Supported models

Any model supported by LangChain's `init_chat_model` works with smelt. Tested with:

| Provider | `provider` value | Example models | Install extra |
|---|---|---|---|
| OpenAI | `"openai"` | `gpt-5.2`, `gpt-4.1-mini`, `gpt-4.1`, `gpt-4o`, `o4-mini` | `[openai]` |
| Anthropic | `"anthropic"` | `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-haiku-4-5-20251001` | `[anthropic]` |
| Google Gemini | `"google_genai"` | `gemini-3-flash-preview`, `gemini-3-pro-preview`, `gemini-2.5-flash` | `[google]` |

!!! tip "Using other providers"
    Any LangChain chat model provider works — just install the corresponding `langchain-*` package manually. For example, for Azure OpenAI:
    ```bash
    pip install smelt-ai langchain-openai
    ```
    ```python
    model = Model(provider="azure_openai", name="my-deployment", params={
        "azure_endpoint": "https://my-resource.openai.azure.com/",
        "api_version": "2024-02-15-preview",
    })
    ```

## API key precedence

Smelt resolves API keys in this order:

1. **Explicit `api_key` parameter** — `Model(api_key="sk-...")`
2. **Environment variable** — provider-specific (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`)
3. **LangChain config** — any provider-level configuration LangChain supports

!!! warning
    Never hardcode API keys in source code. Use environment variables or a secrets manager (e.g. `.env` files with `python-dotenv`).
