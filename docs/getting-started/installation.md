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

=== "Anthropic"

    ```bash
    pip install smelt-ai[anthropic]
    ```

    Set your API key:
    ```bash
    export ANTHROPIC_API_KEY="sk-ant-..."
    ```

=== "Google Gemini"

    ```bash
    pip install smelt-ai[google]
    ```

    Set your API key:
    ```bash
    export GOOGLE_API_KEY="..."
    ```

## Multiple providers

Install multiple extras at once:

```bash
pip install smelt-ai[openai,anthropic,google]
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

## Supported models

Any model supported by LangChain's `init_chat_model`. Tested with:

| Provider | `provider` value | Example models |
|---|---|---|
| OpenAI | `"openai"` | `gpt-5.2`, `gpt-4.1-mini`, `gpt-4.1`, `gpt-4o`, `o4-mini` |
| Anthropic | `"anthropic"` | `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-haiku-4-5-20251001` |
| Google Gemini | `"google_genai"` | `gemini-3-flash-preview`, `gemini-3-pro-preview`, `gemini-2.5-flash` |
