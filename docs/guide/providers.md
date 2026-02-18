# Providers

Smelt works with any LLM provider supported by LangChain's `init_chat_model`. This page covers setup, recommended models, and provider-specific tips.

## Overview

| Provider | `provider` value | Install extra | Env var |
|---|---|---|---|
| OpenAI | `"openai"` | `pip install smelt-ai[openai]` | `OPENAI_API_KEY` |
| Anthropic | `"anthropic"` | `pip install smelt-ai[anthropic]` | `ANTHROPIC_API_KEY` |
| Google Gemini | `"google_genai"` | `pip install smelt-ai[google]` | `GOOGLE_API_KEY` |

## OpenAI

### Recommended models

| Model | Best for | Notes |
|---|---|---|
| `gpt-4.1-mini` | Most tasks — fast, cheap, good quality | Best price/performance ratio |
| `gpt-4.1` | Complex schemas, nuanced classification | Higher quality, higher cost |
| `gpt-4o` | General purpose | Solid all-rounder |
| `gpt-5.2` | Maximum quality | Latest and most capable |
| `o4-mini` | Reasoning-heavy tasks | Good for complex logic |

### Setup

```python
from smelt import Model

# Using environment variable (recommended)
# export OPENAI_API_KEY="sk-..."
model = Model(provider="openai", name="gpt-4.1-mini")

# Explicit API key
model = Model(provider="openai", name="gpt-4.1-mini", api_key="sk-...")

# With parameters
model = Model(
    provider="openai",
    name="gpt-4.1-mini",
    params={"temperature": 0, "max_tokens": 4096},
)
```

### Tips

- **Use `gpt-4.1-mini` as your default** — it handles structured output very well and is cost-effective
- **Set `temperature=0`** for classification and extraction tasks
- **Use `temperature=0.5–0.7`** for creative tasks (summaries, marketing copy)
- **Watch for rate limits** — at high concurrency, you may hit tokens-per-minute limits. Smelt retries 429 errors automatically

## Anthropic

### Recommended models

| Model | Best for | Notes |
|---|---|---|
| `claude-sonnet-4-6` | Most tasks — fast, great quality | Best balance for structured output |
| `claude-opus-4-6` | Complex reasoning, nuanced output | Highest quality, slower |
| `claude-haiku-4-5-20251001` | High volume, simple schemas | Fastest, most cost-effective |

### Setup

```python
from smelt import Model

# Using environment variable (recommended)
# export ANTHROPIC_API_KEY="sk-ant-..."
model = Model(provider="anthropic", name="claude-sonnet-4-6")

# Explicit API key
model = Model(provider="anthropic", name="claude-sonnet-4-6", api_key="sk-ant-...")

# With parameters
model = Model(
    provider="anthropic",
    name="claude-sonnet-4-6",
    params={"temperature": 0, "max_tokens": 4096},
)
```

### Tips

- **Claude excels at following complex instructions** — great for detailed prompts with many rules
- **Anthropic models require `max_tokens`** in some configurations — if you get truncated output, add `params={"max_tokens": 4096}`
- **Claude tends to be conservative** — if you want more creative output, increase temperature

## Google Gemini

### Recommended models

| Model | Best for | Notes |
|---|---|---|
| `gemini-3-flash-preview` | Most tasks — very fast, good quality | Best speed/quality ratio |
| `gemini-3-pro-preview` | Complex tasks, highest quality | Slower but more capable |
| `gemini-2.5-flash` | Budget-friendly, simple tasks | Previous generation, still good |

### Setup

```python
from smelt import Model

# Using environment variable (recommended)
# export GOOGLE_API_KEY="..."
model = Model(provider="google_genai", name="gemini-3-flash-preview")

# Explicit API key
model = Model(provider="google_genai", name="gemini-3-flash-preview", api_key="...")

# With parameters
model = Model(
    provider="google_genai",
    name="gemini-3-flash-preview",
    params={"temperature": 0},
)
```

### Tips

- **Gemini Flash is excellent for high-volume tasks** — fast response times, generous rate limits
- **Provider name is `google_genai`** (not `google` or `gemini`)
- **Gemini handles large batches well** — you can often use `batch_size=20-30` without issues

## Other providers

Smelt works with any LangChain-supported chat model. Install the provider's LangChain package and use the appropriate `provider` value.

### Azure OpenAI

```bash
pip install smelt-ai langchain-openai
```

```python
model = Model(
    provider="azure_openai",
    name="my-gpt4-deployment",
    params={
        "azure_endpoint": "https://my-resource.openai.azure.com/",
        "api_version": "2024-02-15-preview",
        "api_key": "...",
    },
)
```

### AWS Bedrock

```bash
pip install smelt-ai langchain-aws
```

```python
model = Model(
    provider="bedrock",
    name="anthropic.claude-3-5-sonnet-20241022-v2:0",
    params={
        "region_name": "us-east-1",
    },
)
```

### Ollama (local models)

```bash
pip install smelt-ai langchain-ollama
```

```python
model = Model(
    provider="ollama",
    name="llama3.1:8b",
    params={"base_url": "http://localhost:11434"},
)
```

!!! warning "Local model quality"
    Smaller local models may struggle with structured output, especially complex schemas or large batch sizes. Use `batch_size=1-3` and simpler schemas for best results.

## Choosing a provider

### By use case

| Use case | Recommended | Why |
|---|---|---|
| **General classification** | `gpt-4.1-mini` or `gemini-3-flash-preview` | Fast, cheap, good quality |
| **Complex extraction** | `claude-sonnet-4-6` or `gpt-4.1` | Better at following complex instructions |
| **High volume (1000+ rows)** | `gemini-3-flash-preview` | Fastest, generous rate limits |
| **Maximum quality** | `gpt-5.2` or `claude-opus-4-6` | Best output quality |
| **Budget-conscious** | `gpt-4.1-mini` or `gemini-3-flash-preview` | Lowest cost per token |
| **Privacy-sensitive** | Ollama (local) | Data stays on your machine |

### Cost comparison

Approximate cost to process 1000 rows with `batch_size=10` (100 LLM calls):

| Model | Input cost | Output cost | Approx total |
|---|---|---|---|
| `gpt-4.1-mini` | Lowest | Lowest | ~$0.10–0.50 |
| `gemini-3-flash-preview` | Low | Low | ~$0.10–0.50 |
| `claude-sonnet-4-6` | Medium | Medium | ~$0.50–2.00 |
| `gpt-4.1` | Medium-high | Medium-high | ~$1.00–5.00 |
| `claude-opus-4-6` | Highest | Highest | ~$5.00–20.00 |

!!! note
    Actual costs depend on your schema complexity (affects output tokens), input data size (affects input tokens), and retry rate. Check `result.metrics.input_tokens` and `result.metrics.output_tokens` for exact usage.
