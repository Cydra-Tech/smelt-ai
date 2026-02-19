# Smelt AI

<p align="center">
  <img src="assets/banner.png" alt="smelt-ai banner" width="100%">
</p>

[![PyPI](https://img.shields.io/pypi/v/smelt-ai)](https://pypi.org/project/smelt-ai/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://cydra-tech.github.io/smelt-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

LLM-powered structured data transformation. Feed in rows of data, get back strictly typed Pydantic models — batched, concurrent, and validated.

```python
from smelt import Model, Job
from pydantic import BaseModel

class Classification(BaseModel):
    sector: str
    sub_sector: str
    is_public: bool

model = Model(provider="openai", name="gpt-4.1-mini")
job = Job(
    prompt="Classify each company by industry sector and whether it's publicly traded.",
    output_model=Classification,
)

result = job.run(model, data=[
    {"name": "Apple", "desc": "Consumer electronics and software"},
    {"name": "Stripe", "desc": "Payment processing platform"},
    {"name": "Mayo Clinic", "desc": "Nonprofit medical center"},
])

for row in result.data:
    print(row)  # Classification(sector='Technology', sub_sector='Consumer Electronics', is_public=True)
```

## Install

```bash
pip install smelt-ai[openai]      # OpenAI models
pip install smelt-ai[anthropic]   # Anthropic models
pip install smelt-ai[google]      # Google Gemini models
```

Requires Python 3.10+.

---

## How It Works

```
list[dict] → Tag with row_id → Split into batches → Concurrent LLM calls → Validate → Reorder → SmeltResult[T]
```

1. Each input row gets a `row_id` for tracking
2. Rows are split into batches of configurable size
3. Batches run concurrently through the LLM with structured output
4. Each response is validated (schema, row IDs, count)
5. Results are reordered to match original input order
6. Everything is returned as a typed `SmeltResult` with metrics

---

## API

### `Model`

Wraps a LangChain chat model provider. Any LangChain-supported provider works.

```python
model = Model(
    provider="openai",          # LangChain provider name
    name="gpt-4.1-mini",       # Model identifier
    api_key="sk-...",           # Optional — falls back to env var (e.g. OPENAI_API_KEY)
    params={"temperature": 0},  # Forwarded to the chat model constructor
)
```

### `Job`

Defines what transformation to run and how to batch it.

```python
job = Job(
    prompt="Your transformation instructions here",
    output_model=MyPydanticModel,  # Schema for each output row
    batch_size=10,                 # Rows per LLM request (default: 10)
    concurrency=3,                 # Max concurrent requests (default: 3)
    max_retries=3,                 # Retries per failed batch (default: 3)
    shuffle=False,                 # Shuffle rows before batching (default: False)
    stop_on_exhaustion=True,       # Raise on failure vs collect errors (default: True)
)
```

**Run:**

```python
result = job.run(model, data=rows)          # Sync
result = await job.arun(model, data=rows)   # Async
```

**Test with a single row first:**

```python
result = job.test(model, data=rows)         # Sync — runs only the first row
result = await job.atest(model, data=rows)  # Async
```

### `SmeltResult[T]`

```python
result.data       # list[T] — transformed rows in original order
result.errors     # list[BatchError] — failed batches
result.metrics    # SmeltMetrics — tokens, timing, retries
result.success    # bool — True if no errors
```

---

## Error Handling

All exceptions inherit from `SmeltError`.

| Exception | When |
|---|---|
| `SmeltConfigError` | Invalid config (bad provider, empty prompt, etc.) |
| `SmeltValidationError` | LLM output fails schema validation |
| `SmeltAPIError` | Non-retriable API error (401, 403) |
| `SmeltExhaustionError` | Batch exhausted all retries (`stop_on_exhaustion=True`) |

```python
from smelt.errors import SmeltExhaustionError

try:
    result = job.run(model, data=rows)
except SmeltExhaustionError as e:
    print(f"Partial: {len(e.partial_result.data)} rows succeeded")
```

Or collect errors without raising:

```python
job = Job(prompt="...", output_model=MyModel, stop_on_exhaustion=False)
result = job.run(model, data=rows)

if not result.success:
    for err in result.errors:
        print(f"Batch {err.batch_index} failed: {err.message}")
```

---

## Supported Providers

| Provider | `provider` value | Example models |
|---|---|---|
| OpenAI | `"openai"` | `gpt-5.2`, `gpt-4.1-mini`, `gpt-4.1`, `gpt-4o`, `o4-mini` |
| Anthropic | `"anthropic"` | `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-haiku-4-5-20251001` |
| Google Gemini | `"google_genai"` | `gemini-3-flash-preview`, `gemini-3-pro-preview`, `gemini-2.5-flash` |

---

## Links

- [Documentation](https://cydra-tech.github.io/smelt-ai/)
- [PyPI](https://pypi.org/project/smelt-ai/)
- [Issue Tracker](https://github.com/Cydra-Tech/smelt-ai/issues)

## License

MIT
