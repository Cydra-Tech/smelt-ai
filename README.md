# Smelt

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
    batch_size=20,
    concurrency=3,
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
pip install smelt[openai]      # OpenAI models
pip install smelt[anthropic]   # Anthropic models
pip install smelt[google]      # Google Gemini models
```

Requires Python 3.10+.

## How It Works

1. You define a Pydantic model for your desired output schema
2. Smelt splits your input rows into batches
3. Each batch is sent to the LLM concurrently (up to `concurrency` limit)
4. Responses are validated against your schema with row-level ID tracking
5. Results are reassembled in original input order

```
Input rows ──► Tag with row_id ──► Split into batches ──► LLM (concurrent) ──► Validate ──► Reorder ──► SmeltResult
```

## API

### `Model`

Wraps a LangChain chat model provider. Uses `init_chat_model` under the hood, so any LangChain-supported provider works.

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
    stop_on_exhaustion=True,       # Raise on failure vs collect errors (default: True)
)
```

Run synchronously or asynchronously:

```python
# Sync — use in scripts
result = job.run(model, data=rows)

# Async — use in notebooks, async apps
result = await job.arun(model, data=rows)
```

> **Note:** `job.run()` cannot be called from within an async event loop (e.g. Jupyter). Use `await job.arun()` in those contexts.

### `SmeltResult[T]`

```python
result.data       # list[T] — transformed rows in original order
result.errors     # list[BatchError] — failed batches
result.metrics    # SmeltMetrics — tokens, timing, retries
result.success    # bool — True if no errors
```

### `SmeltMetrics`

```python
result.metrics.total_rows         # Total input rows
result.metrics.successful_rows    # Rows with valid output
result.metrics.failed_rows        # Rows in failed batches
result.metrics.total_retries      # Cumulative retries across all batches
result.metrics.input_tokens       # Total input tokens consumed
result.metrics.output_tokens      # Total output tokens consumed
result.metrics.wall_time_seconds  # Wall-clock duration
```

## Error Handling

All exceptions inherit from `SmeltError`.

| Exception | When |
|---|---|
| `SmeltConfigError` | Invalid config (bad provider, empty prompt, etc.) |
| `SmeltValidationError` | LLM output fails schema validation |
| `SmeltAPIError` | Non-retriable API error (401, 403) |
| `SmeltExhaustionError` | Batch exhausted all retries (`stop_on_exhaustion=True`) |

`SmeltExhaustionError` carries a `partial_result` with any successfully processed batches:

```python
from smelt.errors import SmeltExhaustionError

try:
    result = job.run(model, data=rows)
except SmeltExhaustionError as e:
    print(f"Partial: {len(e.partial_result.data)} rows succeeded")
    print(f"Failed: {len(e.partial_result.errors)} batches")
```

Set `stop_on_exhaustion=False` to collect errors without raising:

```python
job = Job(prompt="...", output_model=MyModel, stop_on_exhaustion=False)
result = job.run(model, data=rows)

if not result.success:
    for err in result.errors:
        print(f"Batch {err.batch_index} failed: {err.message}")
```

## Retry & Backoff

Failed batches are retried with exponential backoff (1s base, 60s cap, with jitter).

**Retriable:** 429 (rate limit), 500/502/503/504 (server errors), timeouts, connection errors, validation failures.

**Not retriable:** 400, 401, 403 — these fail the batch immediately.

## Concurrency

Concurrency is async I/O based (`asyncio.Semaphore`), not thread based. With `concurrency=3`, up to 3 batches have outstanding HTTP requests at the same time on a single thread. This is efficient for LLM calls since they're I/O-bound.

## Supported Providers

Any provider supported by LangChain's `init_chat_model`. Tested with:

| Provider | `provider` value | Example models |
|---|---|---|
| OpenAI | `"openai"` | `gpt-4.1-mini`, `gpt-4.1`, `gpt-4o`, `o4-mini` |
| Anthropic | `"anthropic"` | `claude-sonnet-4-20250514`, `claude-haiku-4-5-20251001` |
| Google Gemini | `"google_genai"` | `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash` |

## Development

```bash
git clone <repo>
cd llm-data-transformation
uv sync --all-extras

# Unit tests (mocked, no API keys needed)
uv run pytest tests/ --ignore=tests/test_live.py

# Live API tests (requires .env with API keys)
uv run pytest tests/test_live.py -v

# Lint
uv run ruff check src/ tests/
```

## License

MIT
