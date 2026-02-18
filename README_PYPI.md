# Smelt AI

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
pip install smelt-ai[openai]      # OpenAI models
pip install smelt-ai[anthropic]   # Anthropic models
pip install smelt-ai[google]      # Google Gemini models
```

Requires Python 3.10+.

---

## Architecture

### Pipeline Overview

```
list[dict] → Tag rows with row_id → Shuffle? → Split into batches → Concurrent LLM calls → Validate schema + IDs → Reorder by row_id → SmeltResult[T]
```

### How a Job Executes

1. **Prepare** — Tag each row with a `row_id`, optionally shuffle, create internal Pydantic model, build system prompt, split into batches
2. **Execute** — Send batches concurrently to the LLM (controlled by `concurrency` semaphore)
3. **Validate** — For each batch response: validate row IDs, check schema conformance
4. **Assemble** — Sort all rows by `row_id`, strip internal fields, aggregate metrics into `SmeltResult[T]`

### Retry & Backoff

Each batch independently retries on failure. Validation errors (bad schema) and transient API errors (429, 5xx) trigger retries with exponential backoff (`1s × 2^attempt + jitter`, capped at 60s). Client errors (400, 401, 403) fail immediately without retrying.

### Concurrency Model

Smelt uses `asyncio.Semaphore` for cooperative async concurrency — no threads, no process pools. While one batch awaits an LLM response, others can fire off their requests on the same thread.

```
concurrency=3, batch_size=5, 15 rows:

  Batch 0 (rows 0–4):   ████████░░░░
  Batch 1 (rows 5–9):   ████████████
  Batch 2 (rows 10–14): ██████░░░░░░
                         ↑ 3 slots    ↑ 1 slot
```

### Row ID Tracking

Smelt injects a `row_id` field into your model, tells the LLM to echo it back, then validates and strips it. This ensures correct ordering even when batches complete out of order.

```
Input:                    Tagged:                    LLM Output (unordered):    Final (reordered):
  {name: Apple}     →      {row_id: 0, name: Apple}    {row_id: 1, Fintech}  →    Classification(Tech)
  {name: Stripe}    →      {row_id: 1, name: Stripe}   {row_id: 0, Tech}     →    Classification(Fintech)
  {name: Mayo}      →      {row_id: 2, name: Mayo}      {row_id: 2, Health}   →    Classification(Health)
```

### Dynamic Model Creation

Under the hood, smelt dynamically extends your Pydantic model to add `row_id`, then wraps it in a batch container for `with_structured_output`:

```
YourModel                    _SmeltYourModel (extends YourModel)     _SmeltBatch
  sector: str          →       row_id: int                             rows: list[_SmeltYourModel]
  sub_sector: str                sector: str
  is_public: bool                sub_sector: str
                                 is_public: bool
```

### Error Handling Modes

**`stop_on_exhaustion=True` (default):**
Batch fails → set cancel event → pending batches skip → raise `SmeltExhaustionError` → `e.partial_result` has successful batches

**`stop_on_exhaustion=False`:**
Batch fails → record `BatchError` → continue processing → return `SmeltResult` → `result.errors` has failures, `result.data` has successes

---

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
    shuffle=False,                 # Shuffle rows before batching (default: False)
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

Test with a single row before committing to a full run:

```python
# Quick validation — runs only the first row, ignores shuffle/batch_size/concurrency
result = job.test(model, data=rows)
print(result.data[0])  # see one sample output

# Async version
result = await job.atest(model, data=rows)
```

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

---

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

---

## Supported Providers

Any provider supported by LangChain's `init_chat_model`. Tested with:

| Provider | `provider` value | Example models |
|---|---|---|
| OpenAI | `"openai"` | `gpt-5.2`, `gpt-4.1-mini`, `gpt-4.1`, `gpt-4o`, `o4-mini` |
| Anthropic | `"anthropic"` | `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-haiku-4-5-20251001` |
| Google Gemini | `"google_genai"` | `gemini-3-flash-preview`, `gemini-3-pro-preview`, `gemini-2.5-flash` |

---

## Project Structure

```
src/smelt/
├── __init__.py        # Public API exports
├── model.py           # Model — LLM provider config
├── job.py             # Job — transformation definition + run/arun
├── batch.py           # Async batch engine, retry, concurrency
├── prompt.py          # System/human message construction
├── validation.py      # Dynamic Pydantic model creation, row ID validation
├── types.py           # SmeltResult, SmeltMetrics, BatchError
└── errors.py          # Exception hierarchy
```

---

## Development

```bash
git clone https://github.com/Cydra-Tech/smelt-ai.git
cd smelt
uv sync --all-extras

# Unit tests (mocked, no API keys needed)
uv run pytest tests/ --ignore=tests/test_live.py

# Live API tests (requires .env with API keys)
uv run pytest tests/test_live.py -v

# Lint
uv run ruff check src/ tests/
```

## Links

- [GitHub Repository](https://github.com/Cydra-Tech/smelt-ai)
- [Issue Tracker](https://github.com/Cydra-Tech/smelt-ai/issues)
- [Documentation](https://cydra-tech.github.io/smelt-ai/)

## License

MIT
