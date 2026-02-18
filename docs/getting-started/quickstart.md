# Quickstart

Build your first transformation in 5 minutes.

## 1. Define your output schema

Use a standard Pydantic `BaseModel` to define the shape of each output row:

```python
from pydantic import BaseModel, Field

class Classification(BaseModel):
    sector: str = Field(description="Primary industry sector")
    sub_sector: str = Field(description="More specific sub-sector")
    is_public: bool = Field(description="Whether the company is publicly traded")
```

!!! tip "Use Field descriptions"
    Adding `Field(description=...)` helps the LLM understand what each field means. Smelt includes these descriptions in the system prompt automatically. Better descriptions = better output quality.

## 2. Configure a model

```python
from smelt import Model

model = Model(
    provider="openai",
    name="gpt-4.1-mini",
    # api_key="sk-..."  # Optional — falls back to OPENAI_API_KEY env var
)
```

You can pass additional parameters to the underlying LangChain model:

```python
model = Model(
    provider="openai",
    name="gpt-4.1-mini",
    params={"temperature": 0},  # Deterministic output
)
```

## 3. Create a job

```python
from smelt import Job

job = Job(
    prompt="Classify each company by its primary industry sector and sub-sector. "
           "Determine if the company is publicly traded.",
    output_model=Classification,
    batch_size=10,    # 10 rows per LLM call
    concurrency=3,    # Up to 3 concurrent calls
)
```

## 4. Test with a single row

Before running the full dataset, validate that your prompt and schema work together:

```python
data = [
    {"name": "Apple", "description": "Consumer electronics and software"},
    {"name": "Stripe", "description": "Payment processing platform"},
    {"name": "Mayo Clinic", "description": "Nonprofit medical center"},
]

# Quick test — sends only the first row
result = job.test(model, data=data)
print(result.data[0])
# Classification(sector='Technology', sub_sector='Consumer Electronics', is_public=True)
print(result.metrics)
# SmeltMetrics(total_rows=1, successful_rows=1, ...)
```

!!! info "Why test first?"
    `test()` / `atest()` sends only the first row with `batch_size=1`, `concurrency=1`, and `shuffle=False`. This catches prompt issues, schema mismatches, and auth errors before you burn through tokens on a full run.

## 5. Run the full transformation

```python
result = job.run(model, data=data)

# Check results
print(f"Success: {result.success}")
print(f"Rows: {result.metrics.successful_rows}/{result.metrics.total_rows}")
print(f"Tokens: {result.metrics.input_tokens} in / {result.metrics.output_tokens} out")
print(f"Time: {result.metrics.wall_time_seconds:.2f}s")

for row in result.data:
    print(row)
# Classification(sector='Technology', sub_sector='Consumer Electronics', is_public=True)
# Classification(sector='Financial Technology', sub_sector='Payments', is_public=False)
# Classification(sector='Healthcare', sub_sector='Hospitals', is_public=False)
```

**Key properties of the result:**

| Property | Type | Description |
|---|---|---|
| `result.data` | `list[Classification]` | Transformed rows, in original input order |
| `result.errors` | `list[BatchError]` | Failed batches (empty if all succeeded) |
| `result.metrics` | `SmeltMetrics` | Token usage, timing, retry counts |
| `result.success` | `bool` | `True` if zero errors |

## 6. Handle errors

By default, smelt raises `SmeltExhaustionError` when a batch fails after all retries:

```python
from smelt.errors import SmeltExhaustionError

try:
    result = job.run(model, data=data)
except SmeltExhaustionError as e:
    print(f"Some batches failed after retries")
    print(f"Partial results: {len(e.partial_result.data)} rows")
    print(f"Errors: {len(e.partial_result.errors)} batches")
```

Or collect errors without raising:

```python
job = Job(
    prompt="...",
    output_model=Classification,
    stop_on_exhaustion=False,  # Don't raise, collect errors instead
)

result = job.run(model, data=data)
if not result.success:
    for err in result.errors:
        print(f"Batch {err.batch_index} failed: {err.message}")
```

## Async usage

In Jupyter notebooks or async applications, use the async methods:

```python
# Async test
result = await job.atest(model, data=data)

# Async run
result = await job.arun(model, data=data)
```

!!! warning "Sync vs async"
    `job.run()` and `job.test()` cannot be called from within an async event loop (e.g. Jupyter notebooks). Use `await job.arun()` and `await job.atest()` in those contexts. Smelt will raise a `RuntimeError` with a helpful message if you mix them up.

## Complete example

Here's a full working script you can copy-paste:

```python
"""Classify companies by industry sector using smelt."""
import os
from pydantic import BaseModel, Field
from smelt import Model, Job

# 1. Define output schema
class Classification(BaseModel):
    sector: str = Field(description="Primary industry sector")
    sub_sector: str = Field(description="More specific sub-sector")
    is_public: bool = Field(description="Whether the company is publicly traded")

# 2. Configure model
model = Model(provider="openai", name="gpt-4.1-mini")

# 3. Prepare data
companies = [
    {"name": "Apple Inc.", "description": "Consumer electronics, software, and services"},
    {"name": "JPMorgan Chase", "description": "Global financial services and investment banking"},
    {"name": "Pfizer", "description": "Pharmaceutical company developing medicines and vaccines"},
    {"name": "Tesla", "description": "Electric vehicles and clean energy products"},
    {"name": "Spotify", "description": "Digital music and podcast streaming platform"},
]

# 4. Create and test job
job = Job(
    prompt="Classify each company by its primary industry sector and sub-sector. "
           "Determine if the company is publicly traded.",
    output_model=Classification,
    batch_size=5,
    concurrency=1,
)

# Quick single-row test
test_result = job.test(model, data=companies)
print(f"Test: {test_result.data[0]}")
print(f"Tokens used: {test_result.metrics.input_tokens + test_result.metrics.output_tokens}")
print()

# 5. Full run
result = job.run(model, data=companies)

# 6. Print results
print(f"Success: {result.success}")
print(f"Rows: {result.metrics.successful_rows}/{result.metrics.total_rows}")
print(f"Time: {result.metrics.wall_time_seconds:.2f}s")
print()

for company, classification in zip(companies, result.data):
    print(f"{company['name']:20s} → {classification.sector} / {classification.sub_sector} (public: {classification.is_public})")
```

## Next steps

- [Architecture](../guide/architecture.md) — understand how smelt processes your data
- [Batching & Concurrency](../guide/batching.md) — tune batch_size, concurrency, and shuffle
- [Error Handling](../guide/errors.md) — robust error management strategies
- [Cookbook](../cookbook/classification.md) — real-world examples
- [API Reference](../api/job.md) — full API documentation
