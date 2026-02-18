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

!!! tip
    Adding `Field(description=...)` helps the LLM understand what each field means. Smelt includes these descriptions in the system prompt automatically.

## 2. Configure a model

```python
from smelt import Model

model = Model(
    provider="openai",
    name="gpt-4.1-mini",
    # api_key="sk-..."  # Optional — falls back to OPENAI_API_KEY env var
)
```

## 3. Create a job

```python
from smelt import Job

job = Job(
    prompt="Classify each company by its primary industry sector and sub-sector. "
           "Determine if the company is publicly traded.",
    output_model=Classification,
    batch_size=10,
    concurrency=3,
)
```

## 4. Test with a single row

Before running the full dataset, validate that your prompt and schema work:

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
```

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
```

## 6. Handle errors

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
    stop_on_exhaustion=False,  # Don't raise, collect errors
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

!!! note
    `job.run()` and `job.test()` cannot be called from within an async event loop (e.g. Jupyter). Use `await job.arun()` and `await job.atest()` in those contexts.

## Next steps

- [Architecture](../guide/architecture.md) — understand how smelt processes your data
- [Batching & Concurrency](../guide/batching.md) — tune performance
- [Error Handling](../guide/errors.md) — robust error management
- [API Reference](../api/job.md) — full API documentation
