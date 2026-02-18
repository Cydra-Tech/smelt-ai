# Job

The `Job` class defines a transformation — what prompt to use, what output schema to expect, and how to configure batching and retries.

## Quick reference

```python
from smelt import Job

job = Job(
    prompt="Classify each company by industry sector",
    output_model=Classification,
    batch_size=10,
    concurrency=3,
    max_retries=3,
    shuffle=False,
    stop_on_exhaustion=True,
)

# Sync
result = job.run(model, data=rows)

# Async
result = await job.arun(model, data=rows)

# Single-row test
result = job.test(model, data=rows)
result = await job.atest(model, data=rows)
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | `str` | *required* | Transformation instructions sent to the LLM |
| `output_model` | `Type[BaseModel]` | *required* | Pydantic model defining the output schema per row |
| `batch_size` | `int` | `10` | Number of input rows per LLM request |
| `concurrency` | `int` | `3` | Maximum concurrent batch requests |
| `max_retries` | `int` | `3` | Retry attempts per failed batch |
| `shuffle` | `bool` | `False` | Randomize row order before batching (results are always in original order) |
| `stop_on_exhaustion` | `bool` | `True` | Raise `SmeltExhaustionError` on failure vs collect errors |

## Validation

Job configuration is validated on creation. Invalid values raise `SmeltConfigError`:

```python
# These all raise SmeltConfigError:
Job(prompt="", output_model=MyModel)          # Empty prompt
Job(prompt="ok", output_model=dict)           # Not a BaseModel
Job(prompt="ok", output_model=MyModel, batch_size=0)    # batch_size < 1
Job(prompt="ok", output_model=MyModel, concurrency=0)   # concurrency < 1
Job(prompt="ok", output_model=MyModel, max_retries=-1)  # max_retries < 0
```

## Methods

### `run(model, *, data)` {: #run }

Run the transformation synchronously. Creates an event loop internally.

```python
result = job.run(model, data=[
    {"name": "Apple", "desc": "Tech company"},
    {"name": "Stripe", "desc": "Payments"},
])
print(result.data)      # [Classification(...), Classification(...)]
print(result.success)   # True
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `model` | `Model` | LLM provider configuration |
| `data` | `list[dict[str, Any]]` | Input rows as dictionaries |

**Returns:** `SmeltResult[T]` where `T` is your `output_model`

**Raises:**

- `RuntimeError` if called from an async context (use `arun` instead)
- `SmeltConfigError` if the model can't be initialized
- `SmeltExhaustionError` if `stop_on_exhaustion=True` and a batch fails

### `arun(model, *, data)` {: #arun }

Run the transformation asynchronously.

```python
result = await job.arun(model, data=rows)
```

Same parameters and return type as `run()`. Use this in Jupyter notebooks or async applications.

### `test(model, *, data)` {: #test }

Run a single-row test synchronously. Useful for validating your setup before a full run.

```python
result = job.test(model, data=rows)
print(result.data[0])  # Single output row
```

**Behavior:**

- Only processes the **first row** of `data`
- Ignores `batch_size`, `concurrency`, and `shuffle` settings
- Uses `batch_size=1`, `concurrency=1`, `shuffle=False` internally
- Respects `max_retries` and `stop_on_exhaustion`

**Raises:** `SmeltConfigError` if `data` is empty

### `atest(model, *, data)` {: #atest }

Run a single-row test asynchronously.

```python
result = await job.atest(model, data=rows)
print(result.data[0])
```

Same behavior as `test()`. Use this in Jupyter notebooks or async applications.

## Usage patterns

### Test → Run workflow

```python
# 1. Test with one row to verify setup
test_result = job.test(model, data=data)
print(f"Test output: {test_result.data[0]}")
print(f"Test tokens: {test_result.metrics.input_tokens + test_result.metrics.output_tokens}")

# 2. If test looks good, run the full dataset
result = job.run(model, data=data)
```

### Async Jupyter notebook

```python
# In Jupyter, use await with async methods
result = await job.atest(model, data=data)   # Test
result = await job.arun(model, data=data)    # Full run
```

### Error collection mode

```python
job = Job(
    prompt="...",
    output_model=MyModel,
    stop_on_exhaustion=False,
)
result = job.run(model, data=data)
print(f"Succeeded: {len(result.data)}, Failed: {len(result.errors)}")
```

## Source

::: smelt.job.Job
    options:
      show_source: true
      members:
        - __post_init__
        - atest
        - test
        - arun
        - run
