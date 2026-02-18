# Errors

All smelt exceptions inherit from `SmeltError`. You can catch any smelt error with:

```python
from smelt.errors import SmeltError

try:
    result = job.run(model, data=data)
except SmeltError as e:
    print(f"Smelt error: {e}")
```

## Hierarchy

```
SmeltError
├── SmeltConfigError         # Bad configuration
├── SmeltValidationError     # LLM output validation failure
├── SmeltAPIError            # Non-retriable API error
└── SmeltExhaustionError     # Retries exhausted
```

---

## SmeltError

Base exception class. All smelt exceptions inherit from this.

```python
from smelt.errors import SmeltError
```

::: smelt.errors.SmeltError
    options:
      show_source: true

---

## SmeltConfigError

Raised when configuration is invalid. This happens at job creation time or model initialization — **before** any LLM calls are made.

### Common causes

| Cause | Example |
|---|---|
| Empty prompt | `Job(prompt="", output_model=MyModel)` |
| Invalid output_model | `Job(prompt="ok", output_model=dict)` |
| Bad batch_size | `Job(prompt="ok", output_model=MyModel, batch_size=0)` |
| Reserved field name | Output model has a `row_id` field |
| Bad provider | `Model(provider="nonexistent", name="fake")` |
| Missing package | Provider package not installed |
| Empty data | `job.test(model, data=[])` |

### Example

```python
from smelt.errors import SmeltConfigError

try:
    job = Job(prompt="", output_model=MyModel)
except SmeltConfigError as e:
    print(e)  # "Job prompt must be a non-empty string."
```

::: smelt.errors.SmeltConfigError
    options:
      show_source: true

---

## SmeltValidationError

Raised internally when LLM output fails Pydantic schema validation or row ID checks. This exception is caught by the batch engine's retry loop — you typically won't see it directly.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `raw_response` | `Any` | The raw LLM response that failed validation |

### When it triggers

- LLM returned wrong number of rows
- Missing, duplicate, or unexpected row IDs
- Pydantic field validation failure (including custom validators)
- JSON parsing failure

```python
from smelt.errors import SmeltValidationError

# You typically won't catch this directly — it's internal to the retry loop
# But it's available if you need it:
try:
    # ... direct validation call ...
    pass
except SmeltValidationError as e:
    print(e)
    print(f"Raw response: {e.raw_response}")
```

::: smelt.errors.SmeltValidationError
    options:
      show_source: true

---

## SmeltAPIError

Raised internally for non-retriable API errors (400, 401, 403). Like `SmeltValidationError`, this is caught by the batch engine.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `status_code` | `int \| None` | HTTP status code, if available |

::: smelt.errors.SmeltAPIError
    options:
      show_source: true

---

## SmeltExhaustionError

Raised when a batch exhausts all retries and `stop_on_exhaustion=True` (the default). This is the primary user-facing error for batch failures.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `partial_result` | `SmeltResult` | Results accumulated before the failure, including successful batches |

### Example

```python
from smelt.errors import SmeltExhaustionError

try:
    result = job.run(model, data=data)
except SmeltExhaustionError as e:
    print(f"Error: {e}")

    # Access partial results
    partial = e.partial_result
    print(f"Succeeded: {len(partial.data)} rows")
    print(f"Failed: {len(partial.errors)} batches")
    print(f"Tokens used: {partial.metrics.input_tokens + partial.metrics.output_tokens}")

    # Use successful rows
    for row in partial.data:
        process(row)

    # Inspect failures
    for err in partial.errors:
        print(f"  Batch {err.batch_index}: {err.error_type} — {err.message}")
```

### Avoiding this exception

Set `stop_on_exhaustion=False` to collect errors instead of raising:

```python
job = Job(prompt="...", output_model=MyModel, stop_on_exhaustion=False)
result = job.run(model, data=data)
# No exception raised — check result.success and result.errors instead
```

::: smelt.errors.SmeltExhaustionError
    options:
      show_source: true
