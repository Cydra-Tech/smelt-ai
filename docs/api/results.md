# Results & Metrics

## SmeltResult

The result container returned by `job.run()`, `job.arun()`, `job.test()`, and `job.atest()`. Generic over `T`, your Pydantic output model.

### Quick reference

```python
result = job.run(model, data=rows)

result.data       # list[T] — transformed rows in original input order
result.errors     # list[BatchError] — failed batches
result.metrics    # SmeltMetrics — token usage, timing, retries
result.success    # bool — True if no errors
```

### Properties

| Property | Type | Description |
|---|---|---|
| `data` | `list[T]` | Successfully transformed rows, ordered by original input position |
| `errors` | `list[BatchError]` | One entry per failed batch (empty if all succeeded) |
| `metrics` | `SmeltMetrics` | Aggregated metrics for the entire run |
| `success` | `bool` | `True` if `errors` is empty |

### Examples

```python
result = job.run(model, data=companies)

# Check overall status
if result.success:
    print(f"All {len(result.data)} rows processed successfully")
else:
    print(f"{len(result.data)} succeeded, {len(result.errors)} batches failed")

# Iterate over typed results
for row in result.data:
    print(row.sector, row.sub_sector, row.is_public)

# Access metrics
print(f"Tokens: {result.metrics.input_tokens:,} in / {result.metrics.output_tokens:,} out")
print(f"Time: {result.metrics.wall_time_seconds:.2f}s")
```

### Source

::: smelt.types.SmeltResult
    options:
      show_source: true
      members:
        - data
        - errors
        - metrics
        - success

---

## SmeltMetrics

Aggregated metrics for a completed run.

### Quick reference

```python
m = result.metrics

# Row counts
m.total_rows         # Total input rows
m.successful_rows    # Rows with valid output
m.failed_rows        # Rows in failed batches

# Batch counts
m.total_batches      # Total number of batches
m.successful_batches # Batches that succeeded
m.failed_batches     # Batches that exhausted retries

# Performance
m.total_retries      # Cumulative retries across all batches
m.input_tokens       # Total input tokens consumed
m.output_tokens      # Total output tokens consumed
m.wall_time_seconds  # Wall-clock duration of the run
```

### Fields

| Field | Type | Description |
|---|---|---|
| `total_rows` | `int` | Number of input rows |
| `successful_rows` | `int` | Rows that produced valid output |
| `failed_rows` | `int` | Rows that were in failed batches |
| `total_batches` | `int` | Total batches processed |
| `successful_batches` | `int` | Batches that succeeded |
| `failed_batches` | `int` | Batches that failed after all retries |
| `total_retries` | `int` | Total retry count across all batches |
| `input_tokens` | `int` | Total input tokens (from LLM usage metadata) |
| `output_tokens` | `int` | Total output tokens (from LLM usage metadata) |
| `wall_time_seconds` | `float` | Wall-clock time in seconds |

### Derived metrics

```python
m = result.metrics

# Throughput
if m.wall_time_seconds > 0:
    rows_per_sec = m.successful_rows / m.wall_time_seconds
    print(f"Throughput: {rows_per_sec:.1f} rows/sec")

# Average tokens per batch
if m.total_batches > 0:
    avg_input = m.input_tokens // m.total_batches
    avg_output = m.output_tokens // m.total_batches
    print(f"Avg tokens/batch: {avg_input} in, {avg_output} out")

# Total cost estimate (example for GPT-4.1-mini)
input_cost = m.input_tokens * 0.15 / 1_000_000   # $0.15/M input tokens
output_cost = m.output_tokens * 0.60 / 1_000_000  # $0.60/M output tokens
print(f"Estimated cost: ${input_cost + output_cost:.4f}")

# Retry rate
if m.total_batches > 0:
    retry_rate = m.total_retries / m.total_batches
    print(f"Retry rate: {retry_rate:.1f} retries/batch")
```

### Source

::: smelt.types.SmeltMetrics
    options:
      show_source: true

---

## BatchError

A record of a failed batch after all retries were exhausted. Frozen dataclass (immutable).

### Quick reference

```python
for err in result.errors:
    print(f"Batch {err.batch_index}: {err.error_type}")
    print(f"  Rows: {err.row_ids}")
    print(f"  Message: {err.message}")
    print(f"  Attempts: {err.attempts}")
```

### Fields

| Field | Type | Description |
|---|---|---|
| `batch_index` | `int` | Zero-based batch position in the run |
| `row_ids` | `tuple[int, ...]` | Row IDs that were in this batch |
| `error_type` | `str` | Error classification: `"validation"`, `"api"`, or `"cancelled"` |
| `message` | `str` | Human-readable error description |
| `attempts` | `int` | Total attempts made (1 initial + N retries) |
| `raw_response` | `str \| None` | Raw LLM response if available |

### Error types

| `error_type` | Meaning |
|---|---|
| `"validation"` | LLM output failed schema or row ID validation after all retries |
| `"api"` | API error (retriable exhausted or non-retriable like 401/403) |
| `"cancelled"` | Batch was skipped due to `stop_on_exhaustion` cancellation |

### Examples

```python
# Identify which input rows failed
failed_indices = set()
for err in result.errors:
    failed_indices.update(err.row_ids)

# Get the original data for failed rows
failed_data = [data[i] for i in sorted(failed_indices)]
print(f"{len(failed_data)} rows need manual review")
```

### Source

::: smelt.types.BatchError
    options:
      show_source: true
