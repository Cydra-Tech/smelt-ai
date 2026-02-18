# Cookbook: Advanced Patterns

Advanced usage patterns and techniques.

## Chaining transformations

Run multiple jobs in sequence, feeding the output of one into the next:

```python
from pydantic import BaseModel, Field
from smelt import Model, Job

# Step 1: Extract entities
class ExtractedEntity(BaseModel):
    company_name: str
    industry: str
    sentiment: str

# Step 2: Enrich with additional data
class EnrichedEntity(BaseModel):
    market_position: str = Field(description="Market position: leader, challenger, niche, emerging")
    growth_potential: str = Field(description="Growth potential: high, medium, low")
    recommendation: str = Field(description="Brief investment recommendation")

model = Model(provider="openai", name="gpt-4.1-mini")

# First pass: extract
extract_job = Job(
    prompt="Extract the company name, industry, and overall sentiment from each news article.",
    output_model=ExtractedEntity,
    batch_size=10,
)

articles = [
    {"headline": "Apple reports record Q4 revenue driven by iPhone sales", "body": "..."},
    {"headline": "Stripe expands to 10 new markets in Southeast Asia", "body": "..."},
]

extract_result = extract_job.run(model, data=articles)

# Convert Pydantic models back to dicts for the second pass
enrichment_data = [row.model_dump() for row in extract_result.data]

# Second pass: enrich
enrich_job = Job(
    prompt="Based on the extracted entity information, assess the company's market position, "
           "growth potential, and provide a brief recommendation.",
    output_model=EnrichedEntity,
    batch_size=10,
)

enrich_result = enrich_job.run(model, data=enrichment_data)

# Combine results
for extracted, enriched in zip(extract_result.data, enrich_result.data):
    print(f"{extracted.company_name}: {enriched.market_position} — {enriched.recommendation}")
```

## Comparing providers

Run the same job across multiple providers to compare quality:

```python
from smelt import Model, Job

models = {
    "gpt-4.1-mini": Model(provider="openai", name="gpt-4.1-mini"),
    "claude-sonnet": Model(provider="anthropic", name="claude-sonnet-4-6"),
    "gemini-flash": Model(provider="google_genai", name="gemini-3-flash-preview"),
}

job = Job(
    prompt="Classify each company by industry sector",
    output_model=Classification,
    batch_size=10,
)

for name, model in models.items():
    result = job.run(model, data=companies)
    print(f"\n{name}: {result.metrics.wall_time_seconds:.2f}s, "
          f"{result.metrics.input_tokens + result.metrics.output_tokens} tokens")
    for row in result.data:
        print(f"  {row}")
```

## Temperature comparison

Compare deterministic vs creative outputs:

```python
for temp in [0, 0.5, 1.0]:
    model = Model(
        provider="openai",
        name="gpt-4.1-mini",
        params={"temperature": temp},
    )
    result = job.run(model, data=data[:3])
    print(f"\ntemperature={temp}:")
    for row in result.data:
        print(f"  {row}")
```

## Retrying failed rows

When some rows fail, retry just those:

```python
job = Job(
    prompt="...",
    output_model=MyModel,
    batch_size=10,
    stop_on_exhaustion=False,  # Collect errors, don't raise
)

result = job.run(model, data=data)

if not result.success:
    # Get indices of failed rows
    failed_ids = set()
    for err in result.errors:
        failed_ids.update(err.row_ids)

    print(f"{len(failed_ids)} rows failed, retrying with smaller batches...")

    # Retry with smaller batch_size and more retries
    retry_job = Job(
        prompt="...",
        output_model=MyModel,
        batch_size=1,       # One row at a time
        max_retries=5,      # More retries
        stop_on_exhaustion=False,
    )

    failed_data = [data[i] for i in sorted(failed_ids)]
    retry_result = retry_job.run(model, data=failed_data)

    # Merge results
    all_data = list(result.data)  # Successful rows from first run
    # Note: you'd need to map retry results back to original indices
    print(f"Recovered {len(retry_result.data)} of {len(failed_ids)} failed rows")
```

## Processing large datasets

For very large datasets, process in chunks to manage memory and provide progress:

```python
import time

CHUNK_SIZE = 500  # Process 500 rows at a time

job = Job(
    prompt="...",
    output_model=MyModel,
    batch_size=20,
    concurrency=5,
    stop_on_exhaustion=False,
)

all_results = []
all_errors = []
total_tokens = 0

for i in range(0, len(data), CHUNK_SIZE):
    chunk = data[i:i + CHUNK_SIZE]
    print(f"Processing rows {i}–{i + len(chunk) - 1}...")

    result = job.run(model, data=chunk)

    all_results.extend(result.data)
    all_errors.extend(result.errors)
    total_tokens += result.metrics.input_tokens + result.metrics.output_tokens

    print(f"  Done: {result.metrics.successful_rows}/{result.metrics.total_rows} rows, "
          f"{result.metrics.wall_time_seconds:.1f}s")

print(f"\nTotal: {len(all_results)} rows, {total_tokens:,} tokens, {len(all_errors)} errors")
```

## Using with pandas

```python
import pandas as pd
from smelt import Model, Job

# Load data
df = pd.read_csv("companies.csv")

# Convert DataFrame to list of dicts
data = df.to_dict(orient="records")

# Run transformation
job = Job(prompt="...", output_model=Classification, batch_size=10)
result = job.run(model, data=data)

# Convert results back to DataFrame
result_df = pd.DataFrame([row.model_dump() for row in result.data])

# Combine with original data
combined = pd.concat([df, result_df], axis=1)
print(combined.head())
```

## Using with CSV files

```python
import csv

def load_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_results(path: str, data: list, result_data: list):
    """Save original data + results to CSV."""
    rows = []
    for original, result in zip(data, result_data):
        row = {**original, **result.model_dump()}
        rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

# Usage
data = load_csv("input.csv")
result = job.run(model, data=data)
save_results("output.csv", data, result.data)
```

## Custom Pydantic validators

Your output model's validators run on every row. Use them for business logic:

```python
from pydantic import BaseModel, Field, field_validator, model_validator

class PricingAnalysis(BaseModel):
    suggested_price: float = Field(description="Suggested retail price in USD")
    margin_percent: float = Field(description="Expected profit margin as a percentage (0-100)")
    price_tier: str = Field(description="Price tier: budget, mid-range, premium, luxury")

    @field_validator("margin_percent")
    @classmethod
    def margin_in_range(cls, v: float) -> float:
        if not (0 <= v <= 100):
            raise ValueError(f"margin_percent must be 0-100, got {v}")
        return v

    @field_validator("suggested_price")
    @classmethod
    def price_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"suggested_price must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def tier_matches_price(self) -> "PricingAnalysis":
        if self.price_tier == "budget" and self.suggested_price > 50:
            raise ValueError(f"Budget tier but price is ${self.suggested_price}")
        return self
```

!!! info "Validators trigger retries"
    If a Pydantic validator raises `ValueError`, smelt treats it as a validation error and retries the entire batch. This means your custom validators act as quality gates — the LLM keeps trying until it produces output that passes all your validators.

## Async with asyncio.gather

Run multiple independent jobs concurrently:

```python
import asyncio

async def main():
    model = Model(provider="openai", name="gpt-4.1-mini")

    classify_job = Job(prompt="Classify by industry", output_model=Classification, batch_size=10)
    sentiment_job = Job(prompt="Analyze sentiment", output_model=Sentiment, batch_size=10)
    summary_job = Job(prompt="Summarize each company", output_model=Summary, batch_size=5)

    # Run all three jobs concurrently
    classify_result, sentiment_result, summary_result = await asyncio.gather(
        classify_job.arun(model, data=companies),
        sentiment_job.arun(model, data=reviews),
        summary_job.arun(model, data=companies),
    )

    print(f"Classification: {len(classify_result.data)} rows")
    print(f"Sentiment: {len(sentiment_result.data)} rows")
    print(f"Summaries: {len(summary_result.data)} rows")

asyncio.run(main())
```
