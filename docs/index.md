# Smelt AI

**LLM-powered structured data transformation.**

Feed in rows of data, get back strictly typed Pydantic models — batched, concurrent, and validated.

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
    print(row)
    # Classification(sector='Technology', sub_sector='Consumer Electronics', is_public=True)
```

## Features

- **Structured output** — define your schema with Pydantic, get back validated typed objects
- **Batch processing** — automatically splits data into batches for efficient LLM calls
- **Concurrent execution** — async semaphore-based concurrency, no threads or process pools
- **Automatic retries** — exponential backoff with jitter for transient failures
- **Row ordering** — results always match original input order, regardless of batch completion order
- **Test mode** — validate your setup with a single row before running the full dataset
- **Provider agnostic** — works with any LangChain-supported LLM (OpenAI, Anthropic, Google, etc.)
- **Detailed metrics** — token usage, timing, retry counts, and per-batch error tracking
- **Flexible error handling** — fail fast or collect errors, with partial results always available

## Quick install

```bash
pip install smelt-ai[openai]      # OpenAI models
pip install smelt-ai[anthropic]   # Anthropic models
pip install smelt-ai[google]      # Google Gemini models
```

Requires Python 3.10+.

## How it works

```
list[dict]  →  Tag with row_id  →  Split into batches  →  Concurrent LLM calls  →  Validate  →  Reorder  →  SmeltResult[T]
```

1. Your input rows get tagged with positional IDs for tracking
2. Rows are split into batches of configurable size
3. Batches run concurrently through the LLM with structured output
4. Each response is validated (schema, row IDs, count)
5. Results are reordered to match your original input order
6. Everything is packaged into a typed `SmeltResult` with metrics

[Learn more about the architecture →](guide/architecture.md)

## Documentation

### Getting Started
- [Installation](getting-started/installation.md) — set up smelt with your LLM provider
- [Quickstart](getting-started/quickstart.md) — build your first transformation in 5 minutes

### Guide
- [Architecture](guide/architecture.md) — how smelt works under the hood
- [Batching & Concurrency](guide/batching.md) — tune batch_size, concurrency, shuffle, and retries
- [Writing Prompts](guide/prompts.md) — write prompts that produce consistent results
- [Error Handling](guide/errors.md) — strategies for handling failures
- [Providers](guide/providers.md) — provider setup, model recommendations, and cost comparison

### Cookbook
- [Classification](cookbook/classification.md) — categorize data with fixed or open-ended labels
- [Sentiment Analysis](cookbook/sentiment.md) — extract sentiment, emotions, and opinions
- [Data Extraction](cookbook/extraction.md) — parse structured fields from unstructured text
- [Data Enrichment](cookbook/enrichment.md) — add summaries, translations, and generated content
- [Advanced Patterns](cookbook/advanced.md) — chaining, retries, pandas, large datasets

### API Reference
- [Model](api/model.md) — LLM provider configuration
- [Job](api/job.md) — transformation definition and execution
- [Results & Metrics](api/results.md) — SmeltResult, SmeltMetrics, BatchError
- [Errors](api/errors.md) — exception hierarchy

### [Changelog](changelog.md)
