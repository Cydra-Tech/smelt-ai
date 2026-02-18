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
- **Provider agnostic** — works with any LangChain-supported LLM (OpenAI, Anthropic, Google, etc.)
- **Detailed metrics** — token usage, timing, retry counts, and per-batch error tracking

## Quick Install

```bash
pip install smelt-ai[openai]      # OpenAI models
pip install smelt-ai[anthropic]   # Anthropic models
pip install smelt-ai[google]      # Google Gemini models
```

Requires Python 3.10+.

## Next Steps

- [Installation](getting-started/installation.md) — set up smelt with your provider
- [Quickstart](getting-started/quickstart.md) — build your first transformation in 5 minutes
- [Architecture](guide/architecture.md) — understand how smelt works under the hood
- [API Reference](api/job.md) — full API documentation
