# Cookbook: Data Enrichment

Add new fields to your data using LLM knowledge — summaries, translations, inferences, and generated content.

## Company summaries

Generate concise summaries from raw company data:

```python
from typing import Literal
from pydantic import BaseModel, Field
from smelt import Model, Job

class CompanySummary(BaseModel):
    one_liner: str = Field(description="One sentence description")
    industry: str = Field(description="Primary industry")
    company_size: Literal["startup", "small", "medium", "large", "enterprise"] = Field(
        description="Size classification based on employee count"
    )
    age_years: int = Field(description="Approximate age in years")

model = Model(provider="openai", name="gpt-4.1-mini")
job = Job(
    prompt="Create a concise structured summary for each company. "
           "Calculate age based on the founded year (current year is 2026). "
           "Size classification: startup (<50), small (50-200), medium (200-1000), "
           "large (1000-10000), enterprise (10000+).",
    output_model=CompanySummary,
    batch_size=5,
    concurrency=3,
)

companies = [
    {"name": "Apple Inc.", "description": "Consumer electronics, software, and services", "founded": "1976", "employees": "164000"},
    {"name": "Stripe", "description": "Payment processing platform", "founded": "2010", "employees": "8000"},
    {"name": "Moderna", "description": "Biotechnology company focused on mRNA therapeutics", "founded": "2010", "employees": "3900"},
]

result = job.run(model, data=companies)
for row in result.data:
    print(f"{row.one_liner} [{row.industry}, {row.company_size}, {row.age_years}y]")
```

## Product description generation

Generate marketing copy from product specs:

```python
class ProductCopy(BaseModel):
    headline: str = Field(description="Catchy product headline (max 10 words)")
    description: str = Field(description="Marketing description (2-3 sentences)")
    target_audience: str = Field(description="Primary target audience")
    key_selling_points: list[str] = Field(description="3 key selling points as bullet points")

job = Job(
    prompt="Write compelling marketing copy for each product based on its specifications. "
           "The tone should be professional but approachable. "
           "Focus on benefits, not just features.",
    output_model=ProductCopy,
    batch_size=5,
    # Higher temperature for more creative output
)

model = Model(provider="openai", name="gpt-4.1-mini", params={"temperature": 0.7})

products = [
    {"name": "Sony WH-1000XM5", "price": "$348", "specs": "Wireless noise-cancelling headphones, 30hr battery, USB-C, 250g"},
    {"name": "Herman Miller Aeron", "price": "$1395", "specs": "Ergonomic office chair, mesh back, adjustable lumbar, 12yr warranty"},
]
```

## Translation

Translate text while preserving structure:

```python
class Translation(BaseModel):
    translated_text: str = Field(description="The translated text")
    source_language: str = Field(description="Detected source language (ISO 639-1 code)")
    formality: Literal["formal", "informal"] = Field(
        description="Formality level used in translation"
    )

job = Job(
    prompt="Translate each text to English. Preserve the original tone and meaning. "
           "Use formal English for business/technical content and informal for casual content. "
           "Detect the source language automatically.",
    output_model=Translation,
    batch_size=10,
)

texts = [
    {"text": "Bonjour, je voudrais réserver une table pour deux personnes ce soir."},
    {"text": "この製品は高品質の素材で作られています。"},
    {"text": "Hola! Qué tal? Vamos a la playa mañana?"},
]
```

## Data augmentation

Generate synthetic variations of existing data:

```python
class AugmentedReview(BaseModel):
    paraphrased: str = Field(description="Paraphrased version preserving the same sentiment and key points")
    shorter_version: str = Field(description="Condensed to 1-2 sentences")
    key_points: list[str] = Field(description="2-3 bullet point summary")

job = Job(
    prompt="For each product review, create a paraphrased version (different words, same meaning), "
           "a shorter condensed version (1-2 sentences), and extract 2-3 key points as bullet items.",
    output_model=AugmentedReview,
    batch_size=5,
)
```

## Knowledge enrichment

Add information the LLM knows about the entities:

```python
class CityEnrichment(BaseModel):
    country: str = Field(description="Country name")
    population: str = Field(description="Approximate population (e.g. '8.3 million')")
    timezone: str = Field(description="Primary timezone (e.g. 'America/New_York')")
    known_for: list[str] = Field(description="2-3 things the city is known for")
    cost_of_living: Literal["low", "medium", "high", "very_high"] = Field(
        description="Relative cost of living"
    )

job = Job(
    prompt="Enrich each city with additional information based on your knowledge. "
           "Population should be approximate and recent. "
           "Cost of living should be relative to other major global cities.",
    output_model=CityEnrichment,
    batch_size=10,
)

cities = [
    {"name": "San Francisco", "state": "California"},
    {"name": "Austin", "state": "Texas"},
    {"name": "Tokyo", "country": "Japan"},
]
```

!!! warning "LLM knowledge limitations"
    Enrichment relies on the LLM's training data, which has a knowledge cutoff date. For time-sensitive data (stock prices, population counts, current events), verify the LLM's outputs against authoritative sources.

## Suggested response generation

Generate responses for customer support tickets:

```python
class TicketResponse(BaseModel):
    category: str = Field(description="Ticket category: billing, technical, shipping, account, general")
    priority: Literal["low", "medium", "high", "urgent"] = Field(description="Priority level")
    suggested_response: str = Field(
        description="Professional suggested response to send to the customer (2-4 sentences)"
    )
    internal_notes: str = Field(
        description="Internal notes for the support team (1-2 sentences)"
    )
    requires_escalation: bool = Field(description="Whether this needs human review before sending")

job = Job(
    prompt="Triage each support ticket and draft a professional response. "
           "The response should acknowledge the issue, provide a solution or next step, "
           "and set expectations for resolution time. "
           "Mark for escalation if the issue involves: refunds over $100, legal threats, "
           "security concerns, or VIP customers.",
    output_model=TicketResponse,
    batch_size=5,
)
```

## Tips for enrichment tasks

1. **Use `temperature=0`** for factual enrichment (company info, city data) — you want consistency
2. **Use `temperature=0.5–0.8`** for creative enrichment (marketing copy, paraphrasing)
3. **Be specific about format** — "population as '8.3 million'" not just "population"
4. **Include date context** — "current year is 2026" helps the LLM calculate ages, durations
5. **Validate enriched data** — LLM knowledge can be wrong, especially for niche entities
6. **Batch size trade-off** — larger batches are more efficient but may produce less detailed results per row
