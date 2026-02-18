# Cookbook: Classification

Classify structured data into predefined or open-ended categories.

## Industry classification

Classify companies by their industry sector:

```python
from typing import Literal
from pydantic import BaseModel, Field
from smelt import Model, Job

class IndustryClassification(BaseModel):
    sector: str = Field(description="Primary industry sector (e.g. Technology, Healthcare, Finance)")
    sub_sector: str = Field(description="More specific sub-sector (e.g. Cloud Computing, Pharmaceuticals)")
    is_public: bool = Field(description="Whether the company is publicly traded on a major exchange")

model = Model(provider="openai", name="gpt-4.1-mini")
job = Job(
    prompt="Classify each company by its primary industry sector and sub-sector. "
           "Determine if the company is publicly traded on a major stock exchange (NYSE, NASDAQ, etc.).",
    output_model=IndustryClassification,
    batch_size=10,
)

companies = [
    {"name": "Apple Inc.", "description": "Consumer electronics, software, and services", "founded": "1976"},
    {"name": "JPMorgan Chase", "description": "Global financial services and investment banking", "founded": "1799"},
    {"name": "Pfizer", "description": "Pharmaceutical company developing medicines and vaccines", "founded": "1849"},
    {"name": "Tesla", "description": "Electric vehicles and clean energy products", "founded": "2003"},
    {"name": "Spotify", "description": "Digital music and podcast streaming platform", "founded": "2006"},
]

result = job.run(model, data=companies)

for company, cls in zip(companies, result.data):
    print(f"{company['name']:20s} → {cls.sector} / {cls.sub_sector} (public: {cls.is_public})")
```

## Category assignment with constrained values

Use `Literal` types to restrict outputs to a fixed set of categories:

```python
class TicketCategory(BaseModel):
    category: Literal["billing", "technical", "shipping", "account", "general"] = Field(
        description="Support ticket category"
    )
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        description="Priority level based on urgency and business impact"
    )
    requires_human: bool = Field(
        description="Whether this ticket needs human escalation vs automated response"
    )

job = Job(
    prompt="Triage each support ticket. Assign a category, priority level, "
           "and determine if human escalation is needed. "
           "Urgent priority is for issues affecting billing or complete service outage.",
    output_model=TicketCategory,
    batch_size=10,
)

tickets = [
    {"ticket_id": "TK-1001", "message": "I was charged twice for my subscription", "product": "Pro Plan"},
    {"ticket_id": "TK-1002", "message": "App crashes when I try to export PDF", "product": "Desktop App"},
    {"ticket_id": "TK-1003", "message": "When will my order arrive?", "product": "Wireless Mouse"},
]

result = job.run(model, data=tickets)

for ticket, triage in zip(tickets, result.data):
    print(f"{ticket['ticket_id']}: {triage.category} / {triage.priority} (human: {triage.requires_human})")
```

## Multi-label classification

Use `list[str]` for multi-label outputs:

```python
class ArticleTags(BaseModel):
    primary_topic: str = Field(description="Main topic of the article")
    tags: list[str] = Field(description="1-5 relevant tags for the article")
    reading_level: Literal["beginner", "intermediate", "advanced"] = Field(
        description="Target reading level"
    )

job = Job(
    prompt="Analyze each article and assign a primary topic, relevant tags (1-5), "
           "and a reading level.",
    output_model=ArticleTags,
    batch_size=5,
)

articles = [
    {"title": "Introduction to Neural Networks", "abstract": "A beginner's guide to understanding how neural networks work..."},
    {"title": "Advanced Kubernetes Networking", "abstract": "Deep dive into CNI plugins, service meshes, and network policies..."},
]

result = job.run(model, data=articles)
for article, tags in zip(articles, result.data):
    print(f"{article['title']}")
    print(f"  Topic: {tags.primary_topic}")
    print(f"  Tags: {', '.join(tags.tags)}")
    print(f"  Level: {tags.reading_level}")
```

## Binary classification

Simple yes/no classification:

```python
class SpamCheck(BaseModel):
    is_spam: bool = Field(description="Whether the email is spam")
    confidence: float = Field(description="Confidence score from 0.0 (uncertain) to 1.0 (certain)")
    reason: str = Field(description="Brief explanation for the classification")

job = Job(
    prompt="Determine if each email is spam. Consider common spam indicators: "
           "urgency language, suspicious links, too-good-to-be-true offers, "
           "requests for personal information.",
    output_model=SpamCheck,
    batch_size=20,
)
```

## Tips for classification tasks

1. **Be specific in your prompt** — "Classify by GICS sector" is better than "classify by industry"
2. **Use `Literal` types** when you have a fixed set of categories — prevents hallucinated categories
3. **Add `Field(description=...)` to every field** — tells the LLM exactly what you expect
4. **Add a `confidence` field** — lets you filter low-confidence results for human review
5. **Include examples in your prompt** if the classification criteria are nuanced
6. **Use `temperature=0`** for deterministic, consistent classification:
   ```python
   model = Model(provider="openai", name="gpt-4.1-mini", params={"temperature": 0})
   ```
