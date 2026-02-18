# Cookbook: Sentiment Analysis

Extract sentiment, emotion, and opinion signals from text data.

## Basic sentiment analysis

```python
from typing import Literal
from pydantic import BaseModel, Field
from smelt import Model, Job

class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "mixed", "neutral"] = Field(
        description="Overall sentiment"
    )
    score: float = Field(
        description="Sentiment score from 0.0 (most negative) to 1.0 (most positive)"
    )
    key_themes: list[str] = Field(
        description="1-3 main themes or topics mentioned"
    )

model = Model(provider="openai", name="gpt-4.1-mini", params={"temperature": 0})
job = Job(
    prompt="Analyze the sentiment of each product review. "
           "Identify the overall sentiment, assign a score between 0.0 and 1.0, "
           "and extract 1-3 key themes.",
    output_model=SentimentAnalysis,
    batch_size=10,
)

reviews = [
    {"product": "Sony WH-1000XM5", "review": "Best noise cancelling I've ever used. Battery life is incredible. A bit pricey though."},
    {"product": "Kindle Paperwhite", "review": "Screen is great for reading but the UI is slow and frustrating."},
    {"product": "Instant Pot", "review": "Absolute game changer for weeknight dinners. Easy to clean too."},
]

result = job.run(model, data=reviews)

for review, sentiment in zip(reviews, result.data):
    print(f"{review['product']:25s} → {sentiment.sentiment:8s} ({sentiment.score:.2f}) themes={sentiment.key_themes}")
```

## Aspect-based sentiment

Analyze sentiment for specific aspects of a product:

```python
class AspectSentiment(BaseModel):
    quality: Literal["positive", "negative", "neutral", "not_mentioned"] = Field(
        description="Sentiment about build quality and materials"
    )
    value: Literal["positive", "negative", "neutral", "not_mentioned"] = Field(
        description="Sentiment about price/value ratio"
    )
    usability: Literal["positive", "negative", "neutral", "not_mentioned"] = Field(
        description="Sentiment about ease of use"
    )
    support: Literal["positive", "negative", "neutral", "not_mentioned"] = Field(
        description="Sentiment about customer support experience"
    )
    overall: Literal["positive", "negative", "mixed", "neutral"] = Field(
        description="Overall sentiment considering all aspects"
    )
    summary: str = Field(description="One-sentence summary of the review")

job = Job(
    prompt="Analyze each product review for sentiment across specific aspects: "
           "quality, value, usability, and support. "
           "Mark as 'not_mentioned' if the review doesn't address that aspect.",
    output_model=AspectSentiment,
    batch_size=5,
)
```

## Emotion detection

Go beyond positive/negative to detect specific emotions:

```python
class EmotionAnalysis(BaseModel):
    primary_emotion: Literal[
        "joy", "anger", "sadness", "fear", "surprise", "disgust", "trust", "anticipation"
    ] = Field(description="The dominant emotion expressed")
    intensity: Literal["low", "medium", "high"] = Field(
        description="How strongly the emotion is expressed"
    )
    tone: Literal["formal", "casual", "sarcastic", "urgent"] = Field(
        description="The overall tone of the message"
    )

job = Job(
    prompt="Detect the primary emotion, intensity, and tone in each customer message.",
    output_model=EmotionAnalysis,
    batch_size=10,
)
```

## Competitive sentiment analysis

Compare mentions of your brand vs competitors:

```python
class CompetitorMention(BaseModel):
    mentions_us: bool = Field(description="Whether our product is mentioned")
    mentions_competitor: bool = Field(description="Whether a competitor is mentioned")
    competitor_name: str = Field(description="Name of competitor mentioned, or 'none'")
    sentiment_toward_us: Literal["positive", "negative", "neutral", "not_mentioned"] = Field(
        description="Sentiment specifically toward our product"
    )
    sentiment_toward_competitor: Literal["positive", "negative", "neutral", "not_mentioned"] = Field(
        description="Sentiment specifically toward the competitor"
    )
    switching_intent: bool = Field(
        description="Whether the user expresses intent to switch products"
    )

job = Job(
    prompt="Analyze each social media post for mentions of our product (Acme CRM) "
           "and competitors (Salesforce, HubSpot, Pipedrive). "
           "Determine sentiment toward each and whether the user shows switching intent.",
    output_model=CompetitorMention,
    batch_size=10,
)
```

## Score validation

When using numeric scores, validate them after the run:

```python
result = job.run(model, data=reviews)

# Validate scores are in expected range
for i, row in enumerate(result.data):
    if not (0.0 <= row.score <= 1.0):
        print(f"WARNING: Row {i} has out-of-range score: {row.score}")
    # Verify sentiment matches score
    if row.sentiment == "positive" and row.score < 0.5:
        print(f"WARNING: Row {i} says 'positive' but score is {row.score}")
```

## Tips for sentiment tasks

1. **Use `temperature=0`** for consistent scoring across runs
2. **Add score range in the prompt** — "assign a score between 0.0 and 1.0" is clearer than just "assign a score"
3. **Include calibration examples** if you need consistent scoring: "0.0 = extremely negative, 0.5 = neutral, 1.0 = extremely positive"
4. **Use `Literal` for sentiment labels** — prevents the LLM from inventing labels like "somewhat positive"
5. **Validate numeric outputs** post-run — LLMs occasionally produce out-of-range values
6. **Batch size matters** — larger batches give the LLM context about the scoring distribution, which can improve consistency
