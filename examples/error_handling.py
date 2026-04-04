"""Demonstrate error handling strategies with smelt.

Usage:
    pip install smelt-ai[openai]
    export OPENAI_API_KEY="sk-..."
    python examples/error_handling.py
"""

from pydantic import BaseModel, Field

from smelt import Job, Model
from smelt.errors import SmeltExhaustionError


class Sentiment(BaseModel):
    """Sentiment analysis output."""

    sentiment: str = Field(description="Sentiment: positive, negative, or neutral")
    score: float = Field(description="Confidence score between 0 and 1")


def fail_fast_example() -> None:
    """Stop on first batch failure (default behavior)."""
    print("=== Fail-fast mode (stop_on_exhaustion=True) ===\n")

    model = Model(provider="openai", name="gpt-4.1-mini")
    job = Job(
        prompt="Analyze the sentiment of each review.",
        output_model=Sentiment,
        batch_size=2,
        stop_on_exhaustion=True,
    )

    reviews = [
        {"text": "Absolutely love this product!"},
        {"text": "Terrible experience, would not recommend."},
        {"text": "It's okay, nothing special."},
        {"text": "Best purchase I've ever made."},
    ]

    try:
        result = job.run(model, data=reviews)
        print(f"Success! {len(result.data)} rows processed")
        for review, sentiment in zip(reviews, result.data):
            print(f"  {review['text'][:40]:40s} -> {sentiment.sentiment} ({sentiment.score:.2f})")
    except SmeltExhaustionError as e:
        print(f"Failed: {e}")
        print(f"Partial results: {len(e.partial_result.data)} rows")
        print(f"Errors: {len(e.partial_result.errors)} batch(es)")
        for err in e.partial_result.errors:
            print(f"  Batch {err.batch_index}: {err.message}")


def collect_errors_example() -> None:
    """Continue processing and collect errors."""
    print("\n=== Collect-errors mode (stop_on_exhaustion=False) ===\n")

    model = Model(provider="openai", name="gpt-4.1-mini")
    job = Job(
        prompt="Analyze the sentiment of each review.",
        output_model=Sentiment,
        batch_size=2,
        stop_on_exhaustion=False,
    )

    reviews = [
        {"text": "Great product with amazing features."},
        {"text": "Disappointed with the quality."},
        {"text": "Works as expected, fair price."},
    ]

    result = job.run(model, data=reviews)

    if result.success:
        print(f"All {len(result.data)} rows processed successfully!")
        for review, sentiment in zip(reviews, result.data):
            print(f"  {review['text'][:40]:40s} -> {sentiment.sentiment} ({sentiment.score:.2f})")
    else:
        print(f"Partial success: {result.metrics.successful_rows}/{result.metrics.total_rows}")
        for err in result.errors:
            print(f"  Batch {err.batch_index} failed: {err.message}")


def main() -> None:
    fail_fast_example()
    collect_errors_example()


if __name__ == "__main__":
    main()
