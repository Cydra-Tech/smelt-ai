"""Classify companies by industry sector using smelt.

Usage:
    pip install smelt-ai[openai]
    export OPENAI_API_KEY="sk-..."
    python examples/classification.py
"""

from pydantic import BaseModel, Field

from smelt import Job, Model


class Classification(BaseModel):
    """Classification output for each company."""

    sector: str = Field(description="Primary industry sector")
    sub_sector: str = Field(description="More specific sub-sector")
    is_public: bool = Field(description="Whether the company is publicly traded")


def main() -> None:
    model = Model(provider="openai", name="gpt-4.1-mini")

    job = Job(
        prompt="Classify each company by its primary industry sector and sub-sector. "
        "Determine if the company is publicly traded.",
        output_model=Classification,
        batch_size=5,
        concurrency=2,
    )

    companies = [
        {"name": "Apple Inc.", "description": "Consumer electronics, software, and services"},
        {"name": "JPMorgan Chase", "description": "Global financial services and investment banking"},
        {"name": "Pfizer", "description": "Pharmaceutical company developing medicines and vaccines"},
        {"name": "Tesla", "description": "Electric vehicles and clean energy products"},
        {"name": "Spotify", "description": "Digital music and podcast streaming platform"},
        {"name": "Stripe", "description": "Payment processing platform for internet businesses"},
        {"name": "Mayo Clinic", "description": "Nonprofit academic medical center"},
        {"name": "SpaceX", "description": "Aerospace manufacturer and space transport services"},
    ]

    # Quick test with one row
    print("Testing with first row...")
    test_result = job.test(model, data=companies)
    print(f"  {test_result.data[0]}")
    print()

    # Full run
    print(f"Classifying {len(companies)} companies...")
    result = job.run(model, data=companies)

    print(f"Success: {result.success}")
    print(f"Time: {result.metrics.wall_time_seconds:.2f}s")
    print(f"Tokens: {result.metrics.input_tokens} in / {result.metrics.output_tokens} out")
    print()

    for company, classification in zip(companies, result.data):
        print(f"  {company['name']:20s} -> {classification.sector} / {classification.sub_sector} "
              f"(public: {classification.is_public})")


if __name__ == "__main__":
    main()
