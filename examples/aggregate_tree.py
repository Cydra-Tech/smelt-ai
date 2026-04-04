"""Aggregate a dataset into a single summary using tree-parallel reduction.

Usage:
    pip install smelt-ai[anthropic]
    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/aggregate_tree.py
"""

from pydantic import BaseModel, Field

from smelt import AggregateJob, Model


class PortfolioSummary(BaseModel):
    """Aggregate summary of a company portfolio."""

    total_companies: int = Field(description="Total number of companies")
    sectors: list[str] = Field(description="List of unique industry sectors")
    countries: list[str] = Field(description="List of unique countries")
    public_companies: int = Field(description="Number of publicly traded companies")
    private_companies: int = Field(description="Number of private companies")
    top_5_by_revenue: list[str] = Field(description="Names of the top 5 companies by revenue")


def main() -> None:
    # Sample dataset — 20 companies
    companies = [
        {"name": "Apple", "sector": "Technology", "revenue_m": 394, "is_public": True, "country": "USA"},
        {"name": "Microsoft", "sector": "Technology", "revenue_m": 212, "is_public": True, "country": "USA"},
        {"name": "Amazon", "sector": "Technology", "revenue_m": 575, "is_public": True, "country": "USA"},
        {"name": "Alphabet", "sector": "Technology", "revenue_m": 307, "is_public": True, "country": "USA"},
        {"name": "Samsung", "sector": "Technology", "revenue_m": 244, "is_public": True, "country": "South Korea"},
        {"name": "JPMorgan", "sector": "Finance", "revenue_m": 155, "is_public": True, "country": "USA"},
        {"name": "Goldman Sachs", "sector": "Finance", "revenue_m": 47, "is_public": True, "country": "USA"},
        {"name": "HSBC", "sector": "Finance", "revenue_m": 66, "is_public": True, "country": "UK"},
        {"name": "Pfizer", "sector": "Healthcare", "revenue_m": 100, "is_public": True, "country": "USA"},
        {"name": "Roche", "sector": "Healthcare", "revenue_m": 66, "is_public": True, "country": "Switzerland"},
        {"name": "ExxonMobil", "sector": "Energy", "revenue_m": 414, "is_public": True, "country": "USA"},
        {"name": "Shell", "sector": "Energy", "revenue_m": 386, "is_public": True, "country": "UK"},
        {"name": "Toyota", "sector": "Automotive", "revenue_m": 275, "is_public": True, "country": "Japan"},
        {"name": "Tesla", "sector": "Automotive", "revenue_m": 97, "is_public": True, "country": "USA"},
        {"name": "BMW", "sector": "Automotive", "revenue_m": 154, "is_public": True, "country": "Germany"},
        {"name": "SpaceX", "sector": "Aerospace", "revenue_m": 8, "is_public": False, "country": "USA"},
        {"name": "Stripe", "sector": "Fintech", "revenue_m": 14, "is_public": False, "country": "USA"},
        {"name": "Databricks", "sector": "Technology", "revenue_m": 2, "is_public": False, "country": "USA"},
        {"name": "Revolut", "sector": "Fintech", "revenue_m": 2, "is_public": False, "country": "UK"},
        {"name": "Canva", "sector": "Technology", "revenue_m": 2, "is_public": False, "country": "Australia"},
    ]

    model = Model(provider="anthropic", name="claude-sonnet-4-6")

    job = AggregateJob(
        prompt="Analyze this portfolio of companies. Count totals accurately. "
        "List ALL unique sectors and countries. Identify the top 5 companies by revenue.",
        output_model=PortfolioSummary,
        strategy="tree",
        batch_size=5,
        concurrency=2,
    )

    print(f"Aggregating {len(companies)} companies (tree strategy, batch_size=5)...\n")
    result = job.run(model, data=companies)

    summary = result.data[0]
    print(f"Total companies:  {summary.total_companies}")
    print(f"Sectors:          {summary.sectors}")
    print(f"Countries:        {summary.countries}")
    print(f"Public:           {summary.public_companies}")
    print(f"Private:          {summary.private_companies}")
    print(f"Top 5 by revenue: {summary.top_5_by_revenue}")
    print()
    print(f"Steps: {result.metrics.total_batches}")
    print(f"Time:  {result.metrics.wall_time_seconds:.2f}s")
    print(f"Tokens: {result.metrics.input_tokens} in / {result.metrics.output_tokens} out")


if __name__ == "__main__":
    main()
