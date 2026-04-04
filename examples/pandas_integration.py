"""Use smelt with pandas DataFrames.

Usage:
    pip install smelt-ai[openai] pandas
    export OPENAI_API_KEY="sk-..."
    python examples/pandas_integration.py
"""

import pandas as pd
from pydantic import BaseModel, Field

from smelt import Job, Model


class Enrichment(BaseModel):
    """Enriched company data."""

    sector: str = Field(description="Primary industry sector")
    market_cap_tier: str = Field(description="Market cap tier: mega, large, mid, small, micro")
    risk_level: str = Field(description="Investment risk: low, medium, high")


def main() -> None:
    # Create a DataFrame
    df = pd.DataFrame([
        {"name": "Apple", "revenue_b": 394, "employees": 164000},
        {"name": "Stripe", "revenue_b": 14, "employees": 8000},
        {"name": "Pfizer", "revenue_b": 100, "employees": 83000},
        {"name": "SpaceX", "revenue_b": 8, "employees": 13000},
        {"name": "Toyota", "revenue_b": 275, "employees": 375000},
    ])

    print("Input DataFrame:")
    print(df.to_string(index=False))
    print()

    # Convert to list of dicts for smelt
    data = df.to_dict(orient="records")

    model = Model(provider="openai", name="gpt-4.1-mini")
    job = Job(
        prompt="Classify each company by sector, market cap tier, and investment risk level.",
        output_model=Enrichment,
        batch_size=5,
    )

    result = job.run(model, data=data)

    # Convert results back to DataFrame
    result_df = pd.DataFrame([row.model_dump() for row in result.data])

    # Combine original + enriched data
    combined = pd.concat([df, result_df], axis=1)

    print("Enriched DataFrame:")
    print(combined.to_string(index=False))
    print()
    print(f"Time: {result.metrics.wall_time_seconds:.2f}s")


if __name__ == "__main__":
    main()
