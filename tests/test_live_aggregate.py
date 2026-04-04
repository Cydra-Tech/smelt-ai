"""Live integration tests for AggregateJob with real LLM APIs.

Tests tree-reduction aggregation against real datasets with verifiable
ground truth. Requires API keys in .env file.

Run with: uv run pytest tests/test_live_aggregate.py -v -s
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from smelt import AggregateJob, Model, SmeltResult

load_dotenv()

DATA_DIR: Path = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

ANTHROPIC_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
GEMINI_KEY: str | None = os.getenv("GEMINI_API_KEY")

skip_no_anthropic = pytest.mark.skipif(not ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
skip_no_gemini = pytest.mark.skipif(not GEMINI_KEY, reason="GEMINI_API_KEY not set")


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class PortfolioSummary(BaseModel):
    """Aggregate summary of a company portfolio."""

    total_companies: int = Field(description="Total number of companies")
    sectors: list[str] = Field(description="List of unique industry sectors")
    countries: list[str] = Field(description="List of unique countries")
    public_companies: int = Field(description="Number of publicly traded companies")
    private_companies: int = Field(description="Number of private companies")
    total_revenue_millions: float = Field(description="Sum of all company revenues in millions USD")
    top_5_by_revenue: list[str] = Field(description="Names of the top 5 companies by revenue")


class SurveyAnalysis(BaseModel):
    """Aggregate analysis of employee survey responses."""

    total_responses: int = Field(description="Total number of survey responses")
    average_satisfaction: float = Field(description="Average satisfaction score (1-5 scale)")
    departments: list[str] = Field(description="List of unique departments")
    common_complaints: list[str] = Field(description="Common complaint themes across all responses")
    common_praises: list[str] = Field(description="Common positive themes across all responses")
    lowest_rated_department: str = Field(description="Department with the lowest average satisfaction")
    highest_rated_department: str = Field(description="Department with the highest average satisfaction")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_portfolio() -> list[dict[str, Any]]:
    """Load the 60-company portfolio dataset."""
    path: Path = DATA_DIR / "portfolio_companies.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_survey() -> list[dict[str, Any]]:
    """Load the 50-employee survey dataset."""
    path: Path = DATA_DIR / "employee_survey.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Tests — Portfolio (Anthropic)
# ---------------------------------------------------------------------------


@skip_no_anthropic
class TestPortfolioAggregate:
    """Test portfolio aggregation with Claude."""

    def test_portfolio_60_companies(self) -> None:
        """Should aggregate 60 companies into one PortfolioSummary."""
        data: list[dict[str, Any]] = _load_portfolio()
        assert len(data) == 60

        model = Model(provider="anthropic", name="claude-sonnet-4-6")
        job = AggregateJob(
            prompt=(
                "Analyze this portfolio of companies. Count totals accurately. "
                "For revenue, sum all individual company revenues. "
                "List ALL unique sectors and countries found across the entire dataset."
            ),
            output_model=PortfolioSummary,
            batch_size=15,
            concurrency=2,
            max_retries=2,
        )

        result: SmeltResult[PortfolioSummary] = job.run(model, data=data)

        assert result.success, f"Errors: {result.errors}"
        assert len(result.data) == 1

        summary: PortfolioSummary = result.data[0]
        print(f"\n--- Portfolio Summary ---")
        print(f"  Total companies: {summary.total_companies}")
        print(f"  Sectors ({len(summary.sectors)}): {summary.sectors}")
        print(f"  Countries ({len(summary.countries)}): {summary.countries}")
        print(f"  Public: {summary.public_companies}, Private: {summary.private_companies}")
        print(f"  Total revenue: ${summary.total_revenue_millions:.1f}M")
        print(f"  Top 5: {summary.top_5_by_revenue}")
        print(f"  Metrics: {result.metrics}")

        # Verify hard facts
        assert summary.total_companies == 60
        assert summary.public_companies + summary.private_companies == 60
        assert len(summary.sectors) >= 6  # at least 6 of the 7 sectors
        assert len(summary.countries) >= 10  # at least 10 of the 14 countries
        assert len(summary.top_5_by_revenue) == 5


# ---------------------------------------------------------------------------
# Tests — Survey (Gemini)
# ---------------------------------------------------------------------------


@skip_no_gemini
class TestSurveyAggregate:
    """Test survey aggregation with Gemini."""

    def test_survey_50_responses(self) -> None:
        """Should aggregate 50 survey responses into one SurveyAnalysis."""
        data: list[dict[str, Any]] = _load_survey()
        assert len(data) == 50

        model = Model(
            provider="google_genai",
            name="gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        job = AggregateJob(
            prompt=(
                "Analyze all employee survey responses. Count totals accurately. "
                "Calculate the true average satisfaction across ALL responses. "
                "List ALL unique departments. Identify common complaint and praise themes. "
                "Determine which department has the lowest and highest average satisfaction."
            ),
            output_model=SurveyAnalysis,
            batch_size=10,
            concurrency=2,
            max_retries=2,
        )

        result: SmeltResult[SurveyAnalysis] = job.run(model, data=data)

        assert result.success, f"Errors: {result.errors}"
        assert len(result.data) == 1

        analysis: SurveyAnalysis = result.data[0]
        print(f"\n--- Survey Analysis ---")
        print(f"  Total responses: {analysis.total_responses}")
        print(f"  Avg satisfaction: {analysis.average_satisfaction:.2f}")
        print(f"  Departments ({len(analysis.departments)}): {analysis.departments}")
        print(f"  Lowest rated: {analysis.lowest_rated_department}")
        print(f"  Highest rated: {analysis.highest_rated_department}")
        print(f"  Complaints: {analysis.common_complaints}")
        print(f"  Praises: {analysis.common_praises}")
        print(f"  Metrics: {result.metrics}")

        # Verify hard facts
        assert analysis.total_responses == 50
        assert 2.0 <= analysis.average_satisfaction <= 5.0
        assert len(analysis.departments) >= 5  # at least 5 of the 6 departments
        assert len(analysis.common_complaints) >= 2
        assert len(analysis.common_praises) >= 2


# ---------------------------------------------------------------------------
# Tests — Free-text mode
# ---------------------------------------------------------------------------


@skip_no_anthropic
class TestFreeTextAggregate:
    """Test free-text aggregation."""

    def test_free_text_portfolio_summary(self) -> None:
        """Should produce a single text summary of the portfolio."""
        data: list[dict[str, Any]] = _load_portfolio()

        model = Model(provider="anthropic", name="claude-sonnet-4-6")
        job = AggregateJob(
            prompt="Write an executive summary of this company portfolio.",
            batch_size=20,
            concurrency=2,
            max_retries=2,
        )

        result: SmeltResult[str] = job.run(model, data=data)

        assert result.success, f"Errors: {result.errors}"
        assert len(result.data) == 1
        assert isinstance(result.data[0], str)
        assert len(result.data[0]) > 50  # should be a substantial summary

        print(f"\n--- Free-text Portfolio Summary ---")
        print(f"  Length: {len(result.data[0])} chars")
        print(f"  Preview: {result.data[0][:200]}...")
        print(f"  Metrics: {result.metrics}")
