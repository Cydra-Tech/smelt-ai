"""Exhaustive live integration tests across OpenAI, Anthropic, and Google Gemini.

Tests real API calls with various models, parameters, datasets, and batch
configurations. Requires API keys in .env file.

Run with: uv run pytest tests/test_live.py -v -s
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Literal

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from smelt import Job, Model, SmeltResult

load_dotenv()

DATA_DIR: Path = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

OPENAI_KEY: str | None = os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
GEMINI_KEY: str | None = os.getenv("GEMINI_API_KEY")

skip_no_openai = pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")
skip_no_anthropic = pytest.mark.skipif(not ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
skip_no_gemini = pytest.mark.skipif(not GEMINI_KEY, reason="GEMINI_API_KEY not set")


# ---------------------------------------------------------------------------
# Output models for different test scenarios
# ---------------------------------------------------------------------------


class IndustryClassification(BaseModel):
    """Classification of a company by industry sector."""

    sector: str = Field(description="Primary industry sector (e.g. Technology, Finance, Healthcare)")
    sub_sector: str = Field(description="More specific sub-sector within the primary sector")
    is_public: bool = Field(description="Whether the company is publicly traded")


class SentimentAnalysis(BaseModel):
    """Sentiment analysis of a product review."""

    sentiment: Literal["positive", "negative", "mixed"] = Field(
        description="Overall sentiment of the review"
    )
    score: float = Field(description="Sentiment score from 0.0 (most negative) to 1.0 (most positive)")
    key_themes: list[str] = Field(description="Main themes mentioned in the review (1-3 items)")


class TicketTriage(BaseModel):
    """Support ticket classification and priority assignment."""

    category: str = Field(
        description="Ticket category: billing, technical, shipping, account, or general"
    )
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        description="Priority level based on urgency and impact"
    )
    requires_human: bool = Field(description="Whether this ticket needs human agent escalation")
    suggested_response: str = Field(description="A brief suggested response to the customer")


class ProductTag(BaseModel):
    """Product tagging with structured attributes."""

    primary_category: str = Field(description="Main product category")
    price_tier: Literal["budget", "mid-range", "premium", "luxury"] = Field(
        description="Price tier classification"
    )
    target_audience: str = Field(description="Primary target audience")
    sustainability_mention: bool = Field(
        description="Whether eco/sustainability features are mentioned"
    )


class CompanySummary(BaseModel):
    """Concise structured summary of a company."""

    one_liner: str = Field(description="One sentence description of what the company does")
    industry: str = Field(description="Primary industry")
    company_size: Literal["startup", "small", "medium", "large", "enterprise"] = Field(
        description="Company size classification based on employee count"
    )
    age_years: int = Field(description="Approximate age of the company in years")


# ---------------------------------------------------------------------------
# CSV loader helper
# ---------------------------------------------------------------------------


def load_csv(filename: str) -> list[dict[str, str]]:
    """Load a CSV file from the test data directory.

    Args:
        filename: Name of the CSV file in tests/data/.

    Returns:
        List of row dictionaries.
    """
    filepath: Path = DATA_DIR / filename
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def assert_result_valid(
    result: SmeltResult[Any],
    expected_count: int,
    output_type: type[BaseModel],
) -> None:
    """Assert that a SmeltResult is valid and complete.

    Args:
        result: The SmeltResult to validate.
        expected_count: Expected number of output rows.
        output_type: Expected Pydantic model type for each row.
    """
    assert result.success, f"Run failed with errors: {result.errors}"
    assert len(result.data) == expected_count, (
        f"Expected {expected_count} rows, got {len(result.data)}"
    )
    for i, row in enumerate(result.data):
        assert isinstance(row, output_type), (
            f"Row {i} is {type(row).__name__}, expected {output_type.__name__}"
        )
    assert result.metrics.total_rows == expected_count
    assert result.metrics.successful_rows == expected_count
    assert result.metrics.failed_rows == 0
    assert result.metrics.input_tokens > 0
    assert result.metrics.output_tokens > 0
    assert result.metrics.wall_time_seconds > 0


def print_result_summary(
    provider: str,
    model_name: str,
    result: SmeltResult[Any],
    params: dict[str, Any] | None = None,
) -> None:
    """Print a summary of a test run for visibility.

    Args:
        provider: LLM provider name.
        model_name: Model identifier.
        result: The SmeltResult from the run.
        params: Optional model parameters used.
    """
    params_str: str = f" | params={params}" if params else ""
    print(f"\n{'='*70}")
    print(f"  {provider} / {model_name}{params_str}")
    print(f"  Success: {result.success} | Rows: {len(result.data)}")
    print(f"  Tokens: {result.metrics.input_tokens} in / {result.metrics.output_tokens} out")
    print(f"  Retries: {result.metrics.total_retries} | Time: {result.metrics.wall_time_seconds}s")
    if result.data:
        print(f"  Sample output: {result.data[0]}")
    if result.errors:
        print(f"  Errors: {result.errors}")
    print(f"{'='*70}")


# ===========================================================================
# OpenAI Tests
# ===========================================================================


class TestOpenAI:
    """Live tests against OpenAI API."""

    @skip_no_openai
    def test_gpt4o_mini_company_classification(self) -> None:
        """GPT-4o-mini: classify companies by industry."""
        data: list[dict[str, str]] = load_csv("companies.csv")
        model = Model(provider="openai", name="gpt-4o-mini", api_key=OPENAI_KEY)
        job = Job(
            prompt="Classify each company by its primary industry sector and sub-sector. "
            "Determine if the company is publicly traded.",
            output_model=IndustryClassification,
            batch_size=5,
            concurrency=2,
            max_retries=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("OpenAI", "gpt-4o-mini", result)
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_openai
    def test_gpt4o_mini_sentiment_analysis(self) -> None:
        """GPT-4o-mini: sentiment analysis on product reviews."""
        data: list[dict[str, str]] = load_csv("products.csv")
        model = Model(provider="openai", name="gpt-4o-mini", api_key=OPENAI_KEY)
        job = Job(
            prompt="Analyze the sentiment of each product's customer_review. "
            "Identify the overall sentiment, assign a score, and extract key themes.",
            output_model=SentimentAnalysis,
            batch_size=5,
            concurrency=2,
            max_retries=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[SentimentAnalysis] = job.run(model, data=data)
        print_result_summary("OpenAI", "gpt-4o-mini", result)
        assert_result_valid(result, len(data), SentimentAnalysis)
        for row in result.data:
            assert 0.0 <= row.score <= 1.0, f"Score {row.score} out of range"
            assert row.sentiment in {"positive", "negative", "mixed"}

    @skip_no_openai
    def test_gpt4o_mini_ticket_triage(self) -> None:
        """GPT-4o-mini: triage support tickets."""
        data: list[dict[str, str]] = load_csv("support_tickets.csv")
        model = Model(provider="openai", name="gpt-4o-mini", api_key=OPENAI_KEY)
        job = Job(
            prompt="Triage each support ticket. Classify by category, assign priority, "
            "determine if human escalation is needed, and suggest a brief response.",
            output_model=TicketTriage,
            batch_size=5,
            concurrency=2,
            max_retries=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[TicketTriage] = job.run(model, data=data)
        print_result_summary("OpenAI", "gpt-4o-mini", result)
        assert_result_valid(result, len(data), TicketTriage)
        for row in result.data:
            assert row.priority in {"low", "medium", "high", "urgent"}

    @skip_no_openai
    def test_gpt4o_company_summary(self) -> None:
        """GPT-4o: structured company summaries."""
        data: list[dict[str, str]] = load_csv("companies.csv")
        model = Model(provider="openai", name="gpt-4o", api_key=OPENAI_KEY)
        job = Job(
            prompt="Create a concise structured summary for each company. "
            "Calculate the approximate age based on the founded year (current year is 2026).",
            output_model=CompanySummary,
            batch_size=10,
            concurrency=1,
            max_retries=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[CompanySummary] = job.run(model, data=data)
        print_result_summary("OpenAI", "gpt-4o", result)
        assert_result_valid(result, len(data), CompanySummary)
        for row in result.data:
            assert row.company_size in {"startup", "small", "medium", "large", "enterprise"}
            assert row.age_years > 0

    @skip_no_openai
    def test_gpt4o_mini_temperature_zero(self) -> None:
        """GPT-4o-mini with temperature=0: deterministic output."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:3]
        model = Model(
            provider="openai",
            name="gpt-4o-mini",
            api_key=OPENAI_KEY,
            params={"temperature": 0},
        )
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("OpenAI", "gpt-4o-mini (temp=0)", result, {"temperature": 0})
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_openai
    def test_gpt4o_mini_temperature_high(self) -> None:
        """GPT-4o-mini with temperature=1.0: creative output still validates."""
        data: list[dict[str, str]] = load_csv("products.csv")[:3]
        model = Model(
            provider="openai",
            name="gpt-4o-mini",
            api_key=OPENAI_KEY,
            params={"temperature": 1.0},
        )
        job = Job(
            prompt="Analyze the sentiment of each product review.",
            output_model=SentimentAnalysis,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[SentimentAnalysis] = job.run(model, data=data)
        print_result_summary("OpenAI", "gpt-4o-mini (temp=1.0)", result, {"temperature": 1.0})
        assert_result_valid(result, len(data), SentimentAnalysis)

    @skip_no_openai
    def test_gpt4o_mini_top_p(self) -> None:
        """GPT-4o-mini with top_p=0.5: restricted sampling."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:3]
        model = Model(
            provider="openai",
            name="gpt-4o-mini",
            api_key=OPENAI_KEY,
            params={"top_p": 0.5},
        )
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("OpenAI", "gpt-4o-mini (top_p=0.5)", result, {"top_p": 0.5})
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_openai
    def test_gpt4o_mini_max_tokens(self) -> None:
        """GPT-4o-mini with max_tokens set: output fits within limit."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:2]
        model = Model(
            provider="openai",
            name="gpt-4o-mini",
            api_key=OPENAI_KEY,
            params={"max_tokens": 4096},
        )
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("OpenAI", "gpt-4o-mini (max_tokens=4096)", result, {"max_tokens": 4096})
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_openai
    def test_gpt4o_single_row_batch(self) -> None:
        """GPT-4o: batch_size=1, one row per LLM call."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:3]
        model = Model(provider="openai", name="gpt-4o", api_key=OPENAI_KEY)
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=1,
            concurrency=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("OpenAI", "gpt-4o (batch=1, conc=3)", result)
        assert_result_valid(result, len(data), IndustryClassification)
        assert result.metrics.total_batches == 3

    @skip_no_openai
    def test_gpt4o_full_dataset_concurrent(self) -> None:
        """GPT-4o: full 10-row dataset with concurrency=3, batch_size=4."""
        data: list[dict[str, str]] = load_csv("companies.csv")
        model = Model(
            provider="openai", name="gpt-4o", api_key=OPENAI_KEY,
            params={"temperature": 0},
        )
        job = Job(
            prompt="Classify each company by industry sector and determine if publicly traded.",
            output_model=IndustryClassification,
            batch_size=4,
            concurrency=3,
            max_retries=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("OpenAI", "gpt-4o (batch=4, conc=3, full)", result)
        assert_result_valid(result, len(data), IndustryClassification)
        assert result.metrics.total_batches == 3  # ceil(10/4)


# ===========================================================================
# Anthropic Tests
# ===========================================================================


class TestAnthropic:
    """Live tests against Anthropic API."""

    @skip_no_anthropic
    def test_haiku_company_classification(self) -> None:
        """Claude Haiku 3.5: classify companies by industry."""
        data: list[dict[str, str]] = load_csv("companies.csv")
        model = Model(
            provider="anthropic", name="claude-3-5-haiku-20241022", api_key=ANTHROPIC_KEY,
        )
        job = Job(
            prompt="Classify each company by its primary industry sector and sub-sector. "
            "Determine if the company is publicly traded.",
            output_model=IndustryClassification,
            batch_size=5,
            concurrency=2,
            max_retries=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("Anthropic", "claude-haiku-3.5", result)
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_anthropic
    def test_haiku_sentiment_analysis(self) -> None:
        """Claude Haiku 3.5: sentiment analysis."""
        data: list[dict[str, str]] = load_csv("products.csv")
        model = Model(
            provider="anthropic", name="claude-3-5-haiku-20241022", api_key=ANTHROPIC_KEY,
        )
        job = Job(
            prompt="Analyze the sentiment of each product's customer_review. "
            "Identify the overall sentiment, assign a score, and extract key themes.",
            output_model=SentimentAnalysis,
            batch_size=5,
            concurrency=2,
            max_retries=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[SentimentAnalysis] = job.run(model, data=data)
        print_result_summary("Anthropic", "claude-haiku-3.5", result)
        assert_result_valid(result, len(data), SentimentAnalysis)
        for row in result.data:
            assert 0.0 <= row.score <= 1.0

    @skip_no_anthropic
    def test_sonnet_ticket_triage(self) -> None:
        """Claude Sonnet 3.7: triage support tickets."""
        data: list[dict[str, str]] = load_csv("support_tickets.csv")
        model = Model(
            provider="anthropic", name="claude-3-7-sonnet-20250219", api_key=ANTHROPIC_KEY,
        )
        job = Job(
            prompt="Triage each support ticket. Classify by category, assign priority, "
            "determine if human escalation is needed, and suggest a brief response.",
            output_model=TicketTriage,
            batch_size=5,
            concurrency=2,
            max_retries=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[TicketTriage] = job.run(model, data=data)
        print_result_summary("Anthropic", "claude-sonnet-3.7", result)
        assert_result_valid(result, len(data), TicketTriage)

    @skip_no_anthropic
    def test_sonnet_product_tagging(self) -> None:
        """Claude Sonnet 3.7: tag products with structured attributes."""
        data: list[dict[str, str]] = load_csv("products.csv")
        model = Model(
            provider="anthropic", name="claude-3-7-sonnet-20250219", api_key=ANTHROPIC_KEY,
        )
        job = Job(
            prompt="Tag each product with structured attributes. Classify the price tier, "
            "identify the target audience, and note if sustainability/eco features are mentioned.",
            output_model=ProductTag,
            batch_size=10,
            concurrency=1,
            max_retries=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[ProductTag] = job.run(model, data=data)
        print_result_summary("Anthropic", "claude-sonnet-3.7", result)
        assert_result_valid(result, len(data), ProductTag)
        for row in result.data:
            assert row.price_tier in {"budget", "mid-range", "premium", "luxury"}

    @skip_no_anthropic
    def test_haiku_temperature_zero(self) -> None:
        """Claude Haiku 3.5 with temperature=0: deterministic."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:3]
        model = Model(
            provider="anthropic",
            name="claude-3-5-haiku-20241022",
            api_key=ANTHROPIC_KEY,
            params={"temperature": 0},
        )
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("Anthropic", "claude-haiku-3.5 (temp=0)", result, {"temperature": 0})
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_anthropic
    def test_haiku_temperature_high(self) -> None:
        """Claude Haiku 3.5 with temperature=1.0: creative output still validates."""
        data: list[dict[str, str]] = load_csv("products.csv")[:3]
        model = Model(
            provider="anthropic",
            name="claude-3-5-haiku-20241022",
            api_key=ANTHROPIC_KEY,
            params={"temperature": 1.0},
        )
        job = Job(
            prompt="Analyze the sentiment of each product review.",
            output_model=SentimentAnalysis,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[SentimentAnalysis] = job.run(model, data=data)
        print_result_summary("Anthropic", "claude-haiku-3.5 (temp=1.0)", result, {"temperature": 1.0})
        assert_result_valid(result, len(data), SentimentAnalysis)

    @skip_no_anthropic
    def test_haiku_top_p_and_top_k(self) -> None:
        """Claude Haiku 3.5 with top_p=0.9 and top_k=40."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:3]
        model = Model(
            provider="anthropic",
            name="claude-3-5-haiku-20241022",
            api_key=ANTHROPIC_KEY,
            params={"top_p": 0.9, "top_k": 40},
        )
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary(
            "Anthropic", "claude-haiku-3.5 (top_p=0.9, top_k=40)",
            result, {"top_p": 0.9, "top_k": 40},
        )
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_anthropic
    def test_haiku_max_tokens(self) -> None:
        """Claude Haiku 3.5 with max_tokens=4096."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:2]
        model = Model(
            provider="anthropic",
            name="claude-3-5-haiku-20241022",
            api_key=ANTHROPIC_KEY,
            params={"max_tokens": 4096},
        )
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary(
            "Anthropic", "claude-haiku-3.5 (max_tokens=4096)",
            result, {"max_tokens": 4096},
        )
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_anthropic
    def test_haiku_single_row_batch(self) -> None:
        """Claude Haiku 3.5: batch_size=1 with concurrency."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:3]
        model = Model(
            provider="anthropic", name="claude-3-5-haiku-20241022", api_key=ANTHROPIC_KEY,
        )
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=1,
            concurrency=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("Anthropic", "claude-haiku-3.5 (batch=1, conc=3)", result)
        assert_result_valid(result, len(data), IndustryClassification)
        assert result.metrics.total_batches == 3

    @skip_no_anthropic
    def test_sonnet_full_dataset_concurrent(self) -> None:
        """Claude Sonnet 3.7: full 10-row dataset, batch_size=4, concurrency=3."""
        data: list[dict[str, str]] = load_csv("companies.csv")
        model = Model(
            provider="anthropic",
            name="claude-3-7-sonnet-20250219",
            api_key=ANTHROPIC_KEY,
            params={"temperature": 0},
        )
        job = Job(
            prompt="Create a concise structured summary for each company. "
            "Calculate age based on founded year (current year is 2026).",
            output_model=CompanySummary,
            batch_size=4,
            concurrency=3,
            max_retries=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[CompanySummary] = job.run(model, data=data)
        print_result_summary("Anthropic", "claude-sonnet-3.7 (batch=4, conc=3, full)", result)
        assert_result_valid(result, len(data), CompanySummary)
        assert result.metrics.total_batches == 3


# ===========================================================================
# Google Gemini Tests
# ===========================================================================


class TestGemini:
    """Live tests against Google Gemini API."""

    @skip_no_gemini
    def test_flash_company_classification(self) -> None:
        """Gemini 2.0 Flash: classify companies by industry."""
        data: list[dict[str, str]] = load_csv("companies.csv")
        model = Model(
            provider="google_genai", name="gemini-2.0-flash",
            api_key=GEMINI_KEY,
        )
        job = Job(
            prompt="Classify each company by its primary industry sector and sub-sector. "
            "Determine if the company is publicly traded.",
            output_model=IndustryClassification,
            batch_size=5,
            concurrency=2,
            max_retries=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("Gemini", "gemini-2.0-flash", result)
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_gemini
    def test_flash_sentiment_analysis(self) -> None:
        """Gemini 2.0 Flash: sentiment analysis."""
        data: list[dict[str, str]] = load_csv("products.csv")
        model = Model(
            provider="google_genai", name="gemini-2.0-flash",
            api_key=GEMINI_KEY,
        )
        job = Job(
            prompt="Analyze the sentiment of each product's customer_review. "
            "Identify the overall sentiment, assign a score, and extract key themes.",
            output_model=SentimentAnalysis,
            batch_size=5,
            concurrency=2,
            max_retries=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[SentimentAnalysis] = job.run(model, data=data)
        print_result_summary("Gemini", "gemini-2.0-flash", result)
        assert_result_valid(result, len(data), SentimentAnalysis)
        for row in result.data:
            assert 0.0 <= row.score <= 1.0

    @skip_no_gemini
    def test_flash_ticket_triage(self) -> None:
        """Gemini 2.0 Flash: triage support tickets."""
        data: list[dict[str, str]] = load_csv("support_tickets.csv")
        model = Model(
            provider="google_genai", name="gemini-2.0-flash",
            api_key=GEMINI_KEY,
        )
        job = Job(
            prompt="Triage each support ticket. Classify by category, assign priority, "
            "determine if human escalation is needed, and suggest a brief response.",
            output_model=TicketTriage,
            batch_size=5,
            concurrency=2,
            max_retries=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[TicketTriage] = job.run(model, data=data)
        print_result_summary("Gemini", "gemini-2.0-flash", result)
        assert_result_valid(result, len(data), TicketTriage)

    @skip_no_gemini
    def test_flash_product_tagging(self) -> None:
        """Gemini 2.0 Flash: tag products."""
        data: list[dict[str, str]] = load_csv("products.csv")
        model = Model(
            provider="google_genai", name="gemini-2.0-flash",
            api_key=GEMINI_KEY,
        )
        job = Job(
            prompt="Tag each product with structured attributes. Classify the price tier, "
            "identify the target audience, and note if sustainability/eco features are mentioned.",
            output_model=ProductTag,
            batch_size=10,
            concurrency=1,
            max_retries=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[ProductTag] = job.run(model, data=data)
        print_result_summary("Gemini", "gemini-2.0-flash", result)
        assert_result_valid(result, len(data), ProductTag)

    @skip_no_gemini
    def test_flash_temperature_zero(self) -> None:
        """Gemini 2.0 Flash with temperature=0."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:3]
        model = Model(
            provider="google_genai",
            name="gemini-2.0-flash",
            api_key=GEMINI_KEY,
            params={"temperature": 0},
        )
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("Gemini", "gemini-2.0-flash (temp=0)", result, {"temperature": 0})
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_gemini
    def test_flash_temperature_high(self) -> None:
        """Gemini 2.0 Flash with temperature=1.0."""
        data: list[dict[str, str]] = load_csv("products.csv")[:3]
        model = Model(
            provider="google_genai",
            name="gemini-2.0-flash",
            api_key=GEMINI_KEY,
            params={"temperature": 1.0},
        )
        job = Job(
            prompt="Analyze the sentiment of each product review.",
            output_model=SentimentAnalysis,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[SentimentAnalysis] = job.run(model, data=data)
        print_result_summary("Gemini", "gemini-2.0-flash (temp=1.0)", result, {"temperature": 1.0})
        assert_result_valid(result, len(data), SentimentAnalysis)

    @skip_no_gemini
    def test_flash_top_p(self) -> None:
        """Gemini 2.0 Flash with top_p=0.5."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:3]
        model = Model(
            provider="google_genai",
            name="gemini-2.0-flash",
            api_key=GEMINI_KEY,
            params={"top_p": 0.5},
        )
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("Gemini", "gemini-2.0-flash (top_p=0.5)", result, {"top_p": 0.5})
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_gemini
    def test_flash_max_tokens(self) -> None:
        """Gemini 2.0 Flash with max_output_tokens=4096."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:2]
        model = Model(
            provider="google_genai",
            name="gemini-2.0-flash",
            api_key=GEMINI_KEY,
            params={"max_output_tokens": 4096},
        )
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary(
            "Gemini", "gemini-2.0-flash (max_output_tokens=4096)",
            result, {"max_output_tokens": 4096},
        )
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_gemini
    def test_flash_single_row_batch(self) -> None:
        """Gemini 2.0 Flash: batch_size=1 with concurrency."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:3]
        model = Model(
            provider="google_genai", name="gemini-2.0-flash",
            api_key=GEMINI_KEY,
        )
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=1,
            concurrency=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("Gemini", "gemini-2.0-flash (batch=1, conc=3)", result)
        assert_result_valid(result, len(data), IndustryClassification)
        assert result.metrics.total_batches == 3

    @skip_no_gemini
    def test_flash_full_dataset_concurrent(self) -> None:
        """Gemini 2.0 Flash: full 10-row dataset, batch_size=4, concurrency=3."""
        data: list[dict[str, str]] = load_csv("companies.csv")
        model = Model(
            provider="google_genai",
            name="gemini-2.0-flash",
            api_key=GEMINI_KEY,
            params={"temperature": 0},
        )
        job = Job(
            prompt="Create a concise structured summary for each company. "
            "Calculate age based on founded year (current year is 2026).",
            output_model=CompanySummary,
            batch_size=4,
            concurrency=3,
            max_retries=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[CompanySummary] = job.run(model, data=data)
        print_result_summary("Gemini", "gemini-2.0-flash (batch=4, conc=3, full)", result)
        assert_result_valid(result, len(data), CompanySummary)
        assert result.metrics.total_batches == 3


# ===========================================================================
# Latest 2026 Models
# ===========================================================================


class TestLatestOpenAI:
    """Tests with the latest OpenAI models (2025-2026 generation)."""

    @skip_no_openai
    def test_gpt4_1_company_classification(self) -> None:
        """GPT-4.1: classify companies."""
        data: list[dict[str, str]] = load_csv("companies.csv")
        model = Model(provider="openai", name="gpt-4.1", api_key=OPENAI_KEY)
        job = Job(
            prompt="Classify each company by its primary industry sector and sub-sector. "
            "Determine if the company is publicly traded.",
            output_model=IndustryClassification,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("OpenAI", "gpt-4.1", result)
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_openai
    def test_gpt4_1_mini_sentiment(self) -> None:
        """GPT-4.1-mini: sentiment analysis."""
        data: list[dict[str, str]] = load_csv("products.csv")
        model = Model(
            provider="openai", name="gpt-4.1-mini", api_key=OPENAI_KEY,
            params={"temperature": 0},
        )
        job = Job(
            prompt="Analyze the sentiment of each product's customer_review. "
            "Identify the overall sentiment, assign a score, and extract key themes.",
            output_model=SentimentAnalysis,
            batch_size=5,
            concurrency=2,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[SentimentAnalysis] = job.run(model, data=data)
        print_result_summary("OpenAI", "gpt-4.1-mini", result)
        assert_result_valid(result, len(data), SentimentAnalysis)

    @skip_no_openai
    def test_gpt4_1_nano_ticket_triage(self) -> None:
        """GPT-4.1-nano: fast, cheapest OpenAI model for triage."""
        data: list[dict[str, str]] = load_csv("support_tickets.csv")
        model = Model(
            provider="openai", name="gpt-4.1-nano", api_key=OPENAI_KEY,
            params={"temperature": 0},
        )
        job = Job(
            prompt="Triage each support ticket. Classify by category, assign priority, "
            "determine if human escalation is needed, and suggest a brief response.",
            output_model=TicketTriage,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[TicketTriage] = job.run(model, data=data)
        print_result_summary("OpenAI", "gpt-4.1-nano", result)
        assert_result_valid(result, len(data), TicketTriage)

    @skip_no_openai
    def test_o3_mini_company_summary(self) -> None:
        """o3-mini: reasoning model for structured summaries."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:5]
        model = Model(provider="openai", name="o3-mini", api_key=OPENAI_KEY)
        job = Job(
            prompt="Create a concise structured summary for each company. "
            "Calculate the approximate age based on the founded year (current year is 2026).",
            output_model=CompanySummary,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[CompanySummary] = job.run(model, data=data)
        print_result_summary("OpenAI", "o3-mini", result)
        assert_result_valid(result, len(data), CompanySummary)

    @skip_no_openai
    def test_o4_mini_product_tagging(self) -> None:
        """o4-mini: latest reasoning model for product tagging."""
        data: list[dict[str, str]] = load_csv("products.csv")[:5]
        model = Model(provider="openai", name="o4-mini", api_key=OPENAI_KEY)
        job = Job(
            prompt="Tag each product with structured attributes. Classify the price tier, "
            "identify the target audience, and note if sustainability/eco features are mentioned.",
            output_model=ProductTag,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[ProductTag] = job.run(model, data=data)
        print_result_summary("OpenAI", "o4-mini", result)
        assert_result_valid(result, len(data), ProductTag)


class TestLatestAnthropic:
    """Tests with the latest Anthropic Claude models (2025-2026 generation)."""

    @skip_no_anthropic
    def test_sonnet_4_company_classification(self) -> None:
        """Claude Sonnet 4: classify companies."""
        data: list[dict[str, str]] = load_csv("companies.csv")
        model = Model(
            provider="anthropic", name="claude-sonnet-4-20250514", api_key=ANTHROPIC_KEY,
        )
        job = Job(
            prompt="Classify each company by its primary industry sector and sub-sector. "
            "Determine if the company is publicly traded.",
            output_model=IndustryClassification,
            batch_size=5,
            concurrency=2,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("Anthropic", "claude-sonnet-4", result)
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_anthropic
    def test_sonnet_4_sentiment_analysis(self) -> None:
        """Claude Sonnet 4: sentiment analysis."""
        data: list[dict[str, str]] = load_csv("products.csv")
        model = Model(
            provider="anthropic", name="claude-sonnet-4-20250514", api_key=ANTHROPIC_KEY,
            params={"temperature": 0},
        )
        job = Job(
            prompt="Analyze the sentiment of each product's customer_review. "
            "Identify the overall sentiment, assign a score, and extract key themes.",
            output_model=SentimentAnalysis,
            batch_size=5,
            concurrency=2,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[SentimentAnalysis] = job.run(model, data=data)
        print_result_summary("Anthropic", "claude-sonnet-4", result)
        assert_result_valid(result, len(data), SentimentAnalysis)

    @skip_no_anthropic
    def test_haiku_4_5_ticket_triage(self) -> None:
        """Claude Haiku 4.5: fast triage."""
        data: list[dict[str, str]] = load_csv("support_tickets.csv")
        model = Model(
            provider="anthropic", name="claude-haiku-4-5-20251001", api_key=ANTHROPIC_KEY,
        )
        job = Job(
            prompt="Triage each support ticket. Classify by category, assign priority, "
            "determine if human escalation is needed, and suggest a brief response.",
            output_model=TicketTriage,
            batch_size=5,
            concurrency=2,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[TicketTriage] = job.run(model, data=data)
        print_result_summary("Anthropic", "claude-haiku-4.5", result)
        assert_result_valid(result, len(data), TicketTriage)

    @skip_no_anthropic
    def test_haiku_4_5_product_tagging(self) -> None:
        """Claude Haiku 4.5: product tagging with temperature variations."""
        data: list[dict[str, str]] = load_csv("products.csv")
        model = Model(
            provider="anthropic", name="claude-haiku-4-5-20251001", api_key=ANTHROPIC_KEY,
            params={"temperature": 0.5},
        )
        job = Job(
            prompt="Tag each product with structured attributes. Classify the price tier, "
            "identify the target audience, and note if sustainability/eco features are mentioned.",
            output_model=ProductTag,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[ProductTag] = job.run(model, data=data)
        print_result_summary("Anthropic", "claude-haiku-4.5 (temp=0.5)", result, {"temperature": 0.5})
        assert_result_valid(result, len(data), ProductTag)

    @skip_no_anthropic
    def test_sonnet_4_full_dataset_concurrent(self) -> None:
        """Claude Sonnet 4: full dataset, multi-batch, concurrent."""
        data: list[dict[str, str]] = load_csv("companies.csv")
        model = Model(
            provider="anthropic", name="claude-sonnet-4-20250514", api_key=ANTHROPIC_KEY,
            params={"temperature": 0},
        )
        job = Job(
            prompt="Create a concise structured summary for each company. "
            "Calculate age based on founded year (current year is 2026).",
            output_model=CompanySummary,
            batch_size=4,
            concurrency=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[CompanySummary] = job.run(model, data=data)
        print_result_summary("Anthropic", "claude-sonnet-4 (batch=4, conc=3)", result)
        assert_result_valid(result, len(data), CompanySummary)


class TestLatestGemini:
    """Tests with the latest Google Gemini models (2025-2026 generation)."""

    @skip_no_gemini
    def test_gemini_25_flash_company_classification(self) -> None:
        """Gemini 2.5 Flash: classify companies."""
        data: list[dict[str, str]] = load_csv("companies.csv")
        model = Model(
            provider="google_genai", name="gemini-2.5-flash",
            api_key=GEMINI_KEY,
        )
        job = Job(
            prompt="Classify each company by its primary industry sector and sub-sector. "
            "Determine if the company is publicly traded.",
            output_model=IndustryClassification,
            batch_size=5,
            concurrency=2,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("Gemini", "gemini-2.5-flash-preview", result)
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_gemini
    def test_gemini_25_flash_sentiment(self) -> None:
        """Gemini 2.5 Flash: sentiment analysis."""
        data: list[dict[str, str]] = load_csv("products.csv")
        model = Model(
            provider="google_genai", name="gemini-2.5-flash",
            api_key=GEMINI_KEY,
            params={"temperature": 0},
        )
        job = Job(
            prompt="Analyze the sentiment of each product's customer_review. "
            "Identify the overall sentiment, assign a score, and extract key themes.",
            output_model=SentimentAnalysis,
            batch_size=5,
            concurrency=2,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[SentimentAnalysis] = job.run(model, data=data)
        print_result_summary("Gemini", "gemini-2.5-flash-preview", result)
        assert_result_valid(result, len(data), SentimentAnalysis)

    @skip_no_gemini
    def test_gemini_25_pro_ticket_triage(self) -> None:
        """Gemini 2.5 Pro: complex ticket triage."""
        data: list[dict[str, str]] = load_csv("support_tickets.csv")[:5]
        model = Model(
            provider="google_genai", name="gemini-2.5-pro",
            api_key=GEMINI_KEY,
        )
        job = Job(
            prompt="Triage each support ticket. Classify by category, assign priority, "
            "determine if human escalation is needed, and suggest a brief response.",
            output_model=TicketTriage,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[TicketTriage] = job.run(model, data=data)
        print_result_summary("Gemini", "gemini-2.5-pro-preview", result)
        assert_result_valid(result, len(data), TicketTriage)

    @skip_no_gemini
    def test_gemini_20_flash_lite_product_tagging(self) -> None:
        """Gemini 2.0 Flash Lite: cheapest Gemini model."""
        data: list[dict[str, str]] = load_csv("products.csv")
        model = Model(
            provider="google_genai", name="gemini-2.0-flash-lite",
            api_key=GEMINI_KEY,
        )
        job = Job(
            prompt="Tag each product with structured attributes. Classify the price tier, "
            "identify the target audience, and note if sustainability/eco features are mentioned.",
            output_model=ProductTag,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[ProductTag] = job.run(model, data=data)
        print_result_summary("Gemini", "gemini-2.0-flash-lite", result)
        assert_result_valid(result, len(data), ProductTag)

    @skip_no_gemini
    def test_gemini_25_flash_full_dataset(self) -> None:
        """Gemini 2.5 Flash: full dataset, multi-batch, concurrent."""
        data: list[dict[str, str]] = load_csv("companies.csv")
        model = Model(
            provider="google_genai", name="gemini-2.5-flash",
            api_key=GEMINI_KEY,
            params={"temperature": 0},
        )
        job = Job(
            prompt="Create a concise structured summary for each company. "
            "Calculate age based on founded year (current year is 2026).",
            output_model=CompanySummary,
            batch_size=4,
            concurrency=3,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[CompanySummary] = job.run(model, data=data)
        print_result_summary("Gemini", "gemini-2.5-flash (batch=4, conc=3)", result)
        assert_result_valid(result, len(data), CompanySummary)


# ===========================================================================
# Cross-provider comparison tests (latest models)
# ===========================================================================


class TestCrossProviderLatest:
    """Cross-provider tests using the latest models from each provider."""

    @skip_no_openai
    @skip_no_anthropic
    @skip_no_gemini
    def test_latest_models_same_task(self) -> None:
        """Run identical task across latest model from each provider."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:5]

        providers: list[tuple[str, str, str | None]] = [
            ("openai", "gpt-4.1-mini", OPENAI_KEY),
            ("anthropic", "claude-sonnet-4-20250514", ANTHROPIC_KEY),
            ("google_genai", "gemini-2.5-flash", GEMINI_KEY),
        ]

        for provider, model_name, key in providers:
            model = Model(
                provider=provider, name=model_name, api_key=key,
                params={"temperature": 0},
            )
            job = Job(
                prompt="Classify each company by industry sector.",
                output_model=IndustryClassification,
                batch_size=10,
                stop_on_exhaustion=False,
            )
            result: SmeltResult[IndustryClassification] = job.run(model, data=data)
            print_result_summary(f"{provider} (latest)", model_name, result)
            assert_result_valid(result, len(data), IndustryClassification)


# ===========================================================================
# Sync vs Async tests
# ===========================================================================


class TestSyncAsync:
    """Tests verifying both sync (run) and async (arun) paths work with real APIs."""

    @skip_no_openai
    def test_openai_sync(self) -> None:
        """OpenAI via synchronous job.run()."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:3]
        model = Model(provider="openai", name="gpt-4o-mini", api_key=OPENAI_KEY)
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("OpenAI (sync)", "gpt-4o-mini", result)
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_openai_async(self) -> None:
        """OpenAI via asynchronous job.arun()."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:3]
        model = Model(provider="openai", name="gpt-4o-mini", api_key=OPENAI_KEY)
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = await job.arun(model, data=data)
        print_result_summary("OpenAI (async)", "gpt-4o-mini", result)
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_anthropic
    def test_anthropic_sync(self) -> None:
        """Anthropic via synchronous job.run()."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:3]
        model = Model(
            provider="anthropic", name="claude-sonnet-4-20250514", api_key=ANTHROPIC_KEY,
        )
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("Anthropic (sync)", "claude-sonnet-4", result)
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_async(self) -> None:
        """Anthropic via asynchronous job.arun()."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:3]
        model = Model(
            provider="anthropic", name="claude-sonnet-4-20250514", api_key=ANTHROPIC_KEY,
        )
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = await job.arun(model, data=data)
        print_result_summary("Anthropic (async)", "claude-sonnet-4", result)
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_gemini
    def test_gemini_sync(self) -> None:
        """Gemini via synchronous job.run()."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:3]
        model = Model(
            provider="google_genai", name="gemini-2.0-flash", api_key=GEMINI_KEY,
        )
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = job.run(model, data=data)
        print_result_summary("Gemini (sync)", "gemini-2.0-flash", result)
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_gemini_async(self) -> None:
        """Gemini via asynchronous job.arun()."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:3]
        model = Model(
            provider="google_genai", name="gemini-2.0-flash", api_key=GEMINI_KEY,
        )
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=10,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = await job.arun(model, data=data)
        print_result_summary("Gemini (async)", "gemini-2.0-flash", result)
        assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_async_multi_batch_concurrent(self) -> None:
        """Async with multiple batches and high concurrency."""
        data: list[dict[str, str]] = load_csv("companies.csv")
        model = Model(
            provider="openai", name="gpt-4o-mini", api_key=OPENAI_KEY,
            params={"temperature": 0},
        )
        job = Job(
            prompt="Classify each company by industry sector.",
            output_model=IndustryClassification,
            batch_size=3,
            concurrency=4,
            stop_on_exhaustion=False,
        )
        result: SmeltResult[IndustryClassification] = await job.arun(model, data=data)
        print_result_summary("OpenAI (async, batch=3, conc=4)", "gpt-4o-mini", result)
        assert_result_valid(result, len(data), IndustryClassification)
        assert result.metrics.total_batches == 4  # ceil(10/3)


# ===========================================================================
# Cross-provider comparison tests
# ===========================================================================


class TestCrossProvider:
    """Tests that run the same task across all available providers."""

    @skip_no_openai
    @skip_no_anthropic
    @skip_no_gemini
    def test_same_task_all_providers(self) -> None:
        """Run identical classification task across all 3 providers, verify all succeed."""
        data: list[dict[str, str]] = load_csv("companies.csv")[:5]

        providers: list[tuple[str, str, str | None]] = [
            ("openai", "gpt-4o-mini", OPENAI_KEY),
            ("anthropic", "claude-3-5-haiku-20241022", ANTHROPIC_KEY),
            ("google_genai", "gemini-2.0-flash", GEMINI_KEY),
        ]

        for provider, model_name, key in providers:
            model = Model(
                provider=provider, name=model_name, api_key=key,
                params={"temperature": 0},
            )
            job = Job(
                prompt="Classify each company by industry sector.",
                output_model=IndustryClassification,
                batch_size=10,
                stop_on_exhaustion=False,
            )
            result: SmeltResult[IndustryClassification] = job.run(model, data=data)
            print_result_summary(provider, model_name, result)
            assert_result_valid(result, len(data), IndustryClassification)

    @skip_no_openai
    @skip_no_anthropic
    @skip_no_gemini
    def test_complex_output_all_providers(self) -> None:
        """Run ticket triage (complex schema) across all providers."""
        data: list[dict[str, str]] = load_csv("support_tickets.csv")[:5]

        providers: list[tuple[str, str, str | None]] = [
            ("openai", "gpt-4o-mini", OPENAI_KEY),
            ("anthropic", "claude-3-5-haiku-20241022", ANTHROPIC_KEY),
            ("google_genai", "gemini-2.0-flash", GEMINI_KEY),
        ]

        for provider, model_name, key in providers:
            model = Model(
                provider=provider, name=model_name, api_key=key,
                params={"temperature": 0},
            )
            job = Job(
                prompt="Triage each support ticket with category, priority, escalation need, "
                "and a suggested response.",
                output_model=TicketTriage,
                batch_size=10,
                max_retries=3,
                stop_on_exhaustion=False,
            )
            result: SmeltResult[TicketTriage] = job.run(model, data=data)
            print_result_summary(provider, model_name, result)
            assert_result_valid(result, len(data), TicketTriage)
