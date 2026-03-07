"""Live integration tests for vision/image processing with smelt.

Tests real API calls with PIL images sent to vision-capable LLMs.
Requires OPENAI_API_KEY in .env and Pillow installed.

Run with: uv run pytest tests/test_live_vision.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from smelt import Job, Model, SmeltResult

load_dotenv()

PIL = pytest.importorskip("PIL", reason="Pillow not installed")
from PIL import Image

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

OPENAI_KEY: str | None = os.getenv("OPENAI_API_KEY")

skip_no_openai = pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")

ECG_DIR: Path = Path("/Users/jeevanprakash/Desktop/work/cydratech/ECG/test-imgaes")

skip_no_ecg_dir = pytest.mark.skipif(
    not ECG_DIR.exists(), reason=f"ECG test images directory not found: {ECG_DIR}"
)

# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class ECGAnalysis(BaseModel):
    """Structured analysis of an ECG image."""

    heart_rhythm: str = Field(description="Detected heart rhythm (e.g. normal sinus rhythm)")
    heart_rate_bpm: int = Field(description="Estimated heart rate in beats per minute")
    abnormalities: list[str] = Field(
        description="List of detected abnormalities; empty list if none"
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence level of the analysis"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_no_openai
@skip_no_ecg_dir
class TestSingleECGVision:
    """Test single ECG image analysis with GPT-4o."""

    def test_single_ecg_analysis(self) -> None:
        """Should analyze a single ECG image and return structured output."""
        ecg_path: Path = ECG_DIR / "ecg_1.jpeg"
        assert ecg_path.exists(), f"ECG image not found: {ecg_path}"

        img: Image.Image = Image.open(ecg_path)
        data: list[dict[str, object]] = [{"patient_id": "P001", "ecg": img}]

        model = Model(provider="openai", name="gpt-4o")
        job = Job(
            prompt="Analyze the ECG image and provide a structured cardiac assessment.",
            output_model=ECGAnalysis,
            batch_size=1,
        )

        result: SmeltResult[ECGAnalysis] = job.run(model, data=data)

        assert result.success, f"Errors: {result.errors}"
        assert len(result.data) == 1

        ecg: ECGAnalysis = result.data[0]
        print(f"\n--- Single ECG Result ---")
        print(f"  Heart rhythm: {ecg.heart_rhythm}")
        print(f"  Heart rate: {ecg.heart_rate_bpm} bpm")
        print(f"  Abnormalities: {ecg.abnormalities}")
        print(f"  Confidence: {ecg.confidence}")

        assert isinstance(ecg.heart_rhythm, str)
        assert len(ecg.heart_rhythm) > 0
        assert 20 <= ecg.heart_rate_bpm <= 300
        assert isinstance(ecg.abnormalities, list)
        assert ecg.confidence in ("low", "medium", "high")


@skip_no_openai
@skip_no_ecg_dir
class TestBatchECGVision:
    """Test batch ECG image analysis with GPT-4o."""

    def test_all_ecg_images(self) -> None:
        """Should analyze all 5 ECG images and return structured output for each."""
        ecg_files: list[Path] = sorted(ECG_DIR.glob("ecg_*.*"))
        assert len(ecg_files) >= 1, f"No ECG images found in {ECG_DIR}"

        data: list[dict[str, object]] = []
        for i, ecg_path in enumerate(ecg_files):
            img: Image.Image = Image.open(ecg_path)
            data.append({"patient_id": f"P{i + 1:03d}", "ecg": img})

        model = Model(provider="openai", name="gpt-4o")
        job = Job(
            prompt="Analyze the ECG image and provide a structured cardiac assessment.",
            output_model=ECGAnalysis,
            batch_size=1,
        )

        result: SmeltResult[ECGAnalysis] = job.run(model, data=data)

        assert result.success, f"Errors: {result.errors}"
        assert len(result.data) == len(ecg_files)

        print(f"\n--- Batch ECG Results ({len(result.data)} images) ---")
        for i, ecg in enumerate(result.data):
            print(f"\n  Patient P{i + 1:03d}:")
            print(f"    Heart rhythm: {ecg.heart_rhythm}")
            print(f"    Heart rate: {ecg.heart_rate_bpm} bpm")
            print(f"    Abnormalities: {ecg.abnormalities}")
            print(f"    Confidence: {ecg.confidence}")

            assert isinstance(ecg.heart_rhythm, str)
            assert 20 <= ecg.heart_rate_bpm <= 300
            assert isinstance(ecg.abnormalities, list)
            assert ecg.confidence in ("low", "medium", "high")

        print(f"\n  Metrics: {result.metrics}")
