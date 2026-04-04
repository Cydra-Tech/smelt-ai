"""Analyze ECG images using vision-capable LLMs.

Usage:
    pip install smelt-ai[anthropic,vision]
    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/vision_ecg.py path/to/ecg_image.jpeg
"""

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from smelt import Job, Model

try:
    from PIL import Image
except ImportError:
    print("Pillow is required: pip install smelt-ai[vision]")
    sys.exit(1)


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


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python examples/vision_ecg.py <image_path> [image_path ...]")
        sys.exit(1)

    image_paths = [Path(p) for p in sys.argv[1:]]
    for p in image_paths:
        if not p.exists():
            print(f"File not found: {p}")
            sys.exit(1)

    model = Model(provider="anthropic", name="claude-sonnet-4-6")
    job = Job(
        prompt="Analyze the ECG image and provide a structured cardiac assessment.",
        output_model=ECGAnalysis,
        batch_size=1,
    )

    data = [
        {"patient_id": f"P{i + 1:03d}", "ecg": Image.open(p)}
        for i, p in enumerate(image_paths)
    ]

    print(f"Analyzing {len(data)} ECG image(s)...\n")
    result = job.run(model, data=data)

    for i, ecg in enumerate(result.data):
        print(f"Patient P{i + 1:03d}:")
        print(f"  Heart rhythm:   {ecg.heart_rhythm}")
        print(f"  Heart rate:     {ecg.heart_rate_bpm} bpm")
        print(f"  Abnormalities:  {ecg.abnormalities}")
        print(f"  Confidence:     {ecg.confidence}")
        print()

    print(f"Time: {result.metrics.wall_time_seconds:.2f}s")
    print(f"Tokens: {result.metrics.input_tokens} in / {result.metrics.output_tokens} out")


if __name__ == "__main__":
    main()
