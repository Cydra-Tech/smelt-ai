# Cookbook: Vision / Image Processing

Process images through vision-capable LLMs with smelt. Pass PIL images directly in your data dicts — smelt handles base64 encoding and multimodal message construction automatically.

## Install

```bash
pip install smelt-ai[openai,vision]    # or [anthropic,vision] or [google,vision]
```

## How it works

When smelt detects `PIL.Image.Image` objects in your data:

1. Images are extracted and base64-encoded
2. Placeholders like `"[image: field_name]"` replace images in the JSON payload
3. The system message gets an addendum explaining image placeholders
4. The human message becomes a multimodal content block with text + image data

No new classes to learn — just pass images in your data dicts.

## ECG analysis

Analyze medical ECG images with structured output:

```python
from PIL import Image
from pydantic import BaseModel, Field
from typing import Literal
from smelt import Model, Job

class ECGAnalysis(BaseModel):
    heart_rhythm: str = Field(description="Detected heart rhythm (e.g. normal sinus rhythm)")
    heart_rate_bpm: int = Field(description="Estimated heart rate in beats per minute")
    abnormalities: list[str] = Field(description="List of detected abnormalities; empty if none")
    confidence: Literal["low", "medium", "high"] = Field(description="Confidence level")

model = Model(provider="anthropic", name="claude-sonnet-4-6")
job = Job(
    prompt="Analyze the ECG image and provide a structured cardiac assessment.",
    output_model=ECGAnalysis,
    batch_size=1,
)

data = [
    {"patient_id": "P001", "ecg": Image.open("ecg_1.jpeg")},
    {"patient_id": "P002", "ecg": Image.open("ecg_2.jpeg")},
]

result = job.run(model, data=data)

for row in result.data:
    print(f"{row.heart_rhythm} — {row.heart_rate_bpm} bpm")
    print(f"  Abnormalities: {row.abnormalities}")
```

## Document classification

Classify scanned documents by type:

```python
from PIL import Image
from pydantic import BaseModel, Field
from smelt import Model, Job

class DocumentType(BaseModel):
    category: str = Field(description="Document type: invoice, receipt, contract, letter, form, other")
    language: str = Field(description="Primary language of the document")
    has_signature: bool = Field(description="Whether the document contains a visible signature")

model = Model(provider="openai", name="gpt-4o")
job = Job(
    prompt="Classify the scanned document image by type, language, and whether it's signed.",
    output_model=DocumentType,
    batch_size=1,
)

scans = [
    {"doc_id": "D001", "scan": Image.open("scan_001.png")},
    {"doc_id": "D002", "scan": Image.open("scan_002.png")},
]

result = job.run(model, data=scans)
```

## Product image analysis

Extract product attributes from photos:

```python
from PIL import Image
from pydantic import BaseModel, Field
from smelt import Model, Job

class ProductAttributes(BaseModel):
    primary_color: str = Field(description="Primary color of the product")
    material: str = Field(description="Apparent material (leather, fabric, metal, etc.)")
    condition: str = Field(description="Condition: new, like-new, good, fair, poor")
    brand_visible: bool = Field(description="Whether a brand logo/name is visible")

model = Model(provider="google_genai", name="gemini-3.1-pro-preview")
job = Job(
    prompt="Analyze the product photo and extract visual attributes.",
    output_model=ProductAttributes,
    batch_size=1,
)

products = [
    {"sku": "SKU-001", "photo": Image.open("product_1.jpg")},
    {"sku": "SKU-002", "photo": Image.open("product_2.jpg")},
]

result = job.run(model, data=products)
```

## Multiple images per row

You can include multiple images in a single row:

```python
from PIL import Image
from pydantic import BaseModel, Field
from smelt import Model, Job

class DamageAssessment(BaseModel):
    damage_type: str = Field(description="Type of damage observed")
    severity: str = Field(description="Severity: minor, moderate, severe")
    repair_needed: bool = Field(description="Whether professional repair is needed")

model = Model(provider="anthropic", name="claude-sonnet-4-6")
job = Job(
    prompt="Assess vehicle damage from the front and rear photos.",
    output_model=DamageAssessment,
    batch_size=1,
)

data = [
    {
        "claim_id": "CLM-001",
        "front_photo": Image.open("car_front.jpg"),
        "rear_photo": Image.open("car_rear.jpg"),
    },
]

result = job.run(model, data=data)
```

## Mixing images and text

Image fields and text fields work together seamlessly:

```python
data = [
    {
        "patient_id": "P001",
        "age": 65,
        "symptoms": "chest pain, shortness of breath",
        "ecg": Image.open("ecg_1.jpeg"),        # PIL Image
        "notes": "Patient presents with acute onset",  # Regular text
    },
]
```

Smelt sends the text fields as JSON and the image as a multimodal block. The LLM sees both.

## Tips

!!! tip "Use `batch_size=1` for images"
    Images are large payloads. Smelt warns when `batch_size > 5` with image data. For best results, use `batch_size=1` to send one image per request.

!!! tip "Supported providers"
    Vision works with any LLM that supports multimodal input via LangChain:

    - **OpenAI**: `gpt-4o`, `gpt-4.1`, `gpt-4.1-mini`
    - **Anthropic**: `claude-sonnet-4-6`, `claude-opus-4-6`
    - **Google**: `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-2.5-flash`

!!! tip "Check if Pillow is available"
    ```python
    from smelt import PILLOW_AVAILABLE
    if not PILLOW_AVAILABLE:
        print("Install Pillow: pip install smelt-ai[vision]")
    ```

!!! warning "Image format handling"
    Smelt auto-detects image format from the PIL Image object. RGBA images are encoded as PNG; RGB images as JPEG. If you need a specific format, convert before passing to smelt.
