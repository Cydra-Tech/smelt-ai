# Cookbook: Data Extraction

Extract structured fields from unstructured or semi-structured text.

## Entity extraction

Extract structured entities from free-text descriptions:

```python
from pydantic import BaseModel, Field
from smelt import Model, Job

class CompanyInfo(BaseModel):
    one_liner: str = Field(description="One sentence description of the company")
    industry: str = Field(description="Primary industry")
    company_size: str = Field(description="Size: startup, small, medium, large, or enterprise")
    age_years: int = Field(description="Approximate age in years")
    is_b2b: bool = Field(description="Whether the company primarily serves businesses (B2B)")
    is_b2c: bool = Field(description="Whether the company primarily serves consumers (B2C)")

model = Model(provider="openai", name="gpt-4.1-mini")
job = Job(
    prompt="Extract structured information about each company. "
           "Calculate age based on the founded year (current year is 2026). "
           "A company can be both B2B and B2C.",
    output_model=CompanyInfo,
    batch_size=10,
)

companies = [
    {"name": "Stripe", "description": "Payment processing platform for internet businesses", "founded": "2010", "employees": "8000"},
    {"name": "Airbnb", "description": "Online marketplace for short-term lodging and tourism", "founded": "2008", "employees": "6900"},
]

result = job.run(model, data=companies)
for company, info in zip(companies, result.data):
    print(f"{company['name']}: {info.one_liner}")
    print(f"  {info.industry} | {info.company_size} | {info.age_years}y | B2B={info.is_b2b} B2C={info.is_b2c}")
```

## Contact information extraction

Parse contact details from unstructured text:

```python
class ContactInfo(BaseModel):
    full_name: str = Field(description="Person's full name")
    email: str = Field(description="Email address, or 'not found'")
    phone: str = Field(description="Phone number in E.164 format, or 'not found'")
    company: str = Field(description="Company name, or 'not found'")
    job_title: str = Field(description="Job title, or 'not found'")

job = Job(
    prompt="Extract contact information from each text block. "
           "Normalize phone numbers to E.164 format (+1XXXXXXXXXX for US). "
           "If a field is not present in the text, return 'not found'.",
    output_model=ContactInfo,
    batch_size=10,
)

texts = [
    {"text": "Hi, I'm Sarah Chen, VP of Engineering at Acme Corp. Reach me at sarah@acme.com or (415) 555-0123."},
    {"text": "John Smith here from BigCo. My number is 212-555-4567. Email: jsmith@bigco.io"},
]
```

## Key-value extraction from documents

Extract specific fields from semi-structured documents:

```python
class InvoiceData(BaseModel):
    invoice_number: str = Field(description="Invoice/reference number")
    date: str = Field(description="Invoice date in YYYY-MM-DD format")
    vendor_name: str = Field(description="Name of the vendor/seller")
    total_amount: float = Field(description="Total amount in USD")
    currency: str = Field(description="Currency code (e.g. USD, EUR)")
    line_items_count: int = Field(description="Number of line items")

job = Job(
    prompt="Extract structured data from each invoice text. "
           "Normalize dates to YYYY-MM-DD format. "
           "Convert all amounts to their numeric value (remove currency symbols).",
    output_model=InvoiceData,
    batch_size=5,
)
```

## Resume/CV parsing

```python
from typing import Literal

class ResumeExtract(BaseModel):
    name: str = Field(description="Candidate's full name")
    current_title: str = Field(description="Most recent job title")
    years_experience: int = Field(description="Approximate total years of professional experience")
    top_skills: list[str] = Field(description="Top 3-5 technical skills")
    education_level: Literal["high_school", "bachelors", "masters", "phd", "other"] = Field(
        description="Highest education level"
    )
    languages: list[str] = Field(description="Programming languages mentioned")

job = Job(
    prompt="Parse each resume text and extract structured information. "
           "For years_experience, count from the earliest job mentioned to present. "
           "For top_skills, prioritize skills mentioned most frequently or prominently.",
    output_model=ResumeExtract,
    batch_size=3,  # Resumes can be long, use smaller batches
)
```

## Address normalization

```python
class NormalizedAddress(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    state: str = Field(description="State/province (2-letter code for US)")
    postal_code: str = Field(description="Postal/ZIP code")
    country: str = Field(description="Country (ISO 3166-1 alpha-2 code)")

job = Job(
    prompt="Normalize each address into structured components. "
           "Use 2-letter state codes for US addresses (e.g. CA, NY). "
           "Use ISO 3166-1 alpha-2 country codes (e.g. US, GB, DE). "
           "If a component is missing, make your best inference from context.",
    output_model=NormalizedAddress,
    batch_size=20,  # Addresses are short, can batch more
)

addresses = [
    {"raw_address": "1600 Amphitheatre Pkwy, Mountain View, California 94043"},
    {"raw_address": "221B Baker St, London, UK"},
    {"raw_address": "Friedrichstraße 43-45, 10117 Berlin"},
]
```

## Tips for extraction tasks

1. **Specify formats explicitly** — "YYYY-MM-DD" not just "date format", "E.164" not just "phone format"
2. **Handle missing data** — always tell the LLM what to return when a field isn't found: `"return 'not found'"`, `"return null"`, or `"return empty string"`
3. **Use smaller batch_size for long texts** — each row consumes more input tokens
4. **Add validation** — extract first, then validate with Python:
   ```python
   import re
   for row in result.data:
       if row.email != "not found" and not re.match(r".+@.+\..+", row.email):
           print(f"Invalid email: {row.email}")
   ```
5. **Be explicit about normalization rules** — "convert to lowercase", "remove leading/trailing whitespace", "use title case"
