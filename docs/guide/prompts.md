# Writing Effective Prompts

Your `prompt` is the most important factor in output quality. This page covers how smelt uses your prompt and how to write prompts that produce consistent, high-quality results.

## How smelt uses your prompt

Your prompt string is embedded in a system message that smelt constructs:

```
You are a structured data transformation assistant.

## Task
{your prompt here}          ← This is what you control

## Rules
- You will receive a JSON array of objects. Each object has a "row_id" field.
- For EACH input object, produce exactly one output object.
- Every output object MUST include the same "row_id" as the corresponding input object.
- Do NOT skip, merge, duplicate, or reorder rows.
- Return ALL rows — the count of output rows must equal the count of input rows.

## Output Schema
Each output row must conform to this schema:
- row_id (int, required)
- sector (str, required) — Primary industry sector
- sub_sector (str, required) — More specific sub-sector
- is_public (bool, required) — Whether the company is publicly traded

Return your response as a JSON object with a single "rows" key...
```

Smelt automatically handles:

- Row-level instructions (row_id tracking, count matching)
- Schema description (generated from your Pydantic model's field names, types, and descriptions)
- Output format (JSON structure)

**You only need to describe what transformation to apply.**

## Prompt anatomy

A good prompt has three parts:

### 1. Task description

What to do with each row:

```python
# Bad — vague
prompt = "Classify the companies"

# Good — specific
prompt = "Classify each company by its primary industry sector and sub-sector"

# Better — specific + criteria
prompt = (
    "Classify each company by its primary GICS industry sector and sub-sector. "
    "Determine if the company is publicly traded on a major stock exchange "
    "(NYSE, NASDAQ, LSE, or equivalent)."
)
```

### 2. Rules and constraints

Business logic and edge cases:

```python
prompt = (
    "Analyze the sentiment of each product review. "
    "Rules: "
    "- Score must be between 0.0 (most negative) and 1.0 (most positive). "
    "- 'mixed' sentiment means both positive and negative aspects are present. "
    "- 'neutral' means neither positive nor negative. "
    "- Extract 1-3 key themes, not more."
)
```

### 3. Context and calibration

Help the LLM understand your domain:

```python
prompt = (
    "Create a concise structured summary for each company. "
    "Calculate the approximate age based on the founded year (current year is 2026). "
    "Size classification: startup (<50 employees), small (50-200), "
    "medium (200-1000), large (1000-10000), enterprise (10000+)."
)
```

## Field descriptions matter

Smelt includes your `Field(description=...)` values in the system prompt. These are the LLM's primary guide for each output field.

```python
# Without descriptions — LLM guesses what "tier" means
class Pricing(BaseModel):
    tier: str
    score: float

# With descriptions — LLM knows exactly what you want
class Pricing(BaseModel):
    tier: str = Field(description="Price tier: budget (<$50), mid-range ($50-200), premium ($200-500), luxury ($500+)")
    score: float = Field(description="Value-for-money score from 0.0 (poor value) to 1.0 (excellent value)")
```

!!! tip
    Think of `Field(description=...)` as micro-prompts for individual fields. Be as specific as possible — include ranges, formats, examples, and edge cases.

## Prompt patterns

### Classification with fixed categories

```python
prompt = (
    "Classify each support ticket into exactly one category. "
    "Categories are: billing (payment, charges, invoices), "
    "technical (bugs, errors, performance), "
    "shipping (delivery, tracking, returns), "
    "account (login, settings, permissions), "
    "general (everything else)."
)
```

### Extraction with format rules

```python
prompt = (
    "Extract contact information from each text block. "
    "Normalize phone numbers to E.164 format (+1XXXXXXXXXX for US numbers). "
    "Normalize email addresses to lowercase. "
    "If a field is not present in the text, return 'not_found'."
)
```

### Scoring with calibration

```python
prompt = (
    "Rate each product review on a scale from 0.0 to 1.0. "
    "Calibration: "
    "0.0 = extremely negative (product is broken, dangerous, or useless) "
    "0.25 = mostly negative (significant issues, would not recommend) "
    "0.5 = neutral or mixed (equal positives and negatives) "
    "0.75 = mostly positive (minor issues, would recommend) "
    "1.0 = extremely positive (perfect, no complaints)"
)
```

### Enrichment with knowledge boundaries

```python
prompt = (
    "Enrich each company with market analysis. "
    "Base your analysis on publicly available information. "
    "If you are not confident about a fact, indicate uncertainty. "
    "Do not speculate about future performance."
)
```

### Multi-step reasoning

```python
prompt = (
    "For each support ticket: "
    "1. Identify the core issue (what went wrong). "
    "2. Categorize by department (billing, technical, shipping, account, general). "
    "3. Assess priority based on: urgency (explicit deadline), impact (number of users affected), "
    "   and severity (data loss > functionality > cosmetic). "
    "4. Draft a response that acknowledges the issue and provides a next step."
)
```

## Common mistakes

### Too vague

```python
# Bad
prompt = "Process the data"

# Good
prompt = "Classify each company by GICS sector and determine if publicly traded"
```

### Contradicting the schema

```python
# Bad — prompt says 3 categories but Literal has 5
prompt = "Classify as positive, negative, or neutral"

class Output(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "mixed", "unknown"]

# Good — prompt matches schema
prompt = "Classify sentiment as positive, negative, neutral, mixed, or unknown"
```

### Missing edge cases

```python
# Bad — what if the review is in another language? What if it's empty?
prompt = "Analyze the sentiment of each review"

# Good — covers edge cases
prompt = (
    "Analyze the sentiment of each review. "
    "If the review is in a non-English language, analyze it in its original language. "
    "If the review is empty or unintelligible, classify as 'neutral' with score 0.5."
)
```

### Overloading the prompt

```python
# Bad — too many instructions, LLM may miss some
prompt = (
    "Classify by industry and sub-industry and determine if public and calculate market cap "
    "and find the CEO name and headquarters city and founding year and number of employees "
    "and annual revenue and profit margin and stock ticker and..."
)

# Good — focus on what matters, let the schema handle field definitions
prompt = (
    "Classify each company by industry sector. "
    "Use GICS classification for sector and sub-sector."
)
```

## Testing your prompt

Always test with `job.test()` before a full run:

```python
# Test with one row
result = job.test(model, data=data)
print(result.data[0])

# Check if the output makes sense
# If not, refine your prompt and test again
```

For systematic testing, compare outputs across temperature settings:

```python
for temp in [0, 0.5, 1.0]:
    m = Model(provider="openai", name="gpt-4.1-mini", params={"temperature": temp})
    result = job.test(m, data=data)
    print(f"temp={temp}: {result.data[0]}")
```

If results vary wildly across temperatures, your prompt may be too ambiguous. Tighten the instructions until `temperature=0` and `temperature=0.5` produce similar results.
