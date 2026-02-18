# Smelt — Technical Specification

> LLM-powered data transformation library for Python.
> Raw data in, structured typed output out.

---

## Overview

Smelt takes structured data (`list[dict]`), runs it through an LLM with user-defined instructions, and returns strictly validated structured output (`list[PydanticModel]`). It handles batching, concurrency, retries, and validation so the user doesn't have to.

**Current scope:** Transform (1:1 row mapping). Aggregate (many:1) is deferred.

```mermaid
graph LR
    A["list[dict]"] -->|data| S[smelt]
    P["Prompt"] -->|instructions| S
    M["Model Config"] -->|LLM| S
    O["Pydantic Model"] -->|schema| S
    S --> R["list[PydanticModel]"]
```

---

## Core Concepts

```mermaid
graph TB
    subgraph User Defines
        MODEL["Model<br/>─────────<br/>provider<br/>name<br/>api_key<br/>params"]
        JOB["Job<br/>─────────<br/>prompt<br/>output_model<br/>batch_size<br/>concurrency<br/>max_retries<br/>stop_on_exhaustion<br/>input_format (future)"]
        DATA["Data<br/>─────────<br/>list[dict]"]
    end

    subgraph Execution
        JOB -->|".run(model, data=...)"| ENGINE["Batch Engine"]
        MODEL --> ENGINE
        DATA --> ENGINE
    end

    ENGINE --> RESULT["SmeltResult[T]<br/>─────────<br/>data: list[T]<br/>errors: list[BatchError]<br/>metrics: SmeltMetrics<br/>success: bool"]

    style MODEL fill:#4a9eff,color:#fff
    style JOB fill:#ff6b6b,color:#fff
    style DATA fill:#51cf66,color:#fff
    style RESULT fill:#ffd43b,color:#000
```

**Model** — LLM provider configuration (Pydantic model). Wraps LangChain's `init_chat_model`. Reusable across jobs. Serializable to JSON.

**Job** — Transformation recipe (Pydantic model). Defines what to do (prompt + output schema) and how (batch config + retry policy). Reusable across datasets. Serializable to JSON for sharing.

**Data** — Provided at execution time. Plain `list[dict]`.

**SmeltResult** — Contains typed output, errors, and metrics.

---

## API Surface

```python
from pydantic import BaseModel
from smelt import Model, Job

# Configure LLM
model = Model(
    provider="openai",
    name="gpt-4o",
    api_key="sk-...",
    params={"temperature": 0, "max_tokens": 4096},
)

# Define transformation
class Classification(BaseModel):
    industry: str
    confidence: float
    reasoning: str

job = Job(
    prompt="Classify each company by industry sector",
    output_model=Classification,
    batch_size=20,
    concurrency=3,
    max_retries=3,
    stop_on_exhaustion=True,
)

# Execute (sync)
result = job.run(model, data=rows)

# Execute (async)
result = await job.arun(model, data=rows)

# Access results
result.data       # list[Classification]
result.errors     # list[BatchError]
result.metrics    # SmeltMetrics
result.success    # bool

# Serialize job config for sharing
job_json = job.model_dump_json(indent=2)
loaded_job = Job.model_validate_json(job_json)
```

---

## Execution Pipeline

```mermaid
flowchart TD
    START["job.run(model, data)"] --> INIT["Initialize LLM via<br/>init_chat_model()"]
    INIT --> TAG["Assign row_id to each row<br/>(0-indexed)"]
    TAG --> SCHEMA["Generate JSON schema from<br/>output_model via model_json_schema()"]
    SCHEMA --> SYSMSG["Build system message<br/>(prompt + schema + output rules)"]
    SYSMSG --> SPLIT["Split rows into batches<br/>of batch_size"]
    SPLIT --> CONCURRENT["Run batches concurrently<br/>(semaphore = concurrency)"]

    CONCURRENT --> B1["Batch 0"]
    CONCURRENT --> B2["Batch 1"]
    CONCURRENT --> BN["Batch N"]

    B1 --> COLLECT["Collect results"]
    B2 --> COLLECT
    BN --> COLLECT

    COLLECT --> ORDER["Sort by row_id<br/>(original order)"]
    ORDER --> STRIP["Strip row_id from each result"]
    STRIP --> METRICS["Compute metrics"]
    METRICS --> RESULT["Return SmeltResult"]

    style START fill:#4a9eff,color:#fff
    style RESULT fill:#ffd43b,color:#000
    style CONCURRENT fill:#ff6b6b,color:#fff
```

---

## Prompt Construction

The prompt is built by stitching together three layers into a single system message. No `with_structured_output()` — we own the full prompt and parse raw LLM text ourselves.

```mermaid
flowchart TD
    subgraph "System Message (built once per job)"
        L1["Layer 1: User's Prompt<br/>─────────<br/>The transformation instructions<br/>provided by the user"]
        L2["Layer 2: JSON Schema<br/>─────────<br/>output_model.model_json_schema()<br/>Exact schema the LLM must follow"]
        L3["Layer 3: Output Rules<br/>─────────<br/>Row ID instructions<br/>JSON format rules<br/>Markdown code block format"]
    end

    L1 --> STITCH["Stitched into<br/>single system message"]
    L2 --> STITCH
    L3 --> STITCH

    subgraph "Human Message (built per batch)"
        HM["Batch data as JSON array<br/>with row_id in each row"]
    end

    STITCH --> LLM["LLM Call<br/>(ainvoke)"]
    HM --> LLM
    LLM --> RAW["Raw text response"]
    RAW --> EXTRACT["Extract JSON from<br/>```json ... ``` block"]
    EXTRACT --> PARSE["json.loads()"]
    PARSE --> VALIDATE["Validate each object<br/>via Pydantic model_validate()"]

    style L1 fill:#4a9eff,color:#fff
    style L2 fill:#ff6b6b,color:#fff
    style L3 fill:#51cf66,color:#fff
    style VALIDATE fill:#ffd43b,color:#000
```

### System Message Template

```
{user's prompt}

## Output Schema

You must output a list of JSON objects. Each object must conform to the following JSON schema:

{output_model.model_json_schema() as formatted JSON}

Additionally, each object MUST include a `row_id` integer field matching the
`row_id` from the corresponding input row.

## Output Rules

- Return EXACTLY one object per input row.
- Each object MUST include the `row_id` from its input row.
- Do NOT skip, duplicate, or invent row_ids.
- Do NOT reorder — maintain original row_id ordering.
- Return ALL rows in a single response.

Your output must be a JSON array wrapped in a markdown code block:

```json
[
  {"row_id": 0, ...},
  {"row_id": 1, ...}
]
```
```

### Human Message Template (per batch)

Input data is serialized as a raw JSON array (default). Each row includes its assigned `row_id`.

```
Transform the following rows:

[
  {"row_id": 0, "name": "Acme Corp", "description": "Makes rockets"},
  {"row_id": 1, "name": "BioGen", "description": "Gene therapy startup"}
]
```

> **Future:** `input_format` Job config option to control how batch data is serialized
> (e.g., `"json"` (default), `"markdown_table"`, `"csv"`). For v0.1, JSON only.

### Response Parsing Pipeline

```mermaid
flowchart LR
    RAW["Raw LLM text"] --> REGEX["Regex extract<br/>```json ... ```"]
    REGEX --> JSON["json.loads()"]
    JSON --> LOOP["For each object<br/>in array"]
    LOOP --> PYDANTIC["model_validate(obj)<br/>against internal model<br/>(user model + row_id)"]
    PYDANTIC --> CHECK["Validate row_ids:<br/>count, duplicates,<br/>missing, unexpected"]
    CHECK --> STRIP["Strip row_id<br/>return user model"]

    style RAW fill:#868e96,color:#fff
    style STRIP fill:#51cf66,color:#fff
```

The parsing extracts JSON from the first `` ```json ... ``` `` block in the response, runs `json.loads()` on it, then validates each object against the internal Pydantic model (user's model + `row_id` field). If any step fails, it triggers a retry.

---

## Batch Processing (Per Batch)

```mermaid
flowchart TD
    START["Receive batch<br/>(N rows with row_ids)"] --> BUILD["Build human message<br/>with tagged row data"]
    BUILD --> CALL["LLM call via<br/>chat_model.ainvoke()"]

    CALL --> EXTRACT{Extract JSON<br/>from ```json block?}
    EXTRACT -->|Yes| PARSE{json.loads<br/>succeeded?}
    EXTRACT -->|No| ERROR["SmeltValidationError"]
    PARSE -->|Yes| PYDANTIC{Pydantic<br/>validation?}
    PARSE -->|No| ERROR
    PYDANTIC -->|Yes| VALIDATE{Row ID checks:<br/>1. count == N?<br/>2. all IDs present?<br/>3. no duplicates?}
    PYDANTIC -->|No| ERROR

    VALIDATE -->|Pass| SUCCESS["Return _BatchResult<br/>(rows + metrics)"]
    VALIDATE -->|Fail| ERROR

    ERROR --> RETRIABLE{Retriable?<br/>Attempts left?}
    RETRIABLE -->|Yes| BACKOFF["Exponential backoff<br/>1s x 2^attempt + jitter"]
    RETRIABLE -->|No| FAIL["Return _BatchResult<br/>(with BatchError)"]

    BACKOFF --> CALL

    FAIL --> EXHAUSTION{stop_on_exhaustion<br/>enabled?}
    EXHAUSTION -->|Yes| CANCEL["Set cancel_event<br/>(stop pending batches)"]
    EXHAUSTION -->|No| CONTINUE["Continue other batches"]

    style SUCCESS fill:#51cf66,color:#fff
    style FAIL fill:#ff6b6b,color:#fff
    style CANCEL fill:#ff6b6b,color:#fff
```

---

## Row ID System

Each input row is assigned a unique `row_id` (0-indexed integer) before batching. This ID flows through the entire pipeline for integrity checking and order preservation.

```mermaid
sequenceDiagram
    participant U as User Data
    participant S as Smelt
    participant L as LLM

    U->>S: [{"name": "Acme"}, {"name": "Beta"}]

    Note over S: Assign row_ids
    S->>S: [{"row_id": 0, "name": "Acme"},<br/>{"row_id": 1, "name": "Beta"}]

    S->>L: System: prompt + schema + rules<br/>Human: batch data with row_ids

    L->>S: ```json<br/>[{"row_id": 0, "industry": "Tech", ...},<br/>{"row_id": 1, "industry": "Pharma", ...}]<br/>```

    Note over S: Extract JSON from code block
    Note over S: json.loads() → list[dict]
    Note over S: Validate each via Pydantic
    Note over S: Check count, IDs, duplicates

    Note over S: Strip row_id
    S->>U: [Classification(industry="Tech", ...),<br/>Classification(industry="Pharma", ...)]
```

**Validation checks after each LLM call:**

| Check | Failure Triggers |
|-------|-----------------|
| JSON code block found in response | Retry |
| `json.loads()` succeeds | Retry |
| Each object passes `model_validate()` | Retry |
| Response list length == batch size | Retry |
| All expected `row_id` values present | Retry |
| No duplicate `row_id` values | Retry |
| No unexpected `row_id` values | Retry |

---

## Dynamic Model Creation

User's Pydantic model is extended at runtime with a `row_id` field for internal tracking. The JSON schema is generated via `model_json_schema()` and stitched into the prompt.

```mermaid
graph LR
    subgraph User Defines
        UM["Classification<br/>─────────<br/>industry: str<br/>confidence: float<br/>reasoning: str"]
    end

    subgraph Smelt Creates
        IM["_SmeltClassification<br/>─────────<br/>row_id: int ← injected<br/>industry: str<br/>confidence: float<br/>reasoning: str"]
    end

    subgraph Prompt Stitching
        JS["model_json_schema()<br/>→ JSON Schema dict<br/>→ embedded in prompt"]
    end

    UM -->|"create_model(<br/>__base__=user_model,<br/>row_id=(int, ...)<br/>)"| IM
    IM --> JS

    style UM fill:#4a9eff,color:#fff
    style IM fill:#ff6b6b,color:#fff
    style JS fill:#ffd43b,color:#000
```

- Uses `pydantic.create_model` with `__base__` — inherits all validators and config from user's model
- `model_json_schema()` on the internal model produces the schema embedded in the prompt
- LLM response is parsed from raw text (no `with_structured_output`)
- Each response object is validated via `model_validate()` against the internal model
- `row_id` is stripped via `model_dump(exclude={"row_id"})` before returning to user
- Errors if user's model already has a `row_id` field (reserved)

---

## Serializable Configs

Both `Model` and `Job` are Pydantic `BaseModel` subclasses, making them natively serializable to JSON. This allows users to save, share, and version their configs.

```mermaid
graph LR
    subgraph "Pydantic BaseModel"
        JOB["Job"]
        MODEL["Model"]
    end

    JOB -->|"model_dump_json()"| JSON_J["job.json"]
    MODEL -->|"model_dump_json()"| JSON_M["model.json"]
    JSON_J -->|"Job.model_validate_json()"| JOB2["Job instance"]
    JSON_M -->|"Model.model_validate_json()"| MODEL2["Model instance"]

    style JOB fill:#ff6b6b,color:#fff
    style MODEL fill:#4a9eff,color:#fff
    style JSON_J fill:#ffd43b,color:#000
    style JSON_M fill:#ffd43b,color:#000
```

```python
# Save job config
with open("classify_job.json", "w") as f:
    f.write(job.model_dump_json(indent=2))

# Load and reuse
with open("classify_job.json") as f:
    loaded_job = Job.model_validate_json(f.read())
result = loaded_job.run(model, data=new_data)
```

**Note:** `output_model` in Job is stored as the fully qualified class name (e.g., `"myapp.models.Classification"`) when serialized, and resolved via import when deserialized.

---

## Retry and Error Handling

```mermaid
flowchart LR
    subgraph "Error Categories"
        V["Validation Errors<br/>No JSON block, bad JSON,<br/>wrong count, bad IDs,<br/>Pydantic parse failure"]
        A["API Errors<br/>429, 500, 502, 503, 504"]
        T["Transport Errors<br/>Timeout, connection<br/>refused, DNS"]
        N["Non-Retriable<br/>401, 403, 400"]
    end

    V -->|Retriable| R["Retry with<br/>exponential backoff"]
    A -->|Retriable| R
    T -->|Retriable| R
    N -->|Not retriable| F["Fail immediately"]

    R --> R2["1s → 2s → 4s → 8s<br/>(+ jitter, capped 60s)"]
```

### Retry Timeline (max_retries=3)

```mermaid
gantt
    title Batch Retry Timeline
    dateFormat X
    axisFormat %s

    section Attempts
    Attempt 0 (first try)     :a0, 0, 1
    Backoff ~1s               :crit, b0, 1, 2
    Attempt 1 (retry 1)       :a1, 2, 3
    Backoff ~2s               :crit, b1, 3, 5
    Attempt 2 (retry 2)       :a2, 5, 6
    Backoff ~4s               :crit, b2, 6, 10
    Attempt 3 (retry 3)       :a3, 10, 11
    Exhausted → BatchError    :done, e, 11, 12
```

### Exhaustion Behavior

| `stop_on_exhaustion` | Behavior |
|---------------------|----------|
| `True` (default) | Cancel pending batches. Raise `SmeltExhaustionError` with `partial_result` attached so caller can recover successful rows. |
| `False` | Mark batch as failed. Continue other batches. Return `SmeltResult` with `errors` populated. |

```python
# Handling exhaustion (stop_on_exhaustion=True)
try:
    result = job.run(model, data=rows)
except SmeltExhaustionError as e:
    print(f"Failed: {e}")
    partial = e.partial_result
    print(f"Recovered {len(partial.data)} of {partial.metrics.total_rows} rows")

# Handling partial failure (stop_on_exhaustion=False)
result = job.run(model, data=rows)
if not result.success:
    for err in result.errors:
        print(f"Batch {err.batch_index} failed: {err.message}")
        print(f"  Lost row_ids: {err.row_ids}")
```

---

## Concurrency Model

```mermaid
graph TD
    subgraph "asyncio.Semaphore(concurrency=3)"
        S1["Slot 1"]
        S2["Slot 2"]
        S3["Slot 3"]
    end

    B0["Batch 0"] --> S1
    B1["Batch 1"] --> S2
    B2["Batch 2"] --> S3
    B3["Batch 3"] -.->|waiting| S1
    B4["Batch 4"] -.->|waiting| S2

    S1 --> LLM["LLM API"]
    S2 --> LLM
    S3 --> LLM

    style S1 fill:#51cf66,color:#fff
    style S2 fill:#51cf66,color:#fff
    style S3 fill:#51cf66,color:#fff
    style B3 fill:#868e96,color:#fff
    style B4 fill:#868e96,color:#fff
```

- All batches launched as `asyncio.Task` immediately
- Semaphore limits active LLM calls to `concurrency`
- Pending batches check `cancel_event` before acquiring semaphore
- In-flight batches are allowed to finish (cooperative cancellation)

---

## Package Architecture

```mermaid
graph TD
    subgraph "Public API (smelt/__init__.py)"
        PUB_MODEL["Model"]
        PUB_JOB["Job"]
        PUB_RESULT["SmeltResult"]
        PUB_ERR["SmeltError<br/>SmeltConfigError<br/>SmeltValidationError<br/>SmeltAPIError<br/>SmeltExhaustionError"]
        PUB_TYPES["SmeltMetrics<br/>BatchError"]
    end

    subgraph "Internal Modules"
        BATCH["batch.py<br/>─────────<br/>execute_batches()<br/>_process_batch()<br/>_is_retriable()<br/>_compute_backoff()"]
        PROMPT["prompt.py<br/>─────────<br/>build_system_message()<br/>build_human_message()<br/>extract_json_block()"]
        VALID["validation.py<br/>─────────<br/>create_internal_model()<br/>parse_and_validate()<br/>validate_row_ids()<br/>strip_row_id()"]
    end

    subgraph "External Dependencies"
        LC["langchain<br/>init_chat_model"]
        LCC["langchain-core<br/>BaseChatModel<br/>Messages"]
        PYD["pydantic v2<br/>BaseModel<br/>model_json_schema"]
    end

    PUB_JOB -->|"arun()/run()"| BATCH
    BATCH --> PROMPT
    BATCH --> VALID
    PUB_MODEL -->|"get_chat_model()"| LC
    BATCH --> LCC
    PROMPT --> LCC
    VALID --> PYD
    PUB_MODEL --> PYD
    PUB_JOB --> PYD

    style PUB_MODEL fill:#4a9eff,color:#fff
    style PUB_JOB fill:#ff6b6b,color:#fff
    style PUB_RESULT fill:#ffd43b,color:#000
```

---

## Dependency Stack

```mermaid
graph BT
    SMELT["smelt"]
    LC["langchain"]
    LCC["langchain-core"]
    PYD["pydantic >= 2.0"]
    PROVIDER["langchain-openai<br/>langchain-anthropic<br/>langchain-google-genai<br/>(user installs)"]

    SMELT --> LC
    SMELT --> LCC
    SMELT --> PYD
    LC --> LCC
    LC --> PYD
    PROVIDER -.->|optional| LC

    style SMELT fill:#4a9eff,color:#fff
    style PROVIDER fill:#868e96,color:#fff
```

```
pip install smelt                      # core, no providers
pip install smelt[openai]              # + langchain-openai
pip install smelt[anthropic]           # + langchain-anthropic
pip install "smelt[openai,anthropic]"  # multiple
```

---

## Result Object

```mermaid
classDiagram
    class SmeltResult~T~ {
        +list~T~ data
        +list~BatchError~ errors
        +SmeltMetrics metrics
        +bool success
    }

    class BatchError {
        +int batch_index
        +list~int~ row_ids
        +str error_type
        +str message
        +int attempts
        +str|None raw_response
    }

    class SmeltMetrics {
        +int total_rows
        +int successful_rows
        +int failed_rows
        +int total_batches
        +int successful_batches
        +int failed_batches
        +int total_retries
        +int input_tokens
        +int output_tokens
        +int total_tokens
        +float wall_time_seconds
    }

    SmeltResult --> BatchError
    SmeltResult --> SmeltMetrics
```

---

## Exception Hierarchy

```mermaid
classDiagram
    class SmeltError {
        Base exception
    }
    class SmeltConfigError {
        Bad config / missing provider
    }
    class SmeltValidationError {
        +str|None raw_response
        Output validation failure
    }
    class SmeltAPIError {
        +int|None status_code
        LLM API error
    }
    class SmeltExhaustionError {
        +SmeltResult|None partial_result
        +list~BatchError~ failed_batch_errors
        Retries exhausted
    }

    SmeltError <|-- SmeltConfigError
    SmeltError <|-- SmeltValidationError
    SmeltError <|-- SmeltAPIError
    SmeltError <|-- SmeltExhaustionError
```

---

## File Manifest

| File | Purpose | Key Contents |
|------|---------|-------------|
| `pyproject.toml` | Package config | hatchling build, deps, Python >=3.10 |
| `src/smelt/__init__.py` | Public API | Exports Model, Job, SmeltResult, errors |
| `src/smelt/types.py` | Data containers | SmeltResult[T], SmeltMetrics, BatchError, internal types |
| `src/smelt/errors.py` | Exceptions | SmeltError hierarchy (5 classes) |
| `src/smelt/model.py` | LLM config | Model (Pydantic), lazy init_chat_model |
| `src/smelt/job.py` | Entry point | Job (Pydantic), run() / arun() |
| `src/smelt/batch.py` | Core engine | execute_batches(), retry loop, semaphore concurrency |
| `src/smelt/prompt.py` | Prompt building | System/human messages, JSON extraction, schema stitching |
| `src/smelt/validation.py` | Output validation | Dynamic model creation, row_id checks, Pydantic validation |

---

## Implementation Order

```mermaid
graph LR
    T["1. types.py"] --> E["2. errors.py"]
    E --> V["3. validation.py"]
    E --> P["4. prompt.py"]
    E --> M["5. model.py"]
    V --> B["6. batch.py"]
    P --> B
    M --> B
    B --> J["7. job.py"]
    J --> I["8. __init__.py"]
    T --> I

    style T fill:#51cf66,color:#fff
    style I fill:#ffd43b,color:#000
```

Each module only depends on modules to its left. Zero circular dependencies.
