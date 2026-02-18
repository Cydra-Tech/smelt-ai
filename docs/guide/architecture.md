# Architecture

## Pipeline overview

```mermaid
flowchart LR
    A["list[dict]"] --> B["Tag rows\nwith row_id"]
    B --> S{"shuffle?"}
    S -->|Yes| Sh["Shuffle\ntagged rows"]
    S -->|No| C
    Sh --> C["Split into\nbatches"]
    C --> D["Concurrent\nLLM calls"]
    D --> E["Validate\nschema + IDs"]
    E --> F["Reorder by\nrow_id"]
    F --> G["SmeltResult[T]"]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

## How a job executes

```mermaid
sequenceDiagram
    participant User
    participant Job
    participant BatchEngine
    participant LLM

    User->>Job: job.run(model, data)
    Job->>BatchEngine: execute_batches()

    Note over BatchEngine: Tag rows with row_id<br/>Shuffle if enabled<br/>Create internal Pydantic model<br/>Build system prompt<br/>Split into batches

    par Batch 0
        BatchEngine->>LLM: [system_msg, human_msg]
        LLM-->>BatchEngine: structured response
    and Batch 1
        BatchEngine->>LLM: [system_msg, human_msg]
        LLM-->>BatchEngine: structured response
    and Batch N
        BatchEngine->>LLM: [system_msg, human_msg]
        LLM-->>BatchEngine: structured response
    end

    Note over BatchEngine: Validate row IDs per batch<br/>Sort by row_id<br/>Strip row_id field<br/>Aggregate metrics

    BatchEngine-->>Job: SmeltResult[T]
    Job-->>User: result
```

## Row ID tracking

Smelt injects a `row_id` field into your Pydantic model, instructs the LLM to echo it back, then validates and strips it before returning results. This ensures correct ordering even when batches complete out of order.

```mermaid
flowchart LR
    subgraph Input
        direction TB
        R0["row 0: {name: Apple}"]
        R1["row 1: {name: Stripe}"]
        R2["row 2: {name: Mayo}"]
    end

    subgraph Tagged
        direction TB
        T0["{row_id: 0, name: Apple}"]
        T1["{row_id: 1, name: Stripe}"]
        T2["{row_id: 2, name: Mayo}"]
    end

    subgraph "LLM Output (may be unordered)"
        direction TB
        L1["{row_id: 1, sector: Fintech}"]
        L0["{row_id: 0, sector: Tech}"]
        L2["{row_id: 2, sector: Health}"]
    end

    subgraph "Final (reordered)"
        direction TB
        F0["Classification(sector=Tech)"]
        F1["Classification(sector=Fintech)"]
        F2["Classification(sector=Health)"]
    end

    Input --> Tagged --> L1
    L1 ~~~ L0
    L0 ~~~ L2
    L2 --> F0

    style F0 fill:#9f9,stroke:#333
    style F1 fill:#9f9,stroke:#333
    style F2 fill:#9f9,stroke:#333
```

## Dynamic model creation

Under the hood, smelt dynamically extends your Pydantic model to add `row_id`, then wraps it in a batch container for LangChain's `with_structured_output`:

```mermaid
classDiagram
    class YourModel {
        +str sector
        +str sub_sector
        +bool is_public
    }

    class _SmeltYourModel {
        +int row_id
        +str sector
        +str sub_sector
        +bool is_public
    }

    class _SmeltBatch {
        +list~_SmeltYourModel~ rows
    }

    YourModel <|-- _SmeltYourModel : extends via create_model()
    _SmeltYourModel --* _SmeltBatch : rows

    note for _SmeltYourModel "Injected row_id for tracking.\nStripped before returning to user."
    note for _SmeltBatch "Wrapper required by LangChain's\nwith_structured_output()."
```

This approach:

1. **Preserves all your validators** — the internal model inherits from your model, so all field validators and model validators run
2. **Keeps the public API clean** — `row_id` is never visible in your results
3. **Enables batch processing** — LangChain's `with_structured_output` requires a single model, not a list. The batch wrapper provides this.
