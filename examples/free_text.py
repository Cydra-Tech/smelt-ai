"""Generate free-text summaries without a Pydantic schema.

Usage:
    pip install smelt-ai[anthropic]
    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/free_text.py
"""

from smelt import Job, Model


def main() -> None:
    model = Model(provider="anthropic", name="claude-sonnet-4-6")

    # No output_model — returns plain strings
    job = Job(
        prompt="Write a concise one-paragraph summary for each company, "
        "highlighting their core business and market position.",
        batch_size=5,
    )

    companies = [
        {"name": "Apple", "founded": 1976, "description": "Consumer electronics, software, services"},
        {"name": "Stripe", "founded": 2010, "description": "Online payment processing platform"},
        {"name": "SpaceX", "founded": 2002, "description": "Aerospace manufacturer, space transport"},
        {"name": "Anthropic", "founded": 2021, "description": "AI safety research company"},
        {"name": "Toyota", "founded": 1937, "description": "Automobile manufacturer"},
    ]

    print(f"Generating summaries for {len(companies)} companies...\n")
    result = job.run(model, data=companies)

    for company, summary in zip(companies, result.data):
        print(f"--- {company['name']} ---")
        print(summary)
        print()

    print(f"Time: {result.metrics.wall_time_seconds:.2f}s")
    print(f"Tokens: {result.metrics.input_tokens} in / {result.metrics.output_tokens} out")


if __name__ == "__main__":
    main()
