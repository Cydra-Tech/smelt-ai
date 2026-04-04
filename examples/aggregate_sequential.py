"""Aggregate a dataset using sequential fold — each step builds on the previous.

Usage:
    pip install smelt-ai[google]
    export GEMINI_API_KEY="..."
    python examples/aggregate_sequential.py
"""

from pydantic import BaseModel, Field

from smelt import AggregateJob, Model


class SurveyAnalysis(BaseModel):
    """Aggregate analysis of employee survey responses."""

    total_responses: int = Field(description="Total number of survey responses")
    average_satisfaction: float = Field(description="Average satisfaction score (1-5)")
    departments: list[str] = Field(description="List of unique departments")
    common_complaints: list[str] = Field(description="Common complaint themes")
    common_praises: list[str] = Field(description="Common positive themes")
    lowest_rated_department: str = Field(description="Department with lowest avg satisfaction")
    highest_rated_department: str = Field(description="Department with highest avg satisfaction")


def main() -> None:
    survey_data = [
        {"id": "E01", "dept": "Engineering", "satisfaction": 5, "comment": "Great tech stack and team culture"},
        {"id": "E02", "dept": "Sales", "satisfaction": 2, "comment": "Unrealistic quotas, no career path"},
        {"id": "E03", "dept": "Engineering", "satisfaction": 4, "comment": "Good mentorship, tight deadlines"},
        {"id": "E04", "dept": "HR", "satisfaction": 3, "comment": "Decent balance but too much paperwork"},
        {"id": "E05", "dept": "Marketing", "satisfaction": 4, "comment": "Creative freedom, tight budget"},
        {"id": "E06", "dept": "Support", "satisfaction": 2, "comment": "Burnout from high ticket volume"},
        {"id": "E07", "dept": "Engineering", "satisfaction": 5, "comment": "Best engineering culture ever"},
        {"id": "E08", "dept": "Finance", "satisfaction": 4, "comment": "Stable team, clear quarterly goals"},
        {"id": "E09", "dept": "Sales", "satisfaction": 1, "comment": "Toxic management, impossible targets"},
        {"id": "E10", "dept": "Support", "satisfaction": 3, "comment": "Team is great but tools are outdated"},
        {"id": "E11", "dept": "Engineering", "satisfaction": 4, "comment": "Flexible remote work policy"},
        {"id": "E12", "dept": "Marketing", "satisfaction": 5, "comment": "Data-driven approach works well"},
        {"id": "E13", "dept": "HR", "satisfaction": 4, "comment": "New onboarding process is great"},
        {"id": "E14", "dept": "Finance", "satisfaction": 3, "comment": "Repetitive work but supportive team"},
        {"id": "E15", "dept": "Sales", "satisfaction": 3, "comment": "New CRM helped, quota still high"},
    ]

    model = Model(provider="google_genai", name="gemini-2.0-flash")

    job = AggregateJob(
        prompt="Analyze all employee survey responses. Count totals accurately. "
        "Calculate the average satisfaction. List all departments. "
        "Identify common complaint and praise themes. "
        "Determine which department has the lowest and highest average satisfaction.",
        output_model=SurveyAnalysis,
        strategy="sequential",
        batch_size=5,
    )

    print(f"Analyzing {len(survey_data)} survey responses (sequential, batch_size=5)...\n")
    result = job.run(model, data=survey_data)

    analysis = result.data[0]
    print(f"Total responses:      {analysis.total_responses}")
    print(f"Avg satisfaction:     {analysis.average_satisfaction:.2f}")
    print(f"Departments:          {analysis.departments}")
    print(f"Lowest rated:         {analysis.lowest_rated_department}")
    print(f"Highest rated:        {analysis.highest_rated_department}")
    print(f"Common complaints:    {analysis.common_complaints}")
    print(f"Common praises:       {analysis.common_praises}")
    print()
    print(f"Steps: {result.metrics.total_batches}")
    print(f"Time:  {result.metrics.wall_time_seconds:.2f}s")
    print(f"Tokens: {result.metrics.input_tokens} in / {result.metrics.output_tokens} out")


if __name__ == "__main__":
    main()
