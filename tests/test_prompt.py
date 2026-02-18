"""Tests for smelt.prompt module."""

from __future__ import annotations

from pydantic import BaseModel, Field

from smelt.prompt import build_human_message, build_system_message, describe_output_schema
from smelt.types import _TaggedRow


class AnnotatedModel(BaseModel):
    """Model with field descriptions for testing schema description."""

    category: str = Field(description="The industry category")
    confidence: float = Field(description="Confidence score between 0 and 1")
    optional_note: str | None = Field(default=None, description="Optional notes")


class TestDescribeOutputSchema:
    """Tests for describe_output_schema."""

    def test_includes_all_fields(self) -> None:
        """Should list every field in the model."""
        description = describe_output_schema(AnnotatedModel)
        assert "category" in description
        assert "confidence" in description
        assert "optional_note" in description

    def test_shows_required_status(self) -> None:
        """Should indicate which fields are required vs optional."""
        description = describe_output_schema(AnnotatedModel)
        assert "required" in description
        assert "optional" in description

    def test_includes_field_descriptions(self) -> None:
        """Should include Pydantic field descriptions."""
        description = describe_output_schema(AnnotatedModel)
        assert "industry category" in description
        assert "Confidence score" in description


class TestBuildSystemMessage:
    """Tests for build_system_message."""

    def test_includes_user_prompt(self) -> None:
        """System message should contain the user's instruction."""
        msg = build_system_message("Classify companies", "schema info here")
        assert "Classify companies" in msg.content

    def test_includes_schema(self) -> None:
        """System message should contain the schema description."""
        msg = build_system_message("task", "- category (str, required)")
        assert "category" in msg.content

    def test_includes_row_id_rules(self) -> None:
        """System message should explain row_id tracking rules."""
        msg = build_system_message("task", "schema")
        assert "row_id" in msg.content

    def test_is_system_message_type(self) -> None:
        """Should return a LangChain SystemMessage."""
        msg = build_system_message("task", "schema")
        assert msg.type == "system"


class TestBuildHumanMessage:
    """Tests for build_human_message."""

    def test_includes_row_ids(self) -> None:
        """Human message should include row_id in each row object."""
        rows = [
            _TaggedRow(row_id=0, data={"name": "Apple"}),
            _TaggedRow(row_id=1, data={"name": "Google"}),
        ]
        msg = build_human_message(rows)
        assert '"row_id": 0' in msg.content
        assert '"row_id": 1' in msg.content

    def test_includes_row_data(self) -> None:
        """Human message should include the original row data."""
        rows = [_TaggedRow(row_id=0, data={"company": "Tesla", "revenue": 81000})]
        msg = build_human_message(rows)
        assert "Tesla" in msg.content
        assert "81000" in msg.content

    def test_is_human_message_type(self) -> None:
        """Should return a LangChain HumanMessage."""
        rows = [_TaggedRow(row_id=0, data={"x": 1})]
        msg = build_human_message(rows)
        assert msg.type == "human"
