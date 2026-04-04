"""Tests for smelt.prompt module."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, Field

from smelt.prompt import (
    build_aggregate_human_message,
    build_aggregate_system_message,
    build_human_message,
    build_system_message,
    describe_output_schema,
)
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


class TestBuildSystemMessageTextMode:
    """Tests for build_system_message in free-text mode."""

    def test_text_mode_includes_user_prompt(self) -> None:
        """Text mode system message should contain the user's instruction."""
        msg = build_system_message("Summarize each company", text_mode=True)
        assert "Summarize each company" in msg.content

    def test_text_mode_includes_text_field_instruction(self) -> None:
        """Text mode should instruct LLM to return a 'text' field."""
        msg = build_system_message("task", text_mode=True)
        assert '"text"' in msg.content

    def test_text_mode_includes_row_id_rules(self) -> None:
        """Text mode should still include row_id tracking rules."""
        msg = build_system_message("task", text_mode=True)
        assert "row_id" in msg.content

    def test_text_mode_does_not_include_output_schema(self) -> None:
        """Text mode should NOT include an Output Schema section."""
        msg = build_system_message("task", text_mode=True)
        assert "## Output Schema" not in msg.content

    def test_text_mode_includes_rows_key_instruction(self) -> None:
        """Text mode should instruct LLM to return rows key."""
        msg = build_system_message("task", text_mode=True)
        assert '"rows"' in msg.content

    def test_text_mode_with_images(self) -> None:
        """Text mode with images should append the image addendum."""
        msg = build_system_message("task", text_mode=True, has_images=True)
        assert "image blocks" in msg.content.lower()

    def test_text_mode_is_system_message_type(self) -> None:
        """Text mode should return a LangChain SystemMessage."""
        msg = build_system_message("task", text_mode=True)
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


# ---------------------------------------------------------------------------
# Image / multimodal tests
# ---------------------------------------------------------------------------

PIL = pytest.importorskip("PIL")
from PIL import Image


def _make_test_image() -> Image.Image:
    """Create a small RGB test image."""
    return Image.new("RGB", (4, 4), color=(0, 128, 255))


class TestBuildSystemMessageWithImages:
    """Tests for build_system_message image addendum."""

    def test_no_image_addendum_by_default(self) -> None:
        """System message should NOT contain image instructions by default."""
        msg = build_system_message("task", "schema")
        assert "image placeholder" not in msg.content.lower()
        assert "[image:" not in msg.content

    def test_image_addendum_when_has_images(self) -> None:
        """System message should include image instructions when has_images=True."""
        msg = build_system_message("task", "schema", has_images=True)
        assert "[image: field_name]" in msg.content
        assert "image blocks" in msg.content.lower()

    def test_backward_compat_without_has_images(self) -> None:
        """Calling without has_images kwarg should work (defaults to False)."""
        msg = build_system_message("task", "schema")
        assert "Images" not in msg.content


class TestBuildHumanMessageWithImages:
    """Tests for build_human_message multimodal content blocks."""

    def test_text_only_returns_string_content(self) -> None:
        """Text-only rows should produce a plain string content (backward compat)."""
        rows = [_TaggedRow(row_id=0, data={"name": "Alice"})]
        msg = build_human_message(rows)
        assert isinstance(msg.content, str)

    def test_image_rows_return_list_content(self) -> None:
        """Rows with images should produce a list of content blocks."""
        rows = [_TaggedRow(row_id=0, data={"name": "Alice", "photo": _make_test_image()})]
        msg = build_human_message(rows)
        assert isinstance(msg.content, list)

    def test_content_blocks_structure(self) -> None:
        """Content blocks should have text + (label, image_url) pairs."""
        rows = [_TaggedRow(row_id=0, data={"ecg": _make_test_image()})]
        msg = build_human_message(rows)
        blocks: list[dict[str, Any]] = msg.content  # type: ignore[assignment]

        assert blocks[0]["type"] == "text"
        assert "[image: ecg]" in blocks[0]["text"]

        assert blocks[1]["type"] == "text"
        assert "Row 0" in blocks[1]["text"]
        assert "ecg" in blocks[1]["text"]

        assert blocks[2]["type"] == "image_url"
        assert blocks[2]["image_url"]["url"].startswith("data:image/")

    def test_image_label_includes_row_id(self) -> None:
        """Image label should reference the correct row_id."""
        rows = [_TaggedRow(row_id=5, data={"scan": _make_test_image()})]
        msg = build_human_message(rows)
        blocks: list[dict[str, Any]] = msg.content  # type: ignore[assignment]

        label_block = blocks[1]
        assert "Row 5" in label_block["text"]
        assert "scan" in label_block["text"]

    def test_multiple_images_produce_multiple_blocks(self) -> None:
        """Multiple images across rows should produce separate blocks for each."""
        rows = [
            _TaggedRow(row_id=0, data={"img": _make_test_image()}),
            _TaggedRow(row_id=1, data={"img": _make_test_image()}),
        ]
        msg = build_human_message(rows)
        blocks: list[dict[str, Any]] = msg.content  # type: ignore[assignment]

        image_blocks = [b for b in blocks if b["type"] == "image_url"]
        assert len(image_blocks) == 2

    def test_json_payload_contains_placeholders(self) -> None:
        """JSON text block should have placeholders, not raw image data."""
        rows = [_TaggedRow(row_id=0, data={"photo": _make_test_image(), "name": "Bob"})]
        msg = build_human_message(rows)
        blocks: list[dict[str, Any]] = msg.content  # type: ignore[assignment]

        text_json: str = blocks[0]["text"]
        parsed: list[dict[str, Any]] = json.loads(text_json)
        assert parsed[0]["photo"] == "[image: photo]"
        assert parsed[0]["name"] == "Bob"

    def test_mixed_image_and_text_rows(self) -> None:
        """Rows with a mix of image and text fields should work correctly."""
        rows = [
            _TaggedRow(row_id=0, data={"id": "P001", "ecg": _make_test_image()}),
            _TaggedRow(row_id=1, data={"id": "P002", "notes": "normal"}),
        ]
        msg = build_human_message(rows)
        blocks: list[dict[str, Any]] = msg.content  # type: ignore[assignment]

        image_blocks = [b for b in blocks if b["type"] == "image_url"]
        assert len(image_blocks) == 1


# ---------------------------------------------------------------------------
# Aggregate prompt tests
# ---------------------------------------------------------------------------


class TestBuildAggregateSystemMessage:
    """Tests for build_aggregate_system_message."""

    def test_map_structured_includes_prompt(self) -> None:
        """Map system message should contain the user's instruction."""
        msg = build_aggregate_system_message("Analyze companies", "schema here")
        assert "Analyze companies" in msg.content

    def test_map_structured_includes_schema(self) -> None:
        """Map system message should contain the schema description."""
        msg = build_aggregate_system_message("task", "- total_items (int)")
        assert "total_items" in msg.content

    def test_map_structured_includes_context(self) -> None:
        """Map system message should explain subset/merge context."""
        msg = build_aggregate_system_message("task", "schema")
        assert "subset" in msg.content
        assert "merged" in msg.content.lower()

    def test_map_text_mode(self) -> None:
        """Text mode map message should not include schema section."""
        msg = build_aggregate_system_message("task", text_mode=True)
        assert "## Output Schema" not in msg.content
        assert "subset" in msg.content

    def test_merge_structured_includes_merge_context(self) -> None:
        """Merge system message should explain merging two partial results."""
        msg = build_aggregate_system_message("task", "schema", is_merge=True)
        assert "merging two partial results" in msg.content.lower()

    def test_merge_text_mode(self) -> None:
        """Text mode merge message should explain merging."""
        msg = build_aggregate_system_message("task", text_mode=True, is_merge=True)
        assert "merging two partial results" in msg.content.lower()
        assert "## Output Schema" not in msg.content

    def test_no_row_id_in_aggregate(self) -> None:
        """Aggregate prompts should not mention row_id."""
        msg = build_aggregate_system_message("task", "schema")
        assert "row_id" not in msg.content

    def test_sequential_structured(self) -> None:
        """Sequential system message should mention sequential processing."""
        msg = build_aggregate_system_message("task", "schema", is_sequential=True)
        assert "sequential" in msg.content.lower()
        assert "## Output Schema" in msg.content

    def test_sequential_text_mode(self) -> None:
        """Sequential text mode should not include schema."""
        msg = build_aggregate_system_message("task", text_mode=True, is_sequential=True)
        assert "sequential" in msg.content.lower()
        assert "## Output Schema" not in msg.content


class TestBuildAggregateHumanMessage:
    """Tests for build_aggregate_human_message."""

    def test_map_step_contains_json_rows(self) -> None:
        """Map step should serialize rows as JSON."""
        rows = [{"name": "Apple"}, {"name": "Google"}]
        msg = build_aggregate_human_message(rows=rows)
        parsed = json.loads(msg.content)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "Apple"

    def test_merge_step_contains_both_results(self) -> None:
        """Merge step should contain both partial results."""
        msg = build_aggregate_human_message(
            previous_result='{"total": 5}',
            second_result='{"total": 3}',
        )
        assert "Partial result 1" in msg.content
        assert "Partial result 2" in msg.content
        assert '{"total": 5}' in msg.content
        assert '{"total": 3}' in msg.content

    def test_map_step_is_human_message(self) -> None:
        """Should return a HumanMessage."""
        msg = build_aggregate_human_message(rows=[{"x": 1}])
        assert msg.type == "human"

    def test_merge_step_is_human_message(self) -> None:
        """Merge step should return a HumanMessage."""
        msg = build_aggregate_human_message(
            previous_result="result A",
            second_result="result B",
        )
        assert msg.type == "human"

    def test_sequential_step_contains_previous_and_rows(self) -> None:
        """Sequential step should contain both previous result and new rows."""
        msg = build_aggregate_human_message(
            rows=[{"name": "Apple"}],
            previous_result='{"total": 5}',
        )
        assert "Previous result" in msg.content
        assert '{"total": 5}' in msg.content
        assert "New data to incorporate" in msg.content
        assert "Apple" in msg.content

    def test_sequential_first_step_no_previous(self) -> None:
        """First sequential step (no previous) should just serialize rows."""
        msg = build_aggregate_human_message(rows=[{"x": 1}])
        parsed = json.loads(msg.content)
        assert parsed == [{"x": 1}]
