"""Tests for smelt.prompt module."""

from __future__ import annotations

import json
from typing import Any

import pytest
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
