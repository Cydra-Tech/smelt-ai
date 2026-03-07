"""Tests for smelt.image module."""

from __future__ import annotations

import base64
import io
from unittest.mock import patch

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image

from smelt.image import (
    PILLOW_AVAILABLE,
    _ImageRef,
    _detect_image_format,
    _require_pillow,
    batch_has_images,
    encode_image_to_base64,
    extract_images_from_rows,
    is_pil_image,
)
from smelt.types import _TaggedRow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_image(width: int = 10, height: int = 10) -> Image.Image:
    """Create a small RGB test image."""
    return Image.new("RGB", (width, height), color=(255, 0, 0))


def _make_rgba_image(width: int = 10, height: int = 10) -> Image.Image:
    """Create a small RGBA test image."""
    return Image.new("RGBA", (width, height), color=(255, 0, 0, 128))


# ---------------------------------------------------------------------------
# TestPillowAvailability
# ---------------------------------------------------------------------------


class TestPillowAvailability:
    """Tests for Pillow availability detection."""

    def test_pillow_available_flag_is_true(self) -> None:
        """PILLOW_AVAILABLE should be True when Pillow is installed."""
        assert PILLOW_AVAILABLE is True

    def test_require_pillow_succeeds(self) -> None:
        """_require_pillow should not raise when Pillow is available."""
        _require_pillow()

    def test_require_pillow_raises_when_missing(self) -> None:
        """_require_pillow should raise ImportError with install instructions."""
        with patch("smelt.image.PILLOW_AVAILABLE", False):
            with pytest.raises(ImportError, match="smelt-ai\\[vision\\]"):
                _require_pillow()


# ---------------------------------------------------------------------------
# TestIsPilImage
# ---------------------------------------------------------------------------


class TestIsPilImage:
    """Tests for is_pil_image."""

    def test_detects_pil_rgb_image(self) -> None:
        """Should return True for an RGB PIL Image."""
        assert is_pil_image(_make_rgb_image()) is True

    def test_detects_pil_rgba_image(self) -> None:
        """Should return True for an RGBA PIL Image."""
        assert is_pil_image(_make_rgba_image()) is True

    def test_rejects_string(self) -> None:
        """Should return False for a string."""
        assert is_pil_image("not an image") is False

    def test_rejects_integer(self) -> None:
        """Should return False for an integer."""
        assert is_pil_image(42) is False

    def test_rejects_none(self) -> None:
        """Should return False for None."""
        assert is_pil_image(None) is False

    def test_rejects_dict(self) -> None:
        """Should return False for a dict."""
        assert is_pil_image({"key": "value"}) is False

    def test_returns_false_when_pillow_missing(self) -> None:
        """Should return False when Pillow is not installed."""
        with patch("smelt.image.PILLOW_AVAILABLE", False):
            assert is_pil_image(_make_rgb_image()) is False


# ---------------------------------------------------------------------------
# TestDetectImageFormat
# ---------------------------------------------------------------------------


class TestDetectImageFormat:
    """Tests for _detect_image_format."""

    def test_uses_existing_format(self) -> None:
        """Should use the image's format attribute when set."""
        img = _make_rgb_image()
        img.format = "PNG"
        assert _detect_image_format(img) == "png"

    def test_rgba_defaults_to_png(self) -> None:
        """RGBA images without format should default to png."""
        img = _make_rgba_image()
        assert _detect_image_format(img) == "png"

    def test_rgb_defaults_to_jpeg(self) -> None:
        """RGB images without format should default to jpeg."""
        img = _make_rgb_image()
        assert _detect_image_format(img) == "jpeg"

    def test_jpg_normalized_to_jpeg(self) -> None:
        """Format 'JPG' should be normalized to 'jpeg'."""
        img = _make_rgb_image()
        img.format = "JPG"
        assert _detect_image_format(img) == "jpeg"

    def test_jpeg_format_preserved(self) -> None:
        """Format 'JPEG' should be preserved as 'jpeg'."""
        img = _make_rgb_image()
        img.format = "JPEG"
        assert _detect_image_format(img) == "jpeg"

    def test_grayscale_defaults_to_jpeg(self) -> None:
        """Grayscale (L) images without format should default to jpeg."""
        img = Image.new("L", (10, 10), color=128)
        assert _detect_image_format(img) == "jpeg"

    def test_palette_mode_defaults_to_jpeg(self) -> None:
        """Palette (P) mode images without format should default to jpeg."""
        img = Image.new("P", (10, 10))
        assert _detect_image_format(img) == "jpeg"

    def test_la_mode_defaults_to_png(self) -> None:
        """LA (grayscale + alpha) mode images should default to png."""
        img = Image.new("LA", (10, 10))
        assert _detect_image_format(img) == "png"


# ---------------------------------------------------------------------------
# TestEncodeImageToBase64
# ---------------------------------------------------------------------------


class TestEncodeImageToBase64:
    """Tests for encode_image_to_base64."""

    def test_rgb_image_produces_jpeg(self) -> None:
        """RGB image should be encoded as JPEG with correct MIME type."""
        img = _make_rgb_image()
        b64, mime = encode_image_to_base64(img)
        assert mime == "image/jpeg"
        assert len(b64) > 0

    def test_rgba_image_produces_png(self) -> None:
        """RGBA image should be encoded as PNG with correct MIME type."""
        img = _make_rgba_image()
        b64, mime = encode_image_to_base64(img)
        assert mime == "image/png"
        assert len(b64) > 0

    def test_rgba_as_jpeg_converts_to_rgb(self) -> None:
        """RGBA image with JPEG format should be converted to RGB."""
        img = _make_rgba_image()
        img.format = "JPEG"
        b64, mime = encode_image_to_base64(img)
        assert mime == "image/jpeg"
        assert len(b64) > 0

    def test_output_is_valid_base64(self) -> None:
        """Output should be valid base64 that can be decoded."""
        img = _make_rgb_image()
        b64, _ = encode_image_to_base64(img)
        decoded: bytes = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_roundtrip_rgb(self) -> None:
        """Should be able to decode base64 back to a valid image."""
        img = _make_rgb_image(20, 20)
        b64, _ = encode_image_to_base64(img)
        decoded_bytes: bytes = base64.b64decode(b64)
        restored: Image.Image = Image.open(io.BytesIO(decoded_bytes))
        assert restored.size == (20, 20)

    def test_roundtrip_rgba(self) -> None:
        """Should be able to decode base64 back to a valid PNG image."""
        img = _make_rgba_image(15, 15)
        b64, _ = encode_image_to_base64(img)
        decoded_bytes: bytes = base64.b64decode(b64)
        restored: Image.Image = Image.open(io.BytesIO(decoded_bytes))
        assert restored.size == (15, 15)

    def test_palette_mode_encodes_without_error(self) -> None:
        """Palette (P) mode images should be converted and encoded."""
        img = Image.new("P", (10, 10))
        b64, mime = encode_image_to_base64(img)
        assert len(b64) > 0
        assert mime == "image/jpeg"

    def test_grayscale_mode_encodes(self) -> None:
        """Grayscale (L) mode images should encode as JPEG."""
        img = Image.new("L", (10, 10), color=128)
        b64, mime = encode_image_to_base64(img)
        assert mime == "image/jpeg"
        assert len(b64) > 0

    def test_la_mode_encodes_as_png(self) -> None:
        """LA (grayscale + alpha) mode images should encode as PNG."""
        img = Image.new("LA", (10, 10))
        b64, mime = encode_image_to_base64(img)
        assert mime == "image/png"
        assert len(b64) > 0

    def test_encode_raises_when_pillow_missing(self) -> None:
        """Should raise ImportError when Pillow is not installed."""
        with patch("smelt.image.PILLOW_AVAILABLE", False):
            with pytest.raises(ImportError, match="smelt-ai\\[vision\\]"):
                encode_image_to_base64(_make_rgb_image())


# ---------------------------------------------------------------------------
# TestBatchHasImages
# ---------------------------------------------------------------------------


class TestBatchHasImages:
    """Tests for batch_has_images."""

    def test_no_images(self) -> None:
        """Should return False when no rows contain images."""
        rows = [{"name": "Alice"}, {"name": "Bob"}]
        assert batch_has_images(rows) is False

    def test_with_images(self) -> None:
        """Should return True when a row contains a PIL Image."""
        rows = [{"name": "Alice", "photo": _make_rgb_image()}]
        assert batch_has_images(rows) is True

    def test_mixed_rows(self) -> None:
        """Should return True when only some rows have images."""
        rows = [
            {"name": "Alice"},
            {"name": "Bob", "photo": _make_rgb_image()},
        ]
        assert batch_has_images(rows) is True

    def test_empty_rows(self) -> None:
        """Should return False for an empty list."""
        assert batch_has_images([]) is False

    def test_none_values_in_rows(self) -> None:
        """Should handle None values in row dicts without error."""
        rows = [{"photo": None, "name": "Alice"}]
        assert batch_has_images(rows) is False

    def test_empty_dict_rows(self) -> None:
        """Should handle rows with empty dicts."""
        rows = [{}]
        assert batch_has_images(rows) is False


# ---------------------------------------------------------------------------
# TestExtractImagesFromRows
# ---------------------------------------------------------------------------


class TestExtractImagesFromRows:
    """Tests for extract_images_from_rows."""

    def test_no_images_passthrough(self) -> None:
        """Rows without images should pass through unchanged."""
        rows = [_TaggedRow(row_id=0, data={"name": "Alice", "age": 30})]
        cleaned, refs = extract_images_from_rows(rows)
        assert len(cleaned) == 1
        assert cleaned[0].data == {"name": "Alice", "age": 30}
        assert refs == []

    def test_single_image_extraction(self) -> None:
        """Should extract a single image and leave a placeholder."""
        img = _make_rgb_image()
        rows = [_TaggedRow(row_id=0, data={"name": "Alice", "photo": img})]
        cleaned, refs = extract_images_from_rows(rows)

        assert cleaned[0].data["photo"] == "[image: photo]"
        assert cleaned[0].data["name"] == "Alice"
        assert len(refs) == 1
        assert refs[0].row_id == 0
        assert refs[0].field_name == "photo"
        assert refs[0].mime_type == "image/jpeg"
        assert len(refs[0].base64_data) > 0

    def test_multiple_images_per_row(self) -> None:
        """Should extract multiple images from a single row."""
        rows = [
            _TaggedRow(
                row_id=0,
                data={
                    "front": _make_rgb_image(),
                    "back": _make_rgba_image(),
                },
            )
        ]
        cleaned, refs = extract_images_from_rows(rows)

        assert cleaned[0].data["front"] == "[image: front]"
        assert cleaned[0].data["back"] == "[image: back]"
        assert len(refs) == 2

    def test_images_across_rows(self) -> None:
        """Should extract images from multiple rows."""
        rows = [
            _TaggedRow(row_id=0, data={"img": _make_rgb_image()}),
            _TaggedRow(row_id=1, data={"img": _make_rgba_image()}),
        ]
        cleaned, refs = extract_images_from_rows(rows)

        assert len(cleaned) == 2
        assert len(refs) == 2
        assert refs[0].row_id == 0
        assert refs[1].row_id == 1

    def test_placeholder_format(self) -> None:
        """Placeholder should match the '[image: field_name]' format."""
        rows = [_TaggedRow(row_id=0, data={"ecg": _make_rgb_image()})]
        cleaned, _ = extract_images_from_rows(rows)
        assert cleaned[0].data["ecg"] == "[image: ecg]"

    def test_non_image_fields_preserved(self) -> None:
        """Non-image fields should be preserved unchanged."""
        rows = [
            _TaggedRow(
                row_id=0,
                data={
                    "patient_id": "P001",
                    "age": 45,
                    "ecg": _make_rgb_image(),
                    "notes": "Normal sinus rhythm",
                },
            )
        ]
        cleaned, _ = extract_images_from_rows(rows)

        assert cleaned[0].data["patient_id"] == "P001"
        assert cleaned[0].data["age"] == 45
        assert cleaned[0].data["notes"] == "Normal sinus rhythm"
        assert cleaned[0].data["ecg"] == "[image: ecg]"

    def test_row_ids_preserved(self) -> None:
        """Row IDs should be preserved in cleaned rows."""
        rows = [
            _TaggedRow(row_id=5, data={"img": _make_rgb_image()}),
            _TaggedRow(row_id=10, data={"img": _make_rgb_image()}),
        ]
        cleaned, _ = extract_images_from_rows(rows)
        assert cleaned[0].row_id == 5
        assert cleaned[1].row_id == 10

    def test_empty_input_returns_empty(self) -> None:
        """Empty tagged_rows should return empty results."""
        cleaned, refs = extract_images_from_rows([])
        assert cleaned == []
        assert refs == []

    def test_row_with_empty_data(self) -> None:
        """Row with empty data dict should pass through."""
        rows = [_TaggedRow(row_id=0, data={})]
        cleaned, refs = extract_images_from_rows(rows)
        assert len(cleaned) == 1
        assert cleaned[0].data == {}
        assert refs == []

    def test_image_ref_is_frozen(self) -> None:
        """_ImageRef should be immutable."""
        ref = _ImageRef(row_id=0, field_name="img", base64_data="abc", mime_type="image/png")
        with pytest.raises(AttributeError):
            ref.row_id = 1  # type: ignore[misc]
