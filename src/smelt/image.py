"""Image utilities for multimodal smelt processing.

Provides PIL image detection, base64 encoding, and extraction helpers
for passing images through vision-capable LLMs. Pillow is an optional
dependency — install with ``pip install smelt-ai[vision]``.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

try:
    from PIL import Image as PILImage

    PILLOW_AVAILABLE: bool = True
except ImportError:
    PILLOW_AVAILABLE = False

from smelt.types import _TaggedRow


def _require_pillow() -> None:
    """Raise ``ImportError`` if Pillow is not installed.

    Raises:
        ImportError: When Pillow is not available, with install instructions.
    """
    if not PILLOW_AVAILABLE:
        raise ImportError(
            "Pillow is required for image support. "
            "Install it with: pip install smelt-ai[vision]"
        )


def is_pil_image(value: Any) -> bool:
    """Check whether a value is a PIL Image instance.

    Returns ``False`` without error if Pillow is not installed.

    Args:
        value: The value to check.

    Returns:
        ``True`` if the value is a ``PIL.Image.Image``, ``False`` otherwise.
    """
    if not PILLOW_AVAILABLE:
        return False
    return isinstance(value, PILImage.Image)


def _detect_image_format(image: Any) -> str:
    """Detect the image format suitable for encoding.

    Uses the image's ``format`` attribute if set, otherwise infers from
    the image mode (RGBA → png, everything else → jpeg).

    Args:
        image: A PIL Image instance.

    Returns:
        A lowercase format string (``"png"`` or ``"jpeg"``).
    """
    fmt: str | None = getattr(image, "format", None)
    if fmt is not None:
        normalized: str = fmt.lower()
        if normalized == "jpg":
            return "jpeg"
        return normalized
    return "png" if image.mode == "RGBA" else "jpeg"


def encode_image_to_base64(image: Any) -> tuple[str, str]:
    """Encode a PIL Image to a base64 string with MIME type.

    RGBA images requested as JPEG are automatically converted to RGB
    before encoding.

    Args:
        image: A PIL Image instance.

    Returns:
        A tuple of ``(base64_string, mime_type)`` where mime_type is
        e.g. ``"image/jpeg"`` or ``"image/png"``.

    Raises:
        ImportError: If Pillow is not installed.
    """
    _require_pillow()
    fmt: str = _detect_image_format(image)
    save_format: str = fmt.upper()
    if save_format == "JPEG" and image.mode == "RGBA":
        image = image.convert("RGB")
    buffer: io.BytesIO = io.BytesIO()
    image.save(buffer, format=save_format)
    b64_string: str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime_type: str = f"image/{fmt}"
    return b64_string, mime_type


def batch_has_images(rows: list[dict[str, Any]]) -> bool:
    """Check whether any row in a batch contains a PIL Image.

    Args:
        rows: The input data rows as dictionaries.

    Returns:
        ``True`` if at least one value in any row is a PIL Image.
    """
    return any(is_pil_image(v) for row in rows for v in row.values())


@dataclass(frozen=True)
class _ImageRef:
    """Reference to an extracted image from a data row.

    Attributes:
        row_id: The tagged row ID the image came from.
        field_name: The dictionary key that held the image.
        base64_data: Base64-encoded image data.
        mime_type: MIME type string (e.g. ``"image/jpeg"``).
    """

    row_id: int
    field_name: str
    base64_data: str
    mime_type: str


def extract_images_from_rows(
    tagged_rows: list[_TaggedRow],
) -> tuple[list[_TaggedRow], list[_ImageRef]]:
    """Extract PIL images from tagged rows, replacing them with placeholders.

    Each image value is replaced with a string placeholder of the form
    ``"[image: field_name]"`` in the returned row data. The extracted
    images are returned as ``_ImageRef`` objects for downstream
    multimodal message construction.

    Args:
        tagged_rows: Input rows tagged with positional row IDs.

    Returns:
        A tuple of ``(cleaned_rows, image_refs)`` where cleaned_rows
        have images replaced with placeholders and image_refs contains
        the extracted image data.
    """
    cleaned_rows: list[_TaggedRow] = []
    image_refs: list[_ImageRef] = []

    for row in tagged_rows:
        new_data: dict[str, Any] = {}
        for key, value in row.data.items():
            if is_pil_image(value):
                b64_data, mime_type = encode_image_to_base64(value)
                image_refs.append(
                    _ImageRef(
                        row_id=row.row_id,
                        field_name=key,
                        base64_data=b64_data,
                        mime_type=mime_type,
                    )
                )
                new_data[key] = f"[image: {key}]"
            else:
                new_data[key] = value
        cleaned_rows.append(_TaggedRow(row_id=row.row_id, data=new_data))

    return cleaned_rows, image_refs
