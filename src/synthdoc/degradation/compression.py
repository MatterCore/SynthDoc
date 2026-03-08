"""JPEG compression artifact simulation."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def apply_jpeg_compression(
    image: Image.Image, quality_min: int, rng: np.random.Generator
) -> Image.Image:
    """Simulate JPEG compression artifacts by re-encoding at low quality.

    Args:
        image: Input PIL Image.
        quality_min: Minimum JPEG quality (1-100).
        rng: NumPy random generator.

    Returns:
        Re-compressed PIL Image with JPEG artifacts.
    """
    quality = int(rng.integers(max(1, quality_min), min(quality_min + 20, 96)))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")

    # Force the image data to be loaded before closing buffer
    compressed.load()
    return compressed
